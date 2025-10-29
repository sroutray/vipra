import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import lpips
import torch
import torch.nn.functional as F
from beartype import beartype
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from model.attention import ContinuousPositionBias, FeedForward, STTransformer, Transformer
from model.nsvq import NSVQ
from model.utils import PatchEmbed, exists, get_vq_encoder, leaky_relu, pair
from model.vit import DINOv2Encoder, VisionTransformerEncoder
from torch import nn
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

logger = logging.getLogger(__name__)


class LAQ(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        quant_dim: int,
        codebook_size: int,
        image_size: Union[int, Tuple[int, int]],
        patch_size: Union[int, Tuple[int, int]],
        enc_depth: int,
        dec_depth: int,
        dim_head: int = 64,
        heads: int = 8,
        channels: int = 3,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        code_seq_len: int = 1,
        encode_deltas: bool = True,
        discarding_threshold: float = 0.1,
        max_codebook_update_step: int = 10000,
        spatial_enc_type: str = "dino_base",  # ["dino_base", "vit_base"]
        use_lpips_loss: bool = True,
        lpips_loss_weight: float = 1.0,
        use_flow_loss: bool = True,
        flow_loss_weight: float = 1.0,
        flow_loss_kickin_step: int = 0,
        flow_loss_warmup_steps: int = 10_000,
    ) -> None:
        super().__init__()

        self.code_seq_len = code_seq_len
        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        patch_height, patch_width = self.patch_size
        self.patch_grid = (self.image_size[0] // patch_height, self.image_size[1] // patch_width)
        self.channels = channels

        action_h = int(math.sqrt(self.code_seq_len))
        action_w = self.code_seq_len // action_h
        self.action_size = (action_h, action_w)

        self.enc_spatial_rel_pos_bias = ContinuousPositionBias(dim=dim, heads=heads, num_dims=2)
        self.dec_spatial_rel_pos_bias = ContinuousPositionBias(dim=dim, heads=heads, num_dims=2)

        image_height, image_width = self.image_size
        assert (image_height % patch_height) == 0 and (image_width % patch_width) == 0

        enc_st_transformer_kwargs = dict(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            causal=False,
            peg=True,
            peg_causal=False,
        )

        dec_st_transformer_kwargs = dict(
            dim=dim,
            dim_cond=dim,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            causal=True,
            peg=True,
            peg_causal=True,
            enable_conditioning=True,
        )

        spatial_enc_type_parts = spatial_enc_type.split("_")
        vit_size = "_".join(spatial_enc_type_parts[1:])
        self.spatial_enc_type = spatial_enc_type_parts[0]

        if self.spatial_enc_type == "dino":
            self.enc_spatial_transformer = DINOv2Encoder(
                image_size=image_size,
                patch_size=patch_size,
                vit_size=vit_size,
            )
        elif self.spatial_enc_type == "vit":
            self.enc_spatial_transformer = VisionTransformerEncoder(
                image_size=image_size,
                patch_size=patch_size,
                vit_size=vit_size,
            )
        else:
            raise ValueError("Invalid spatial_enc_type. Choose 'dino' or 'vit'.")

        self.enc_st_transformer = STTransformer(depth=enc_depth, **enc_st_transformer_kwargs)

        # Quantization
        self.encode_deltas = encode_deltas
        self.vq_project_in = nn.Linear(dim, quant_dim)
        self.vq_encoder = get_vq_encoder(code_seq_len, quant_dim)
        self.vq = NSVQ(
            num_embeddings=codebook_size,
            embedding_dim=quant_dim,
            discarding_threshold=discarding_threshold,
        )
        self.max_codebook_update_step = max_codebook_update_step

        self.vq_to_cond = nn.Sequential(
            Rearrange("b t hq wq d -> b t (hq wq d)", hq=action_h, wq=action_w, d=quant_dim),
            nn.Linear(quant_dim * code_seq_len, dim),
            FeedForward(dim, mult=4.0, dropout=ff_dropout),
        )
        self.dec_patchify = PatchEmbed(image_size, patch_size, channels, dim, norm_layer=nn.LayerNorm)
        fusion_hidden_dim = int(8 * (2 / 3) * dim)
        self.fusion_layer = nn.Sequential(
            nn.Linear(2 * dim, fusion_hidden_dim), leaky_relu(), nn.Linear(fusion_hidden_dim, dim)
        )
        self.dec_transformer = STTransformer(depth=dec_depth, **dec_st_transformer_kwargs)
        self.to_pixels = nn.Sequential(
            nn.Linear(dim, channels * patch_width * patch_height),
            Rearrange(
                "b t (h w) (c p1 p2) -> b t c (h p1) (w p2)",
                p1=patch_height,
                p2=patch_width,
                h=self.patch_grid[0],
                w=self.patch_grid[1],
            ),
        )

        # Perceptual loss
        self.use_lpips_loss = use_lpips_loss
        self.lpips_loss_weight = lpips_loss_weight
        if self.use_lpips_loss:
            self.lpips = lpips.LPIPS(net="vgg").eval().requires_grad_(False)

        # Optical flow loss
        self.use_flow_loss = use_flow_loss
        self.flow_loss_weight = flow_loss_weight
        self.flow_loss_kickin_step = flow_loss_kickin_step
        self.flow_loss_warmup_steps = flow_loss_warmup_steps
        if self.use_flow_loss:
            self.flow_model = (
                raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2, progress=False).eval().requires_grad_(False)
            )

    def forward(
        self,
        videos: torch.Tensor,  # (B, T, C, H, W)
        mask: torch.BoolTensor = None,  # (B, T)
        step: int = 0,  # for logging
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        B, T, C, H, W = videos.shape
        device = videos.device
        mask = mask if mask is not None else torch.ones((B, T), device=device, dtype=torch.bool)
        mask_gt = mask[:, :-1]  # (B, T-1)
        mask_recon = mask[:, 1:]  # (B, T-1)
        mask_loss = torch.logical_and(mask_gt, mask_recon)  # (B, T-1)

        # Inverse model encoder
        frame_tokens, tokens = self.encode(videos)  # (B, T, Hp, Wp, D)

        if self.encode_deltas:
            vq_inputs = tokens[:, 1:] - tokens[:, :-1]
        else:
            vq_inputs = tokens[:, 1:]

        # Inverse model quantizer
        quantized_actions, perplexity, num_unique_indices = self.quantize(vq_inputs)

        # Scheduled codebook replacement
        # if (
        #     step > 0
        #     and step < self.max_codebook_update_step
        #     and (
        #         (100 <= step <= 1000 and step % 100 == 0)
        #         or (1000 < step <= 10_000 and step % 1000 == 0)
        #         or (10_000 < step <= 20_000 and step % 2000 == 0)
        #         or (20_000 < step <= 90_000 and step % 3000 == 0)
        #         or (step > 90_000 and step % 10_000 == 0)
        #     )
        #     and self.training
        # ):
        if (
            step > 0
            and step < self.max_codebook_update_step
            and (
                (100 <= step <= 1000 and step % 100 == 0)
                or (1000 < step <= 10_000 and step % 1000 == 0)
                or (10_000 < step <= 40_000 and step % 2000 == 0)
                or (40_000 < step <= 90_000 and step % 3000 == 0)
                or (step > 90_000 and step % 5_000 == 0)
            )
            and self.training
        ):
            self.vq.replace_unused_codebooks()

        # Forward model decoder
        recon_videos = self.decode(videos[:, :-1], quantized_actions)
        gt_videos = videos[:, 1:]

        rec_loss = self.compute_reconstruction_loss(recon_videos, gt_videos, mask_loss)

        lpips_loss = 0.0
        if self.use_lpips_loss:
            lpips_loss = self.compute_lpips_loss(recon_videos, gt_videos, mask_loss)

        flow_loss = 0.0
        flow_loss_weight = 0.0
        if self.use_flow_loss and step >= self.flow_loss_kickin_step:
            flow_loss = self.compute_flow_loss(recon_videos, gt_videos, mask_loss)
            flow_loss_weight = self.compute_flow_loss_weight(step)

        loss = rec_loss + self.lpips_loss_weight * lpips_loss + flow_loss_weight * flow_loss

        log_dict = {
            "loss": loss.item() if isinstance(loss, torch.Tensor) else loss,
            "rec_loss": rec_loss.item() if isinstance(rec_loss, torch.Tensor) else rec_loss,
            "lpips_loss": lpips_loss.item() if isinstance(lpips_loss, torch.Tensor) else lpips_loss,
            "flow_loss": flow_loss.item() if isinstance(flow_loss, torch.Tensor) else flow_loss,
            "perplexity": perplexity.item() if isinstance(perplexity, torch.Tensor) else perplexity,
            "num_unique_indices": (
                num_unique_indices.item() if isinstance(num_unique_indices, torch.Tensor) else num_unique_indices
            ),
        }

        return loss, log_dict

    def encode(self, videos: torch.Tensor, mask: torch.BoolTensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        device = videos.device
        B = videos.shape[0]
        Hp, Wp = self.patch_grid

        # Feature extraction
        videos_unrolled = rearrange(videos, "b t c h w -> (b t) c h w")
        frame_tokens, tokens = self.enc_spatial_transformer(videos_unrolled)

        frame_tokens = rearrange(frame_tokens, "(b t) (hp wp) d -> b t hp wp d", b=B, hp=Hp, wp=Wp)
        tokens = rearrange(tokens, "(b t) (hp wp) d -> b t (hp wp) d", b=B, hp=Hp, wp=Wp)

        # Spatio-temporal encoding
        tokens_shape = tuple(frame_tokens.shape[:-1])  # (B, T, Hp, Wp)
        attn_bias = self.enc_spatial_rel_pos_bias(Hp, Wp, device=device, dtype=tokens.dtype)
        tokens = self.enc_st_transformer(
            tokens, tokens_shape, spatial_attn_bias=attn_bias, attn_mask=mask
        )  # (B, T, Hp*Wp, D)
        tokens = rearrange(tokens, "b t (hp wp) d -> b t hp wp d", b=B, hp=Hp, wp=Wp)

        return frame_tokens, tokens

    def quantize(
        self, tokens: torch.Tensor, inference_mode: bool = False
    ) -> Union[Tuple[torch.Tensor, float, int], Tuple[torch.Tensor, torch.Tensor]]:
        B = tokens.shape[0]
        Hp, Wp = self.patch_grid
        Hq, Wq = self.action_size

        tokens = self.vq_project_in(tokens)  # (B, T, Hp, Wp, D)
        tokens = rearrange(tokens, "b t hp wp d -> b d t hp wp", b=B, hp=Hp, wp=Wp)
        tokens = self.vq_encoder(tokens)  # (B, D, T, Hq, Wq)

        tokens = rearrange(tokens, "b d t hq wq -> (b t hq wq) d", b=B, hq=Hq, wq=Wq)

        if inference_mode:
            quantized_actions, quantized_action_idxs = self.vq.inference(tokens)

            quantized_actions = rearrange(quantized_actions, "(b t hq wq) d -> b t hq wq d", b=B, hq=Hq, wq=Wq)
            quantized_action_idxs = rearrange(quantized_action_idxs, "(b t hq wq) -> b t hq wq", b=B, hq=Hq, wq=Wq)

            return quantized_actions, quantized_action_idxs

        quantized_actions, perplexity, num_unique_indices = self.vq(tokens)  # (B*T*Hq*Wq, D)

        quantized_actions = rearrange(quantized_actions, "(b t hq wq) d -> b t hq wq d", b=B, hq=Hq, wq=Wq)

        return quantized_actions, perplexity, num_unique_indices

    def decode(
        self, videos: torch.Tensor, quantized_actions: torch.Tensor, mask_recon: torch.BoolTensor = None
    ) -> torch.Tensor:
        device = videos.device
        B, Tm1 = videos.shape[0], videos.shape[1]
        Hp, Wp = self.patch_grid

        videos = rearrange(videos, "b t c h w -> (b t) c h w", b=B)
        patch_tokens = self.dec_patchify(videos)  # (B*Tm1, Hp*Wp, D)

        # VQ to condition
        cond = self.vq_to_cond(quantized_actions)  # (B, Tm1, D)

        # Concatenate condition as additional spatial token
        tokens = torch.cat([patch_tokens, rearrange(cond, "b t d -> (b t) 1 d")], dim=1)  # (B*Tm1, Hp*Wp + 1, D)
        tokens = rearrange(tokens, "(b t) np d -> b t np d", b=B, t=Tm1, np=Hp * Wp + 1)  # (B, Tm1, Np + 1, D)
        videos_shape = (B, Tm1, Hp, Wp)

        # Decoder
        attn_bias = self.dec_spatial_rel_pos_bias(Hp, Wp, device=device, dtype=tokens.dtype)  # (h, Np, Np)
        attn_bias = F.pad(attn_bias, (0, 1, 0, 1), value=0.0)  # (h, Np + 1, Np + 1)
        tokens = self.dec_transformer(
            tokens,
            videos_shape,
            cond=cond,
            spatial_attn_bias=attn_bias,
            attn_mask=mask_recon,
        )  # (B, Tm1, Np + 1, D)

        visual_tokens = tokens[:, :, :-1, :]
        recon_videos = self.to_pixels(visual_tokens)  # (B, Tm1, C, H, W)

        return recon_videos

    def compute_reconstruction_loss(
        self, recon_videos: torch.Tensor, gt_videos: torch.Tensor, mask_loss: torch.BoolTensor
    ) -> torch.Tensor:
        """
        Computes L1 reconstruction loss between the reconstructed and ground truth videos.
        """
        B, Tm1, C, H, W = recon_videos.shape
        l1_loss = F.l1_loss(recon_videos, gt_videos, reduction="none")
        if exists(mask_loss):
            mult_mask = mask_loss[..., None, None, None].to(l1_loss.dtype)  # (B, T, 1, 1, 1)
            denom = (mask_loss.sum() * C * H * W).clamp_min(1).to(l1_loss.dtype)
            return (l1_loss * mult_mask).sum() / denom
        else:
            return l1_loss.mean()

    def compute_lpips_loss(
        self, recon_videos: torch.Tensor, gt_videos: torch.Tensor, mask_loss: torch.BoolTensor
    ) -> torch.Tensor:
        """
        Computes LPIPS loss between the reconstructed and ground truth videos.
        """
        B, T, C, H, W = recon_videos.shape

        flat_recon_videos = rearrange(recon_videos, "b t c h w -> (b t) c h w")
        flat_gt_videos = rearrange(gt_videos, "b t c h w -> (b t) c h w")

        # Scale to [-1, 1] range for LPIPS
        flat_recon_videos = 2 * flat_recon_videos - 1
        flat_gt_videos = 2 * flat_gt_videos - 1

        lpips_loss = self.lpips(flat_recon_videos, flat_gt_videos).squeeze()  # (B*T, 1)

        if exists(mask_loss):
            lpips_loss = lpips_loss[rearrange(mask_loss, "b t -> (b t)")]
            denom = mask_loss.sum().clamp_min(1).to(lpips_loss.dtype)
            return lpips_loss.sum() / denom
        else:
            return lpips_loss.mean()

    def compute_flow_loss_weight(self, step: int) -> float:
        if self.flow_loss_warmup_steps > 0:
            warmup_progress = (step - self.flow_loss_kickin_step) / self.flow_loss_warmup_steps
            warmup_progress = min(warmup_progress, 1.0)
            current_flow_weight = warmup_progress * self.flow_loss_weight
        else:
            current_flow_weight = self.flow_loss_weight
        return current_flow_weight

    def get_flow(self, vid0, vid1):
        """
        Computes RAFT flow between two videos of shape (B, C, T, H, W).
        Returns flow of shape (B, 2, T-1, H, W).
        """
        B, C, T, H, W = vid0.shape
        v0 = rearrange(vid0, "b c t h w -> (b t) c h w")
        v1 = rearrange(vid1, "b c t h w -> (b t) c h w")
        # RAFT expects inputs in the range [-1, 1]
        # inputs are alredy in [-1, 1]
        flow = self.flow_model(v0, v1)[-1]
        return rearrange(flow, "(b t) c h w -> b c t h w", b=B)

    def compute_flow_loss(self, recon_videos: torch.Tensor, gt_videos: torch.Tensor, mask_loss: torch.BoolTensor):
        """
        Computes flow loss between the reconstructed and ground truth videos.
        """
        B, T, C, H, W = recon_videos.shape
        recon_videos_flow = rearrange(recon_videos, "b t c h w -> b c t h w")
        gt_videos_flow = rearrange(gt_videos, "b t c h w -> b c t h w")

        # Scale to [-1, 1] range for flow model
        recon_videos_flow = 2 * recon_videos_flow - 1
        gt_videos_flow = 2 * gt_videos_flow - 1

        # GT flow don't need grad
        with torch.no_grad():
            flow_gt_fwd = self.get_flow(gt_videos_flow[:, :, :-1], gt_videos_flow[:, :, 1:])
            flow_gt_bwd = self.get_flow(gt_videos_flow[:, :, 1:], gt_videos_flow[:, :, :-1])

        flow_recon_fwd = self.get_flow(recon_videos_flow[:, :, :-1], recon_videos_flow[:, :, 1:])
        flow_recon_bwd = self.get_flow(recon_videos_flow[:, :, 1:], recon_videos_flow[:, :, :-1])

        flow_loss_fwd = F.l1_loss(flow_recon_fwd, flow_gt_fwd, reduction="none")
        flow_loss_bwd = F.l1_loss(flow_recon_bwd, flow_gt_bwd, reduction="none")
        flow_loss = flow_loss_fwd + flow_loss_bwd  # (B, 2, T-1, H, W)

        if exists(mask_loss):
            flow_mask = torch.logical_and(mask_loss[:, 1:], mask_loss[:, :-1])  # (B, T-1)
            mult_mask = flow_mask[:, None, :, None, None].to(flow_loss.dtype)  # (B, 1, T-1, 1, 1)
            denom = (flow_mask.sum() * 2 * H * W).clamp_min(1).to(flow_loss.dtype)
            return 0.5 * (flow_loss * mult_mask).sum() / denom
        else:
            return 0.5 * flow_loss.mean()

    def state_dict(
        self, destination: Optional[Dict[str, Any]] = None, prefix: str = "", keep_vars: bool = False
    ) -> Dict[str, Any]:
        """
        Override state_dict to exclude LPIPS and flow_model parameters.
        This makes checkpoints more compact by default.
        """
        # Get the original state dict from parent class
        full_state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

        # Filter out LPIPS and flow_model parameters
        filtered_state = {
            k: v for k, v in full_state.items() if not (k.startswith("lpips.") or k.startswith("flow_model."))
        }

        return filtered_state

    def get_trainable_parameters(
        self,
        lr: float,
        no_decay_keywords: List[str] = ["nsvq.", "codebooks"],
        filter_keywords: List[str] = [],
        pretrained_init_keywords: List[str] = [],
        pretrained_init_lr_mult_factor: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Identifies parameters for four groups:
        1. Pre-trained with decay
        2. Pre-trained without decay
        3. From-scratch with decay
        4. From-scratch without decay

        Applies a learning rate multiplier to pre-trained parameters.
        """
        pt_decay, pt_no_decay = [], []
        new_decay, new_no_decay = [], []

        pt_decay_names, pt_no_decay_names = [], []
        new_decay_names, new_no_decay_names = [], []

        # Filter out non-trainable and auxiliary parameters
        excluded_param_names: set[str] = set()
        if hasattr(self, "use_lpips_loss") and self.use_lpips_loss and hasattr(self, "lpips"):
            for name, _ in self.lpips.named_parameters():
                excluded_param_names.add(f"lpips.{name}")
        if hasattr(self, "use_flow_loss") and self.use_flow_loss and hasattr(self, "flow_model"):
            for name, _ in self.flow_model.named_parameters():
                excluded_param_names.add(f"flow_model.{name}")

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if name in excluded_param_names:
                logger.info(f"Excluding {name} from optimizer (LPIPS/Flow).")
                continue

            if any(keyword in name for keyword in filter_keywords):
                logger.info(f"Filtering out {name} from optimizer due to filter_keywords.")
                continue

            # Check 1: Is the parameter pre-trained?
            is_pretrained = any(keyword in name for keyword in pretrained_init_keywords)

            # Check 2: Should the parameter have weight decay?
            # param.ndim == 1 is a good heuristic for biases and norm parameters
            is_no_decay = param.ndim == 1 or any(keyword in name for keyword in no_decay_keywords)

            if is_pretrained:
                if is_no_decay:
                    pt_no_decay.append(param)
                    pt_no_decay_names.append(name)
                else:
                    pt_decay.append(param)
                    pt_decay_names.append(name)
            else:
                if is_no_decay:
                    new_no_decay.append(param)
                    new_no_decay_names.append(name)
                else:
                    new_decay.append(param)
                    new_decay_names.append(name)

        logger.info(f"Pre-trained params with decay: {pt_decay_names}")
        logger.info(f"Pre-trained params without decay: {pt_no_decay_names}")
        logger.info(f"New params with decay: {new_decay_names}")
        logger.info(f"New params without decay: {new_no_decay_names}")

        param_groups = []

        # Pretrained groups with a learning rate multiplier
        pretrain_lr = lr * pretrained_init_lr_mult_factor
        if len(pt_decay) > 0:
            param_groups.append({"params": pt_decay, "lr": pretrain_lr})
        if len(pt_no_decay) > 0:
            param_groups.append({"params": pt_no_decay, "weight_decay": 0.0, "lr": pretrain_lr})

        # From scratch groups with the base learning rate
        if len(new_decay) > 0:
            param_groups.append({"params": new_decay})
        if len(new_no_decay) > 0:
            param_groups.append({"params": new_no_decay, "weight_decay": 0.0})

        return param_groups

    def load_weights(
        self,
        ckpt_path: Union[str, Path],
        map_location: Union[str, torch.device] = "cpu",
        strict: bool = False,
        verbose: bool = True,
    ) -> Optional[nn.modules.module._IncompatibleKeys]:
        """
        Load model parameters from a checkpoint produced by LAQTrainer or
        by `torch.save(model.state_dict(), ...)`.
        """
        ckpt_path = Path(ckpt_path)

        if not ckpt_path.exists():
            if verbose:
                logger.error(f"Checkpoint not found: {ckpt_path}")
            return None

        try:
            if verbose:
                logger.info(f"Loading weights from {ckpt_path} …")

            payload = torch.load(ckpt_path, map_location=map_location)

            # Support several common checkpoint layouts
            if isinstance(payload, dict):
                if "model" in payload:  # trainer.save()
                    state_dict = payload["model"]
                elif "module" in payload:  # DDP wrapper saved
                    state_dict = payload["module"]
                elif "state_dict" in payload:  # lightning style
                    state_dict = payload["state_dict"]
                else:  # raw state_dict
                    state_dict = payload
            else:
                state_dict = payload  # unusual but possible

            incompatible = self.load_state_dict(state_dict, strict=strict)

            if verbose:
                miss, unexp = incompatible.missing_keys, incompatible.unexpected_keys
                logger.info(f"Loaded with " f"{len(miss)} missing / {len(unexp)} unexpected keys.")

            return incompatible

        except Exception as exc:
            if verbose:
                logger.error(f"Failed to load checkpoint: {exc}")
            return None

    @torch.no_grad()
    def inference(
        self,
        videos: torch.Tensor,  # (B, T, C, H, W)
        mask: Optional[torch.BoolTensor] = None,  # (B, T)
        return_reconstructions: bool = True,
        return_quantized_actions: bool = False,
        return_quantized_actions_idxs: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Run inference on the model with the given videos and optional mask.
        Returns reconstructed videos.
        """
        B, T, C, H, W = videos.shape
        device = videos.device
        mask = mask if mask is not None else torch.ones((B, T), device=device, dtype=torch.bool)
        mask_gt = mask[:, :-1]  # (B, T-1)
        mask_recon = mask[:, 1:]  # (B, T-1)
        mask_loss = torch.logical_and(mask_gt, mask_recon)  # (B, T-1)

        # Inverse model encoder
        frame_tokens, tokens = self.encode(videos)  # (B, T, Hp, Wp, D)

        if self.encode_deltas:
            vq_inputs = tokens[:, 1:] - tokens[:, :-1]
        else:
            vq_inputs = tokens[:, 1:]

        # Inverse model quantizer
        quantized_actions, quantized_actions_idxs = self.quantize(vq_inputs, inference_mode=True)

        # Forward model decoder
        recon_videos = self.decode(videos[:, :-1], quantized_actions)
        gt_videos = videos[:, 1:]

        return_dict = {}
        if return_reconstructions:
            return_dict["recon_videos"] = recon_videos
        if return_quantized_actions and exists(quantized_actions):
            return_dict["quantized_actions"] = quantized_actions  # (B, T-1, Hq, Wq, d)
        if return_quantized_actions_idxs:
            return_dict["quantized_actions_idxs"] = quantized_actions_idxs  # (B, T-1, Hq, Wq)

        return return_dict

    @torch.no_grad()
    def rollout(
        self,
        videos: torch.Tensor,  # (B, T-1, C, H, W)
        quantized_actions: torch.Tensor,  # (B, T-1, Hq, Wq, d)
    ):
        recon_videos = self.decode(videos, quantized_actions)  # (B, T-1, C, H, W)
        return recon_videos

    @torch.no_grad()
    def rollout_ar(
        self,
        videos: torch.Tensor,  # (B, T_init, C, H, W)
        quantized_actions: torch.Tensor,  # (B, T_init + T_gen, Hq, Wq, d)
    ):
        T_init = videos.shape[1]
        T_total = quantized_actions.shape[1]
        T_gen = T_total - T_init + 1  # Generate T_gen frames

        recon_videos = []
        history_frames = [videos[:, i] for i in range(T_init)]  # Start with initial frames
        for t in range(T_gen):
            # Stack all history frames
            history_tensor = torch.stack(history_frames, dim=1)  # (B, len(history_frames), C, H, W)

            # Get action and decode
            action_t = quantized_actions[:, : (t + T_init)]  # (B, len(history_frames), Hq, Wq, d)

            next_frame = self.decode(history_tensor, action_t)  # (B, len(history_frames), C, H, W)
            next_frame = next_frame[:, -1:]  # Take the last frame as the next frame
            recon_videos.append(next_frame)
            history_frames.append(next_frame.squeeze(1))  # Add to history

        return torch.cat(recon_videos, dim=1)  # (B, T_gen, C, H, W)
