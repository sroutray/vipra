import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torchvision
from accelerate import Accelerator
from accelerate.utils import DistributedType
from einops import rearrange
from ema_pytorch import EMA
from model.latent_action_quantization import LAQ
from model.utils import default, exists
from torch import nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def cycle(dl: DataLoader, skipped_dl: Optional[DataLoader] = None):
    if skipped_dl is not None:
        for data in skipped_dl:
            yield data
    while True:
        for data in dl:
            yield data


class LAQTrainer(nn.Module):
    def __init__(
        self,
        model: LAQ,
        accelerator: Accelerator,
        dataset: Dataset,
        num_train_steps: int,
        results_folder: str,
        batch_size: int = 32,
        val_dataset: Optional[Dataset] = None,
        val_batch_size: Optional[int] = 4,
        lr: float = 1e-4,
        pretrained_init_lr_mult_factor: float = 0.1,
        weight_decay: float = 0.0,
        grad_accum_every: int = 1,
        max_grad_norm: float = 1.0,
        use_ema: bool = False,
        ema_update_every: int = 10,
        ema_beta: float = 0.9999,
        save_model_every: int = 1000,
        save_milestone_every: int = 10000,
        val_every_n_steps: int = 1000,
        num_val_batches_to_log: int = 5,
        num_val_samples_to_save: int = 4,
        num_workers: int = 4,
        prefetch_factor: int = 4,
        pin_memory: bool = True,
        resume_checkpoint_path: Optional[str] = None,
        milestone_optim_state: bool = True,
        wandb_kwargs: dict = {},
    ):
        super().__init__()
        self.accelerator = accelerator

        # wandb config
        config = {}
        arguments = locals()
        for key in arguments.keys():
            if key not in [
                "self",
                "config",
                "__class__",
                "model",
                "wandb_kwargs",
                "val_dataset",
                "dataset",
                "accelerator",
            ]:
                config[key] = arguments[key]

        # Merge model hyperparameters into config
        if hasattr(model, "__dict__"):
            model_config = {
                k: v
                for k, v in vars(model).items()
                if isinstance(v, (int, float, str, bool))  # filter to avoid large tensors or objects
            }
            config.update(model_config)

        wandb_kwargs["wandb"]["config"] = config
        self.accelerator.init_trackers(project_name="vipra-laq", config=config, init_kwargs=wandb_kwargs)
        if self.accelerator.is_main_process:
            logger.info(f"Config:\n{config}")

        self.model = model
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)

        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            prefetch_factor=prefetch_factor,
        )

        self.val_dataloader = None
        if exists(val_dataset):
            effective_val_batch_size = default(val_batch_size, batch_size)
            self.val_dataloader = DataLoader(
                val_dataset,
                batch_size=effective_val_batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=True,
            )

        self.lr = lr
        self.pretrained_init_lr_mult_factor = pretrained_init_lr_mult_factor
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.AdamW(
            model.get_trainable_parameters(
                lr,
                pretrained_init_keywords=["enc_spatial_transformer"],
                pretrained_init_lr_mult_factor=pretrained_init_lr_mult_factor,
            ),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Prepare with accelerator
        # Exclude val_dataloader from preparation
        self.model, self.optimizer, self.dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.dataloader
        )

        self.grad_accum_every = grad_accum_every
        self.max_grad_norm = max_grad_norm
        self.save_model_every = save_model_every
        self.save_milestone_every = save_milestone_every
        self.milestone_optim_state = milestone_optim_state
        self.val_every_n_steps = val_every_n_steps
        self.num_val_batches_to_log = num_val_batches_to_log
        self.num_val_samples_to_save = min(num_val_samples_to_save, default(val_batch_size, batch_size))

        self.num_train_steps = num_train_steps
        self.current_step = 0
        self.current_val_step = 0

        self.use_ema = use_ema
        if self.use_ema:
            model_to_ema = self.accelerator.unwrap_model(self.model)
            self.ema_model = EMA(model_to_ema, beta=ema_beta, update_every=ema_update_every)
        else:
            self.ema_model = None

        self.resume_checkpoint_path = resume_checkpoint_path
        skipped_dl_for_resume = None
        skipped_val_dl_for_resume = None
        # Checkpoint resumption (Map style dataset)
        if self.resume_checkpoint_path is not None and Path(self.resume_checkpoint_path).exists():
            self.load(self.resume_checkpoint_path)
            if self.current_step > 0:
                logger.info(f"Map style resuming training dataloader from step {self.current_step}...")
                skipped_dl_for_resume = self.maybe_skip_batches_for_resume(self.current_step, self.dataloader)
            if self.val_dataloader is not None and self.current_val_step > 0:
                logger.info(f"Map style resuming validation dataloader from step {self.current_val_step}...")
                skipped_val_dl_for_resume = self.maybe_skip_batches_for_resume(
                    self.current_val_step, self.val_dataloader
                )

        self.dl_iter = cycle(self.dataloader, skipped_dl_for_resume)
        if self.val_dataloader is not None:
            self.val_dl_iter = cycle(self.val_dataloader, skipped_val_dl_for_resume)
        else:
            self.val_dl_iter = None

        self.accelerator.wait_for_everyone()

    def maybe_skip_batches_for_resume(self, step: int, dataloader: DataLoader) -> Optional[DataLoader]:
        """
        Skip batches in the dataloader based on current step and grad accumulation,
        used to resume from a checkpoint in both training and validation.
        """
        num_batches_processed = step * self.grad_accum_every
        num_batches_to_skip = num_batches_processed % len(dataloader)

        if num_batches_to_skip > 0:
            skipped_loader = self.accelerator.skip_first_batches(dataloader, num_batches_to_skip)
            logger.info(f"Resuming: Skipping {num_batches_to_skip} batches from dataloader.")
            return skipped_loader
        else:
            return None

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def load(self, path: str):
        p = Path(path)
        if not p.exists():
            logger.info(f"Checkpoint not found at {str(p)}, starting from scratch.")
            return

        try:
            logger.info(f"Loading checkpoint from {str(p)}...")
            data = torch.load(p, map_location="cpu")

            model_to_load = self.accelerator.unwrap_model(self.model)

            # Handle potential mismatch if model was saved directly vs. via get_state_dict
            if "model" in data:
                model_state = data["model"]
            elif "module" in data:  # If it was a DDP model state_dict
                model_state = data["module"]
            else:
                model_state = data  # Assume entire data is model state if 'model' key is missing

            msg = model_to_load.load_state_dict(model_state, strict=False)
            logger.info(f"Model loaded with message: {msg}")

            if "optimizer" in data:
                self.optimizer.load_state_dict(data["optimizer"])
            else:
                logger.info("Warning: Optimizer state not found in checkpoint.")

            self.current_step = int(data.get("steps", data.get("step", 0))) + 1
            self.current_val_step = int(
                data.get("val_steps", data.get("val_step", self.current_step // self.val_every_n_steps))
            )

            if self.use_ema and self.ema_model is not None and "ema_model" in data:
                self.ema_model.load_state_dict(data["ema_model"])

            logger.info(f"Resumed training from checkpoint {str(p)} at step {self.current_step}")

        except Exception as e:
            logger.error(f"Failed to load checkpoint from {str(p)}: {e}. Starting from scratch.")
            self.current_step = 0

    def save(self, path: str, is_milestone: bool = False):
        if not self.is_main:
            return

        p = Path(path)
        logger.info(f"Saving checkpoint to {str(p)} at step {self.current_step}...")

        save_data = {
            "model": self.accelerator.get_state_dict(self.model),
            "steps": self.current_step,
            "val_steps": self.current_val_step,
        }

        if not is_milestone or (is_milestone and self.milestone_optim_state):
            save_data["optimizer"] = self.optimizer.state_dict()

        if self.use_ema and self.ema_model is not None:
            save_data["ema_model"] = self.ema_model.state_dict()

        try:
            # Safe atomic overwrite for non-milestone checkpoints
            tmp_path = p.with_suffix(p.suffix + ".tmp")

            self.accelerator.save(save_data, tmp_path)  # Write to temp first
            tmp_path.replace(p)  # Atomically replace existing file

            logger.info(f"Checkpoint saved successfully to {str(p)}.")
        except Exception as e:
            logger.error(f"Failed to save checkpoint to {str(p)}: {e}")
            if tmp_path.exists():
                tmp_path.unlink()  # Clean up partial file

    def train_step(self):
        self.model.train()
        total_loss_value_accum = 0.0

        for i in range(self.grad_accum_every):
            is_last_accum_step = i == self.grad_accum_every - 1

            with self.accelerator.accumulate(self.model):
                batch_data = next(self.dl_iter)
                videos, mask = batch_data[0], batch_data[1]

                loss, logs_dict = self.model(videos, mask, step=self.current_step)

                # Check for any NaNs in loss tensor
                if torch.isnan(loss).any():
                    logger.warning(
                        f"NaN loss detected at step {self.current_step}. Skipping gradient update for this batch."
                    )
                    self.accelerator.skip_gradient_allreduce = True  # Skip allreduce for this step
                    # Do not accumulate loss if NaN to avoid polluting average
                    # Ensure backward is not called on NaN loss
                    # The loop will continue to next accumulation step or finish
                    if is_last_accum_step:  # If it's the step where optimizer_step happens
                        self.optimizer.zero_grad()  # Still zero grad to prevent issues on next real step
                    continue  # Skip backward and loss accumulation for this specific micro-batch

                loss_to_backward = loss / self.grad_accum_every
                self.accelerator.backward(loss_to_backward)

                total_loss_value_accum += loss_to_backward.item()

        grad_norm_val = 0.0
        if self.max_grad_norm is not None:
            grad_norm_val = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            grad_norm_val = grad_norm_val.item()

        self.optimizer.step()

        # Logging
        log_payload = {"step": self.current_step}
        # Add all scalar losses from logs_dict
        for key, value in logs_dict.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                log_payload[key] = value.item()
            elif isinstance(value, (int, float)):
                log_payload[key] = value

        # Add accumulated loss if not already in logs_dict or if different
        if "total_loss_accumulated_step" not in log_payload:  # Use a distinct name
            log_payload["total_loss_accumulated_step"] = total_loss_value_accum

        # Parameter and Gradient Norms
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        param_norm = torch.norm(
            torch.stack([torch.norm(p.detach().float(), 2.0) for p in unwrapped_model.parameters() if p.requires_grad])
        ).item()

        log_payload["param_norm"] = param_norm
        log_payload["grad_norm"] = grad_norm_val

        self.optimizer.zero_grad()

        # EMA Update on main process only
        if self.use_ema and self.ema_model is not None:
            if self.is_main:
                self.ema_model.update()

        return total_loss_value_accum, log_payload

    @torch.no_grad()
    def run_validation_and_log(self, train_step: int):
        if self.val_dl_iter is None:
            return {}

        logger.info(f"Running validation step {self.current_val_step} at training step {train_step}...")
        model_for_eval = self.ema_model if self.use_ema and self.ema_model else self.model
        model_for_eval = self.accelerator.unwrap_model(model_for_eval)
        model_for_eval.eval()

        total_val_loss = 0.0
        all_val_logs = {}  # To average all reported losses

        first_batch_videos = None
        first_batch_recons = None

        batch_idx = 0
        while batch_idx < self.num_val_batches_to_log:
            val_batch_data = next(self.val_dl_iter)
            videos, mask = val_batch_data[0], val_batch_data[1]

            # Move validation data to the correct device and dtype
            videos = videos.to(self.device)
            mask = mask.to(self.device)

            val_loss, val_logs_dict = model_for_eval(videos, mask, step=train_step)

            # Store first batch for visualization
            if batch_idx == 0 and self.num_val_samples_to_save > 0:
                recon_videos = model_for_eval.inference(videos, return_reconstructions=True)["recon_videos"]

                batch_size = videos.shape[0]
                num_to_save = min(batch_size, self.num_val_samples_to_save)

                # Randomly permute the indices (on the same device as videos)
                indices = torch.randperm(batch_size, device=videos.device)[:num_to_save]

                # Use the shuffled indices to index into your tensors
                first_batch_videos = videos[indices, ...].cpu()
                first_batch_recons = torch.cat([videos[indices, 0:1, ...], recon_videos[indices, ...]], dim=1).cpu()

            for key, value in val_logs_dict.items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    all_val_logs[key] = all_val_logs.get(key, 0.0) + value.item()
                elif isinstance(value, (int, float)):
                    all_val_logs[key] = all_val_logs.get(key, 0.0) + value
            total_val_loss += val_loss.item()

            batch_idx += 1

        # Average losses
        avg_val_logs = {}
        if batch_idx > 0:  # Ensure at least one batch was processed
            for key, value in all_val_logs.items():
                avg_val_logs[f"val/{key}"] = value / batch_idx
            avg_val_logs["val/total_loss_avg"] = total_val_loss / batch_idx
        avg_val_logs["val/train_step"] = train_step
        avg_val_logs["val/val_step"] = self.current_val_step

        logger.info(f"Validation at step {train_step}: {avg_val_logs}")

        # Save reconstruction grid
        if first_batch_videos is not None and first_batch_recons is not None:
            self.save_reconstruction_grid(first_batch_videos, first_batch_recons, train_step)

        self.current_val_step += 1
        return avg_val_logs

    def save_reconstruction_grid(self, gt_videos: torch.Tensor, recon_videos: torch.Tensor, current_train_step: int):
        if not self.is_main:
            return

        num_saved, T_recon, C_img, H, W = gt_videos.shape

        gt_videos = torch.clamp(gt_videos, 0.0, 1.0)
        recon_videos = torch.clamp(recon_videos, 0.0, 1.0)

        output_strips = []
        for i in range(num_saved):
            # Concatenate all time steps horizontally for GT
            gt_strip = torch.cat([gt_videos[i, t] for t in range(T_recon)], dim=2)  # (C, H, T*W)
            # Concatenate all time steps horizontally for Recon
            recon_strip = torch.cat([recon_videos[i, t] for t in range(T_recon)], dim=2)  # (C, H, T*W)
            # Stack GT strip above Recon strip
            comparison_strip = torch.cat((gt_strip, recon_strip), dim=1)  # (C, 2*H, T*W)
            output_strips.append(comparison_strip)

        # Make a grid of these comparison strips (num_saved rows, 1 col)
        # Add padding between samples if desired
        grid = torchvision.utils.make_grid(output_strips, nrow=1, padding=5, pad_value=0.5)

        try:
            save_path = self.results_folder / f"reconstructions_step_{current_train_step}.png"
            torchvision.utils.save_image(grid, save_path)
            logger.info(f"Saved reconstruction grid to {save_path}")

            # Optionally log to WandB if an image logger is configured
            # if self.accelerator.trackers:
            #      try:
            #          wandb_tracker = self.accelerator.get_tracker("wandb", unwrap=True)
            #          wandb_tracker.log({"val/reconstructions": wandb_tracker.Image(save_path)}, step=current_train_step)
            #      except Exception as e:
            #          self.print(f"Could not log reconstructions to WandB: {e}")

        except Exception as e:
            logger.error(f"Error saving reconstruction grid: {e}")

    def train(self):
        logger.info(f"Starting training from step {self.current_step} up to {self.num_train_steps} steps.")

        while self.current_step < self.num_train_steps:
            avg_loss_this_step, logs_train = self.train_step()

            logger.info(f"Step {self.current_step}/{self.num_train_steps}: {logs_train}")

            # Validation must run on main process
            logs_val = {}
            if self.val_dataloader is not None and self.current_step % self.val_every_n_steps == 0:
                logs_val = self.run_validation_and_log(self.current_step)

            if self.is_main:
                # Overwrite the latest checkpoint
                if self.current_step % self.save_model_every == 0:
                    self.save(self.results_folder / "laq_model_latest.pt")

                # Milestone: keep permanent
                if self.current_step % self.save_milestone_every == 0:
                    self.save(self.results_folder / f"laq_model_milestone_{self.current_step}.pt", is_milestone=True)
            # merge logs for training and validation
            if len(logs_val) > 0:
                logs = {**logs_train, **logs_val}
            else:
                logs = logs_train
            self.accelerator.log(logs)

            self.current_step += 1
            self.accelerator.wait_for_everyone()

        if self.is_main:
            self.save(self.results_folder / "laq_model_final.pt", is_milestone=True)
            logger.info("Training complete.")
            if self.accelerator.trackers:
                self.accelerator.end_training()
