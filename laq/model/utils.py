import logging
import sys
from typing import Any, Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator


def exists(val: Any) -> bool:
    return val is not None


def default(val: Optional[Any], d: Any) -> Any:
    return val if exists(val) else d


def pair(val: Any) -> Tuple[Any, Any]:
    ret = (val, val) if not isinstance(val, tuple) else val
    assert len(ret) == 2
    return ret


def leaky_relu(p: float = 0.1) -> nn.LeakyReLU:
    return nn.LeakyReLU(p)


def l2norm(t: torch.Tensor) -> torch.Tensor:
    return F.normalize(t, dim=-1)


def get_vq_encoder(code_seq_len: int, quant_dim: int) -> nn.Sequential:
    """
    Returns a VQ encoder that processes a 3D (T, H, W) input grid.
    The encoder uses 3D convolutions to reduce only the spatial dimensions (H, W),
    while preserving the temporal dimension (T).

    Assumes a spatial input of 16x16.
    """
    if code_seq_len == 4:
        # Spatial: 16x16 -> 8x8 -> 4x4 -> 2x2. Temporal: T -> T -> T -> T
        return nn.Sequential(
            nn.Conv3d(quant_dim, quant_dim, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            leaky_relu(),
            nn.Conv3d(quant_dim, quant_dim, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            leaky_relu(),
            nn.Conv3d(quant_dim, quant_dim, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
        )
    elif code_seq_len == 16:
        # Spatial: 16x16 -> 8x8 -> 4x4. Temporal: T -> T -> T
        return nn.Sequential(
            nn.Conv3d(quant_dim, quant_dim, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            leaky_relu(),
            nn.Conv3d(quant_dim, quant_dim, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
        )
    elif code_seq_len == 64:
        # Spatial: 16x16 -> 8x8 -> 8x8. Temporal: T -> T -> T
        return nn.Sequential(
            nn.Conv3d(quant_dim, quant_dim, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            leaky_relu(),
            nn.Conv3d(quant_dim, quant_dim, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
        )
    elif code_seq_len == 256:
        # Spatial: 16x16 -> 16x16 -> 16x16. Temporal: T -> T -> T
        return nn.Sequential(
            nn.Conv3d(quant_dim, quant_dim, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            leaky_relu(),
            nn.Conv3d(quant_dim, quant_dim, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
        )
    else:
        raise ValueError(f"Unsupported code sequence length: {code_seq_len}. Supported are 4, 16, 64, 256.")


class RankFilter(logging.Filter):
    """
    This filter only allows messages from the main process.
    """

    def __init__(self, accelerator: Accelerator):
        super().__init__()
        self.accelerator = accelerator

    def filter(self, record):
        return self.accelerator.is_main_process


def setup_logging(accelerator: Accelerator):
    """
    Configures a minimal root logger for distributed training
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RankFilter(accelerator))

    formatter = logging.Formatter("[%(levelname)s] [%(name)s] - %(message)s")
    handler.setFormatter(formatter)

    root_logger.addHandler(handler)


class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HW = pair(img_size)
        patch_HW = pair(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size

        assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x
