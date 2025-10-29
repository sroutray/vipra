import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from beartype import beartype
from einops import rearrange, repeat
from model.utils import default, exists, l2norm, leaky_relu
from torch import nn


def precompute_freqs_cis_1d(
    dim: int, seq_len: int, theta: float = 10000.0, scale: float = 1.0, use_cls: bool = False
) -> torch.Tensor:
    """
    Precompute LLaMA-style RoPE "cis" factors for a 1D sequence.

    Args:
        dim (int): Head dimension (must be even)
        seq_len (int): Maximum sequence length
        theta (float): Base for inverse frequency
        scale (float): Position scaling factor
        use_cls (bool): Whether to prepend CLS token row with no rotation

    Returns:
        torch.Tensor: Complex tensor of shape (seq_len + use_cls, dim//2)
    """
    assert dim % 2 == 0, "RoPE dimension must be even"
    half = dim // 2

    # Build inverse frequencies for each dimension pair
    idx = torch.arange(0, half, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (idx / half))

    # Generate scaled positions
    positions = torch.arange(seq_len, dtype=torch.float32) / scale

    # Compute angles for each position-frequency pair
    angles = torch.einsum("i,j->ij", positions, inv_freq)

    # Convert to complex exponentials
    freqs_cis = torch.polar(torch.ones_like(angles), angles)

    # Prepend CLS token (no rotation) if requested
    if use_cls:
        cls_row = torch.ones((1, half), dtype=torch.complex64)
        freqs_cis = torch.cat([cls_row, freqs_cis], dim=0)

    return freqs_cis


def apply_rope_1d(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply 1D LLaMA-style RoPE to query and key tensors.

    Args:
        xq (torch.Tensor): Query tensor of shape (..., seq_len, dim)
        xk (torch.Tensor): Key tensor of shape (..., seq_len, dim)
        freqs_cis (torch.Tensor): Precomputed complex frequencies from precompute_freqs_cis_1d

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Rotated query and key tensors
    """
    assert xq.shape == xk.shape, "Query and key must have the same shape"
    *prefix, seq_len, dim = xq.shape
    assert dim % 2 == 0, "RoPE dimension must be even"
    half = dim // 2

    # Handle CLS token offset if freqs_cis includes it
    if freqs_cis.shape[0] == seq_len + 1:
        freqs_cis = freqs_cis[1:]
    else:
        assert freqs_cis.shape[0] == seq_len, f"freqs_cis shape mismatch: expected {seq_len}, got {freqs_cis.shape[0]}"

    assert freqs_cis.dtype == torch.complex64, "freqs_cis must be complex64"
    assert freqs_cis.shape[1] == half, "freqs_cis dimension mismatch"

    # Split tensors into even and odd dimensions
    xq_even, xq_odd = xq[..., :half], xq[..., half:]
    xk_even, xk_odd = xk[..., :half], xk[..., half:]

    # Extract rotation components
    cos_pair = freqs_cis.real
    sin_pair = freqs_cis.imag

    # Broadcast to match input dimensions
    expand_shape = [1] * len(prefix) + [seq_len, half]
    cos_broad = cos_pair.view(*expand_shape).expand(*prefix, seq_len, half)
    sin_broad = sin_pair.view(*expand_shape).expand(*prefix, seq_len, half)

    # Apply rotation
    q_rot_even = xq_even * cos_broad - xq_odd * sin_broad
    q_rot_odd = xq_even * sin_broad + xq_odd * cos_broad
    k_rot_even = xk_even * cos_broad - xk_odd * sin_broad
    k_rot_odd = xk_even * sin_broad + xk_odd * cos_broad

    # Reconstruct full tensors
    q_rot = torch.empty_like(xq)
    k_rot = torch.empty_like(xk)
    q_rot[..., :half] = q_rot_even
    q_rot[..., half:] = q_rot_odd
    k_rot[..., :half] = k_rot_even
    k_rot[..., half:] = k_rot_odd

    return q_rot, k_rot


class LayerNorm(nn.Module):
    """
    Bias-less Layer Normalization module.

    Args:
        dim (int): The dimension of the input features to normalize
    """

    @beartype
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    @beartype
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization to input tensor.

        Args:
            x (torch.Tensor): Input tensor to normalize

        Returns:
            torch.Tensor: Layer normalized tensor
        """
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class GEGLU(nn.Module):
    """
    Gated Linear Unit with GELU activation.

    Splits input tensor in half along the last dimension, applies GELU to one half
    (gate) and multiplies with the other half (value).
    """

    @beartype
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply GEGLU activation.

        Args:
            x (torch.Tensor): Input tensor with even last dimension

        Returns:
            torch.Tensor: GEGLU activated tensor with half the last dimension
        """
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


@beartype
def FeedForward(dim: int, mult: float = 4.0, dropout: float = 0.0) -> nn.Sequential:
    """
    Create a feedforward network with GEGLU activation.

    Args:
        dim (int): Input/output dimension
        mult (float): Multiplier for inner dimension calculation
        dropout (float): Dropout probability

    Returns:
        nn.Sequential: Feedforward network module
    """
    inner_dim = int(mult * (2 / 3) * dim)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias=False),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias=False),
    )


# TemporalPEG - position generating module for temporal attention


class TemporalPEG(nn.Module):
    """
    TemporalPEG module using a 3D convolution, adapted to work on the 3D tensor
    format used by the temporal transformer (B*H*W, T, D).
    """

    @beartype
    def __init__(self, dim: int, causal: bool = False):
        super().__init__()
        self.causal = causal
        self.dsconv = nn.Conv3d(dim, dim, 3, groups=dim)

    @beartype
    def forward(self, x: torch.Tensor, shape: Tuple[int, int, int, int]) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B*H*W, T, D).
            shape (Tuple[int, int, int, int]): Original shape (B, T, H, W).

        Returns:
            torch.Tensor: Output tensor with the same shape as input.
        """
        B, T, H, W = shape
        N, T_in, D = x.shape
        assert N == B * H * W and T_in == T, f"Got {x.shape}, expected {(B*H*W, T, 'D')}"
        assert D == self.dsconv.in_channels, f"D={D} must match Conv3d in_channels={self.dsconv.in_channels}"

        x_5d = rearrange(x, "(b h w) t d -> b t h w d", b=B, h=H, w=W)
        x_permuted = rearrange(x_5d, "b t h w d -> b d t h w")

        frame_padding = (2, 0) if self.causal else (1, 1)
        x_padded = F.pad(x_permuted, (1, 1, 1, 1, *frame_padding), value=0.0)

        x_convolved = self.dsconv(x_padded)

        x_processed_5d = rearrange(x_convolved, "b d t h w -> b t h w d")
        output = rearrange(x_processed_5d, "b t h w d -> (b h w) t d")

        return output


# PEG - position generating module


class PEG(nn.Module):
    """
    Position Encoding Generator (PEG) using a 3D depthwise separable convolution.

    This module operates on 5D video tensors (B, T, H, W, D) to generate
    positional encodings, supporting both causal and non-causal variants.

    Args:
        dim (int): Input feature dimension.
        causal (bool): Whether to use causal padding for the temporal dimension.
    """

    @beartype
    def __init__(self, dim: int, causal: bool = False) -> None:
        super().__init__()
        self.causal = causal
        self.dsconv = nn.Conv3d(dim, dim, 3, groups=dim)

    @beartype
    def forward(self, x: torch.Tensor, shape: Tuple[int, int, int, int]) -> torch.Tensor:
        """
        Apply positional encoding generation to the input video tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, H, W, D).

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        assert x.ndim == 5, "Input tensor must be 5D (Batch, Time, Height, Width, Dim)"
        B, T, H, W = shape
        B, T_in, H_in, W_in, D = x.shape
        assert (B, T_in, H_in, W_in) == (B, T, H, W), f"Got {x.shape}, expected {(B, T, H, W, 'D')}"

        # Rearrange for Conv3d, which expects (Batch, Dim, Time, Height, Width)
        x = rearrange(x, "b t h w d -> b d t h w", b=B, t=T, h=H, w=W)

        # Apply causal or symmetric padding
        frame_padding = (2, 0) if self.causal else (1, 1)
        x = F.pad(x, (1, 1, 1, 1, *frame_padding), value=0.0)

        x = self.dsconv(x)

        # Rearrange back to the original format
        x = rearrange(x, "b d t h w -> b t h w d", b=B, t=T, h=H, w=W)

        return x


class ContinuousPositionBias(nn.Module):
    """
    Continuous Position Bias module for spatial and temporal attention.

    Generates continuous positional biases using MLPs, supporting both 2D (images)
    and 3D (video) data with optional logarithmic distance scaling.

    Reference: https://arxiv.org/abs/2111.09883

    Args:
        dim (int): Hidden dimension for MLP layers
        heads (int): Number of attention heads
        num_dims (int): Number of spatial/temporal dimensions (2 for images, 3 for video)
        layers (int): Number of MLP layers
        log_dist (bool): Whether to apply logarithmic distance scaling
        cache_rel_pos (bool): Whether to cache relative position embeddings
    """

    def __init__(
        self,
        *,
        dim: int,
        heads: int,
        num_dims: int = 2,
        layers: int = 2,
        log_dist: bool = True,
        cache_rel_pos: bool = True,  # keep the flag; now it's per-shape
        normalize: bool = True,  # NEW: normalize coords into [0, 1]
        use_centers: bool = True,  # NEW: use (i+0.5)/size instead of i/size
    ) -> None:
        super().__init__()
        self.num_dims = num_dims
        self.log_dist = log_dist
        self.normalize = normalize
        self.use_centers = use_centers
        self.cache_rel_pos = cache_rel_pos

        # MLP: R^num_dims -> R^heads
        mlp = [nn.Linear(self.num_dims, dim), leaky_relu()]
        for _ in range(layers - 1):
            mlp += [nn.Linear(dim, dim), leaky_relu()]
        mlp += [nn.Linear(dim, heads)]
        self.net = nn.Sequential(*mlp)

        # Per-shape CPU fp32 cache for relative positions
        # key: tuple(dimensions) -> tensor of shape (P, P, num_dims)
        self._rel_cache = {}

    @torch.no_grad()
    def _axis(self, n: int, device: torch.device, dtype: torch.dtype):
        """Build 1D coordinate axis."""
        if not self.normalize:
            # raw index units: 0..n-1
            return torch.arange(n, device=device, dtype=dtype)
        # normalized units: [0,1], centers recommended
        if self.use_centers:
            return (torch.arange(n, device=device, dtype=dtype) + 0.5) / n
        return torch.linspace(0, 1, steps=n, device=device, dtype=dtype)

    @torch.no_grad()
    def _rel_positions(self, dims: tuple[int, ...], device: torch.device) -> torch.Tensor:
        """
        Build (P, P, num_dims) relative positions for given dims on `device`.
        Uses CPU fp32 cache; returns a device copy.
        """
        key = tuple(dims)
        if self.cache_rel_pos and key in self._rel_cache:
            return self._rel_cache[key].to(device)

        axes = [self._axis(n, device=device, dtype=torch.float32) for n in dims]
        # grid: num_dims x (d1 x d2 x ...), then flatten to (P, num_dims)
        mesh = torch.stack(torch.meshgrid(*axes, indexing="ij"))  # (num_dims, d1, d2, ...)
        coords = rearrange(mesh, "c ... -> (...) c")  # (P, num_dims)

        # square relative diffs for self-attn
        rel = rearrange(coords, "i c -> i 1 c") - rearrange(coords, "j c -> 1 j c")  # (P, P, num_dims)

        if self.log_dist:
            rel = torch.sign(rel) * torch.log1p(rel.abs())

        if self.cache_rel_pos:
            # store CPU fp32 copy
            self._rel_cache[key] = rel.detach().to("cpu", torch.float32)

        return rel  # on `device`, fp32

    def clear_cache(self) -> None:
        self._rel_cache.clear()

    def forward(
        self, *dimensions: int, device: torch.device = torch.device("cpu"), dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """
        Args:
            *dimensions: grid sizes (e.g., Hp, Wp) or (T, Hp, Wp)
            device: target device for the returned bias
            dtype: optional dtype for the returned bias (match Q/K dtype)

        Returns:
            bias: (heads, P, P), where P = prod(*dimensions)
        """
        assert len(dimensions) == self.num_dims, f"expected {self.num_dims} dims, got {len(dimensions)}"
        rel = self._rel_positions(tuple(dimensions), device=device)  # (P, P, num_dims)

        # MLP over last dim → per-head bias
        x = self.net(rel.float())  # (P, P, heads)
        bias = rearrange(x, "i j h -> h i j")  # (heads, P, P)

        if dtype is not None:
            bias = bias.to(dtype)
        return bias


class AdaLayerNorm(nn.Module):
    """
    AdaLN: y = (1 + scale(c)) * LN(x) + shift(c)
    Zero-initialized last layer so it starts as plain LN.
    """

    def __init__(self, dim: int, cond_dim: int, mult: float = 4.0, zero_init: bool = True) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(dim, elementwise_affine=False)
        hidden = int(mult * max(dim, cond_dim))
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * dim),
        )
        if zero_init:
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)
        else:
            nn.init.normal_(self.mlp[-1].weight, std=0.02)
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        scale, shift = self.mlp(c).chunk(2, dim=-1)
        # broadcast cond over extra token dims if needed (e.g., H*W)
        while scale.ndim < x.ndim:
            scale = scale.unsqueeze(-2)
            shift = shift.unsqueeze(-2)
        return self.ln(x) * (1.0 + scale) + shift


class AdaFeedForward(nn.Module):
    """
      LayerNorm -> AdaLayerNorm (conditioned on c)
      Linear(dim, inner*2) -> GEGLU -> Dropout -> Linear(inner, dim)

    Call as: out = ada_ff(x, c)
      x: (..., D)
      c: (..., cond_dim) aligned to the same leading dims as x, but can omit token dims
         e.g., x:(B,T,H*W,D) and c:(B,T,cond_dim) -> broadcasts over H*W
    """

    def __init__(self, dim: int, cond_dim: int, mult: float = 4, dropout: float = 0.0):
        super().__init__()
        inner_dim = int(mult * (2 / 3) * dim)
        self.pre = AdaLayerNorm(dim, cond_dim)  # replaces LayerNorm
        self.proj_in = nn.Linear(dim, inner_dim * 2, bias=False)
        self.act = GEGLU()
        self.dropout = nn.Dropout(dropout)
        self.proj_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.pre(x, c)  # AdaLN pre-norm (conditioned)
        h = self.proj_in(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.proj_out(h)
        return h


class Attention(nn.Module):
    """
    Multi-head attention module with optional causal masking and positional bias.

    Supports both self-attention and cross-attention with configurable null key-value pairs,
    causal masking, and various positional encoding schemes.

    Supports PyTorch's native scaled dot-product attention (SDPA) for efficiency.

    Args:
        dim (int): Input dimension
        dim_context (Optional[int]): Context dimension for cross-attention
        dim_head (int): Dimension per attention head
        heads (int): Number of attention heads
        causal (bool): Whether to apply causal masking
        num_null_kv (int): Number of null key-value pairs to prepend
        norm_context (bool): Whether to apply layer normalization to context
        dropout (float): Dropout probability for attention weights
        scale (float): Scaling factor for attention scores
        use_sdpa (bool): Whether to use PyTorch's scaled dot-product attention (SDPA)
        is_temporal (bool): Whether the attention is temporal, enabling RoPE for 1D sequences
        dim_cond (Optional[int]): Conditioning dimension for AdaLN
        enable_conditioning (bool): Whether to enable conditioning. Replace LN with AdaLN if True.
    """

    @beartype
    def __init__(
        self,
        dim: int,
        dim_context: Optional[int] = None,
        dim_head: int = 64,
        heads: int = 8,
        causal: bool = False,
        num_null_kv: int = 0,
        norm_context: bool = True,
        dropout: float = 0.0,
        scale: float = 8.0,
        use_sdpa: bool = True,
        is_temporal: bool = False,
        dim_cond: Optional[int] = None,
        enable_conditioning: bool = False,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.scale = scale
        self.dim_head = dim_head
        self.use_sdpa = use_sdpa
        self.is_temporal = is_temporal
        self.enable_conditioning = enable_conditioning

        # Precomputed frequencies for RoPE, if needed
        self.freqs_cis = None

        inner_dim = dim_head * heads
        dim_context = default(dim_context, dim)

        self.attn_dropout = nn.Dropout(dropout)

        if enable_conditioning:
            self.norm = AdaLayerNorm(dim, cond_dim=dim_cond)
        else:
            self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(dim_context) if norm_context else nn.Identity()

        self.num_null_kv = num_null_kv
        if num_null_kv > 0:
            self.null_kv = nn.Parameter(torch.randn(heads, 2 * num_null_kv, dim_head))

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_context, inner_dim * 2, bias=False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    @beartype
    def _build_additive_mask(
        self,
        *,
        b: int,
        h: int,
        q_len: int,
        k_len: int,
        device: torch.device,
        dtype: torch.dtype,
        attn_bias: Optional[torch.Tensor] = None,
        key_mask_bool: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build additive mask for attention computation.

        Returns an additive mask of shape (b, h, q_len, k_len) that encodes:
        - External bias (padded for null kv)
        - Key padding (False -> -inf)
        - Causal triangular (leaves null kv unmasked)

        Args:
            b (int): Batch size
            h (int): Number of attention heads
            q_len (int): Query sequence length
            k_len (int): Key sequence length
            device (torch.device): Target device for tensor creation
            dtype (torch.dtype): Target data type for tensor creation
            attn_bias (Optional[torch.Tensor]): External attention bias
            key_mask_bool (Optional[torch.Tensor]): Key padding mask (True=keep)

        Returns:
            torch.Tensor: Additive mask of shape (b, h, q_len, k_len)
        """
        mask_add = torch.zeros((b, h, q_len, k_len), device=device, dtype=dtype)

        # External additive bias
        if exists(attn_bias):
            bias = F.pad(attn_bias.to(dtype), (self.num_null_kv, 0), value=0.0)
            # Rely on broadcasting: (b,h,q,k) + (b,1,q,k) / (b,q,k) / (1,1,q,k)
            mask_add = mask_add + bias

        # Key padding: True = keep
        if exists(key_mask_bool):
            key_mask_bool = F.pad(key_mask_bool, (self.num_null_kv, 0), value=True)
            neg_inf = torch.full((), -torch.finfo(dtype).max, device=device, dtype=dtype)
            pad = torch.where(key_mask_bool, torch.zeros((), device=device, dtype=dtype), neg_inf)
            mask_add = mask_add + pad.view(b, 1, 1, k_len)

        # Causal mask
        if self.causal:
            # Causal mask over real keys, keep null kv unmasked
            k_real = k_len - self.num_null_kv
            tri = torch.ones((q_len, k_real), device=device, dtype=torch.bool).triu(1)
            tri = F.pad(tri, (self.num_null_kv, 0), value=False)
            neg_inf = torch.full((), -torch.finfo(dtype).max, device=device, dtype=dtype)
            mask_add = mask_add + torch.where(tri, neg_inf, 0).view(1, 1, q_len, k_len)

        return mask_add

    @beartype
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply multi-head attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, dim)
            mask (Optional[torch.Tensor]): Attention mask where True means keep
            context (Optional[torch.Tensor]): Context tensor for cross attention
            attn_bias (Optional[torch.Tensor]): Additional attention bias. Should be broadcastable to (batch, heads, seq_len, key_seq_len)
            cond (Optional[torch.Tensor]): Conditioning tensor for AdaLN, if enabled

        Returns:
            torch.Tensor: Output tensor with same shape as input
        """
        batch, seq_len, device, dtype = x.shape[0], x.shape[1], x.device, x.dtype

        if exists(context):
            context = self.context_norm(context)
            if self.causal:
                assert (
                    context.shape[1] == seq_len
                ), f"Context length {context.shape[1]} must match input sequence length {seq_len} for causal attention"
            assert (
                context.shape[0] == batch
            ), f"Context batch size {context.shape[0]} must match input batch size {batch}"

        kv_input = context if exists(context) else x
        if self.enable_conditioning:
            assert exists(cond), "Conditioning tensor is required when enable_conditioning=True"
            x = self.norm(x, cond)
        else:
            x = self.norm(x)

        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim=-1)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)

        if self.is_temporal or self.causal:
            if not exists(self.freqs_cis) or self.freqs_cis.shape[0] < seq_len:
                # Precompute RoPE frequencies if not cached or too short
                self.freqs_cis = precompute_freqs_cis_1d(
                    dim=self.dim_head,
                    seq_len=seq_len,
                    use_cls=False,
                ).to(device)
            q, k = apply_rope_1d(q, k, self.freqs_cis[:seq_len])

        if self.num_null_kv > 0:
            nk, nv = repeat(self.null_kv, "h (n r) d -> b h n r d", b=batch, r=2).unbind(dim=-2)
            k = torch.cat((nk, k), dim=-2)
            v = torch.cat((nv, v), dim=-2)

        q = l2norm(q) * self.q_scale
        k = l2norm(k) * self.k_scale

        q_len, k_len = q.shape[-2], k.shape[-2]

        # Build common additive mask used by both paths
        add_mask = self._build_additive_mask(
            b=batch,
            h=self.heads,
            q_len=q_len,
            k_len=k_len,
            device=device,
            dtype=dtype,
            attn_bias=attn_bias,
            key_mask_bool=mask,
        )

        if self.use_sdpa:
            # SDPA path
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=add_mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=False,
                scale=self.scale,
            )
            out = rearrange(attn_out, "b h n d -> b n (h d)")
            return self.to_out(out)

        # non-SDPA path
        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        sim = sim + add_mask
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    """
    Transformer block with self-attention, cross-attention, and feed-forward layers.

    Supports causal and non-causal attention, positional encoding generation (PEG),
    and optional cross-attention for conditioning on external context.

    Args:
        dim (int): Input/output dimension
        depth (int): Number of transformer layers
        dim_context (Optional[int]): Context dimension for cross-attention
        causal (bool): Whether to use causal masking in self-attention
        dim_head (int): Dimension per attention head
        heads (int): Number of attention heads
        ff_mult (float): Multiplier for feedforward inner dimension
        peg (bool): Whether to use Position Encoding Generation
        peg_causal (bool): Whether PEG should use causal padding
        attn_num_null_kv (int): Number of null key-value pairs for cross-attention
        has_cross_attn (bool): Whether to include cross-attention layers
        attn_dropout (float): Dropout probability for attention weights
        ff_dropout (float): Dropout probability for feedforward layers
        use_sdpa (bool): Whether to use PyTorch's scaled dot-product attention (SDPA)
        is_temporal (bool): Whether the attention is temporal, enabling RoPE for 1D sequences
    """

    @beartype
    def __init__(
        self,
        dim: int,
        *,
        depth: int,
        dim_context: Optional[int] = None,
        causal: bool = False,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        peg: bool = False,
        peg_causal: bool = False,
        attn_num_null_kv: int = 0,
        has_cross_attn: bool = False,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        use_sdpa: bool = True,
        is_temporal: bool = False,
    ) -> None:
        super().__init__()
        assert not (peg and not is_temporal), "PEG should only be used with temporal attention (is_temporal=True)"
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        TemporalPEG(dim=dim, causal=peg_causal) if peg else None,
                        Attention(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            causal=causal,
                            dropout=attn_dropout,
                            use_sdpa=use_sdpa,
                            is_temporal=is_temporal,
                        ),
                        (
                            Attention(
                                dim=dim,
                                dim_head=dim_head,
                                dim_context=dim_context,
                                heads=heads,
                                causal=False,
                                num_null_kv=attn_num_null_kv,
                                dropout=attn_dropout,
                                use_sdpa=use_sdpa,
                            )
                            if has_cross_attn
                            else None
                        ),
                        FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout),
                    ]
                )
            )

        self.norm_out = LayerNorm(dim)

    @beartype
    def forward(
        self,
        x: torch.Tensor,
        video_shape: Optional[Tuple[int, int, int, int]] = None,
        attn_bias: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_context_mask: Optional[torch.Tensor] = None,
        cross_attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply transformer layers to input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, dim)
            video_shape (Optional[Tuple[int, int, int, int]]): Shape for PEG if input is 2D
            attn_bias (Optional[torch.Tensor]): Additional attention bias
            context (Optional[torch.Tensor]): Context tensor for cross-attention
            self_attn_mask (Optional[torch.Tensor]): Mask for self-attention
            cross_attn_context_mask (Optional[torch.Tensor]): Mask for cross-attention context
            cross_attn_bias (Optional[torch.Tensor]): Additional bias for cross-attention

        Returns:
            torch.Tensor: Output tensor with same shape as input
        """
        for peg, self_attn, cross_attn, ff in self.layers:
            if exists(peg):
                x = peg(x, shape=video_shape) + x

            x = self_attn(x, attn_bias=attn_bias, mask=self_attn_mask) + x

            if exists(cross_attn) and exists(context):
                x = cross_attn(x, context=context, mask=cross_attn_context_mask, attn_bias=cross_attn_bias) + x

            x = ff(x) + x

        return self.norm_out(x)


class ConditioningModule(nn.Module):
    """
    Lightweight conditioning via per-frame scale and shift modulation.
    Applies: x_i = x_i + FFN(LayerNorm(x_i) * (1 + alpha_i) + beta_i)
    where alpha_i and beta_i are regressed from the external context (e.g., action embedding).
    """

    def __init__(self, dim: int, cond_dim: int, ff_mult: float = 4.0):
        super().__init__()
        hidden_dim = int(dim * ff_mult)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.to_alpha_beta = nn.Linear(cond_dim, 2 * dim)  # Output (alpha, beta)
        # Initalize alpha and beta to zero
        nn.init.zeros_(self.to_alpha_beta.weight)
        nn.init.zeros_(self.to_alpha_beta.bias)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, C) video feature tokens
            cond: (B, T, C_cond) conditioning input per frame

        Returns:
            (B, T, N, C) modulated x
        """
        B, T, N, C = x.shape
        assert cond.shape[:2] == (B, T), f"Expected cond shape (B, T, ...), got {cond.shape}"

        alpha_beta = self.to_alpha_beta(cond)  # (B, T, 2*C)
        alpha, beta = alpha_beta.chunk(2, dim=-1)  # each is (B, T, C)
        alpha, beta = rearrange(alpha, "b t c -> b t 1 c"), rearrange(beta, "b t c -> b t 1 c")

        x_norm = self.norm(x)
        mod = x_norm * (1 + alpha) + beta
        return x + self.ffn(mod)


class STTransformer(nn.Module):
    """
    Spatio-Temporal transformer block with spatial attention, temporal attention and feed-forward layers.

    Supports causal and non-causal temporal attention, positional encoding generation (PEG),
    and conditioning temporal attention on external context.

    Args:
        dim (int): Input/output dimension
        depth (int): Number of transformer layers
        dim_context (Optional[int]): Context dimension for cross-attention
        causal (bool): Whether to use causal masking in self-attention
        dim_head (int): Dimension per attention head
        heads (int): Number of attention heads
        ff_mult (float): Multiplier for feedforward inner dimension
        peg (bool): Whether to use Position Encoding Generation
        peg_causal (bool): Whether PEG should use causal padding
        attn_num_null_kv (int): Number of null key-value pairs for cross-attention
        attn_dropout (float): Dropout probability for attention weights
        ff_dropout (float): Dropout probability for feedforward layers
        use_sdpa (bool): Whether to use PyTorch's scaled dot-product attention (SDPA)
        dim_cond (Optional[int]): Conditioning dimension
        enable_conditioning (bool): Whether to enable conditioning
    """

    @beartype
    def __init__(
        self,
        dim: int,
        *,
        depth: int,
        dim_context: Optional[int] = None,
        causal: bool = False,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: float = 4.0,
        peg: bool = False,
        peg_causal: bool = False,
        attn_num_null_kv: int = 0,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        use_sdpa: bool = True,
        dim_cond: Optional[int] = None,
        enable_conditioning: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        ConditioningModule(dim, dim_cond) if enable_conditioning else None,
                        PEG(dim=dim, causal=peg_causal) if peg else None,
                        Attention(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            causal=False,
                            dropout=attn_dropout,
                            use_sdpa=use_sdpa,
                            num_null_kv=attn_num_null_kv,
                        ),  # spatial attention
                        Attention(
                            dim=dim,
                            dim_context=dim_context,
                            dim_head=dim_head,
                            heads=heads,
                            causal=causal,
                            dropout=attn_dropout,
                            use_sdpa=use_sdpa,
                            is_temporal=True,
                            # disabling, may be causing stability issues
                            # dim_cond=dim_cond,
                            # enable_conditioning=enable_conditioning,
                        ),  # temporal attention
                        (
                            FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
                            # if not enable_conditioning
                            # else AdaFeedForward(dim=dim, cond_dim=dim_cond, mult=ff_mult)
                        ),
                    ]
                )
            )

        self.norm_out = LayerNorm(dim)

    @beartype
    def forward(
        self,
        x: torch.Tensor,
        video_shape: Optional[Tuple[int, int, int, int]] = None,
        spatial_attn_bias: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply transformer layers to input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (b, t, n, d)
            video_shape (Optional[Tuple[int, int, int, int]]): Shape for PEG, must be (b, t, h, w)
            spatial_attn_bias (Optional[torch.Tensor]): Additional spatial attention bias of shape (heads, Np, Np)
            context (Optional[torch.Tensor]): Context tensor of shape (b, t, n, d_ctx) for cross attention
            cond (Optional[torch.Tensor]): Conditioning tensor of shape (b, t, d_cond)
            attn_mask (Optional[torch.Tensor]): Mask for attention of shape to handle padding in temporal dimension. Shape (b, t)

        Returns:
            torch.Tensor: Output tensor with same shape as input
        """
        B, T, H, W = video_shape
        N = x.shape[2]
        num_spatial_tokens = H * W
        num_extra_spatial_tokens = N - num_spatial_tokens

        for cond_mod, peg, spatial_attn, temporal_attn, ff in self.layers:
            # Conditioning Module
            if exists(cond_mod):
                x = cond_mod(x, cond)  # Already does residual connection

            # PEG
            if exists(peg):
                if num_extra_spatial_tokens > 0:
                    x, x_extra = x[:, :, :-num_extra_spatial_tokens], x[:, :, -num_extra_spatial_tokens:]
                x_grid = rearrange(x, "b t (h w) d -> b t h w d", b=B, t=T, h=H, w=W)
                x_grid = peg(x_grid, video_shape) + x_grid
                x = rearrange(x_grid, "b t h w d -> b t (h w) d", b=B, t=T, h=H, w=W)
                if num_extra_spatial_tokens > 0:
                    x = torch.cat((x, x_extra), dim=2)

            # Temporal Attention
            temporal_mask = None
            if exists(attn_mask):
                temporal_mask = repeat(attn_mask, "b t -> (b n) t", n=N)
            context_temp = None
            if exists(context):
                context_temp = rearrange(context, "b t n d -> (b n) t d", b=B, t=T, n=context.shape[2])
            cond_temp = None
            # if exists(cond):
            #     cond_temp = repeat(cond, "b t c -> (b n) t c", n=N)
            x_temp = rearrange(x, "b t n d -> (b n) t d", b=B, n=N)
            temp_out = temporal_attn(x_temp, context=context_temp, mask=temporal_mask, cond=cond_temp)
            x = rearrange(temp_out, "(b n) t d -> b t n d", b=B, n=N) + x

            # Spatial Attention
            x_spat = rearrange(x, "b t n d -> (b t) n d", b=B, t=T, n=N)
            spat_out = spatial_attn(x_spat, attn_bias=spatial_attn_bias)
            x = rearrange(spat_out, "(b t) n d -> b t n d", b=B, t=T, n=N) + x

            # FeedForward Network
            # if exists(cond):
            #     ff_out = ff(x, cond)
            # else:
            #     ff_out = ff(x)
            ff_out = ff(x)
            x = ff_out + x

        return self.norm_out(x)
