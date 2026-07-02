from __future__ import annotations

import math
from typing import Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import DropPath, LayerNorm, LayerScale, use_fused_attn


GridSize = Tuple[int, int]
Rect = Union[Tuple[float, float, float, float], torch.Tensor]


def _cross_attn_mask(
    attn_mask: Optional[torch.Tensor],
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """
    Bool mask semantics:
        True  -> keep
        False -> mask out

    Accepts shapes broadcastable to (B, num_heads, Nq, Nk).
    """
    if attn_mask is None:
        return None

    if attn_mask.dtype == torch.bool:
        bias = torch.zeros_like(attn_mask, dtype=dtype)
        bias.masked_fill_(~attn_mask, float("-inf"))
        return bias

    return attn_mask


def _infer_square_grid_size(num_patch_tokens: int, *, name: str) -> GridSize:
    side = math.isqrt(num_patch_tokens)
    if side * side != num_patch_tokens:
        raise ValueError(
            f"Cannot infer square grid for {name} from {num_patch_tokens} patch tokens. "
            f"Pass {name}_grid_size=(H, W)."
        )
    return side, side


def _as_batched_rect(
    rect: Rect,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    """
    Converts rect to shape [B, 4].

    rect format:
        (y0, x0, y1, x1)

    Coordinates are in the global patch-grid coordinate frame.
    y1/x1 are exclusive-style boundaries.
    """
    rect = torch.as_tensor(rect, device=device, dtype=dtype)

    if rect.ndim == 1:
        if rect.shape[0] != 4:
            raise ValueError(f"{name}_rect must have shape [4] or [B, 4].")
        rect = rect[None, :].expand(batch_size, 4)

    elif rect.ndim == 2:
        if rect.shape[1] != 4:
            raise ValueError(f"{name}_rect must have shape [4] or [B, 4].")

        if rect.shape[0] == 1:
            rect = rect.expand(batch_size, 4)
        elif rect.shape[0] != batch_size:
            raise ValueError(
                f"{name}_rect has batch size {rect.shape[0]}, "
                f"but expected {batch_size}."
            )

    else:
        raise ValueError(f"{name}_rect must have shape [4] or [B, 4].")

    return rect


def _native_grid_coords(
    grid_size: GridSize,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Returns global/native grid coordinates with shape [B, H*W, 2].

    Coordinate order:
        [..., 0] = y
        [..., 1] = x
    """
    gh, gw = grid_size

    yy, xx = torch.meshgrid(
        torch.arange(gh, device=device, dtype=dtype),
        torch.arange(gw, device=device, dtype=dtype),
        indexing="ij",
    )

    coords = torch.stack(
        (yy.reshape(-1), xx.reshape(-1)),
        dim=-1,
    )

    return coords[None, :, :].expand(batch_size, -1, -1)


def _rect_grid_coords(
    grid_size: GridSize,
    rect: torch.Tensor,
    *,
    align_corners: bool,
) -> torch.Tensor:
    """
    Creates per-sample local-grid coordinates inside per-sample global rectangles.

    Args:
        grid_size:
            Local or stream grid size, (H, W).

        rect:
            Tensor of shape [B, 4], where each row is:
                (y0, x0, y1, x1)

            y0/x0/y1/x1 are in global patch-grid coordinates.

        align_corners:
            If False, uses patch-center mapping:

                y_i = y0 + (i + 0.5) * (y1 - y0) / H - 0.5

            This makes coordinates exactly match integer global patch centers when
            the local grid resolution equals the rectangle size.

            If True, maps the first and last local tokens to y0 and y1 - 1.

    Returns:
        coords:
            [B, H*W, 2]
    """
    b = rect.shape[0]
    gh, gw = grid_size
    device = rect.device
    dtype = rect.dtype

    y0, x0, y1, x1 = rect.unbind(dim=-1)

    if align_corners:
        if gh == 1:
            ys = ((y0 + y1 - 1.0) * 0.5)[:, None]
        else:
            iy = torch.linspace(0.0, 1.0, gh, device=device, dtype=dtype)
            ys = y0[:, None] + iy[None, :] * ((y1 - 1.0) - y0)[:, None]

        if gw == 1:
            xs = ((x0 + x1 - 1.0) * 0.5)[:, None]
        else:
            ix = torch.linspace(0.0, 1.0, gw, device=device, dtype=dtype)
            xs = x0[:, None] + ix[None, :] * ((x1 - 1.0) - x0)[:, None]

    else:
        iy = torch.arange(gh, device=device, dtype=dtype) + 0.5
        ix = torch.arange(gw, device=device, dtype=dtype) + 0.5

        ys = y0[:, None] + iy[None, :] * ((y1 - y0) / gh)[:, None] - 0.5
        xs = x0[:, None] + ix[None, :] * ((x1 - x0) / gw)[:, None] - 0.5

    yy = ys[:, :, None].expand(b, gh, gw)
    xx = xs[:, None, :].expand(b, gh, gw)

    return torch.stack(
        (yy.reshape(b, -1), xx.reshape(b, -1)),
        dim=-1,
    )


def _as_batched_patch_coords(
    patch_coords: torch.Tensor,
    *,
    batch_size: int,
    num_patch_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    """
    Converts explicit patch coordinates to shape [B, N_patch, 2].

    Accepted:
        [N_patch, 2]
        [1, N_patch, 2]
        [B, N_patch, 2]
    """
    patch_coords = torch.as_tensor(patch_coords, device=device, dtype=dtype)

    if patch_coords.ndim == 2:
        if patch_coords.shape != (num_patch_tokens, 2):
            raise ValueError(
                f"{name}_patch_coords must have shape "
                f"[{num_patch_tokens}, 2] or [B, {num_patch_tokens}, 2], "
                f"got {tuple(patch_coords.shape)}."
            )
        patch_coords = patch_coords[None, :, :].expand(batch_size, -1, -1)

    elif patch_coords.ndim == 3:
        if patch_coords.shape[1:] != (num_patch_tokens, 2):
            raise ValueError(
                f"{name}_patch_coords must have shape "
                f"[B, {num_patch_tokens}, 2], got {tuple(patch_coords.shape)}."
            )

        if patch_coords.shape[0] == 1:
            patch_coords = patch_coords.expand(batch_size, -1, -1)
        elif patch_coords.shape[0] != batch_size:
            raise ValueError(
                f"{name}_patch_coords has batch size {patch_coords.shape[0]}, "
                f"but expected {batch_size}."
            )

    else:
        raise ValueError(
            f"{name}_patch_coords must have shape [N_patch, 2] or [B, N_patch, 2]."
        )

    return patch_coords


def make_stream_patch_coords(
    *,
    batch_size: int,
    num_patch_tokens: int,
    grid_size: Optional[GridSize],
    rect: Optional[Rect],
    patch_coords: Optional[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    align_corners: bool = False,
    name: str,
) -> torch.Tensor:
    """
    Returns coordinates for one token stream as [B, N_patch, 2].

    Priority:
        1. explicit patch_coords
        2. rect + grid_size
        3. native grid coordinates

    Use cases:
        global stream:
            rect=None
            patch_coords=None
            grid_size=(Hg, Wg)

        local stream with per-sample rectangles:
            rect: [B, 4]
            grid_size=(Hl, Wl)

        arbitrary sampled stream:
            patch_coords: [B, N_patch, 2]
    """
    if grid_size is None:
        grid_size = _infer_square_grid_size(num_patch_tokens, name=name)

    gh, gw = grid_size
    expected = gh * gw
    if expected != num_patch_tokens:
        raise ValueError(
            f"{name}_grid_size={grid_size} implies {expected} patch tokens, "
            f"but {name} stream has {num_patch_tokens} patch tokens."
        )

    if patch_coords is not None and rect is not None:
        raise ValueError(
            f"Provide either {name}_patch_coords or {name}_rect, not both."
        )

    if patch_coords is not None:
        return _as_batched_patch_coords(
            patch_coords,
            batch_size=batch_size,
            num_patch_tokens=num_patch_tokens,
            device=device,
            dtype=dtype,
            name=name,
        )

    if rect is not None:
        rect = _as_batched_rect(
            rect,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            name=name,
        )
        return _rect_grid_coords(
            grid_size,
            rect,
            align_corners=align_corners,
        )

    return _native_grid_coords(
        grid_size,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
    )


class Axial2dCrossRotaryEmbedding(nn.Module):
    """
    Axial 2D RoPE for cross attention.

    q shape:
        [B, num_heads, Nq, head_dim]

    k shape:
        [B, num_heads, Nk, head_dim]

    q_coords_yx:
        [B, Nq_patch, 2]

    kv_coords_yx:
        [B, Nk_patch, 2]

    Front tokens are left unrotated independently for q and kv.
    This supports CLS/register tokens at the front of either stream.
    """

    def __init__(
        self,
        dim: int,
        base: float = 100.0,
    ) -> None:
        super().__init__()

        if dim <= 0 or dim % 4 != 0:
            raise ValueError(f"dim must be a positive multiple of 4, got {dim}.")

        self.dim = dim
        self.axis_dim = dim // 2

        inv_freq = 1.0 / (
            base ** (
                torch.arange(0, self.axis_dim, 2, dtype=torch.float32)
                / self.axis_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _axis_cos_sin(
        self,
        coords: torch.Tensor,
        *,
        out_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        coords:
            [B, N]

        returns:
            cos, sin: [B, N, axis_dim // 2]
        """
        if coords.ndim != 2:
            raise ValueError(f"coords must have shape [B, N], got {tuple(coords.shape)}.")

        coords = coords.to(dtype=torch.float32)
        inv_freq = self.inv_freq.to(device=coords.device, dtype=torch.float32)

        freqs = coords[..., None] * inv_freq[None, None, :]
        return freqs.cos().to(dtype=out_dtype), freqs.sin().to(dtype=out_dtype)

    @staticmethod
    def _rotate_axis(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        x:
            [B, H, N, axis_dim]

        cos/sin:
            [B, N, axis_dim // 2]
        """
        x = x.reshape(*x.shape[:-1], -1, 2)
        x_even, x_odd = x.unbind(dim=-1)

        cos = cos[:, None, :, :]
        sin = sin[:, None, :, :]

        x_rot = torch.stack(
            (
                x_even * cos - x_odd * sin,
                x_even * sin + x_odd * cos,
            ),
            dim=-1,
        )

        return x_rot.flatten(-2)

    def rotate_one(
        self,
        x: torch.Tensor,
        coords_yx: torch.Tensor,
        *,
        num_front_tokens: int,
        stream_name: str,
    ) -> torch.Tensor:
        """
        Rotates one stream.

        x:
            [B, H, N_total, head_dim]

        coords_yx:
            [B, N_patch, 2]
        """
        b, _, n_total, head_dim = x.shape

        if self.dim > head_dim:
            raise ValueError(
                f"RoPE dim {self.dim} exceeds head_dim {head_dim} for {stream_name}."
            )

        if num_front_tokens < 0:
            raise ValueError(f"{stream_name}_num_front_tokens must be non-negative.")

        n_patch = n_total - num_front_tokens
        if n_patch <= 0:
            raise ValueError(
                f"{stream_name} has no patch tokens after "
                f"{num_front_tokens} front tokens."
            )

        if coords_yx.shape != (b, n_patch, 2):
            raise ValueError(
                f"{stream_name}_coords_yx must have shape "
                f"[{b}, {n_patch}, 2], got {tuple(coords_yx.shape)}."
            )

        coords_yx = coords_yx.to(device=x.device)

        x_front = x[:, :, :num_front_tokens, :]
        x_patch = x[:, :, num_front_tokens:, :]

        y = coords_yx[..., 0]
        x_coord = coords_yx[..., 1]

        cos_y, sin_y = self._axis_cos_sin(y, out_dtype=x.dtype)
        cos_x, sin_x = self._axis_cos_sin(x_coord, out_dtype=x.dtype)

        x_rope = x_patch[..., : self.dim]
        x_pass = x_patch[..., self.dim :]

        x_y, x_x = x_rope.split(self.axis_dim, dim=-1)

        x_y = self._rotate_axis(x_y, cos_y, sin_y)
        x_x = self._rotate_axis(x_x, cos_x, sin_x)

        x_patch = torch.cat((x_y, x_x, x_pass), dim=-1)

        if num_front_tokens == 0:
            return x_patch

        return torch.cat((x_front, x_patch), dim=2)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        q_coords_yx: torch.Tensor,
        kv_coords_yx: torch.Tensor,
        q_num_front_tokens: int = 0,
        kv_num_front_tokens: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.rotate_one(
            q,
            q_coords_yx,
            num_front_tokens=q_num_front_tokens,
            stream_name="q",
        )
        k = self.rotate_one(
            k,
            kv_coords_yx,
            num_front_tokens=kv_num_front_tokens,
            stream_name="kv",
        )
        return q, k
    

class AxialRoPECrossAttention(nn.Module):
    """
    timm-style multi-head cross attention with axial 2D RoPE.

    Query comes from x_q.
    Key/value come from x_kv.

    Shapes:
        x_q:
            [B, Nq, C]

        x_kv:
            [B, Nk, C]

        out:
            [B, Nq, C]

    RoPE is applied to q and k only.
    v is not rotated.

    Geometry conventions:
        - Front tokens are at the beginning of each sequence and are not rotated.
        - Patch tokens follow front tokens.
        - Rectangles are in global patch-grid coordinates.
        - Rect format is (y0, x0, y1, x1).
        - rect may be shared [4] or per-sample [B, 4].
        - patch_coords may be shared [N_patch, 2] or per-sample [B, N_patch, 2].

    For local crops, pass rect for the local stream.
    For global full-scene streams, leave rect=None.
    """

    fused_attn: bool

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        attn_head_dim: Optional[int] = None,
        dim_out: Optional[int] = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        scale_norm: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: Optional[Type[nn.Module]] = None,
        q_num_front_tokens: int = 0,
        kv_num_front_tokens: int = 0,
        rope_base: float = 100.0,
        rope_dim: Optional[int] = None,
        align_corners: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        dd = {"device": device, "dtype": dtype}

        dim_out = dim_out or dim

        if attn_head_dim is None:
            assert dim % num_heads == 0, "dim should be divisible by num_heads"
            head_dim = dim // num_heads
        else:
            head_dim = attn_head_dim

        if qk_norm or scale_norm:
            assert norm_layer is not None, (
                "norm_layer must be provided if qk_norm or scale_norm is True"
            )

        rope_dim = head_dim if rope_dim is None else rope_dim
        if rope_dim > head_dim:
            raise ValueError(f"rope_dim={rope_dim} exceeds head_dim={head_dim}.")
        if rope_dim % 4 != 0:
            raise ValueError("For axial 2D RoPE, rope_dim must be divisible by 4.")

        if q_num_front_tokens < 0:
            raise ValueError("q_num_front_tokens must be non-negative.")
        if kv_num_front_tokens < 0:
            raise ValueError("kv_num_front_tokens must be non-negative.")

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attn_dim = num_heads * head_dim
        self.scale = head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.q_num_front_tokens = q_num_front_tokens
        self.kv_num_front_tokens = kv_num_front_tokens
        self.align_corners = align_corners

        self.q = nn.Linear(dim, self.attn_dim, bias=qkv_bias, **dd)
        self.k = nn.Linear(dim, self.attn_dim, bias=qkv_bias, **dd)
        self.v = nn.Linear(dim, self.attn_dim, bias=qkv_bias, **dd)

        self.q_norm = norm_layer(head_dim, **dd) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(head_dim, **dd) if qk_norm else nn.Identity()

        self.rope = Axial2dCrossRotaryEmbedding(
            dim=rope_dim,
            base=rope_base,
        )

        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(self.attn_dim, **dd) if scale_norm else nn.Identity()
        self.proj = nn.Linear(self.attn_dim, dim_out, bias=proj_bias, **dd)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        *,
        q_grid_size: Optional[GridSize] = None,
        kv_grid_size: Optional[GridSize] = None,
        q_rect: Optional[Rect] = None,
        kv_rect: Optional[Rect] = None,
        q_patch_coords: Optional[torch.Tensor] = None,
        kv_patch_coords: Optional[torch.Tensor] = None,
        q_num_front_tokens: Optional[int] = None,
        kv_num_front_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            x_q:
                [B, Nq, C]

            x_kv:
                [B, Nk, C]

            attn_mask:
                Optional mask broadcastable to [B, num_heads, Nq, Nk].
                Bool semantics:
                    True  -> keep
                    False -> mask out

            q_grid_size / kv_grid_size:
                Patch grid sizes for q and kv streams.

            q_rect / kv_rect:
                Optional rectangles for local streams.
                Shape [4], [1, 4], or [B, 4].
                Format: (y0, x0, y1, x1).

            q_patch_coords / kv_patch_coords:
                Optional explicit coordinates.
                Shape [N_patch, 2], [1, N_patch, 2], or [B, N_patch, 2].
                If provided, these override rect-based coordinate construction.

            q_num_front_tokens / kv_num_front_tokens:
                Optional per-call overrides for CLS/register/front token counts.
        """
        b, nq, _ = x_q.shape
        b_kv, nk, _ = x_kv.shape

        if b != b_kv:
            raise ValueError(
                f"x_q and x_kv must have the same batch size, got {b} and {b_kv}."
            )

        q_num_front_tokens = (
            self.q_num_front_tokens
            if q_num_front_tokens is None
            else q_num_front_tokens
        )
        kv_num_front_tokens = (
            self.kv_num_front_tokens
            if kv_num_front_tokens is None
            else kv_num_front_tokens
        )

        q_num_patch_tokens = nq - q_num_front_tokens
        kv_num_patch_tokens = nk - kv_num_front_tokens

        if q_num_patch_tokens <= 0:
            raise ValueError(
                f"x_q has no patch tokens after {q_num_front_tokens} front tokens."
            )
        if kv_num_patch_tokens <= 0:
            raise ValueError(
                f"x_kv has no patch tokens after {kv_num_front_tokens} front tokens."
            )

        coord_dtype = torch.float32

        q_coords_yx = make_stream_patch_coords(
            batch_size=b,
            num_patch_tokens=q_num_patch_tokens,
            grid_size=q_grid_size,
            rect=q_rect,
            patch_coords=q_patch_coords,
            device=x_q.device,
            dtype=coord_dtype,
            align_corners=self.align_corners,
            name="q",
        )

        kv_coords_yx = make_stream_patch_coords(
            batch_size=b,
            num_patch_tokens=kv_num_patch_tokens,
            grid_size=kv_grid_size,
            rect=kv_rect,
            patch_coords=kv_patch_coords,
            device=x_kv.device,
            dtype=coord_dtype,
            align_corners=self.align_corners,
            name="kv",
        )

        q = self.q(x_q).reshape(
            b, nq, self.num_heads, self.head_dim
        ).transpose(1, 2)

        k = self.k(x_kv).reshape(
            b, nk, self.num_heads, self.head_dim
        ).transpose(1, 2)

        v = self.v(x_kv).reshape(
            b, nk, self.num_heads, self.head_dim
        ).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k = self.rope(
            q,
            k,
            q_coords_yx=q_coords_yx,
            kv_coords_yx=kv_coords_yx,
            q_num_front_tokens=q_num_front_tokens,
            kv_num_front_tokens=kv_num_front_tokens,
        )

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)

            attn_bias = _cross_attn_mask(attn_mask, attn.dtype)
            if attn_bias is not None:
                attn = attn + attn_bias

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(b, nq, self.attn_dim)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    

class AxialRoPECrossAttentionBlock(nn.Module):
    """
    Cross-attention-only block.

    Layout:
        x_q = x_q + DropPath(
            LayerScale(
                CrossAttn(Norm(x_q), Norm(x_kv))
            )
        )

    No FFN / MLP.

    Only the query stream is updated.
    x_kv is read-only.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        scale_attn_norm: bool = False,
        proj_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        norm_layer: Type[nn.Module] = LayerNorm,
        q_num_front_tokens: int = 0,
        kv_num_front_tokens: int = 0,
        rope_base: float = 100.0,
        rope_dim: Optional[int] = None,
        align_corners: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        dd = {"device": device, "dtype": dtype}
        self.norm_q = norm_layer(dim, **dd)
        self.norm_kv = norm_layer(dim, **dd)

        self.attn = AxialRoPECrossAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            scale_norm=scale_attn_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            q_num_front_tokens=q_num_front_tokens,
            kv_num_front_tokens=kv_num_front_tokens,
            rope_base=rope_base,
            rope_dim=rope_dim,
            align_corners=align_corners,
            **dd,
        )

        self.ls1 = (
            LayerScale(dim, init_values=init_values, **dd)
            if init_values is not None
            else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        *,
        q_grid_size: Optional[GridSize] = None,
        kv_grid_size: Optional[GridSize] = None,
        q_rect: Optional[Rect] = None,
        kv_rect: Optional[Rect] = None,
        q_patch_coords: Optional[torch.Tensor] = None,
        kv_patch_coords: Optional[torch.Tensor] = None,
        q_num_front_tokens: Optional[int] = None,
        kv_num_front_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        x_q = x_q + self.drop_path1(
            self.ls1(
                self.attn(
                    self.norm_q(x_q),
                    self.norm_kv(x_kv),
                    attn_mask=attn_mask,
                    q_grid_size=q_grid_size,
                    kv_grid_size=kv_grid_size,
                    q_rect=q_rect,
                    kv_rect=kv_rect,
                    q_patch_coords=q_patch_coords,
                    kv_patch_coords=kv_patch_coords,
                    q_num_front_tokens=q_num_front_tokens,
                    kv_num_front_tokens=kv_num_front_tokens,
                )
            )
        )
        return x_q