from __future__ import annotations

import math
from typing import Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import (
    DropPath,
    LayerNorm,
    LayerScale,
    Mlp,
    use_fused_attn,
)


GridSize = Optional[Tuple[int, int]]
Rect = Union[Tuple[float, float, float, float], torch.Tensor]


class Axial2dRotaryEmbedding(nn.Module):
    """
    Axial 2D RoPE for ViT patch tokens.

    q/k shape: [B, num_heads, N, head_dim]

    Tokens at the front of the sequence are left unrotated:
      [front tokens | patch tokens]

    Front tokens can be:
      - CLS token
      - register tokens
      - any other prefix tokens
    """

    def __init__(self, dim: int, base: float = 100.0) -> None:
        super().__init__()
        if dim <= 0 or dim % 4 != 0:
            raise ValueError(f"`dim` must be a positive multiple of 4, got {dim}.")

        self.dim = dim
        self.axis_dim = dim // 2

        inv_freq = 1.0 / (
            base ** (torch.arange(0, self.axis_dim, 2, dtype=torch.float32) / self.axis_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _axis_cos_sin(
        self,
        coords: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ):
        inv_freq = self.inv_freq.to(device=device, dtype=torch.float32)
        freqs = coords.to(device=device, dtype=torch.float32)[:, None] * inv_freq[None, :]
        return freqs.cos().to(dtype=dtype), freqs.sin().to(dtype=dtype)

    @staticmethod
    def _rotate_axis(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x.reshape(*x.shape[:-1], -1, 2)
        x_even, x_odd = x.unbind(dim=-1)

        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]

        x_rot = torch.stack(
            (
                x_even * cos - x_odd * sin,
                x_even * sin + x_odd * cos,
            ),
            dim=-1,
        )
        return x_rot.flatten(-2)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        grid_size: tuple[int, int],
        num_front_tokens: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, n, head_dim = q.shape
        gh, gw = grid_size
        num_patch_tokens = gh * gw

        expected = num_front_tokens + num_patch_tokens
        if n != expected:
            raise ValueError(
                f"Token count mismatch: got N={n}, expected "
                f"{num_front_tokens} front tokens + {gh}*{gw} patch tokens = {expected}."
            )

        if self.dim > head_dim:
            raise ValueError(f"RoPE dim {self.dim} exceeds head_dim {head_dim}.")

        q_front, q_patch = q[:, :, :num_front_tokens], q[:, :, num_front_tokens:]
        k_front, k_patch = k[:, :, :num_front_tokens], k[:, :, num_front_tokens:]

        yy, xx = torch.meshgrid(
            torch.arange(gh, device=q.device),
            torch.arange(gw, device=q.device),
            indexing="ij",
        )
        yy = yy.reshape(-1)
        xx = xx.reshape(-1)

        cos_y, sin_y = self._axis_cos_sin(yy, device=q.device, dtype=q.dtype)
        cos_x, sin_x = self._axis_cos_sin(xx, device=q.device, dtype=q.dtype)

        def apply_rope(t: torch.Tensor) -> torch.Tensor:
            t_rope, t_pass = t[..., : self.dim], t[..., self.dim :]
            t_y, t_x = t_rope.split(self.axis_dim, dim=-1)

            t_y = self._rotate_axis(t_y, cos_y, sin_y)
            t_x = self._rotate_axis(t_x, cos_x, sin_x)

            return torch.cat((t_y, t_x, t_pass), dim=-1)

        q_patch = apply_rope(q_patch)
        k_patch = apply_rope(k_patch)

        q = torch.cat((q_front, q_patch), dim=2)
        k = torch.cat((k_front, k_patch), dim=2)

        return q, k
    

class AxialRoPEAttention(nn.Module):
    fused_attn: torch.jit.Final[bool]

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
        norm_layer: Optional[type[nn.Module]] = None,
        grid_size: Optional[tuple[int, int]] = None,
        num_front_tokens: int = 0,
        rope_base: float = 100.0,
        rope_dim: Optional[int] = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        dd = {"device": device, "dtype": dtype}

        dim_out = dim_out or dim
        head_dim = attn_head_dim or dim // num_heads

        if attn_head_dim is None:
            assert dim % num_heads == 0, "`dim` should be divisible by `num_heads`."

        if qk_norm or scale_norm:
            assert norm_layer is not None, "`norm_layer` is required for qk_norm or scale_norm."

        rope_dim = head_dim if rope_dim is None else rope_dim
        if rope_dim > head_dim:
            raise ValueError(f"`rope_dim`={rope_dim} exceeds head_dim={head_dim}.")
        if rope_dim % 4 != 0:
            raise ValueError("For axial 2D RoPE, `rope_dim` must be divisible by 4.")

        if num_front_tokens < 0:
            raise ValueError("`num_front_tokens` must be non-negative.")

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attn_dim = num_heads * head_dim
        self.scale = head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.grid_size = grid_size
        self.num_front_tokens = num_front_tokens

        self.qkv = nn.Linear(dim, self.attn_dim * 3, bias=qkv_bias, **dd)
        self.q_norm = norm_layer(head_dim, **dd) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(head_dim, **dd) if qk_norm else nn.Identity()

        self.rope = Axial2dRotaryEmbedding(rope_dim, base=rope_base)

        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(self.attn_dim, **dd) if scale_norm else nn.Identity()
        self.proj = nn.Linear(self.attn_dim, dim_out, bias=proj_bias, **dd)
        self.proj_drop = nn.Dropout(proj_drop)

    def set_grid_size(self, grid_size: tuple[int, int]) -> None:
        self.grid_size = grid_size

    def _infer_grid_size(self, num_patch_tokens: int) -> tuple[int, int]:
        if self.grid_size is not None:
            gh, gw = self.grid_size
            if gh * gw != num_patch_tokens:
                raise ValueError(
                    f"`grid_size={self.grid_size}` implies {gh * gw} patches, "
                    f"but got {num_patch_tokens} patch tokens."
                )
            return gh, gw

        side = math.isqrt(num_patch_tokens)
        if side * side != num_patch_tokens:
            raise ValueError(
                "Cannot infer a non-square patch grid from the token sequence. "
                "Pass `grid_size=(H, W)` to AxialRoPEBlock."
            )
        return side, side

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        b, n, _ = x.shape

        num_patch_tokens = n - self.num_front_tokens
        if num_patch_tokens <= 0:
            raise ValueError(
                f"Expected patch tokens after {self.num_front_tokens} front tokens, "
                f"but got sequence length N={n}."
            )

        qkv = self.qkv(x).reshape(
            b, n, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = self.q_norm(q)
        k = self.k_norm(k)

        grid_size = self._infer_grid_size(num_patch_tokens)

        q, k = self.rope(
            q,
            k,
            grid_size=grid_size,
            num_front_tokens=self.num_front_tokens,
        )

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=is_causal,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)

            assert attn_mask is None  # No need to implement this - we don't use masks at all

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(b, n, self.attn_dim)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

class AxialRoPEBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        scale_attn_norm: bool = False,
        scale_mlp_norm: bool = False,
        proj_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = LayerNorm,
        mlp_layer: type[nn.Module] = Mlp,
        attn_layer=None,
        depth: int = 0,
        grid_size: Optional[tuple[int, int]] = None,
        num_front_tokens: int = 0,
        rope_base: float = 100.0,
        rope_dim: Optional[int] = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        dd = {"device": device, "dtype": dtype}

        self.norm1 = norm_layer(dim, **dd)
        self.attn = AxialRoPEAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            scale_norm=scale_attn_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            grid_size=grid_size,
            num_front_tokens=num_front_tokens,
            rope_base=rope_base,
            rope_dim=rope_dim,
            **dd,
        )

        self.ls1 = LayerScale(dim, init_values=init_values, **dd) if init_values is not None else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim, **dd)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            norm_layer=norm_layer if scale_mlp_norm else None,
            bias=proj_bias,
            drop=proj_drop,
            **dd,
        )

        self.ls2 = LayerScale(dim, init_values=init_values, **dd) if init_values is not None else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def set_grid_size(self, grid_size: tuple[int, int]) -> None:
        self.attn.set_grid_size(grid_size)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        x = x + self.drop_path1(
            self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal))
        )
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x