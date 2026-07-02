import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.mlp import SwiGLU, Mlp
from timm.layers import DropPath, LayerScale, LayerNorm, use_fused_attn
from typing import Optional, Type


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


class CrossAttention(nn.Module):
    """
    timm-style multi-head cross attention.

    Query comes from x_q, key/value come from x_kv.
    Shapes:
        x_q:  (B, Nq, C)
        x_kv: (B, Nk, C)
        out:  (B, Nq, C)
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
    ) -> None:
        super().__init__()

        dim_out = dim_out or dim
        if attn_head_dim is None:
            assert dim % num_heads == 0, "dim should be divisible by num_heads"
            head_dim = dim // num_heads
        else:
            head_dim = attn_head_dim

        if qk_norm or scale_norm:
            assert norm_layer is not None, "norm_layer must be provided if qk_norm or scale_norm is True"

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attn_dim = num_heads * head_dim
        self.scale = head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.q = nn.Linear(dim, self.attn_dim, bias=qkv_bias)
        self.k = nn.Linear(dim, self.attn_dim, bias=qkv_bias)
        self.v = nn.Linear(dim, self.attn_dim, bias=qkv_bias)

        self.q_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(head_dim) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(self.attn_dim) if scale_norm else nn.Identity()
        self.proj = nn.Linear(self.attn_dim, dim_out, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, Nq, _ = x_q.shape
        Bkv, Nk, _ = x_kv.shape
        assert B == Bkv, "x_q and x_kv must have the same batch size"

        q = self.q(x_q).reshape(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x_kv).reshape(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x_kv).reshape(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

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

        x = x.transpose(1, 2).reshape(B, Nq, self.attn_dim)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionBlock(nn.Module):
    """
    timm ViT Block-style cross-attention block without FFN.

    Layout:
        x_q = x_q + DropPath(LayerScale(CrossAttn(Norm(x_q), Norm(x_kv))))
        x_q = x_q + DropPath(LayerScale(MLP(Norm(x_q))))

    Only the query stream is updated; x_kv is read-only for this block.
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
    ) -> None:
        super().__init__()

        self.norm_q = norm_layer(dim)
        self.norm_kv = norm_layer(dim)

        self.attn = CrossAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            scale_norm=scale_attn_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_q = x_q + self.drop_path(
            self.ls1(self.attn(self.norm_q(x_q), self.norm_kv(x_kv), attn_mask=attn_mask))
        )
        return x_q