"""
Fusion wrappers are kept here.
For the sake of consistency in training and inference, we put all simple backbones (e.g. DINOv2-only) in a SimpleBackboneWrapper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

from page.backbone import *
from page.utils import TransposeLayerNorm


class BackboneWrapper(nn.Module, ABC):
    def __init__(self):
        super(BackboneWrapper, self).__init__()

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def get_dimension(self):
        pass

    @abstractmethod
    def get_out_size(self, in_size):
        pass

    def get_transforms(self):
        pass


class FusionConcatBackboneWrapper(BackboneWrapper):
    """
    A 2-backbone wrapper where the features are concatenated (along the channel dimension) directly.
    Not used in the final version of PaGE. You can play with hybrid backbones (e.g. DINOv3 + CLIP) using this wrapper :)
    Optional LayerNorm before concat 
    Optional Trainable backbone
    """
    def __init__(
        self,
        backbone_a: Backbone,
        backbone_b: Backbone,
        freeze_backbone_a: bool = True,
        freeze_backbone_b: bool = True,
        norm_a_before_cat: bool = True,
        norm_b_before_cat: bool = True,
    ):
        super().__init__()
        self.a = backbone_a
        self.b = backbone_b
        
        # Ensure output size is the same so that the dense features can be concatenated: (Ha, Wa) == (Hb, Wb) == (H, W)
        a_out_size, b_out_size = self.a.get_out_size(self.a.in_size), self.b.get_out_size(self.b.in_size)
        if a_out_size != b_out_size:
            raise ValueError(f'Feature map size mismatch: backbone a has {a_out_size}, backbone b has {b_out_size}.')

        self.freeze_backbone_a = freeze_backbone_a
        self.freeze_backbone_b = freeze_backbone_b
        if freeze_backbone_a:
            self.a.requires_grad_(False)
        if freeze_backbone_b:
            self.b.requires_grad_(False)

        self.norm_a_before_cat = norm_a_before_cat
        self.norm_b_before_cat = norm_b_before_cat
        self.ln_a = TransposeLayerNorm(self.a.get_dimension()) if norm_a_before_cat else nn.Identity()
        self.ln_b = TransposeLayerNorm(self.b.get_dimension()) if norm_b_before_cat else nn.Identity()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone_a:
            self.a.eval()
        if self.freeze_backbone_b:
            self.b.eval()
        return self

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        assert len(x) == 2
        if self.freeze_backbone_a:
            with torch.inference_mode():
                Fa = self.a(x[0])      # [B, Ca, H, W]
        else:
            Fa = self.a(x[0])          # [B, Ca, H, W]
        if self.freeze_backbone_b:
            with torch.inference_mode():
                Fb = self.b(x[1])      # [B, Cb, H, W]
        else:
            Fb = self.b(x[1])          # [B, Cb, H, W]

        # If needed, layer norm the features from different backbones before concatenation
        Fa = self.ln_a(Fa)
        Fb = self.ln_b(Fb)

        # Concatenate along channel dimension -> [B, Ca+Cb, H, W]
        fused = torch.cat([Fa, Fb], dim=1)
        return fused
    
    def trainable_state_prefixes(self):
        ret = []
        if self.norm_a_before_cat:
            ret.append('ln_a')
        if self.norm_b_before_cat:
            ret.append('ln_b')
        if not self.freeze_backbone_a:
            ret.append('a')
        if not self.freeze_backbone_b:
            ret.append('b')
        return ret

    def get_dimension(self):
        return self.a.get_dimension() + self.b.get_dimension()

    def get_out_size(self):
        # Use backbone_a as reference (already checked (Ha, Wa) == (Hb, Wb) during instantiation)
        return self.a.get_out_size(self.a.in_size)

    def get_transforms(self):
        return [self.a.get_transform(), self.b.get_transform()]
    


class SimpleBackboneWrapper(BackboneWrapper):
    """
    A simple backbone wrapper to align the input and output format of a single backbone model 
    with more sophisticated fusion backbones

    Fixed various potential issues present in the v1 implementation
    """
    def __init__(self, backbone: Backbone, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self.backbone.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        assert len(x) == 1
        if self.freeze_backbone:
            with torch.inference_mode():
                out = self.backbone(x[0])  # [B, C, H, W]
        else:
            out = self.backbone(x[0])      # [B, C, H, W]
        return out
        
    def get_dimension(self):
        return self.backbone.get_dimension()

    def get_out_size(self):
        # Use backbone_a as reference (already checked during instantiation)
        return self.backbone.get_out_size(self.backbone.in_size)

    def trainable_state_prefixes(self):
        return [] if self.freeze_backbone else ['backbone']

    def get_transforms(self):
        return [self.backbone.get_transform()]