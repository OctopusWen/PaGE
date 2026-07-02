from abc import ABC, abstractmethod
from typing import Optional
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import transforms
import torch.nn.functional as F
from timm.models.convnext import ConvNeXtBlock

from page.utils import TransposeLayerNorm, positionalencoding2d


# Abstract Backbone class
class Backbone(nn.Module, ABC):
    def __init__(self):
        super(Backbone, self).__init__()

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def get_dimension(self):
        pass

    @abstractmethod
    def get_out_size(self, in_size):
        pass

    def get_transform(self):
        pass

    # def trainable_state_prefixes(self):
    #     return []


# Official DINOv2 backbones from torch hub (https://github.com/facebookresearch/dinov2#pretrained-backbones-via-pytorch-hub)
class DinoV2Backbone(Backbone):
    def __init__(self, model_name, in_size):
        super(DinoV2Backbone, self).__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', model_name, force_reload=True)
        self.in_size = in_size

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        out_h, out_w = self.get_out_size((h, w))
        x = self.model.forward_features(x)['x_norm_patchtokens']
        # "b (out_h out_w) c -> b c out_h out_w"
        x = x.view(x.size(0), out_h, out_w, -1).permute(0, 3, 1, 2)
        return x

    def get_dimension(self):
        return self.model.embed_dim

    def get_out_size(self, in_size):
        h, w = in_size
        return (h // self.model.patch_size, w // self.model.patch_size)

    def get_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            transforms.Resize(self.in_size),
        ])


class DinoV2HFBackbone(Backbone):
    """
    DINOv2 backbone loaded via Hugging Face local checkpoint.
    Output: patch tokens -> feature map [B, C, H', W'].
    """

    def __init__(
        self,
        model_path: str = "/root/data/dinov2_model/dinov2_vitb14",
        in_size=(448, 448),
    ):
        super().__init__()
        self.model_path = model_path
        self.in_size = in_size

        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model.eval()

        config = getattr(self.model, "config", None)
        self.patch_size = getattr(config, "patch_size", 16)
        self.embed_dim = getattr(config, "hidden_size", None)
        self.num_register_tokens = int(getattr(config, "num_register_tokens", 0) or 0)

        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
            transforms.Resize(in_size),
        ])

    def _get_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=x, return_dict=True)
        tokens = getattr(out, "last_hidden_state", None)
        if tokens is None:
            tokens = out[0]

        if not torch.is_tensor(tokens) or tokens.dim() != 3:
            raise RuntimeError("Unexpected DINOv3 HF output format.")

        # If extra tokens (CLS + register) exist, drop them
        b, n, c = tokens.shape
        out_h, out_w = self.get_out_size((x.shape[2], x.shape[3]))
        target = out_h * out_w
        tokens = tokens[:, 1:, :]  # CLS: 1, Register: 0
        return tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        out_h, out_w = self.get_out_size((h, w))
        patch_tokens = self._get_patch_tokens(x)

        if patch_tokens.shape[1] != out_h * out_w:
            raise RuntimeError(
                f"[DinoV3HFBackbone] token count mismatch: {patch_tokens.shape[1]} vs {out_h*out_w}. "
                f"Check patch_size={self.patch_size} and input={(h, w)}"
            )

        feat = patch_tokens.view(
            b, out_h, out_w, -1).permute(0, 3, 1, 2).contiguous()
        return feat

    def get_dimension(self):
        if self.embed_dim is None:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, self.in_size[0], self.in_size[1])
                t = self._get_patch_tokens(dummy)
                self.embed_dim = t.shape[-1]
        return int(self.embed_dim)

    def get_out_size(self, in_size):
        h, w = in_size
        return (h // int(self.patch_size), w // int(self.patch_size))

    def get_transform(self):
        def _tf(pil_img):
            return self._transform(pil_img)
        return _tf
    


class DinoV3HFBackbone(Backbone):
    """
    DINOv3 backbone loaded via Hugging Face local checkpoint.
    Output: patch tokens -> feature map [B, C, H', W'].
    """

    def __init__(
        self,
        model_path: str = "/root/data/dinov3_model/DINOv3/dinov3_vit7b16",
        in_size=(512, 512),
    ):
        super().__init__()
        self.model_path = model_path
        self.in_size = in_size

        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model.eval()

        config = getattr(self.model, "config", None)
        self.patch_size = getattr(config, "patch_size", 16)
        self.embed_dim = getattr(config, "hidden_size", None)
        self.num_register_tokens = int(getattr(config, "num_register_tokens", 0) or 0)

        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
            transforms.Resize(in_size),
        ])

    def _get_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=x, return_dict=True)
        tokens = getattr(out, "last_hidden_state", None)
        if tokens is None:
            tokens = out[0]

        if not torch.is_tensor(tokens) or tokens.dim() != 3:
            raise RuntimeError("Unexpected DINOv3 HF output format.")

        # If extra tokens (CLS + register) exist, drop them
        b, n, c = tokens.shape
        out_h, out_w = self.get_out_size((x.shape[2], x.shape[3]))
        target = out_h * out_w
        tokens = tokens[:, 5:, :]  # CLS: 1, Register: 4
        return tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        out_h, out_w = self.get_out_size((h, w))
        patch_tokens = self._get_patch_tokens(x)

        if patch_tokens.shape[1] != out_h * out_w:
            raise RuntimeError(
                f"[DinoV3HFBackbone] token count mismatch: {patch_tokens.shape[1]} vs {out_h*out_w}. "
                f"Check patch_size={self.patch_size} and input={(h, w)}"
            )

        feat = patch_tokens.view(
            b, out_h, out_w, -1).permute(0, 3, 1, 2).contiguous()
        return feat

    def get_dimension(self):
        if self.embed_dim is None:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, self.in_size[0], self.in_size[1])
                t = self._get_patch_tokens(dummy)
                self.embed_dim = t.shape[-1]
        return int(self.embed_dim)

    def get_out_size(self, in_size):
        h, w = in_size
        return (h // int(self.patch_size), w // int(self.patch_size))

    def get_transform(self):
        def _tf(pil_img):
            return self._transform(pil_img)
        return _tf
    

class TIPSV2HFBackbone(Backbone):
    """
    TIPSv2 backbone loaded via Hugging Face local checkpoint.
    Output: patch tokens -> feature map [B, C, H', W'].
    """

    def __init__(
        self,
        model_path: str = "/root/data/TIPSv2/tipsv2_l14",
        in_size=(448, 448),
    ):
        super().__init__()
        self.model_path = model_path
        self.in_size = in_size

        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model.eval()

        config = getattr(self.model, "config", None)
        self.patch_size = getattr(config, "patch_size", 14)
        self.embed_dim = getattr(config, "embed_dim", None)
        self.num_register_tokens = int(getattr(config, "num_register_tokens", 0) or 0)

        # TIPS requires no ImageNet transform
        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(in_size),
        ])

    def _get_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model.encode_image(x)
        tokens = out.patch_tokens
        return tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        out_h, out_w = self.get_out_size((h, w))
        patch_tokens = self._get_patch_tokens(x)

        if patch_tokens.shape[1] != out_h * out_w:
            raise RuntimeError(
                f"[TIPSv2HFBackbone] token count mismatch: {patch_tokens.shape[1]} vs {out_h*out_w}. "
                f"Check patch_size={self.patch_size} and input={(h, w)}"
            )

        feat = patch_tokens.view(
            b, out_h, out_w, -1).permute(0, 3, 1, 2).contiguous()
        return feat

    def get_dimension(self):
        return int(self.embed_dim)

    def get_out_size(self, in_size):
        h, w = in_size
        return (h // int(self.patch_size), w // int(self.patch_size))

    def get_transform(self):
        def _tf(pil_img):
            return self._transform(pil_img)
        return _tf


class CLIPHFBackbone(Backbone):
    """
    CLIP backbone loaded via Hugging Face local checkpoint.
    Output: patch tokens -> feature map [B, C, H', W'].
    """

    def __init__(
        self,
        model_path: str = "/root/data/CLIP/clip-vit-large-patch14-336",
        in_size=(448, 448),
    ):
        super().__init__()
        self.model_path = model_path
        self.in_size = in_size

        from transformers import CLIPVisionModel

        self.model = CLIPVisionModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model.eval()

        config = getattr(self.model, "config", None)
        self.patch_size = getattr(config, "patch_size", 14)
        self.embed_dim = getattr(config, "hidden_size", None)

        # CLIP uses a custom transform
        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[
                                 0.26862954, 0.26130258, 0.27577711]),
            transforms.Resize(in_size),
        ])

    def _get_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=x, interpolate_pos_encoding=True)
        tokens = out.last_hidden_state[:, 1:, :]  # remove CLS token
        return tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        out_h, out_w = self.get_out_size((h, w))
        patch_tokens = self._get_patch_tokens(x)

        if patch_tokens.shape[1] != out_h * out_w:
            raise RuntimeError(
                f"[CLIPHFBackbone] token count mismatch: {patch_tokens.shape[1]} vs {out_h*out_w}. "
                f"Check patch_size={self.patch_size} and input={(h, w)}"
            )

        feat = patch_tokens.view(
            b, out_h, out_w, -1).permute(0, 3, 1, 2).contiguous()
        return feat

    def get_dimension(self):
        return int(self.embed_dim)

    def get_out_size(self, in_size):
        h, w = in_size
        return (h // int(self.patch_size), w // int(self.patch_size))

    def get_transform(self):
        def _tf(pil_img):
            return self._transform(pil_img)
        return _tf