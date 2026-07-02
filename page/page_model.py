import torch
import torch.nn as nn
import torchvision
from timm.models.vision_transformer import Block
from timm.layers.mlp import SwiGLU, Mlp
from page.rope_self_attention import AxialRoPEBlock
from page.rope_cross_attention import AxialRoPECrossAttentionBlock
from page.cross_attention import CrossAttentionBlock

import page.utils as utils
from page.utils import positionalencoding2d, positionalencoding2d_aspect_batch


def get_vit_block(
    dim=256,
    num_heads=8,
    mlp_ratio=4,
    mlp_layer=SwiGLU,
    drop_path=0.1,
    act_layer=nn.GELU,
    pos_encoding='rope',
    num_front_tokens=4,
    rope_base=100.0,
) -> Block | AxialRoPEBlock:
    if pos_encoding not in {"rope", "sinusoidal", "ape"}:
        raise ValueError(f"pos_encoding must be one of the following: rope, sinusoidal or ape, got {pos_encoding} instead")
    if pos_encoding == 'rope':
        return AxialRoPEBlock(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            mlp_layer=mlp_layer,
            drop_path=drop_path,
            act_layer=act_layer,
            num_front_tokens=num_front_tokens,
            rope_base=rope_base,
        )
    else:
        return Block(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            mlp_layer=mlp_layer,
            drop_path=drop_path,
            act_layer=act_layer,
        )


def get_cross_attn_block(
    dim=256,
    num_heads=8,
    drop_path=0.1,
    pos_encoding='rope',
    q_num_front_tokens=0,
    kv_num_front_tokens=0,
    rope_base=100.0,
) -> Block | AxialRoPEBlock:
    if pos_encoding not in {"rope", "sinusoidal", "ape"}:
        raise ValueError(f"pos_encoding must be one of the following: rope, sinusoidal or ape, got {pos_encoding} instead")
    if pos_encoding == 'rope':
        return AxialRoPECrossAttentionBlock(
            dim=dim,
            num_heads=num_heads,
            drop_path=drop_path,
            q_num_front_tokens=q_num_front_tokens,
            kv_num_front_tokens=kv_num_front_tokens,
            rope_base=rope_base
        )
    else:
        return CrossAttentionBlock(
            dim=dim,
            num_heads=num_heads,
            drop_path=drop_path,
        )


class SceneHeadInteraction(nn.Module):
    """
        Scene-head Interaction Module (SIM)
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4, mlp_layer=SwiGLU, act_layer=nn.GELU, drop_path=0.0, num_front_tokens=0, pos_encoding='rope'):
        super().__init__()

        if pos_encoding not in {"rope", "sinusoidal", "ape"}:
            raise ValueError(f"pos_encoding must be one of the following: rope, sinusoidal or ape, got {pos_encoding} instead")
        self.pos_encoding = pos_encoding
        self.cross_attn_scene = get_cross_attn_block(
            dim=dim,
            num_heads=num_heads,
            pos_encoding=pos_encoding,
            q_num_front_tokens=num_front_tokens,
            kv_num_front_tokens=num_front_tokens,
            drop_path=drop_path,
        )
        self.vit_block_scene = get_vit_block(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            mlp_layer=mlp_layer,
            drop_path=drop_path,
            act_layer=act_layer,
            num_front_tokens=num_front_tokens,
            pos_encoding=pos_encoding,
        )
        self.cross_attn_head = get_cross_attn_block(
            dim=dim,
            num_heads=num_heads,
            pos_encoding=pos_encoding,
            q_num_front_tokens=num_front_tokens,
            kv_num_front_tokens=num_front_tokens,
            drop_path=drop_path,
        )
        self.vit_block_head = get_vit_block(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            mlp_layer=mlp_layer,
            drop_path=drop_path,
            act_layer=act_layer,
            num_front_tokens=num_front_tokens,
            pos_encoding=pos_encoding,
        )
    
    def forward(self, tokens):
        scene_tokens = tokens['scene_tokens']
        head_tokens = tokens['head_tokens']
        head_rects = tokens['head_rects']

        if self.pos_encoding == 'rope':
            out_scene_tokens = self.cross_attn_scene(scene_tokens, head_tokens, kv_rect=head_rects)
            out_head_tokens = self.cross_attn_head(head_tokens, scene_tokens, q_rect=head_rects)
        else:
            out_scene_tokens = self.cross_attn_scene(scene_tokens, head_tokens)
            out_head_tokens = self.cross_attn_head(head_tokens, scene_tokens)
        
        out_scene_tokens = self.vit_block_scene(out_scene_tokens)
        out_head_tokens = self.vit_block_head(out_head_tokens)

        return {
            "scene_tokens": out_scene_tokens,
            "head_tokens": out_head_tokens,
            "head_rects": head_rects
        }
    

class PaGE(nn.Module):
    """
    Gaze target estimation with explicit modeling of interaction between scene and head features through cross attn
    Scene -> DINOv3 -> ViT -|
                            |-> Interaction Layers -> Downstream Tasks
    Head  -> DINOv3 -> ViT -|
    """
    def __init__(self, scene_branch_backbone, head_branch_backbone, inout=False, dim=256, num_heads=8, mlp_ratio=4, mlp_layer='geglu', 
                 pos_encoding="rope", n_scene_self_attn_layers=1, n_head_self_attn_layers=1, n_scene_head_interaction_layers=5, 
                 n_reg_tokens=4, heatmap_out_size=(64, 64), dino_feature_dropout=0.1, drop_path=0.1, use_head_prompt=False):
        super().__init__()
        self.scene_branch_backbone = scene_branch_backbone
        self.head_branch_backbone = head_branch_backbone
        self.dim = dim
        self.n_scene_self_attn_layers = n_scene_self_attn_layers
        self.n_head_self_attn_layers = n_head_self_attn_layers
        self.n_scene_head_interaction_layers = n_scene_head_interaction_layers
        self.scene_featmap_h, self.scene_featmap_w = scene_branch_backbone.get_out_size()
        self.head_featmap_h, self.head_featmap_w = head_branch_backbone.get_out_size()
        self.n_reg_tokens = n_reg_tokens
        self.n_front_tokens = n_reg_tokens + 1 if inout else n_reg_tokens
        self.heatmap_out_size = heatmap_out_size
        self.inout = inout
        self.pos_encoding = pos_encoding
        self.use_head_prompt = use_head_prompt

        if pos_encoding not in {"rope", "sinusoidal", "ape"}:
            raise ValueError(f"pos_encoding must be one of the following: rope, sinusoidal or ape, got {pos_encoding} instead")

        self.scene_proj = nn.Sequential(
            nn.Dropout2d(dino_feature_dropout),
            nn.Conv2d(scene_branch_backbone.get_dimension(), self.dim, 1)
        )
        self.head_proj = nn.Sequential(
            nn.Dropout2d(dino_feature_dropout),
            nn.Conv2d(head_branch_backbone.get_dimension(), self.dim, 1)
        )
        if self.use_head_prompt:
            self.head_position_token = nn.Embedding(1, self.dim)  # weight shape: (1, 256), added to the scene feature map, the same as GazeLLE's original design

        if self.pos_encoding == 'ape':
            self.scene_seq_len = self.n_reg_tokens + self.scene_featmap_h * self.scene_featmap_w
            self.head_seq_len = self.n_reg_tokens + self.head_featmap_h * self.head_featmap_w
            self.scene_ape = nn.Parameter(torch.zeros((1, self.scene_seq_len, self.dim)))
            nn.init.trunc_normal_(self.scene_ape, std=0.02, a=-0.06, b=0.06)
            self.head_ape = nn.Parameter(torch.zeros((1, self.head_seq_len, self.dim)))
            nn.init.trunc_normal_(self.head_ape, std=0.02, a=-0.06, b=0.06)
        elif self.pos_encoding == 'sinusoidal':
            self.register_buffer("scene_pos_embed", positionalencoding2d(self.dim, self.scene_featmap_h, self.scene_featmap_w).squeeze(dim=0).squeeze(dim=0))
            self.register_buffer("head_pos_embed", positionalencoding2d(self.dim, self.head_featmap_h, self.head_featmap_w).squeeze(dim=0).squeeze(dim=0))

        if self.inout:
            self.scene_inout_token = nn.Parameter(torch.zeros((1, 1, self.dim)))
            nn.init.trunc_normal_(self.scene_inout_token, std=0.02, a=-0.06, b=0.06)
            self.head_inout_token = nn.Parameter(torch.zeros((1, 1, self.dim)))
            nn.init.trunc_normal_(self.head_inout_token, std=0.02, a=-0.06, b=0.06)   

        if self.n_reg_tokens > 0:
            self.scene_register_tokens = nn.Parameter(torch.zeros((1, self.n_reg_tokens, self.dim)))
            nn.init.trunc_normal_(self.scene_register_tokens, std=0.02, a=-0.06, b=0.06)
            self.head_register_tokens = nn.Parameter(torch.zeros((1, self.n_reg_tokens, self.dim)))
            nn.init.trunc_normal_(self.head_register_tokens, std=0.02, a=-0.06, b=0.06)        

        if mlp_layer == 'mlp':
            mlp_layer = Mlp
            act_layer = nn.GELU
        elif mlp_layer == 'geglu':
            mlp_layer = SwiGLU  # timm's SwiGLU implementation is equivalent to GEGLU when the activation function is GELU
            act_layer = nn.GELU
        elif mlp_layer == 'swiglu':
            mlp_layer = SwiGLU
            act_layer = nn.SiLU
        else:
            raise ValueError(f"mlp_layer must be one of the following: mlp or swiglu, got {mlp_layer} instead.")

        self.scene_self_attn_layers = nn.Sequential(*[
            get_vit_block(
                dim=self.dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                mlp_layer=mlp_layer,
                drop_path=drop_path,
                act_layer=act_layer,
                num_front_tokens=self.n_front_tokens,
                pos_encoding=pos_encoding
            ) for _ in range(n_scene_self_attn_layers)
        ]) if n_scene_self_attn_layers > 0 else nn.Identity()
        self.head_self_attn_layers = nn.Sequential(*[
            get_vit_block(
                dim=self.dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                mlp_layer=mlp_layer,
                drop_path=drop_path,
                act_layer=act_layer,
                num_front_tokens=self.n_front_tokens,
                pos_encoding=pos_encoding
            ) for _ in range(n_head_self_attn_layers)
        ]) if n_head_self_attn_layers > 0 else nn.Identity()
        self.scene_head_interaction_layers = nn.Sequential(*[
            SceneHeadInteraction(
                dim=self.dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                mlp_layer=mlp_layer,
                drop_path=drop_path,
                act_layer=act_layer,
                pos_encoding=pos_encoding,
                num_front_tokens=self.n_front_tokens,
            ) for _ in range(n_scene_head_interaction_layers)
        ])

        if self.heatmap_out_size[0] <= 64:
            self.heatmap_head = nn.Sequential(
                nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
                nn.Conv2d(dim, 1, kernel_size=1, bias=False),
            )
        else:
            self.heatmap_head = nn.Sequential(
                nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
                nn.Conv2d(dim, dim, 3, 1, 1),
                nn.GELU(),
                nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
                nn.Conv2d(dim, dim, 3, 1, 1),
                nn.GELU(),
                nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
                nn.Conv2d(dim, dim, 3, 1, 1),
                nn.GELU(),
                nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
                nn.Conv2d(dim, 1, kernel_size=1, bias=False),
            )

        if self.inout:
            self.inout_head = nn.Sequential(
                nn.Linear(self.dim * 2, 128),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
            )

    def get_logits(self, input, return_tokens=False):
        """
        Forward pass that produces raw logits.
        Used directly in training for superior numerical stability
        """
        num_ppl_per_img = [len(bbox_list) for bbox_list in input["bboxes"]]
        # make sure Np == Nhead for every head encoder backbone stream
        for head_stream_images in input["head_images"]:
            if sum(num_ppl_per_img) != len(head_stream_images):
                raise ValueError(f"bboxes and head crops mismatch: received {sum(num_ppl_per_img)} bboxes and {len(head_stream_images)} head crops.")

        # extract scene features and maybe add sinusoidal PE
        scene_featmap = self.scene_branch_backbone(input["images"])                               # [B, C, H', W']
        scene_featmap = self.scene_proj(scene_featmap)                                            # [B, D, H', W']
        scene_dino_tokens = scene_featmap.flatten(start_dim=2).permute(0, 2, 1)                   # [B, H'*W', D]
        if self.pos_encoding == 'sinusoidal':
            scene_featmap = scene_featmap + self.scene_pos_embed                                  # [B, D, H', W']
        scene_featmap = utils.repeat_tensors(scene_featmap, num_ppl_per_img)                      # [sum(Np), 256, H', W']

        # extract head features and maybe add sinusoidal PE
        head_featmap = self.head_branch_backbone(input["head_images"])                            # [sum(Np), C, Hh', Wh']
        head_featmap = self.head_proj(head_featmap)                                               # [sum(Np), 256, Hh', Wh']
        head_dino_tokens = head_featmap.flatten(start_dim=2).permute(0, 2, 1)                     # [sum(Np), Hh'*Wh', D]
        if self.pos_encoding == 'sinusoidal':
            head_featmap = head_featmap + self.head_pos_embed                                     # [sum(Np), 256, Hh', Wh']

        # add head position prompt to scene features in case sinusoidal or absolute PE is used
        head_maps, head_rects = self.get_input_head_maps(input["bboxes"])
        head_maps = torch.cat(head_maps, dim=0).to(scene_featmap.device)                          # [sum(Np), H', W']
        head_rects = torch.cat(head_rects, dim=0).to(scene_featmap.device)
        if self.use_head_prompt:
            head_map_embeddings = head_maps.unsqueeze(
                dim=1) * self.head_position_token.weight.unsqueeze(-1).unsqueeze(-1)
            scene_featmap = scene_featmap + head_map_embeddings

        # convert 2D feature maps into tokens, append register tokens (if any), append inout tokens (if needed)
        scene_tokens = scene_featmap.flatten(start_dim=2).permute(0, 2, 1)                        # [sum(Np), H'*W', 256]
        head_tokens = head_featmap.flatten(start_dim=2).permute(0, 2, 1)                          # [sum(Np), Hh'*Wh', 256]

        if self.n_reg_tokens > 0:
            scene_register_tokens = self.scene_register_tokens.expand(sum(num_ppl_per_img), -1, -1)
            scene_tokens = torch.cat([scene_register_tokens, scene_tokens], dim=1) 
            head_register_tokens = self.head_register_tokens.expand(sum(num_ppl_per_img), -1, -1)
            head_tokens = torch.cat([head_register_tokens, head_tokens], dim=1)
        
        if self.inout:
            scene_inout_token = self.scene_inout_token.expand(sum(num_ppl_per_img), -1, -1)
            scene_tokens = torch.cat([scene_inout_token, scene_tokens], dim=1) 
            head_inout_token = self.head_inout_token.expand(sum(num_ppl_per_img), -1, -1)
            head_tokens = torch.cat([head_inout_token, head_tokens], dim=1)
        
        # add learned absolute position encoding (if needed)
        if self.pos_encoding == 'ape':
            scene_tokens = scene_tokens + self.scene_ape.expand(sum(num_ppl_per_img), -1, -1)
            head_tokens = head_tokens + self.head_ape.expand(sum(num_ppl_per_img), -1, -1)

        # decode heatmap from scene tokens.
        # decode in/out score from the register tokens
        scene_tokens = self.scene_self_attn_layers(scene_tokens)
        head_tokens = self.head_self_attn_layers(head_tokens)
        tokens = self.scene_head_interaction_layers({
            "scene_tokens": scene_tokens,
            "head_tokens": head_tokens,
            "head_rects": head_rects
        })
        scene_tokens = tokens["scene_tokens"][:, self.n_front_tokens:, :]
        scene_inout_token = tokens["scene_tokens"][:, 0, :]
        head_inout_token = tokens["head_tokens"][:, 0, :]

        # decode inout score from the register tokens (AvgPool + MLP)
        if self.inout:
            inout_features = torch.cat((scene_inout_token, head_inout_token), dim=1)
            inout_preds = self.inout_head(inout_features).squeeze(dim=-1)
            inout_preds = utils.split_tensors(inout_preds, num_ppl_per_img)

        # decode heatmap (and maybe gaze target segmentation mask) from scene tokens
        scene_featmap = scene_tokens.reshape(scene_tokens.shape[0], self.scene_featmap_h,
                        self.scene_featmap_w, scene_tokens.shape[2]).permute(0, 3, 1, 2)  # [sum(Np), 256, H', W']
        heatmap = self.heatmap_head(scene_featmap).squeeze(dim=1)
        heatmap = torchvision.transforms.functional.resize(heatmap, self.heatmap_out_size)
        heatmap_preds = utils.split_tensors(heatmap, num_ppl_per_img)

        if return_tokens:
            return {"scene_tokens": tokens["scene_tokens"], "head_tokens": tokens["head_tokens"], "scene_dino_tokens": scene_dino_tokens, "head_dino_tokens": head_dino_tokens, "heatmap": heatmap_preds, "inout": inout_preds if self.inout else None}
        else:
            return {"heatmap": heatmap_preds, "inout": inout_preds if self.inout else None}

    def forward(self, input):
        """
        This is strictly for inference, applying sigmoid to the logits.
        Do NOT use this for training due to numerical stability issues.
        """
        logits = self.get_logits(input)
        heatmap_preds = [torch.sigmoid(heatmap) for heatmap in logits["heatmap"]]
        inout_preds = [torch.sigmoid(inout) for inout in logits["inout"]] if logits["inout"] is not None else None
        return {"heatmap": heatmap_preds, "inout": inout_preds if self.inout else None}

    def get_input_head_maps(self, bboxes):
        head_maps = []
        head_rects = []
        for bbox_list in bboxes:
            img_head_maps = []
            img_head_rects = []
            for bbox in bbox_list:
                if bbox is None:
                    img_head_maps.append(torch.zeros(
                        self.scene_featmap_h, self.scene_featmap_w))
                else:
                    xmin, ymin, xmax, ymax = bbox
                    width, height = self.scene_featmap_w, self.scene_featmap_h
                    xmin = round(xmin * width)
                    ymin = round(ymin * height)
                    xmax = round(xmax * width)
                    ymax = round(ymax * height)

                    head_map = torch.zeros((height, width))
                    head_map[ymin:ymax, xmin:xmax] = 1
                    img_head_maps.append(head_map)
                    img_head_rects.append(torch.Tensor([ymin, xmin, ymax, xmax]))
            head_maps.append(torch.stack(img_head_maps))
            head_rects.append(torch.stack(img_head_rects))
        return head_maps, head_rects

    def _backbone_trainable_prefixes(self):
        trainable_fn = getattr(self.scene_branch_backbone, "trainable_state_prefixes", None)
        if callable(trainable_fn):
            prefixes = trainable_fn()
        else:
            prefixes = []

        normalized = []
        for prefix in prefixes:
            if not prefix:
                continue
            normalized.append(prefix if prefix.startswith(
                "scene_branch_backbone") else f"scene_branch_backbone.{prefix}")

        trainable_fn = getattr(self.head_branch_backbone, "trainable_state_prefixes", None)
        if callable(trainable_fn):
            prefixes = trainable_fn()
        else:
            prefixes = []
        for prefix in prefixes:
            if not prefix:
                continue
            normalized.append(prefix if prefix.startswith(
                "head_branch_backbone") else f"head_branch_backbone.{prefix}")

        return normalized

    def get_page_state_dict(self, include_backbone=False):
        if include_backbone:
            return self.state_dict()

        allowed_prefixes = self._backbone_trainable_prefixes()
        filtered_state = {}
        for k, v in self.state_dict().items():
            if not (k.startswith("head_branch_backbone") or k.startswith("scene_branch_backbone") or k.startswith("backbone")):
                filtered_state[k] = v
                continue

            if any(k.startswith(prefix) for prefix in allowed_prefixes):
                filtered_state[k] = v

        return filtered_state

    def load_page_state_dict(self, ckpt_state_dict, include_backbone=False, enable_warning=False):
        current_state_dict = self.state_dict()
        keys1 = current_state_dict.keys()
        keys2 = ckpt_state_dict.keys()

        def filter_keys(keys):
            if include_backbone:
                return set(keys)

            allowed_prefixes = self._backbone_trainable_prefixes()
            filtered = set()
            for key in keys:
                if not (key.startswith("head_branch_backbone") or key.startswith("scene_branch_backbone") or key.startswith("backbone")):
                    filtered.add(key)
                    continue
                if any(key.startswith(prefix) for prefix in allowed_prefixes):
                    filtered.add(key)
            return filtered

        keys1 = filter_keys(keys1)
        keys2 = filter_keys(keys2)
        
        if enable_warning:
            if len(keys2 - keys1) > 0:
                print("WARNING unused keys in provided state dict: ", keys2 - keys1)
            if len(keys1 - keys2) > 0:
                print("WARNING provided state dict does not have values for keys: ", keys1 - keys2)

        for k in list(keys1 & keys2):
            current_state_dict[k] = ckpt_state_dict[k]

        self.load_state_dict(current_state_dict, strict=False)


    def load_output_head_state_dict(self, ckpt_state_dict, include_backbone=False):
        current_state_dict = self.state_dict()
        keys1 = current_state_dict.keys()
        keys2 = ckpt_state_dict.keys()

        def filter_keys(keys):
            allowed_prefixes = ["heatmap_head", "inout_head"]
            filtered = set()
            for key in keys:
                if any(key.startswith(prefix) for prefix in allowed_prefixes):
                    filtered.add(key)
            return filtered

        keys1 = filter_keys(keys1)
        keys2 = filter_keys(keys2)

        if len(keys2 - keys1) > 0:
            print("WARNING unused keys in provided state dict: ", keys2 - keys1)
        if len(keys1 - keys2) > 0:
            print(
                "WARNING provided state dict does not have values for keys: ", keys1 - keys2)

        for k in list(keys1 & keys2):
            current_state_dict[k] = ckpt_state_dict[k]

        self.load_state_dict(current_state_dict, strict=False)