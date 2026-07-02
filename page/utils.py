import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import random
from sklearn.metrics import roc_auc_score

def repeat_tensors(tensor, repeat_counts):
    repeated_tensors = [tensor[i:i+1].repeat(repeat, *[1] * (tensor.ndim - 1)) for i, repeat in enumerate(repeat_counts)]
    return torch.cat(repeated_tensors, dim=0)

def split_tensors(tensor, split_counts):
    indices = torch.cumsum(torch.tensor([0] + split_counts), dim=0)
    return [tensor[indices[i]:indices[i+1]] for i in range(len(split_counts))]

def visualize_heatmap(pil_image, heatmap, bbox=None):
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()
    heatmap = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(pil_image.size, Image.Resampling.BILINEAR)
    heatmap = plt.cm.jet(np.array(heatmap) / 255.)
    heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)
    heatmap = Image.fromarray(heatmap).convert("RGBA")
    heatmap.putalpha(128)
    overlay_image = Image.alpha_composite(pil_image.convert("RGBA"), heatmap)

    if bbox is not None:
        width, height = pil_image.size
        xmin, ymin, xmax, ymax = bbox
        draw = ImageDraw.Draw(overlay_image)
        draw.rectangle([xmin * width, ymin * height, xmax * width, ymax * height], outline="green", width=3)
    return overlay_image

def stack_and_pad(tensor_list):
    max_size = max([t.shape[0] for t in tensor_list])
    padded_list = []
    for t in tensor_list:
        if t.shape[0] == max_size:
            padded_list.append(t)
        else:
            padded_list.append(torch.cat([t, torch.zeros(max_size - t.shape[0], *t.shape[1:])], dim=0))
    return torch.stack(padded_list)

def random_crop(img, bbox, gazex, gazey, inout):
    width, height = img.size
    bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = bbox
    # determine feasible crop region (must include bbox and gaze target)
    crop_reg_xmin = min(bbox_xmin, min(gazex)) if inout else bbox_xmin
    crop_reg_ymin = min(bbox_ymin, min(gazey)) if inout else bbox_ymin
    crop_reg_xmax = max(bbox_xmax, max(gazex)) if inout else bbox_xmax
    crop_reg_ymax = max(bbox_ymax, max(gazey)) if inout else bbox_ymax

    try:
        xmin = random.randint(0, int(crop_reg_xmin))
        ymin = random.randint(0, int(crop_reg_ymin))
        xmax = random.randint(int(crop_reg_xmax), width)
        ymax = random.randint(int(crop_reg_ymax), height)
    except:
        import pdb; pdb.set_trace()

    img = torchvision.transforms.functional.crop(img, ymin, xmin, ymax - ymin, xmax - xmin)
    bbox = [bbox_xmin - xmin, bbox_ymin - ymin, bbox_xmax - xmin, bbox_ymax - ymin]
    gazex = [x - xmin for x in gazex]
    gazey = [y - ymin for y in gazey]

    return img, bbox, gazex, gazey

def horiz_flip(img, bbox, gazex, gazey, inout):
    width, height = img.size
    img = torchvision.transforms.functional.hflip(img)
    xmin, ymin, xmax, ymax = bbox
    bbox = [width - xmax, ymin, width - xmin, ymax]
    if inout:
        gazex = [width - x for x in gazex]
    return img, bbox, gazex, gazey

def random_bbox_jitter(img, bbox):
    width, height = img.size
    xmin, ymin, xmax, ymax = bbox
    jitter = 0.2
    xmin_j = (np.random.random_sample() * (jitter*2) - jitter) * (xmax - xmin)
    xmax_j = (np.random.random_sample() * (jitter*2) - jitter) * (xmax - xmin)
    ymin_j = (np.random.random_sample() * (jitter*2) - jitter) * (ymax - ymin)
    ymax_j = (np.random.random_sample() * (jitter*2) - jitter) * (ymax - ymin)

    bbox = [max(0, xmin_j + xmin), max(0, ymin_j + ymin), min(width, xmax_j + xmax), min(height, ymax_j + ymax)]

    return bbox

def random_ground_truth_jitter(img, gazex, gazey):
    width, height = img.size
    jitter = 0.03
    gazex = [x + (np.random.random_sample() * (jitter*2) - jitter) * width for x in gazex]
    gazey = [y + (np.random.random_sample() * (jitter*2) - jitter) * height for y in gazey]
    gazex = [max(0, min(x, width)) for x in gazex]
    gazey = [max(0, min(y, height)) for y in gazey]
    
    return gazex, gazey


def _get_rot_params(width, height, angle_deg):
    """
    Geometry that matches PIL/torchvision rotate(..., expand=True).

    Returns:
        cos_t, sin_t, cx, cy, tx, ty, new_w, new_h
    """
    cx = width / 2.0
    cy = height / 2.0

    # IMPORTANT: image coordinates are x-right, y-down,
    # so point annotations must use -angle to match image rotation.
    theta = math.radians(-angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    def rot_noshift(x, y):
        x0 = x - cx
        y0 = y - cy
        xr = x0 * cos_t - y0 * sin_t + cx
        yr = x0 * sin_t + y0 * cos_t + cy
        return xr, yr

    corners = [
        (0.0, 0.0),
        (width, 0.0),
        (width, height),
        (0.0, height),
    ]
    rot_corners = [rot_noshift(x, y) for x, y in corners]
    xs = [p[0] for p in rot_corners]
    ys = [p[1] for p in rot_corners]

    min_x = min(xs)
    min_y = min(ys)
    max_x = max(xs)
    max_y = max(ys)

    # This matches expand=True behavior
    tx = -math.floor(min_x)
    ty = -math.floor(min_y)
    new_w = int(math.ceil(max_x) - math.floor(min_x))
    new_h = int(math.ceil(max_y) - math.floor(min_y))

    return cos_t, sin_t, cx, cy, tx, ty, new_w, new_h


def _rotate_points(points, cos_t, sin_t, cx, cy, tx, ty):
    out = []
    for x, y in points:
        x0 = x - cx
        y0 = y - cy
        xr = x0 * cos_t - y0 * sin_t + cx + tx
        yr = x0 * sin_t + y0 * cos_t + cy + ty
        out.append((xr, yr))
    return out


def rotate_sample(img, bbox, gazex, gazey, inout, angle_deg,
                  transform_out_of_frame=False):
    """
    Rotate one sample while preserving all pixels with expand=True.

    Args:
        img: PIL image
        bbox: [xmin, ymin, xmax, ymax] in pixel coords
        gazex, gazey: lists of gaze annotations in pixel coords
        inout: True if target is inside frame
        angle_deg: rotation angle in degrees, same convention as PIL/torchvision
        transform_out_of_frame: rotate gaze labels even when inout=False

    Returns:
        img, bbox, gazex, gazey
    """
    width, height = img.size

    cos_t, sin_t, cx, cy, tx, ty, new_w, new_h = _get_rot_params(
        width, height, angle_deg
    )

    # Rotate image without losing content/resolution
    fill = 0 if len(img.getbands()) == 1 else tuple(0 for _ in img.getbands())
    img = TF.rotate(
        img,
        angle_deg,
        interpolation=InterpolationMode.BILINEAR,
        expand=True,
        fill=fill,
    )
    assert img.size == (new_w, new_h)

    # Rotate bbox corners, then take enclosing axis-aligned box
    xmin, ymin, xmax, ymax = bbox
    bbox_corners = [
        (xmin, ymin),
        (xmax, ymin),
        (xmax, ymax),
        (xmin, ymax),
    ]
    rot_bbox = _rotate_points(bbox_corners, cos_t, sin_t, cx, cy, tx, ty)

    xs = [p[0] for p in rot_bbox]
    ys = [p[1] for p in rot_bbox]

    # Use floor/ceil so the rotated head is fully contained for cropping
    bbox = [
        max(0, int(math.floor(min(xs)))),
        max(0, int(math.floor(min(ys)))),
        min(new_w, int(math.ceil(max(xs)))),
        min(new_h, int(math.ceil(max(ys)))),
    ]

    # Rotate gaze annotations
    if inout or transform_out_of_frame:
        gaze_points = list(zip(gazex, gazey))
        rot_gaze = _rotate_points(gaze_points, cos_t, sin_t, cx, cy, tx, ty)
        gazex = [p[0] for p in rot_gaze]
        gazey = [p[1] for p in rot_gaze]

    return img, bbox, gazex, gazey


def random_rotate(img, bbox, gazex, gazey, inout, degrees=(-30, 30), transform_out_of_frame=False):
    angle_deg = random.uniform(degrees[0], degrees[1])
    return rotate_sample(
        img, bbox, gazex, gazey, inout, angle_deg,
        transform_out_of_frame=transform_out_of_frame
    )


def get_heatmap(gazex, gazey, height, width, sigma=3, htype="Gaussian"):
    # Adapted from https://github.com/ejcgt/attention-target-detection/blob/master/utils/imutils.py

    img = torch.zeros(height, width)
    if gazex < 0 or gazey < 0:  # return empty map if out of frame
        return img
    gazex = int(gazex * width)
    gazey = int(gazey * height)

    # scale sigma if width and height are not 64, which is the default used in all datasets
    if height != 64 or width != 64:
        sigma *= round(max(height, width) / 64)

    # Check that any part of the gaussian is in-bounds
    ul = [int(gazex - 3 * sigma), int(gazey - 3 * sigma)]
    br = [int(gazex + 3 * sigma + 1), int(gazey + 3 * sigma + 1)]
    if ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or br[0] < 0 or br[1] < 0:
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if htype == "Gaussian":
        g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))
    elif htype == "Cauchy":
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma**2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0] : img_y[1], img_x[0] : img_x[1]] += g[g_y[0] : g_y[1], g_x[0] : g_x[1]]
    img = img / img.max()  # normalize heatmap so it has max value of 1
    return img

# GazeFollow calculates AUC using original image size with GT (x,y) coordinates set to 1 and everything else as 0
# References:
    # https://github.com/ejcgt/attention-target-detection/blob/acd264a3c9e6002b71244dea8c1873e5c5818500/eval_on_gazefollow.py#L78
    # https://github.com/ejcgt/attention-target-detection/blob/acd264a3c9e6002b71244dea8c1873e5c5818500/utils/imutils.py#L67
    # https://github.com/ejcgt/attention-target-detection/blob/acd264a3c9e6002b71244dea8c1873e5c5818500/utils/evaluation.py#L7
def gazefollow_auc(heatmap, gt_gazex, gt_gazey, height, width):
    target_map = np.zeros((height, width))
    for point in zip(gt_gazex, gt_gazey):
        if point[0] >= 0:
            x, y = map(int, [point[0]*float(width), point[1]*float(height)])
            x = min(x, width - 1)
            y = min(y, height - 1)
            target_map[y, x] = 1
    resized_heatmap = torch.nn.functional.interpolate(heatmap.unsqueeze(dim=0).unsqueeze(dim=0), (height, width), mode='bilinear').squeeze()
    auc = roc_auc_score(target_map.flatten(), resized_heatmap.cpu().flatten())
    
    return auc

# Reference: https://github.com/ejcgt/attention-target-detection/blob/acd264a3c9e6002b71244dea8c1873e5c5818500/eval_on_gazefollow.py#L81
def gazefollow_l2(heatmap, gt_gazex, gt_gazey, no_bias=False):
    argmax = heatmap.flatten().argmax().item()
    pred_y, pred_x = np.unravel_index(argmax, (heatmap.shape[0], heatmap.shape[1]))
    pred_x = pred_x / float(heatmap.shape[1])
    pred_y = pred_y / float(heatmap.shape[0])
    if no_bias:
        pred_x += 0.5
        pred_y += 0.5

    gazex = np.array(gt_gazex)
    gazey = np.array(gt_gazey)

    avg_l2 = np.sqrt((pred_x - gazex.mean())**2 + (pred_y - gazey.mean())**2)
    all_l2s = np.sqrt((pred_x - gazex)**2 + (pred_y - gazey)**2)
    min_l2 = all_l2s.min().item()

    return avg_l2, min_l2

# VideoAttentionTarget calculates AUC on 64x64 heatmap, defining a rectangular tolerance region of 6*(sigma=3) + 1 (uses 2D Gaussian code but binary thresholds > 0 resulting in rectangle)
# References:
    # https://github.com/ejcgt/attention-target-detection/blob/acd264a3c9e6002b71244dea8c1873e5c5818500/eval_on_videoatttarget.py#L106
    # https://github.com/ejcgt/attention-target-detection/blob/acd264a3c9e6002b71244dea8c1873e5c5818500/utils/imutils.py#L31
def vat_auc(heatmap, gt_gazex, gt_gazey, res=64, sigma=3):
    if res != 64:  # rescale sigma so that results remain comparable to standerd procedure when heatmap resolution is not 64
        sigma *= round(res / 64)
    assert heatmap.shape[0] == res and heatmap.shape[1] == res
    target_map = np.zeros((res, res))
    gazex = gt_gazex * res
    gazey = gt_gazey * res
    ul = [max(0, int(gazex - 3 * sigma)), max(0, int(gazey - 3 * sigma))]
    br = [min(int(gazex + 3 * sigma + 1), res-1), min(int(gazey + 3 * sigma + 1), res-1)]
    target_map[ul[1]:br[1], ul[0]:br[0]] = 1
    auc = roc_auc_score(target_map.flatten(), heatmap.cpu().flatten())
    return auc

# Reference: https://github.com/ejcgt/attention-target-detection/blob/acd264a3c9e6002b71244dea8c1873e5c5818500/eval_on_videoatttarget.py#L118
def vat_l2(heatmap, gt_gazex, gt_gazey, res=64, no_bias=False):
    argmax = heatmap.flatten().argmax().item()
    pred_y, pred_x = np.unravel_index(argmax, (res, res))
    pred_x = pred_x / res
    pred_y = pred_y / res
    if no_bias:
        pred_x += 0.5
        pred_y += 0.5

    l2 = np.sqrt((pred_x - gt_gazex)**2 + (pred_y - gt_gazey)**2)

    return l2


class CosineLRWithWarmup(_LRScheduler):
    """
    - Warmup for the first `warmup_iters` optimizer steps (typically first epoch only),
      linearly from `warmup_start_lr` to `base_lr`.
    - After warmup, LR changes only at epoch boundaries, following cosine annealing
      from `base_lr` down to `min_lr` over `max_epochs-1` epochs (epoch 0 is the first epoch).

    Usage:
      - Call scheduler.step_batch() every iteration.
      - Call scheduler.step_epoch(epoch) once at the end of each epoch (or beginning; be consistent).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_iters: int = 10,
        warmup_start_lr: float = 1e-4,
        base_lr: float = 1e-3,
        min_lr: float = 0.0,
        max_epochs: int = 100,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        self.warmup_iters = int(warmup_iters)
        self.warmup_start_lr = float(warmup_start_lr)
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)
        self.max_epochs = int(max_epochs)

        self._global_step = 0
        self._current_epoch = 0

        # Set the optimizer's initial LR to warmup_start_lr (explicit & predictable)
        for pg in optimizer.param_groups:
            pg["lr"] = self.warmup_start_lr

        super().__init__(optimizer, last_epoch=last_epoch)

    def _set_lr(self, lr: float) -> None:
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def get_lr(self):
        # Required by PyTorch internals; we manage LR via step_batch/step_epoch.
        # Return current lrs (do not change anything here).
        return [pg["lr"] for pg in self.optimizer.param_groups]

    def step_batch(self) -> float:
        """
        Call once per optimizer step/iteration.
        Returns the new LR.
        """
        if self._current_epoch == 0 and self._global_step < self.warmup_iters:
            # Linear warmup from warmup_start_lr -> base_lr inclusive
            t = (self._global_step + 1) / max(1, self.warmup_iters)  # (0,1]
            lr = self.warmup_start_lr + t * (self.base_lr - self.warmup_start_lr)
            self._set_lr(lr)
        else:
            # After warmup, LR is epoch-based; do nothing per batch.
            lr = self.optimizer.param_groups[0]["lr"]

        self._global_step += 1
        return lr

    def step_epoch(self, epoch: int | None = None) -> float:
        """
        Call once per epoch boundary.
        - If epoch is provided, uses it; otherwise increments internal epoch counter.
        Returns the new LR.
        """
        if epoch is None:
            self._current_epoch += 1
        else:
            self._current_epoch = int(epoch)

        # Epoch 0: keep base_lr (after warmup, you'll likely be at base_lr already)
        # Epoch 1..max_epochs-1: cosine decay
        if self._current_epoch <= 0:
            lr = self.base_lr
        else:
            # We want epoch 1 < epoch 0, so start cosine decay at epoch 1.
            # progress in [0, 1] across epochs 1..max_epochs-1
            denom = max(1, self.max_epochs - 1)
            progress = min(1.0, self._current_epoch / denom)

            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))

        # Important: if we're *still* in warmup steps inside epoch 0, don't override warmup.
        # So only apply epoch LR if warmup is done OR we are past epoch 0.
        warmup_done = (self._global_step >= self.warmup_iters)
        if self._current_epoch > 0 or warmup_done:
            self._set_lr(lr)

        return self.optimizer.param_groups[0]["lr"]


class TransposeLayerNorm(nn.Module):
    """
        Transpose 2D feature maps for layer norm, and then transpose back to original shape
    """
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
    

def positionalencoding2d(d_model, height, width):
    if d_model % 4 != 0:
        raise ValueError(
            "Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2)
                         * -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(
        pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(
        pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(
        pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :,
        :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    return pe


def positionalencoding2d_aspect_batch(x: torch.Tensor, aspect_ratio: torch.Tensor) -> torch.Tensor:
    """
    Batched 2D sinusoidal positional encoding with per-sample aspect ratio.

    Args:
        x: Tensor of shape [B, C, H, W]
        aspect_ratio: Tensor of shape [B, 1] containing original width / height

    Returns:
        pe: Tensor of shape [B, C, H, W]
    """
    if x.ndim != 4:
        raise ValueError(f"x must have shape [B, C, H, W], got {tuple(x.shape)}")

    B, C, H, W = x.shape

    if aspect_ratio.ndim != 2 or aspect_ratio.shape != (B, 1):
        raise ValueError(
            f"aspect_ratio must have shape [B, 1], got {tuple(aspect_ratio.shape)}"
        )

    if C % 4 != 0:
        raise ValueError(
            f"Cannot use 2D sin/cos positional encoding when C % 4 != 0 (got C={C})"
        )

    device = x.device
    pe_dtype = torch.float32  # do trig in fp32 for stability

    aspect_ratio = aspect_ratio.to(device=device, dtype=pe_dtype).clamp_min(1e-6)

    half_c = C // 2
    quarter_c = C // 4

    # Frequency terms
    div_term = torch.exp(
        torch.arange(0, half_c, 2, device=device, dtype=pe_dtype)
        * -(math.log(10000.0) / half_c)
    )  # [C/4]

    # Max-side-normalized coordinate scaling:
    # landscape (r >= 1): x full scale, y compressed by 1/r
    # portrait  (r <  1): y full scale, x compressed by r
    x_scale = torch.where(aspect_ratio >= 1.0, 1.0, aspect_ratio)      # [B,1]
    y_scale = torch.where(aspect_ratio >= 1.0, 1.0 / aspect_ratio, 1.0)  # [B,1]

    base = float(max(H, W))

    # Centered normalized coordinates, then scaled per sample
    x_pos = ((torch.arange(W, device=device, dtype=pe_dtype) + 0.5) / W)  # [W]
    y_pos = ((torch.arange(H, device=device, dtype=pe_dtype) + 0.5) / H)  # [H]

    x_pos = x_pos.unsqueeze(0) * (x_scale * base)  # [B, W]
    y_pos = y_pos.unsqueeze(0) * (y_scale * base)  # [B, H]

    # Outer product with frequencies
    x_phase = x_pos.unsqueeze(-1) * div_term.view(1, 1, quarter_c)  # [B, W, C/4]
    y_phase = y_pos.unsqueeze(-1) * div_term.view(1, 1, quarter_c)  # [B, H, C/4]

    x_sin = torch.sin(x_phase)  # [B, W, C/4]
    x_cos = torch.cos(x_phase)
    y_sin = torch.sin(y_phase)  # [B, H, C/4]
    y_cos = torch.cos(y_phase)

    pe = torch.zeros((B, C, H, W), device=device, dtype=pe_dtype)

    # Encode x dimension into first half of channels
    pe[:, 0:half_c:2, :, :] = x_sin.permute(0, 2, 1).unsqueeze(2).expand(-1, -1, H, -1)
    pe[:, 1:half_c:2, :, :] = x_cos.permute(0, 2, 1).unsqueeze(2).expand(-1, -1, H, -1)

    # Encode y dimension into second half of channels
    pe[:, half_c::2, :, :] = y_sin.permute(0, 2, 1).unsqueeze(3).expand(-1, -1, -1, W)
    pe[:, half_c + 1::2, :, :] = y_cos.permute(0, 2, 1).unsqueeze(3).expand(-1, -1, -1, W)

    return pe.to(dtype=x.dtype)


def cosine_mean(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.reshape(-1, a.shape[-1])
    b_flat = b.reshape(-1, b.shape[-1])
    return float(F.cosine_similarity(a_flat, b_flat, dim=-1).mean().item())

def relation_kl(pred: torch.Tensor, target: torch.Tensor, temperature: float = 1.0) -> float:
    pred_rel = torch.matmul(pred, pred.transpose(-1, -2)) / max(pred.shape[-1], 1)
    target_rel = torch.matmul(target, target.transpose(-1, -2)) / max(target.shape[-1], 1)
    pred_log_probs = F.log_softmax(pred_rel / temperature, dim=-1)
    target_probs = F.softmax(target_rel / temperature, dim=-1)
    return float(F.kl_div(pred_log_probs, target_probs, reduction="batchmean").item())

class L1CosineLoss(nn.Module):
    def __init__(self, cosine_lambda=1.0):
        super().__init__()
        self.cosine_lambda = cosine_lambda
    
    def forward(self, pred, target):
        l1 = F.l1_loss(pred, target)
        cos = F.cosine_similarity(pred, target, dim=-1).mean()
        return l1 + self.cosine_lambda * (1.0 - cos)
    
class MSECosineLoss(nn.Module):
    def __init__(self, cosine_lambda=1.0):
        super().__init__()
        self.cosine_lambda = cosine_lambda
    
    def forward(self, pred, target):
        mse = F.mse_loss(pred, target)
        cos = F.cosine_similarity(pred, target, dim=-1).mean()
        return mse + self.cosine_lambda * (1.0 - cos)

def make_param_groups(model: nn.Module, weight_decay=0.05):
    decay = []
    no_decay = []
    no_decay_names = []
    
    no_decay_keywords = [
        "inout_token",
        "scene_inout_token",
        "head_inout_token",
        "scene_register_tokens",
        "head_register_tokens",
        "scene_ape",
        "head_ape",
        "head_position_token",
        "head_branch_enc",
        "head_token",
        "fourier",  # DINOv3
        "storage",  # DINOv3
        "cls",      # DINOv3
    ]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Usually no decay for:
        # - 1D params: biases, LayerNorm/BatchNorm weights
        # - explicit special tokens
        # - positional encoding
        if (
            param.ndim <= 1
            or any(k in name for k in no_decay_keywords)
        ):
            no_decay.append(param)
            no_decay_names.append(name)
        else:
            decay.append(param)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]