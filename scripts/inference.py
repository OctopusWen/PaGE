"""
Single-image gaze target inference with a released PaGE HuggingFace checkpoint.

Unlike training/eval (which build the model via page.model_factory and load a
separate DINOv3 backbone), the released PaGE checkpoints are *self-contained*:
the safetensors already include the full DINOv3 backbone weights, and the model
structure is loaded from the checkpoint's remote code via `trust_remote_code`.
So this script does NOT need page.model_factory, page.backbone, or any external
DINOv3 weights.

Input : one RGB frame + one head bounding box (normalized xyxy in [0, 1]).
Output: a visualization overlaying the predicted gaze heatmap, the head box, the
        argmax gaze point, and the in/out-of-frame probability.

Example:
    python scripts/inference.py \\
        --image ./demo/scene.jpg \\
        --bbox 0.30 0.12 0.48 0.40 \\
        --model_path ./checkpoints/page-vitb \\
        --output ./visualization/inference.png
"""

import argparse
import contextlib
import io

import numpy as np
import torch
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def parse_args():
    parser = argparse.ArgumentParser(description="PaGE single-image gaze inference (HF checkpoint)")
    parser.add_argument("--image", type=str, required=True, help="Path to the input RGB frame")
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
        required=True,
        help="Head bounding box as normalized xyxy in [0, 1]",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./checkpoints/page-vitb",
        help="Local directory or HuggingFace repo id of the PaGE checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./visualization/inference.png",
        help="Where to save the visualization",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to run on",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show HuggingFace model loading logs",
    )
    return parser.parse_args()


def validate_bbox(bbox):
    xmin, ymin, xmax, ymax = bbox
    if not all(0.0 <= v <= 1.0 for v in bbox):
        raise ValueError(f"--bbox must be normalized to [0, 1], got {bbox}")
    if xmax <= xmin or ymax <= ymin:
        raise ValueError(f"--bbox must satisfy xmax>xmin and ymax>ymin, got {bbox}")
    return tuple(float(v) for v in bbox)


def crop_head(scene: Image.Image, bbox) -> Image.Image:
    """Crop the head region from the scene using a normalized xyxy bbox."""
    w, h = scene.size
    xmin, ymin, xmax, ymax = bbox
    left = int(round(xmin * w))
    top = int(round(ymin * h))
    right = max(left + 1, int(round(xmax * w)))
    bottom = max(top + 1, int(round(ymax * h)))
    return scene.crop((left, top, right, bottom))


def to_numpy_heatmap(heatmap) -> np.ndarray:
    """Extract a single [H, W] heatmap from the model output (first image, first head)."""
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().float().cpu().numpy()
    else:
        heatmap = np.asarray(heatmap, dtype=np.float32)
    heatmap = np.squeeze(heatmap)  # drop leading singleton (Np=1) dims
    if heatmap.ndim != 2:
        raise RuntimeError(f"Expected a 2D heatmap after squeeze, got shape {heatmap.shape}")
    return heatmap


def move_to_device(obj, device):
    """Move nested processor outputs to the target torch device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    return obj


def main():
    args = parse_args()
    bbox = validate_bbox(args.bbox)

    # Import here so the (large) transformers import only happens when running,
    # and so a missing dependency produces a clear, actionable message.
    try:
        from transformers import AutoModel, AutoImageProcessor
        from transformers.utils import logging as hf_logging
    except ImportError as e:
        raise SystemExit(
            "transformers is required for HF inference. Install the project deps "
            "(`uv sync`) or `pip install 'transformers==5.6.2' safetensors`."
        ) from e

    scene = Image.open(args.image).convert("RGB")
    head = crop_head(scene, bbox)

    print(f"[Info] loading model from {args.model_path}")
    if args.verbose:
        model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True).eval().to(args.device)
        processor = AutoImageProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    else:
        old_verbosity = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True).eval().to(args.device)
                processor = AutoImageProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        finally:
            hf_logging.set_verbosity(old_verbosity)

    inputs = processor(scene, head_crops=[head], bboxes=[[bbox]])
    inputs = move_to_device(inputs, args.device)

    with torch.inference_mode():
        out = model(inputs)

    # out["heatmap"]: list over images; [0] -> [Np, 64, 64] (sigmoid probabilities)
    # out["inout"]  : list over images; [0] -> [Np]        (sigmoid probabilities)
    heatmap = to_numpy_heatmap(out["heatmap"][0])
    inout_prob = None
    if out.get("inout") is not None:
        inout_val = out["inout"][0]
        if isinstance(inout_val, torch.Tensor):
            inout_val = inout_val.detach().float().cpu().numpy()
        inout_prob = float(np.squeeze(inout_val).reshape(-1)[0])

    # Locate the predicted gaze point (argmax of the heatmap), mapped to image pixels.
    hm_h, hm_w = heatmap.shape
    flat_idx = int(np.argmax(heatmap))
    gaze_row, gaze_col = divmod(flat_idx, hm_w)
    img_w, img_h = scene.size
    gaze_x = (gaze_col + 0.5) / hm_w * img_w
    gaze_y = (gaze_row + 0.5) / hm_h * img_h

    # Head box center in pixels (origin of the gaze arrow).
    head_cx = (bbox[0] + bbox[2]) / 2 * img_w
    head_cy = (bbox[1] + bbox[3]) / 2 * img_h

    # Visualization: image + heatmap overlay + head box + gaze arrow.
    heatmap_up = np.asarray(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize((img_w, img_h), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0

    fig, ax = plt.subplots(figsize=(10, 10 * img_h / img_w))
    ax.imshow(scene)
    ax.imshow(heatmap_up, cmap="jet", alpha=0.4)

    rect = patches.Rectangle(
        (bbox[0] * img_w, bbox[1] * img_h),
        (bbox[2] - bbox[0]) * img_w,
        (bbox[3] - bbox[1]) * img_h,
        linewidth=2.5,
        edgecolor="lime",
        facecolor="none",
    )
    ax.add_patch(rect)

    ax.annotate(
        "",
        xy=(gaze_x, gaze_y),
        xytext=(head_cx, head_cy),
        arrowprops=dict(arrowstyle="->", color="white", linewidth=2.5),
    )
    ax.scatter([gaze_x], [gaze_y], s=120, c="red", marker="*", edgecolors="white", linewidths=1.0, zorder=5)

    title = "PaGE gaze prediction"
    if inout_prob is not None:
        title += f"  |  in-frame prob = {inout_prob:.3f}"
    ax.set_title(title)
    ax.axis("off")

    import os
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[Info] gaze point (pixels): ({gaze_x:.1f}, {gaze_y:.1f})")
    if inout_prob is not None:
        print(f"[Info] in-frame probability: {inout_prob:.3f}")
    print(f"[Info] saved visualization to {args.output}")


if __name__ == "__main__":
    main()
