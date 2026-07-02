import argparse
import json
import os
from typing import Any, cast

import scipy.io as sio
from PIL import Image
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./data/mpii")
parser.add_argument("--annotations_mat", type=str, default=None, help="Path to MPII annotations .mat file")
parser.add_argument("--images_dir", type=str, default=None, help="Path to MPII images directory")
parser.add_argument("--max_samples", type=int, default=None)
parser.add_argument("--output_name", type=str, default="train_preprocessed.json")
args = parser.parse_args()


def resolve_annotations_mat(data_path: str, annotations_mat: str | None) -> str:
    if annotations_mat is not None:
        return annotations_mat
    return os.path.join(data_path, "mpii_human_pose_v1_u12_2", "mpii_human_pose_v1_u12_1.mat")


def resolve_images_dir(data_path: str, images_dir: str | None) -> str:
    if images_dir is not None:
        return images_dir
    return os.path.join(data_path, "images")


def mat_to_python(value):
    if hasattr(value, "_fieldnames"):
        return {field: mat_to_python(getattr(value, field)) for field in value._fieldnames}
    if isinstance(value, dict):
        return {k: mat_to_python(v) for k, v in value.items()}
    if hasattr(value, "dtype") and getattr(value.dtype, "names", None):
        return {name: mat_to_python(value[name]) for name in value.dtype.names}
    if isinstance(value, (list, tuple)):
        return [mat_to_python(v) for v in value]
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            if value.dtype == object:
                return [mat_to_python(v) for v in value.reshape(-1).tolist()]
            if value.size == 1:
                return mat_to_python(value.reshape(-1)[0])
            return [mat_to_python(v) for v in value.tolist()]
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return value


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, list):
        if not value:
            return None
        return to_float(value[0])
    try:
        return float(value)
    except Exception:
        return None


def build_head_record(head_bbox: list[float], width: int, height: int, head_id: int) -> dict[str, Any]:
    xmin, ymin, xmax, ymax = head_bbox
    return {
        "bbox": [xmin, ymin, xmax, ymax],
        "bbox_norm": [xmin / float(width), ymin / float(height), xmax / float(width), ymax / float(height)],
        "inout": None,
        "gazex": [],
        "gazey": [],
        "gazex_norm": [],
        "gazey_norm": [],
        "head_id": head_id,
    }


def preprocess_mpii_train(data_path: str, annotations_mat: str, images_dir: str, max_samples: int | None = None) -> list[dict[str, Any]]:
    mat = sio.loadmat(annotations_mat, struct_as_record=False, squeeze_me=True)
    release = mat["RELEASE"]

    annolist = as_list(mat_to_python(release.annolist))
    img_train = as_list(mat_to_python(release.img_train))

    frames: list[dict[str, Any]] = []
    for idx, ann in enumerate(tqdm(annolist, desc="processing mpii train")):
        train_flag = img_train[idx] if idx < len(img_train) else True
        # MPII test annotations are withheld; only the train split is usable
        if not bool(train_flag):
            continue

        image_info = ann.get("image") if isinstance(ann, dict) else None
        image_name = image_info.get("name") if isinstance(image_info, dict) else None
        if image_name is None:
            continue

        image_path = os.path.join(images_dir, str(image_name))
        if not os.path.exists(image_path):
            continue

        with Image.open(image_path) as img:
            width, height = img.size

        annorect = as_list(ann.get("annorect") if isinstance(ann, dict) else None)
        heads = []
        for rect in annorect:
            if not isinstance(rect, dict):
                continue
            x1 = to_float(rect.get("x1"))
            y1 = to_float(rect.get("y1"))
            x2 = to_float(rect.get("x2"))
            y2 = to_float(rect.get("y2"))
            if None in (x1, y1, x2, y2):
                continue
            x1 = float(cast(float, x1))
            y1 = float(cast(float, y1))
            x2 = float(cast(float, x2))
            y2 = float(cast(float, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            heads.append(build_head_record([x1, y1, x2, y2], width, height, len(heads)))

        if not heads:
            continue

        frames.append({
            "path": os.path.relpath(image_path, start=data_path),
            "heads": heads,
            "num_heads": len(heads),
            "width": width,
            "height": height,
            "meta": {
                "source": "mpii_human_pose",
                "split": "train",
                "annotation_index": idx,
                "is_train": bool(train_flag),
                "img_name": image_name,
                "vidx": ann.get("vidx") if isinstance(ann, dict) else None,
                "frame_sec": ann.get("frame_sec") if isinstance(ann, dict) else None,
            },
        })

        if max_samples is not None and len(frames) >= max_samples:
            break

    return frames


def main(data_path: str) -> None:
    annotations_mat = resolve_annotations_mat(data_path, args.annotations_mat)
    images_dir = resolve_images_dir(data_path, args.images_dir)

    if not os.path.exists(annotations_mat):
        raise FileNotFoundError(f"Annotations mat not found: {annotations_mat}")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images dir not found: {images_dir}")

    frames = preprocess_mpii_train(data_path, annotations_mat, images_dir, max_samples=args.max_samples)
    out_path = os.path.join(data_path, args.output_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(frames, f)
    print(f"Saved {len(frames)} train samples to {out_path}")


if __name__ == "__main__":
    main(args.data_path)
