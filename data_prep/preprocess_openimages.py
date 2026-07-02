import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


DEFAULT_CLASSES = ["Person", "Man", "Woman", "Boy", "Girl"]


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./data/OpenImages")
parser.add_argument("--split", type=str, default="train", choices=["train"])
parser.add_argument(
    "--crowdhuman_repo",
    type=str,
    default="./yolov5-crowdhuman",
    help="Local path of MahenderAutonomo/yolov5-crowdhuman.",
)
parser.add_argument(
    "--crowdhuman_weights",
    type=str,
    default="./yolov5-crowdhuman/weights/crowdhuman_yolov5m.pt",
    help="Weights path for yolov5-crowdhuman.",
)
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--score_threshold", type=float, default=0.7)
parser.add_argument("--min_head_size", type=float, default=8.0)
parser.add_argument("--img_size", type=int, default=640)
parser.add_argument("--iou_threshold", type=float, default=0.45)
parser.add_argument("--max_samples", type=int, default=None)
parser.add_argument("--output_name", type=str, default=None)
parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
args = parser.parse_args()


def resolve_yolo_device(device_arg: str) -> str:
    value = str(device_arg).strip().lower()
    if value in {"", "cpu"}:
        return value
    if value == "cuda":
        return "0"
    if value.startswith("cuda:"):
        suffix = value.split(":", 1)[1].strip()
        return suffix if suffix else "0"
    return str(device_arg).strip()


class HeadDetector:
    def __init__(self, repo_path: str, weights_path: str, device: str, score_threshold: float, min_head_size: float, img_size: int, iou_threshold: float) -> None:
        repo_path = os.path.abspath(repo_path)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        from models.experimental import attempt_load
        from utils.general import check_img_size, non_max_suppression, scale_coords
        from utils.torch_utils import select_device
        from utils.datasets import letterbox

        original_torch_load = torch.load

        def torch_load_compat(*load_args, **load_kwargs):
            load_kwargs.setdefault("weights_only", False)
            return original_torch_load(*load_args, **load_kwargs)

        torch.load = torch_load_compat

        normalized_device = resolve_yolo_device(device)
        self.device = torch.device("cpu" if normalized_device == "cpu" else f"cuda:{normalized_device}") if normalized_device != "" else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.non_max_suppression = non_max_suppression
        self.scale_coords = scale_coords
        self.letterbox = letterbox
        self.score_threshold = score_threshold
        self.min_head_size = min_head_size
        self.iou_threshold = iou_threshold

        model_device = select_device(normalized_device)
        try:
            self.model = attempt_load(weights_path, map_location=model_device)
        finally:
            torch.load = original_torch_load
        self.model.eval()
        self.half = model_device.type != "cpu"
        if self.half:
            self.model.half()
        self.model_device = model_device
        self.stride = int(self.model.stride.max())
        self.img_size = check_img_size(img_size, s=self.stride)

        if self.model_device.type != "cpu":
            self.model(torch.zeros(1, 3, self.img_size, self.img_size).to(self.model_device).type_as(next(self.model.parameters())))

        names = self.model.module.names if hasattr(self.model, "module") else self.model.names
        self.names = {int(k): str(v).lower() for k, v in enumerate(names)} if not isinstance(names, dict) else {int(k): str(v).lower() for k, v in names.items()}
        self.head_class_ids = [class_id for class_id, name in self.names.items() if "head" in name]
        if not self.head_class_ids:
            raise RuntimeError(f"No head class found in model names: {self.names}")

    @torch.inference_mode()
    def detect(self, image: Image.Image, person_bbox: list[float]) -> list[float] | None:
        im0 = np.asarray(image)
        if im0.ndim < 2 or im0.shape[0] == 0 or im0.shape[1] == 0:
            return None
        img = self.letterbox(im0, new_shape=self.img_size, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        img_tensor = torch.from_numpy(img).to(self.model_device)
        img_tensor = img_tensor.half() if self.half else img_tensor.float()
        img_tensor /= 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        pred = self.model(img_tensor, augment=False)[0]
        pred = self.non_max_suppression(
            pred,
            conf_thres=self.score_threshold,
            iou_thres=self.iou_threshold,
            classes=self.head_class_ids,
            agnostic=False,
        )
        detections = pred[0]
        if detections is None or len(detections) == 0:
            return None

        detections[:, :4] = self.scale_coords(img_tensor.shape[2:], detections[:, :4], im0.shape).round()

        best_box = None
        best_score = -1.0
        for *xyxy, conf, cls in detections.tolist():
            cls_id = int(cls)
            if cls_id not in self.head_class_ids:
                continue
            xmin, ymin, xmax, ymax = [float(x) for x in xyxy]
            if xmax <= xmin or ymax <= ymin:
                continue
            if (xmax - xmin) < self.min_head_size or (ymax - ymin) < self.min_head_size:
                continue
            overlap = intersection_over_union([xmin, ymin, xmax, ymax], person_bbox)
            if overlap <= 0.0:
                continue
            score_value = float(conf)
            if score_value > best_score:
                best_score = score_value
                best_box = [xmin, ymin, xmax, ymax]
        return best_box

    def detect_in_person_crop(self, image: Image.Image, person_bbox: list[float]) -> list[float] | None:
        width, height = image.size
        px1, py1, px2, py2 = person_bbox
        pw = px2 - px1
        ph = py2 - py1
        if pw <= 0 or ph <= 0:
            return None

        crop_x1 = max(0.0, px1 - 0.10 * pw)
        crop_x2 = min(float(width), px2 + 0.10 * pw)
        crop_y1 = max(0.0, py1 - 0.15 * ph)
        crop_y2 = min(float(height), py1 + 0.60 * ph)
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return None

        crop = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        if crop.size[0] <= 2 or crop.size[1] <= 2:
            return None
        local_head = self.detect(crop, [0.0, 0.0, crop_x2 - crop_x1, crop_y2 - crop_y1])
        if local_head is None:
            return None

        hx1, hy1, hx2, hy2 = local_head
        return [hx1 + crop_x1, hy1 + crop_y1, hx2 + crop_x1, hy2 + crop_y1]


def intersection_over_union(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def clamp_box(box: list[float], width: int, height: int) -> list[float] | None:
    xmin, ymin, xmax, ymax = box
    xmin = max(0.0, min(float(width), xmin))
    ymin = max(0.0, min(float(height), ymin))
    xmax = max(0.0, min(float(width), xmax))
    ymax = max(0.0, min(float(height), ymax))
    if xmax <= xmin or ymax <= ymin:
        return None
    return [xmin, ymin, xmax, ymax]


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


def split_root(data_path: str, split: str) -> str:
    return os.path.join(data_path, "open-images-v7", split)


def load_class_id_map(split_dir: str) -> dict[str, str]:
    classes_csv = os.path.join(split_dir, "metadata", "classes.csv")
    if not os.path.exists(classes_csv):
        raise FileNotFoundError(f"classes.csv not found: {classes_csv}")

    class_id_map: dict[str, str] = {}
    with open(classes_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            class_id_map[row[0]] = row[1]
    return class_id_map


def load_person_detections(data_path: str, split: str, classes: list[str]) -> dict[str, list[dict[str, Any]]]:
    split_dir = split_root(data_path, split)
    detections_csv = os.path.join(split_dir, "labels", "detections.csv")
    image_ids_csv = os.path.join(split_dir, "metadata", "image_ids.csv")
    if not os.path.exists(detections_csv):
        raise FileNotFoundError(f"detections.csv not found: {detections_csv}")
    if not os.path.exists(image_ids_csv):
        raise FileNotFoundError(f"image_ids.csv not found: {image_ids_csv}")

    class_id_map = load_class_id_map(split_dir)
    target_class_ids = {class_id for class_id, class_name in class_id_map.items() if class_name in classes}
    if not target_class_ids:
        raise ValueError(f"None of target classes {classes} found in classes.csv")

    valid_image_ids: set[str] = set()
    with open(image_ids_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            valid_image_ids.add(row[0])

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with open(detections_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row.get("ImageID")
            label_name = row.get("LabelName")
            if image_id is None or label_name is None:
                continue
            if image_id not in valid_image_ids:
                continue
            if label_name not in target_class_ids:
                continue
            try:
                xmin = float(row["XMin"])
                xmax = float(row["XMax"])
                ymin = float(row["YMin"])
                ymax = float(row["YMax"])
            except Exception:
                continue
            grouped[image_id].append({
                "label_id": label_name,
                "label": class_id_map.get(label_name, label_name),
                "bbox_norm": [xmin, ymin, xmax, ymax],
            })
    return grouped


def resolve_image_path(data_path: str, split: str, image_id: str) -> str | None:
    split_data_dir = os.path.join(split_root(data_path, split), "data")
    candidates = [
        os.path.join(split_data_dir, f"{image_id}.jpg"),
        os.path.join(split_data_dir, f"{image_id}.jpeg"),
        os.path.join(split_data_dir, f"{image_id}.png"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def preprocess_split(data_path: str, split: str, detector: HeadDetector, classes: list[str], max_samples: int | None = None) -> list[dict[str, Any]]:
    detections_by_image = load_person_detections(data_path, split, classes)

    frames: list[dict[str, Any]] = []
    for image_id, detections in tqdm(detections_by_image.items(), desc=f"processing open images {split}"):
        img_path = resolve_image_path(data_path, split, image_id)
        if img_path is None:
            continue

        with Image.open(img_path) as img:
            image = img.convert("RGB")
            width, height = image.size

            heads = []
            for det in detections:
                xmin_norm, ymin_norm, xmax_norm, ymax_norm = det["bbox_norm"]
                person_bbox = clamp_box(
                    [xmin_norm * width, ymin_norm * height, xmax_norm * width, ymax_norm * height],
                    width,
                    height,
                )
                if person_bbox is None:
                    continue

                head_bbox = detector.detect_in_person_crop(image, person_bbox)
                if head_bbox is None:
                    continue
                head_bbox = clamp_box(head_bbox, width, height)
                if head_bbox is None:
                    continue
                heads.append(build_head_record(head_bbox, width, height, len(heads)))

        if not heads:
            continue

        rel_path = os.path.relpath(img_path, data_path)
        frames.append({
            "path": rel_path,
            "heads": heads,
            "num_heads": len(heads),
            "width": width,
            "height": height,
            "meta": {
                "source": "open-images-v7",
                "split": split,
                "open_images_id": image_id,
                "classes": classes,
                "head_detector_model": "MahenderAutonomo/yolov5-crowdhuman",
                "head_detector_weights": args.crowdhuman_weights,
            },
        })

        if max_samples is not None and len(frames) >= max_samples:
            break

    return frames


def main(data_path: str) -> None:
    split_dir = split_root(data_path, args.split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Open Images split dir not found: {split_dir}")

    print(f"[Info] Using split dir: {split_dir}")

    detector = HeadDetector(
        repo_path=args.crowdhuman_repo,
        weights_path=args.crowdhuman_weights,
        device=args.device,
        score_threshold=args.score_threshold,
        min_head_size=args.min_head_size,
        img_size=args.img_size,
        iou_threshold=args.iou_threshold,
    )
    frames = preprocess_split(data_path, args.split, detector, args.classes, max_samples=args.max_samples)
    output_name = args.output_name or f"{args.split}_preprocessed.json"
    out_path = os.path.join(data_path, output_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(frames, f)
    print(f"Saved {len(frames)} samples to {out_path}")


if __name__ == "__main__":
    main(args.data_path)