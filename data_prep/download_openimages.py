#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F


DEFAULT_CLASSES = ["Person", "Man", "Woman", "Boy", "Girl"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Open Images V7 train samples with Person/Woman/Man detections via FiftyOne"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./data/OpenImages",
        help="Local download directory, e.g., ./data/OpenImages",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        choices=["train"],
        help="Splits to download. Only the train split is used.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASSES,
        help="Open Images class names (case-sensitive). Default: Person Woman Man Boy Girl",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of images to download per split. None means download all matching samples",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Use with --max-samples to randomly sample",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=51,
        help="Random seed for shuffle",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker processes for downloading images; default is determined by FiftyOne",
    )
    parser.add_argument(
        "--dataset-prefix",
        type=str,
        default="open_images_v7_people",
        help="FiftyOne internal dataset name prefix",
    )
    parser.add_argument(
        "--keep-all-labels",
        action="store_true",
        help=(
            "By default, only Person/Woman/Man detection labels are kept. "
            "With this option, images containing target classes are downloaded but all detection labels on the image are retained"
        ),
    )
    parser.add_argument(
        "--persistent",
        action="store_true",
        help="Mark FiftyOne dataset as persistent for easy reopening later",
    )
    parser.add_argument(
        "--overwrite-datasets",
        action="store_true",
        help="If a FiftyOne dataset with the same name exists, delete it first before reloading",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Do not write additional JSON manifest",
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Launch FiftyOne App to visualize the last split after download completes",
    )
    return parser.parse_args()


def safe_get_sample_field(sample, field_name: str):
    try:
        return sample[field_name]
    except Exception:
        return None


def detection_attributes_to_dict(det) -> dict[str, Any]:
    attrs = {}
    for k, v in det.attributes.items():
        attrs[k] = getattr(v, "value", str(v))
    return attrs


def write_manifest(view, split: str, classes: list[str], out_dir: Path) -> Path:
    """
    Write a simple JSON manifest:
    [
      {
        "filepath": "...jpg",
        "width": W,
        "height": H,
        "detections": [
          {
            "label": "Person",
            "bbox": [xmin, ymin, xmax, ymax],
            "bbox_norm_xywh": [x, y, w, h]
          }
        ]
      }
    ]
    """
    manifest_dir = out_dir / "open-images-v7_people_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{split}_person_woman_man_detections.json"

    records = []

    for sample in view.iter_samples(progress=True):
        detections = safe_get_sample_field(sample, "detections")
        if detections is None or not detections.detections:
            continue

        if sample.metadata is None:
            # Normal zoo loader will have metadata; skip if not available
            continue

        width = int(sample.metadata.width)
        height = int(sample.metadata.height)

        det_records = []
        for det in detections.detections:
            if det.label not in classes:
                continue

            # FiftyOne detection bbox: normalized [top_left_x, top_left_y, width, height]
            x, y, w, h = det.bounding_box
            xmin = x * width
            ymin = y * height
            xmax = (x + w) * width
            ymax = (y + h) * height

            det_records.append(
                {
                    "id": det.id,
                    "label": det.label,
                    "bbox": [xmin, ymin, xmax, ymax],
                    "bbox_norm_xywh": [x, y, w, h],
                    "confidence": det.confidence,
                    "attributes": detection_attributes_to_dict(det),
                }
            )

        if not det_records:
            continue

        records.append(
            {
                "sample_id": sample.id,
                "open_images_id": safe_get_sample_field(sample, "open_images_id"),
                "filepath": sample.filepath,
                "width": width,
                "height": height,
                "detections": det_records,
            }
        )

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    return manifest_path


def main():
    args = parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Set where Open Images will be downloaded to
    # Actual path will typically be:
    #   <out-dir>/open-images-v7/train
    fo.config.dataset_zoo_dir = str(out_dir)

    classes = args.classes
    only_matching = not args.keep_all_labels

    print(f"[Info] FiftyOne dataset_zoo_dir = {fo.config.dataset_zoo_dir}", flush=True)
    print(f"[Info] out_dir = {out_dir}", flush=True)
    print(f"[Info] classes = {classes}", flush=True)
    print(f"[Info] only_matching labels = {only_matching}", flush=True)
    print(f"[Info] splits = {args.splits}", flush=True)
    print(f"[Info] max_samples = {args.max_samples}", flush=True)
    print(f"[Info] shuffle = {args.shuffle}", flush=True)
    print(f"[Info] seed = {args.seed}", flush=True)
    print(f"[Info] num_workers = {args.num_workers}", flush=True)
    print(f"[Info] persistent = {args.persistent}", flush=True)
    print(f"[Info] overwrite_datasets = {args.overwrite_datasets}", flush=True)
    print(f"[Info] no_manifest = {args.no_manifest}", flush=True)
    print(f"[Info] launch = {args.launch}", flush=True)

    last_view = None

    for split in args.splits:
        dataset_name = f"{args.dataset_prefix}_{split}"

        if args.overwrite_datasets and fo.dataset_exists(dataset_name):
            print(f"[Info] Deleting existing FiftyOne dataset: {dataset_name}", flush=True)
            fo.delete_dataset(dataset_name)

        print(f"\n[Download] Loading Open Images V7 split={split}", flush=True)
        print(f"[Download] dataset_name = {dataset_name}", flush=True)
        print(f"[Download] label_types = ['detections']", flush=True)
        print(f"[Download] classes = {classes}", flush=True)
        print(f"[Download] calling foz.load_zoo_dataset(...)", flush=True)

        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split=split,
            label_types=["detections"],
            classes=classes,
            only_matching=only_matching,
            max_samples=args.max_samples,
            shuffle=args.shuffle,
            seed=args.seed,
            num_workers=args.num_workers,
            dataset_name=dataset_name,
        )

        print(f"[Download] foz.load_zoo_dataset(...) returned for split={split}", flush=True)
        print(f"[Download] dataset persistent flag before set = {dataset.persistent}", flush=True)

        if args.persistent:
            dataset.persistent = True
            print(f"[Download] dataset persistent flag set to True", flush=True)

        # Ensure metadata is available for exporting absolute bboxes
        print(f"[Metadata] compute_metadata start for split={split}", flush=True)
        dataset.compute_metadata(num_workers=args.num_workers, skip_failures=True)
        print(f"[Metadata] compute_metadata done for split={split}", flush=True)

        # Even if keep_all_labels=True, create a view that only shows/exports Person/Woman/Man
        print(f"[View] filtering labels for split={split}", flush=True)
        view = dataset.filter_labels(
            "detections",
            F("label").is_in(classes),
            only_matches=True,
        )
        print(f"[View] filter_labels done for split={split}", flush=True)

        print(f"[Done] split={split}", flush=True)
        print(f"       FiftyOne dataset name: {dataset.name}", flush=True)
        print(f"       total dataset samples: {len(dataset)}", flush=True)
        print(f"       matched samples: {len(view)}", flush=True)
        print(f"       local zoo dir: {out_dir / 'open-images-v7' / split}", flush=True)

        if not args.no_manifest:
            print(f"[Manifest] writing manifest for split={split}", flush=True)
            manifest_path = write_manifest(view, split, classes, out_dir)
            print(f"[Manifest] done: {manifest_path}", flush=True)

        last_view = view

    if args.launch and last_view is not None:
        print("\n[Launch] Starting FiftyOne App...", flush=True)
        session = fo.launch_app(last_view)
        session.wait()


if __name__ == "__main__":
    main()
