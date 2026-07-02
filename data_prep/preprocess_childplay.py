import os
import pandas as pd
import json
from PIL import Image
import argparse
from tqdm import tqdm

# preprocessing adapted from https://github.com/ejcgt/attention-target-detection/blob/master/dataset.py

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./data/childplay")
args = parser.parse_args()


def main(DATA_PATH):
    # TRAIN
    multiperson_ex = 0
    TRAIN_FRAMES = []
    for train_csv_dir in [os.path.join(DATA_PATH, "annotations", "train"), os.path.join(DATA_PATH, "annotations", "val")]:
        for csv_path in tqdm(sorted(f for f in os.listdir(train_csv_dir) if f.endswith(".csv"))):
            # HEADERS: clip, frame, person_id, bbox_x, bbox_y, bbox_width, bbox_height, gaze_class, gaze_x, gaze_y, is_child
            df = pd.read_csv(os.path.join(train_csv_dir, csv_path))
            df = df[df['gaze_class'].isin({"inside_visible", "outside_frame"})]  # keep only conventional gaze types, excluding things like gaze shifts or eyes closed
            df = df.groupby("frame").agg(list)
            SEQUENCE = []
            for frame, row in df.iterrows():
                clip = row["clip"][0]
                clip_name = "_".join(clip.split("_")[:-1])
                clip_base_idx = int(clip.split("_")[-1].split("-")[0])
                frame_idx = clip_base_idx + int(frame) - 1
                img_path = os.path.join(DATA_PATH, "images", clip, f"{clip_name}_{frame_idx}.jpg")
                img = Image.open(img_path)
                width, height = img.size
                num_people = len(row["person_id"])
                if num_people > 1:
                    multiperson_ex += 1
                heads = []
                head_cnt = 0
                for i in range(num_people):
                    xmin, ymin, xlen, ylen = row["bbox_x"][i], row["bbox_y"][i], row["bbox_width"][i], row["bbox_height"][i]
                    xmax = xmin + xlen
                    ymax = ymin + ylen
                    gaze_x, gaze_y = row["gaze_x"][i], row["gaze_y"][i]
                    gaze_x_norm = gaze_x / float(width)
                    gaze_y_norm = gaze_y / float(height)
                    if row["gaze_class"][i] == "inside_visible" and gaze_x != -1.0 and gaze_y != -1.0:
                        inout = 1
                    elif row["gaze_class"][i] == "outside_frame":
                        inout = 0
                        gaze_x, gaze_y, gaze_x_norm, gaze_y_norm = 0.0, 0.0, 0.0, 0.0
                    else:
                        continue

                    heads.append({
                        'bbox': [xmin, ymin, xmax, ymax],
                        'bbox_norm': [xmin / float(width), ymin / float(height), xmax / float(width), ymax / float(height)],
                        'inout': inout,
                        'gazex': [gaze_x], # convert to list for consistency with multi-annotation format
                        'gazey': [gaze_y],
                        'gazex_norm': [gaze_x_norm],
                        'gazey_norm': [gaze_y_norm],
                        'head_id': head_cnt
                    })
                    head_cnt += 1
                SEQUENCE.append({
                    'path': os.path.join("images", "images", clip, f"{clip_name}_{frame_idx}.jpg"),
                    'heads': heads,
                    'num_heads': num_people,
                    'width': width,
                    'height': height,
                })
            TRAIN_FRAMES.append(SEQUENCE)

    # print("Train set: {} frames, {} multi-person".format(len(TRAIN_FRAMES), multiperson_ex))
    with open(os.path.join(DATA_PATH, "train_preprocessed.json"), "w", encoding="utf-8") as f:
        json.dump(TRAIN_FRAMES, f)

    # TEST
    test_csv_dir = os.path.join(DATA_PATH, "annotations", "test")
    multiperson_ex = 0
    TEST_FRAMES = []
    for csv_path in tqdm(sorted(f for f in os.listdir(test_csv_dir) if f.endswith(".csv"))):
        # HEADERS: clip, frame, person_id, bbox_x, bbox_y, bbox_width, bbox_height, gaze_class, gaze_x, gaze_y, is_child
        df = pd.read_csv(os.path.join(test_csv_dir, csv_path))
        df = df[df['gaze_class'].isin({"inside_visible", "outside_frame"})]  # keep only conventional gaze types, excluding things like gaze shifts or eyes closed
        df = df.groupby("frame").agg(list)
        SEQUENCE = []
        for frame, row in df.iterrows():
            clip = row["clip"][0]
            clip_name = "_".join(clip.split("_")[:-1])
            clip_base_idx = int(clip.split("_")[-1].split("-")[0])
            frame_idx = clip_base_idx + int(frame) - 1
            img_path = os.path.join(DATA_PATH, "images", clip, f"{clip_name}_{frame_idx}.jpg")
            img = Image.open(img_path)
            width, height = img.size
            num_people = len(row["person_id"])
            if num_people > 1:
                multiperson_ex += 1
            heads = []
            head_cnt = 0
            for i in range(num_people):
                xmin, ymin, xlen, ylen = row["bbox_x"][i], row["bbox_y"][i], row["bbox_width"][i], row["bbox_height"][i]
                xmax = xmin + xlen
                ymax = ymin + ylen
                gaze_x, gaze_y = row["gaze_x"][i], row["gaze_y"][i]
                gaze_x_norm = gaze_x / float(width)
                gaze_y_norm = gaze_y / float(height)
                if row["gaze_class"][i] == "inside_visible" and gaze_x != -1.0 and gaze_y != -1.0:
                    inout = 1
                elif row["gaze_class"][i] == "outside_frame":
                    inout = 0
                    gaze_x, gaze_y, gaze_x_norm, gaze_y_norm = 0.0, 0.0, 0.0, 0.0
                else:
                    continue

                heads.append({
                    'bbox': [xmin, ymin, xmax, ymax],
                    'bbox_norm': [xmin / float(width), ymin / float(height), xmax / float(width), ymax / float(height)],
                    'inout': inout,
                    'gazex': [gaze_x], # convert to list for consistency with multi-annotation format
                    'gazey': [gaze_y],
                    'gazex_norm': [gaze_x_norm],
                    'gazey_norm': [gaze_y_norm],
                    'head_id': head_cnt
                })
                head_cnt += 1
            SEQUENCE.append({
                'path': os.path.join("images", "images", clip, f"{clip_name}_{frame_idx}.jpg"),
                'heads': heads,
                'num_heads': num_people,
                'width': width,
                'height': height,
            })
        TEST_FRAMES.append(SEQUENCE)
    # print("Test set: {} frames, {} multi-person".format(len(TEST_FRAMES), multiperson_ex))
    with open(os.path.join(DATA_PATH, "test_preprocessed.json"), "w", encoding="utf-8") as f:
        json.dump(TEST_FRAMES, f)


if __name__ == "__main__":
    main(args.data_path)