import os
import pandas as pd
import json
from PIL import Image
import argparse
from tqdm import tqdm

# preprocessing adapted from https://github.com/ejcgt/attention-target-detection/blob/master/dataset.py

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./data/GOOReal")
args = parser.parse_args()


def main(DATA_PATH):
    # TEST
    TEST_FRAMES = []
    with open(os.path.join(DATA_PATH, 'testrealhumansNew.json')) as f:
        test_annotations = json.load(f)
    for annotation in tqdm(test_annotations, 'processing test set'):
        img_path = os.path.join(DATA_PATH, 'finalrealdatasetImgsV3', annotation['filename'].replace('\\', '/'))
        path = img_path
        img = Image.open(img_path)
        width, height = img.size

        bbox_lowres = annotation['ann']['bboxes'][-1]  # at 640*480
        bbox_norm = [bbox_lowres[0] / 640, bbox_lowres[1] / 480, bbox_lowres[2] / 640, bbox_lowres[3] / 480]
        bbox = [bbox_norm[0] * 1920, bbox_norm[1] * 1080, bbox_norm[2] * 1920, bbox_norm[3] * 1080]
        gaze_x_lowres, gaze_y_lowres = annotation['gaze_cx'], annotation['gaze_cy']  # at 640*480
        gaze_x_norm, gaze_y_norm = gaze_x_lowres / 640, gaze_y_lowres / 480
        gaze_x, gaze_y = gaze_x_norm * 1920, gaze_y_norm * 1080

        if not (0 <= gaze_x < width and 0 <= gaze_y < height):
            continue
        if not (0 <= bbox[0] < bbox[2] < width and 0 <= bbox[1] < bbox[3] < height):
            continue


        heads=[{
            'bbox': bbox,
            'bbox_norm': bbox_norm,
            'inout': 1,
            'gazex': [gaze_x], # convert to list for consistency with multi-annotation format
            'gazey': [gaze_y],
            'gazex_norm': [gaze_x_norm],
            'gazey_norm': [gaze_y_norm],
            'head_id': 0
        }]
        TEST_FRAMES.append({
            'path': path,
            'heads': heads,
            'num_heads': 1,
            'width': width,
            'height': height,
        })


    print("Test set: {} frames, {} multi-person".format(len(TEST_FRAMES), 0))
    out_file = open(os.path.join(DATA_PATH, "test_preprocessed.json"), "w")
    json.dump(TEST_FRAMES, out_file)



if __name__ == "__main__":
    main(args.data_path)