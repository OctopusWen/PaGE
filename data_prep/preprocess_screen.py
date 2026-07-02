import os
import pandas as pd
import json
from PIL import Image
import argparse
from tqdm import tqdm

# preprocessing adapted from https://github.com/ejcgt/attention-target-detection/blob/master/dataset.py

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./data/gazefollow")
args = parser.parse_args()


def main(DATA_PATH):

    # TRAIN
    with open(os.path.join(DATA_PATH, 'train_annotations.json')) as f:
        train_annotations = json.load(f)
    TRAIN_FRAMES = []
    for annotation in tqdm(train_annotations, 'processing train set'):
        img_path = os.path.join(DATA_PATH, 'images', annotation['image'])
        path = img_path
        img = Image.open(img_path)
        width, height = img.size

        if annotation['inout'] == 0:
            annotation['x'] = annotation['y'] = 0
        if not (0 <= annotation['x'] < width and 0 <= annotation['y'] < height):
            continue
        if not (0 <= annotation['bbox'][0] < annotation['bbox'][2] < width and 0 <= annotation['bbox'][1] < annotation['bbox'][3] < height):
            continue

        heads=[{
            'bbox': annotation['bbox'],
            'bbox_norm': annotation['bbox_norm'],
            'inout': annotation['inout'],
            'gazex': [annotation['x']], # convert to list for consistency with multi-annotation format
            'gazey': [annotation['y']],
            'gazex_norm': [annotation['x_norm']],
            'gazey_norm': [annotation['y_norm']],
            'head_id': 0
        }]
        TRAIN_FRAMES.append({
            'path': path,
            'heads': heads,
            'num_heads': 1,
            'width': width,
            'height': height,
        })

    print("Train set: {} frames, {} multi-person".format(len(TRAIN_FRAMES), 0))
    out_file = open(os.path.join(DATA_PATH, "train_preprocessed.json"), "w")
    json.dump(TRAIN_FRAMES, out_file)

    # TEST
    TEST_FRAMES = []
    with open(os.path.join(DATA_PATH, 'test_annotations.json')) as f:
        test_annotations = json.load(f)
    for annotation in tqdm(test_annotations, 'processing test set'):
        img_path = os.path.join(DATA_PATH, 'images', annotation['image'])
        path = img_path
        img = Image.open(img_path)
        width, height = img.size

        if annotation['inout'] == 0:
            annotation['x'] = annotation['y'] = 0
        if not (0 <= annotation['x'] < width and 0 <= annotation['y'] < height):
            continue
        if not (0 <= annotation['bbox'][0] < annotation['bbox'][2] < width and 0 <= annotation['bbox'][1] < annotation['bbox'][3] < height):
            continue

        heads=[{
            'bbox': annotation['bbox'],
            'bbox_norm': annotation['bbox_norm'],
            'inout': annotation['inout'],
            'gazex': [annotation['x']], # convert to list for consistency with multi-annotation format
            'gazey': [annotation['y']],
            'gazex_norm': [annotation['x_norm']],
            'gazey_norm': [annotation['y_norm']],
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