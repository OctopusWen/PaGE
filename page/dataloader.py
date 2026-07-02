import torch
import json
import os
import copy
from PIL import Image
import numpy as np
from tqdm import tqdm


import page.utils as utils


def load_data_gazefollow(file, sample_rate, path):
    data = json.load(open(file, "r"))
    ret = []
    for i in range(0, len(data), sample_rate):
        ret.append({'basedir': path, 'data': data[i], 'src': 0})
    return ret

def load_data_vat(file, sample_rate, path):
    sequences = json.load(open(file, "r"))
    data = []
    for i in range(len(sequences)):
        for j in range(0, len(sequences[i]['frames']), sample_rate):
            data.append({'basedir': path, 'data': sequences[i]['frames'][j], 'src': 1})
    return data

def load_data_childplay(file, sample_rate, path):
    sequences = json.load(open(file, "r"))
    data = []
    for i in range(len(sequences)):
        for j in range(0, len(sequences[i]), sample_rate):
            data.append({'basedir': path, 'data': sequences[i][j], 'src': 2})
    return data

def load_data_screen(file, sample_rate, path):
    data = json.load(open(file, "r"))
    ret = []
    for i in range(0, len(data), sample_rate):
        ret.append({'basedir': path, 'data': data[i], 'src': 3})
    return ret

def load_data_coco(file, sample_rate, path):
    data = json.load(open(file, "r"))
    ret = []
    for i in range(0, len(data), sample_rate):
        ret.append({'basedir': path, 'data': data[i], 'src': 4})
    return ret

def load_data_mpii(file, sample_rate, path):
    data = json.load(open(file, "r"))
    ret = []
    for i in range(0, len(data), sample_rate):
        ret.append({'basedir': path, 'data': data[i], 'src': 5})
    return ret

def load_data_openimages(file, sample_rate, path):
    data = json.load(open(file, "r"))
    ret = []
    for i in range(0, len(data), sample_rate):
        ret.append({'basedir': path, 'data': data[i], 'src': 6})
    return ret

def load_data_gooreal(file, sample_rate, path):
    data = json.load(open(file, "r"))
    ret = []
    for i in range(0, len(data), sample_rate):
        ret.append({'basedir': path, 'data': data[i], 'src': 7})
    return ret

class GazeDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset_names, paths, split, transforms, head_transforms=[], in_frame_only=True, sample_rates=None, need_heatmap=True, preload_imgs=False, distillation=False, max_images=None, heatmap_res=64):
        self.dataset_names = dataset_names
        self.paths = paths
        self.split = split
        self.aug = self.split == "train"
        self.transforms = transforms
        self.head_transforms = head_transforms
        self.in_frame_only = in_frame_only
        self.preload_imgs = preload_imgs
        self.need_heatmap = need_heatmap
        self.distillation = distillation
        self.max_images = max_images
        self.heatmap_res = heatmap_res

        # default: sample every image
        if sample_rates is None:
            sample_rates = [1 for _ in range(len(dataset_names))]
        self.sample_rates = sample_rates

        if len(dataset_names) != len(paths):
            raise ValueError(f"The number of datasets ({len(dataset_names)}) does not match the number of paths provided ({len(paths)})")
        
        self.images = []
        self.data = []
        for dataset_name, path, sample_rate in zip(dataset_names, paths, sample_rates):
            if dataset_name == "gazefollow":
                self.data += load_data_gazefollow(os.path.join(path, "{}_preprocessed.json".format(split)), sample_rate=sample_rate, path=path)  # GazeFollow consists of individual, unrelated images, so we do not downsample at all
            elif dataset_name == "videoattentiontarget":
                self.data += load_data_vat(os.path.join(path, "{}_preprocessed.json".format(split)), sample_rate=sample_rate, path=path)         # vat has highly homogenous scenes and high native sample rate, so we downsample temporally
            elif dataset_name == "childplay":
                self.data += load_data_childplay(os.path.join(path, "{}_preprocessed.json".format(split)), sample_rate=sample_rate, path=path)   # childplay gets similar treatment
            elif dataset_name == "screen":
                self.data += load_data_screen(os.path.join(path, "{}_preprocessed.json".format(split)), sample_rate=sample_rate, path=path)      # Gaze-on-screen dataset we collected ourselves
            elif dataset_name == "coco":
                self.data += load_data_coco(os.path.join(path, "{}_preprocessed.json".format(split)), sample_rate=sample_rate, path=path)
            elif dataset_name == "mpii":
                self.data += load_data_mpii(os.path.join(path, "{}_preprocessed.json".format(split)), sample_rate=sample_rate, path=path)
            elif dataset_name == "openimages":
                self.data += load_data_openimages(os.path.join(path, "{}_preprocessed.json".format(split)), sample_rate=sample_rate, path=path)
            elif dataset_name == 'gooreal':
                if split == 'train':
                    raise ValueError("GOOReal contains test set images only. Train split is not available.")
                self.data += load_data_gooreal(os.path.join(path, "{}_preprocessed.json".format(split)), sample_rate=sample_rate, path=path)
            else:
                raise ValueError("Invalid dataset: {}".format(dataset_name))

        self.data_idxs = []
        for i in range(len(self.data)):
            distill_max_heads_per_image = 3  # A maximum of 3 heads per scene is used to ensure scene diversity (this leads to ~1.8 heads per scene on average)
            head_cnt = 0
            for j in range(len(self.data[i]['data']['heads'])):
                if not self.in_frame_only or self.data[i]['data']['heads'][j]['inout'] == 1:
                    # Filter out bad head bboxes in COCO, MPII and OpenImages
                    if self.data[i]['src'] in {4, 5, 6}:
                        head = self.data[i]['data']['heads'][j]
                        xmin, ymin, xmax, ymax = head["bbox"]
                        if (xmax - xmin) < 16 or (ymax - ymin) < 16:
                            continue
                    head_cnt += 1
                    self.data_idxs.append((i, j))
                if self.distillation and head_cnt >= distill_max_heads_per_image:
                    break

        if self.max_images is not None:
            self.data_idxs = self.data_idxs[:max_images]

        if self.preload_imgs:
            for idx in tqdm(range(len(self.data_idxs))):
                img_idx, _ = self.data_idxs[idx]
                img_data = self.data[img_idx]['data']
                base_dir = self.data[img_idx]['basedir']
                img_path = os.path.join(base_dir, img_data['path'])
                with Image.open(img_path) as img:
                    img_rgb = img.convert("RGB")
                    self.images.append(img_rgb)

    def __getitem__(self, idx):
        img_idx, head_idx = self.data_idxs[idx]
        img_data = self.data[img_idx]['data']
        base_dir = self.data[img_idx]['basedir']
        data_source = self.data[img_idx]['src']
        head_data = copy.deepcopy(img_data['heads'][head_idx])
        bbox = head_data['bbox']
        bbox_norm = head_data['bbox_norm']
        gazex = head_data['gazex']
        gazey = head_data['gazey']
        gazex_norm = head_data['gazex_norm']
        gazey_norm = head_data['gazey_norm']
        inout = head_data['inout']

        img_path = os.path.join(base_dir, img_data['path'])
        if self.preload_imgs:
            img = self.images[idx]
        else:
            img = Image.open(img_path)
            img = img.convert("RGB")
        width, height = img.size

        # provide placeholders for unlabeled data used in distillation
        # the placeholder is set at the center of the head bbox so that it does not interfere with cropping
        if len(gazex) == 0:
            gazex = [(bbox[0] + bbox[2]) / 2]
            gazey = [(bbox[1] + bbox[3]) / 2]
            gazex_norm = [gazex[0] / float(width)]
            gazey_norm = [gazey[0] / float(height)]
            inout = True

        if self.aug:
            if np.random.sample() <= 0.5:
                img, bbox, gazex, gazey = utils.random_crop(img, bbox, gazex, gazey, inout)
            if np.random.sample() <= 0.5:
                img, bbox, gazex, gazey = utils.horiz_flip(img, bbox, gazex, gazey, inout)
            # if np.random.sample() <= 0.5:
            #     img, bbox, gazex, gazey = utils.random_rotate(img, bbox, gazex, gazey, inout, degrees=(-15, 15))
            if np.random.sample() <= 0.5:
                bbox = utils.random_bbox_jitter(img, bbox)
            # if np.random.sample() <= 0.5:
            #     gazex, gazey = utils.random_ground_truth_jitter(img, gazex, gazey)

            # update width and height and re-normalize
            width, height = img.size
            bbox_norm = [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height]
            gazex_norm = [x / float(width) for x in gazex]
            gazey_norm = [y / float(height) for y in gazey]
        
        # generate transformed image of every size needed by the backbones
        imgs = [transform(img) for transform in self.transforms]
        # generate transformed image of every head (if head crop is needed, otherwise fill in zeros)
        if self.head_transforms is not None and len(self.head_transforms) > 0:
            if bbox[0] != bbox[2] and bbox[1] != bbox[3]:
                head_imgs = [transform(img.crop(bbox)) for transform in self.head_transforms]
                head_aspect_ratio = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])
            else:
                head_imgs = [torch.zeros_like(transform(img)) for transform in self.head_transforms]  # bad bbox, occurs very infrequently
                head_aspect_ratio = 1.0
        else:
            # head_imgs = [torch.zeros_like(self.transforms[0](img))]  # placeholder - nobody cares
            head_imgs = None
            head_aspect_ratio = 1.0

        if self.split == "train":
            if self.need_heatmap:
                heatmap = utils.get_heatmap(gazex_norm[0], gazey_norm[0], self.heatmap_res, self.heatmap_res)  # note for training set, there is only one annotation
            else:
                heatmap = None
            return imgs, head_imgs, bbox_norm, gazex_norm, gazey_norm, torch.tensor(inout), height, width, heatmap, torch.tensor(data_source), torch.tensor([width / height]), torch.tensor([head_aspect_ratio])
        else:
            return imgs, head_imgs, bbox_norm, gazex_norm, gazey_norm, torch.tensor(inout), height, width, torch.tensor(data_source), torch.tensor([width / height]), torch.tensor([head_aspect_ratio])

    def __len__(self):
        return len(self.data_idxs)



def collate_fn(batch):
    # batch is a list of samples returned by __getitem__
    # Each sample:
    #   train: (imgs_list, head_img, bbox_norm, gazex_norm, gazey_norm, inout, height, width, heatmap, data_source)
    #   eval : (imgs_list, head_img, bbox_norm, gazex_norm, gazey_norm, inout, height, width, data_source)

    transposed = list(zip(*batch))

    # 0th element is the list-of-images per sample
    scene_imgs_per_sample = transposed[0]  # length = batch_size; each item is list[T] length = n_transforms

    # number of image streams / backbones
    scene_n_streams = len(scene_imgs_per_sample[0])

    # stack images per stream: output is list of tensors, each [B, C, H, W]
    imgs_batched = [
        torch.stack([sample_imgs[s] for sample_imgs in scene_imgs_per_sample], dim=0)
        for s in range(scene_n_streams)
    ]

    # for consistency with with multi-person inputs at inference time, we write the dimensions as B*1
    head_imgs_per_sample = transposed[1]  
    if head_imgs_per_sample[0] is not None:
        head_n_streams = len(head_imgs_per_sample[0])
        head_imgs_batched = [
            torch.stack([sample_imgs[s] for sample_imgs in head_imgs_per_sample], dim=0)  # [B*1,3,H,W]
            for s in range(head_n_streams)
        ]
    else:
        head_imgs_batched = None
    
    out = [imgs_batched, head_imgs_batched]

    # Collate the remaining fields
    for items in transposed[2:]:
        if isinstance(items[0], torch.Tensor):
            out.append(torch.stack(items, dim=0))
        else:
            out.append(list(items))

    return tuple(out)
    
