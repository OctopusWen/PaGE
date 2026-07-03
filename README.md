# PaGE: Towards Practical Human-Level Gaze Target Estimation


Official PyTorch implementation of **PaGE** (Practical Gaze Estimator), a gaze target estimation model with human-level accuracy and deployable student models.

**Zhoutong Ye<sup>1,*</sup>**, **Chengwen Zhang<sup>1,*</sup>**, Jiaqi Liu<sup>1</sup>, Xiangwu Li<sup>2</sup>, Xutong Wang<sup>1</sup>, Yu Mei<sup>1</sup>, Zhaibin Cui<sup>1</sup>, Mingze Sun<sup>1</sup>, Qingyang Wan<sup>1</sup>, Chang Liu<sup>1</sup>, Chun Yu<sup>1,†</sup>, Yuanchun Shi<sup>1</sup>

<sup>1</sup>Tsinghua University &nbsp;&nbsp; <sup>2</sup>Jinan University &nbsp;&nbsp; <sup>*</sup>Equal contribution &nbsp;&nbsp; <sup>†</sup>Corresponding author

![architecture](.\assets\architecture.jpg)

## Links

- Project page: https://page-26.github.io/
- Models: https://huggingface.co/collections/Octopus1/page
- Paper: arXiv coming soon

## Table of Contents

- [Overview](#overview-of-page)
- [Installation](#installation)
- [Pre-trained Models](#pre-trained-models)
- [Inference](#inference)
- [Data Preparation](#data-preparation)
- [Training Backbones](#training-backbones)
- [Training](#training)
- [Evaluation](#evaluation)
- [Model Architectures](#model-architectures)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

## Overview of PaGE

Gaze target estimation, the task of predicting where a person is looking in a scene, is crucial to understanding human attention and intent. However, human-level performance remains elusive for existing models, limiting their practical application. To this end, we propose **PaGE** (Practical Gaze Estimator), a gaze estimation model that explicitly models the complex interaction between scene and head features. PaGE **outperforms humans in 7 out of 9 metrics** while reducing the human-AI gap by at least 60% in the remaining 2.

![leaderboard](.\assets\leaderboard.jpg)

### Visualized examples

![main-text](.\assets\main-text.jpg)

### Generalization Beyond Real-world Human Gaze

We also find that PaGE can predict the visual attention of animals, cameras, and animated characters **without extra training**. We consider this a first step towards a **universal visual attention estimator**. We hope

![discussion-new](.\assets\discussion-new.jpg)

## Installation

### Requirements

- Python 3.12+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)

### Environment Setup

We use [uv](https://github.com/astral-sh/uv) for fast, reproducible dependency management. The pinned dependency versions are committed in `uv.lock`, so `uv sync` reproduces the exact environment:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create the virtual environment and install PaGE + all dependencies from uv.lock
uv sync
```


### HuggingFace Token Setup

PaGE uses DINOv3 backbones from HuggingFace. You need to set up your HuggingFace token:

```bash
# Option 1: Set environment variable
export HF_TOKEN=your_hf_token_here

# Option 2: Use .env file (copy from example)
cp .env.example .env
# Edit .env and add your token
```

Get your HuggingFace token from: https://huggingface.co/settings/tokens

## Pre-trained Models

We provide pre-trained checkpoints on HuggingFace:

- [page-vits](https://huggingface.co/Octopus1/page-vits) - Small model (ViT-S backbone)
- [page-vitsplus](https://huggingface.co/Octopus1/page-vitsplus) - Small+ model
- [page-vitb](https://huggingface.co/Octopus1/page-vitb) - Base model (ViT-B backbone)
- [page-vithplus](https://huggingface.co/Octopus1/page-vithplus) - Huge+ model (ViT-H+ backbone)

Download checkpoints that you need:

```bash
# Using HuggingFace CLI
hf download Octopus1/page-vits --local-dir ./checkpoints/page-vits
hf download Octopus1/page-vitsplus --local-dir ./checkpoints/page-vitsplus
hf download Octopus1/page-vitb --local-dir ./checkpoints/page-vitb
hf download Octopus1/page-vithplus --local-dir ./checkpoints/page-vithplus
```

## Inference

Run gaze target prediction on a single image given one head bounding box. For inference, you only need the environment and a released PaGE checkpoint. The released HuggingFace checkpoints are **self-contained**: the DINOv3 backbone weights are bundled in the checkpoint and the model structure is loaded from PaGE remote code, so **you do not need the separate DINOv3 backbone weights** used for training.

```bash
python scripts/inference.py \
    --image ./demo/scene.jpg \
    --bbox 0.30 0.12 0.48 0.40 \
    --model_path ./checkpoints/page-vitb \
    --output ./visualization/inference.png
```

- `--bbox` is the head box as normalized `xmin ymin xmax ymax` in `[0, 1]`.
- `--model_path` can be a local checkpoint directory (e.g. `./checkpoints/page-vitb`) or a HuggingFace repo id (e.g. `Octopus1/page-vitb`).
- `--device` defaults to CUDA when available; use `--device cpu` if GPU memory is limited.
- The script crops the head from the frame, runs the model, and saves a visualization with the predicted gaze heatmap, gaze point, and in/out-of-frame probability.

> Inference requires `transformers==5.6.2` (already pinned in `pyproject.toml`; run `uv sync` if it is not installed). The model code is fetched via `trust_remote_code=True` on first run. If you enable verbose HuggingFace loading logs and see `MISSING` / `UNEXPECTED` backbone keys, this is expected across Transformers DINOv3 naming versions; PaGE remote code reloads and remaps the backbone weights after the initial report.

## Data Preparation

### Dataset Structure

Create a `data` directory in the project root:

```
PaGE/
├── data/
│   ├── gazefollow/
│   ├── vat/
│   ├── childplay/
│   ├── OpenImages/
│   └── MPII/
```

### Download Datasets

#### 1. GazeFollow

Download from the [official website](http://gazefollow.csail.mit.edu/):
- `train_annotations_release.txt`
- `test_annotations_release.txt`
- Image files

Place in `data/gazefollow/` with the following layout:

```
data/gazefollow/
├── train/
├── test2/
├── test_annotations_release.txt
└── train_annotations_release.txt
```

#### 2. VideoAttentionTarget (VAT)

Download from the [official repository](https://github.com/ejcgt/attention-target-detection):
- Annotations
- Video frames

Place in `data/vat/` with the following layout:

```
data/vat/
├── annotations/
└── images/
```

#### 3. ChildPlay

Download from the [official website](https://github.com/ejcgt/attention-target-detection):
- Annotations
- Video frames

Place in `data/childplay/` with the following layout:

```
data/childplay/
├── images/
└── annotations/
```


> **Distillation only:** The two datasets below (MPII and OpenImages) are required **only for knowledge distillation**. If you are not running distillation, you can skip them entirely.

#### 6. MPII Human Pose

Download from the [official website](http://human-pose.mpi-inf.mpg.de/):
- `mpii_human_pose_v1_u12_1.mat` (annotations)
- Images

Place in `data/MPII/` with the following layout:

```
data/MPII/
├── images/                              # all .jpg frames
└── mpii_human_pose_v1_u12_2/
    └── mpii_human_pose_v1_u12_1.mat     # annotation file
```


#### 7. OpenImages

We use images from [Open Images V7](https://storage.googleapis.com/openimages/web/index.html), selecting samples annotated with the person-related classes `Person`, `Man`, `Woman`, `Boy`, and `Girl`. Only the **train** split is downloaded.

We provide a download script based on [FiftyOne](https://voxel51.com/fiftyone/):

```bash
# Install FiftyOne into the environment
uv pip install fiftyone

# Download the OpenImages train split (person classes, detections)
python data_prep/download_openimages.py
```

### Data Preprocessing

After downloading datasets, preprocess them:

```bash
# GazeFollow
python data_prep/preprocess_gazefollow.py --data_path ./data/gazefollow

# VideoAttentionTarget
python data_prep/preprocess_vat.py --data_path ./data/vat

# ChildPlay
python data_prep/preprocess_childplay.py --data_path ./data/childplay
```

Each preprocessing script generates `{split}_preprocessed.json` files in the respective dataset directories.

#### Distillation-only preprocessing (MPII & OpenImages)

Only needed if you plan to run knowledge distillation.

```bash
# MPII
python data_prep/preprocess_mpii.py --data_path ./data/MPII

# OpenImages (requires YOLOv5 for head detection)
# First, clone and setup yolov5-crowdhuman:
git clone https://github.com/MahenderAutonomo/yolov5-crowdhuman.git
cd yolov5-crowdhuman
# Download weights as per the repo instructions
cd ..

python data_prep/preprocess_openimages.py \
    --data_path ./data/OpenImages \
    --split train \
    --crowdhuman_repo ./yolov5-crowdhuman \
    --crowdhuman_weights ./yolov5-crowdhuman/weights/crowdhuman_yolov5m.pt
```

## Training Backbones

Training from scratch initializes the scene and head branches from pre-trained DINOv3 backbones. These are **not** needed for inference with a released PaGE checkpoint, but are **required if you train or fine-tune from scratch**.

Download the backbone matching the model size you plan to train into `./checkpoints/`:

```bash
# ViT-S  (page_vits_*)
hf download facebook/dinov3-vits16-pretrain-lvd1689m      --local-dir ./checkpoints/dinov3-vits16-pretrain-lvd1689m

# ViT-S+ (page_vitsplus_*)
hf download facebook/dinov3-vits16plus-pretrain-lvd1689m  --local-dir ./checkpoints/dinov3-vits16plus-pretrain-lvd1689m

# ViT-B  (page_vitb_*)
hf download facebook/dinov3-vitb16-pretrain-lvd1689m      --local-dir ./checkpoints/dinov3-vitb16-pretrain-lvd1689m

# ViT-H+ (page_vithplus_*)
hf download facebook/dinov3-vith16plus-pretrain-lvd1689m  --local-dir ./checkpoints/dinov3-vith16plus-pretrain-lvd1689m
```

The paths above match what [page/model_factory.py](page/model_factory.py) expects (`./checkpoints/dinov3-*`). DINOv3 weights are gated on HuggingFace, so make sure your `HF_TOKEN` is set (see [HuggingFace Token Setup](#huggingface-token-setup)) and that you have accepted the model license.

## Training

### Basic Training

Train a PaGE model on GazeFollow, VideoAttentionTarget, and ChildPlay:

```bash
python scripts/train_all.py \
    --model page_vitb_inout \
    --gf_data_path ./data/gazefollow \
    --vat_data_path ./data/vat \
    --cp_data_path ./data/childplay \
    --inout \
    --exp_name page_vitb_training \
    --lr 1e-3 \
    --batch_size 60 \
    --max_epochs 15 \
    --eval_every_epochs 3 \
    --ckpt_save_dir ./experiments
```

### Fine-tuning

Fine-tune a pre-trained model:

```bash
python scripts/train_all.py \
    --model page_vitb_inout_finetune \
    --model_ckpt_path experiments/page_vitb_training/epoch_14.pt \
    --gf_data_path ./data/gazefollow \
    --vat_data_path ./data/vat \
    --cp_data_path ./data/childplay \
    --inout \
    --exp_name page_vitb_finetune \
    --batch_size 60 \
    --max_epochs 5 \
    --eval_every_epochs 1 \
    --warmup_iters 500 \
    --warmup_start_lr 1e-7 \
    --lr 1e-5 \
    --weight_decay 1e-2 \
    --ckpt_save_dir ./experiments
```

### Knowledge Distillation

Distill a large teacher model to a smaller student:

```bash
python scripts/distill.py \
    --teacher_model page_vithplus_inout_finetune \
    --teacher_ckpt ./experiments/page_vithplus_finetune/epoch_4.pt \
    --student_model page_vitb_inout_student \
    --gf_data_path ./data/gazefollow \
    --vat_data_path ./data/vat \
    --cp_data_path ./data/childplay \
    --mpii_data_path ./data/MPII \
    --openimages_data_path ./data/OpenImages \
    --max_epochs 20 \
    --batch_size 60 \
    --lr 2e-4 \
    --head_loss_lambda 1.0 \
    --dino_loss_lambda 1.0 \
    --exp_name distill_h2b \
    --ckpt_save_dir ./experiments \
    --max_images 2000000 \
    --eval_every_epochs 1
```

After distillation, the student can be further fine-tuned on the full training set (GazeFollow + VideoAttentionTarget + ChildPlay), using the distilled checkpoint as initialization:

```bash
python scripts/train_all.py \
    --model page_vitb_inout_finetune \
    --model_ckpt_path ./experiments/distill_h2b/epoch_19.pt \
    --gf_data_path ./data/gazefollow \
    --vat_data_path ./data/vat \
    --cp_data_path ./data/childplay \
    --inout \
    --exp_name page_vitb_inout_distill_finetune \
    --batch_size 60 \
    --max_epochs 3 \
    --eval_every_epochs 1 \
    --warmup_iters 500 \
    --warmup_start_lr 1e-7 \
    --lr 2e-5 \
    --weight_decay 1e-2 \
    --ckpt_save_dir ./experiments
```

### Training Options

Key arguments:
- `--model`: Model architecture (e.g., `page_vitb_inout`, `page_vits_inout`)
- `--batch_size`: Batch size (automatically adjusted for gradient accumulation)
- `--lr`: Learning rate
- `--max_epochs`: Number of training epochs
- `--inout`: Enable in/out-of-frame classification
- `--heatmap_res`: Output heatmap resolution (default: 64)
- `--eval_every_epochs`: Evaluation frequency

## Evaluation

Evaluate on individual datasets:

```bash
# GazeFollow
python scripts/eval_gazefollow_trainstyle.py \
    --data_path ./data/gazefollow \
    --model_name page_vitb_inout_finetune \
    --ckpt_path ./checkpoints/page-vitb/model.pt \
    --batch_size 60

# VideoAttentionTarget
python scripts/eval_vat_trainstyle.py \
    --data_path ./data/vat \
    --model_name page_vitb_inout_finetune \
    --ckpt_path ./checkpoints/page-vitb/model.pt \
    --batch_size 60

# ChildPlay
python scripts/eval_childplay_trainstyle.py \
    --data_path ./data/childplay \
    --model_name page_vitb_inout_finetune \
    --ckpt_path ./checkpoints/page-vitb/model.pt \
    --batch_size 60

# GOOReal
python scripts/eval_gooreal_trainstyle.py \
    --data_path ./data/GOOReal \
    --model_name page_vitb_inout_finetune \
    --ckpt_path ./checkpoints/page-vitb/model.pt \
    --batch_size 60
```

## Model Zoo

Available models:
- `page_vits_inout`: ViT-S backbone (smallest)
- `page_vitsplus_inout`: ViT-S+ backbone
- `page_vitb_inout`: ViT-B backbone (recommended)
- `page_vitl_inout`: ViT-L backbone
- `page_vithplus_inout`: ViT-H+ backbone (largest)

For student models (used in distillation), append `_student` to the name (e.g. `page_vitb_inout_student`).
For fine-tuning, append `_finetune` (e.g. `page_vitb_inout_finetune`).

## Citation

If you find PaGE useful for your project, consider giving us a star and cite:

```bibtex
@article{page2024,
  title={PaGE: Person-Aware Gaze Estimation with Cross-Modal Attention},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## License

This project is licensed under the terms specified in the LICENSE file.

## Acknowledgments

- PaGE is based on the DINOv3 backbone from [Meta AI](https://github.com/facebookresearch/dinov3)
- We train our model with the following public datasets
  - GazeFollow from [Recasens et al.](https://www.gazefollow.com/)
  - VideoAttentionTarget from [Chong et al.](https://github.com/ejcgt/attention-target-detection)
  - ChildPlay from [Tafasca et al.](https://www.idiap.ch/en/scientific-research/data/childplay-gaze)
  - MPII from [Andriluka et al.](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/software-and-datasets/mpii-human-pose-dataset)
  - OpenImages v7 from [Google](https://storage.googleapis.com/openimages/web/index.html)

- In addition, we evaluated PaGE on the following public benchmarks
  - GOO-Real from [Tomas et al.](https://github.com/upeee/GOO-GAZE2021)
  - Zhang et al.'s [gaze target recognition benchmark](https://zoryzhang.github.io/gaze/)
  - AA from [Liu et al.](https://arxiv.org/abs/2606.31211)


- Some of our implementations are adapted from [Chong et al.](https://github.com/ejcgt/attention-target-detection) and [Gaze-LLE](https://github.com/fkryan/gazelle). We are grateful for their effort to make gaze target estimation open-source, accessible, and reproducible.

## Contact

For questions or issues, please open an issue on GitHub.
