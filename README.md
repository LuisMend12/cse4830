# CSE4830 Person Re-Identification Project

This repository trains and evaluates a person re-identification model on the
Market-1501 dataset. The main model uses RGB person crops with an optional
MediaPipe pose/keypoint branch, plus standard ReID training losses.

## Project Overview

The pipeline has four main stages:

1. Preprocess the raw Market-1501 folders into a consistent dataset layout.
2. Generate MediaPipe pose keypoints and rendered skeleton images.
3. Train a ResNet50-based ReID model with ID loss, batch-hard triplet loss,
   and center loss.
4. Evaluate Rank-1 accuracy and mAP on the Market-1501 query/gallery split.

The recommended setup is RGB images plus keypoints:

```powershell
python train.py --data_root market_1501_data/data --use_keypoints
```

## Repository Structure

```text
.
+-- dataset.py                    # Dataset and torchvision transforms
+-- model.py                      # ResNet50 / ResNet50-IBN ReID model
+-- train.py                      # Training loop and checkpoint saving
+-- evaluate.py                   # Standalone evaluation script
+-- plot_training_curves.py       # Plot metrics from a training log
+-- README.md                     # Project instructions
`-- market_1501_data/
    +-- preprocess.py             # Raw Market-1501 preprocessing
    +-- pose_estimation.py        # MediaPipe keypoint generation
    +-- requirements.txt          # Preprocessing/pose dependencies
    +-- bounding_box_train/       # Raw training images
    +-- bounding_box_test/        # Raw gallery/test images
    +-- query/                    # Raw query images
    `-- data/
        +-- processed/            # Preprocessed RGB images
        +-- keypoints/            # MediaPipe keypoint .npy files
        +-- skeleton_images/      # Rendered skeleton images
        +-- metadata.csv          # Image metadata from preprocessing
        `-- pose_stats.json       # Pose extraction summary
```

## Setup

Create and activate a Python environment, then install the required packages.
The project uses PyTorch, torchvision, NumPy, Pillow, tqdm, matplotlib,
OpenCV, and MediaPipe.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision numpy pillow tqdm matplotlib opencv-python mediapipe
```

If you only need the preprocessing and pose tools, the dependencies listed in
`market_1501_data/requirements.txt` are:

```powershell
pip install -r market_1501_data/requirements.txt
```

## Data Preparation

The raw Market-1501 folders should be inside `market_1501_data/`:

```text
market_1501_data/
+-- bounding_box_train/
+-- bounding_box_test/
`-- query/
```

Run preprocessing from the dataset folder:

```powershell
cd market_1501_data
python preprocess.py
cd ..
```

This creates:

```text
market_1501_data/data/processed/train/
market_1501_data/data/processed/test/
market_1501_data/data/processed/query/
market_1501_data/data/metadata.csv
```

To preview what preprocessing would do without writing files:

```powershell
cd market_1501_data
python preprocess.py --dry-run
cd ..
```

## Pose and Keypoint Generation

After preprocessing, generate pose keypoints and skeleton images:

```powershell
cd market_1501_data
python pose_estimation.py
cd ..
```

Optional flags:

```powershell
python pose_estimation.py --split train
python pose_estimation.py --split query --limit 100
```

Pose generation writes:

```text
market_1501_data/data/keypoints/
market_1501_data/data/skeleton_images/
market_1501_data/data/pose_stats.json
```

Each keypoint file is a `33 x 3` NumPy array containing MediaPipe `x`, `y`,
and `visibility` values. If no pose is detected, the script saves a zero-filled
keypoint array so the dataset layout stays consistent.

## Training

Train the default RGB model:

```powershell
python train.py --data_root market_1501_data/data
```

Train the recommended RGB + keypoint model:

```powershell
python train.py --data_root market_1501_data/data --use_keypoints
```

Train with the ResNet50-IBN-a backbone:

```powershell
python train.py --data_root market_1501_data/data --use_keypoints --use_ibn
```

Useful training options:

```text
--output_dir        Directory for checkpoints, default: ./checkpoints
--epochs            Number of training epochs, default: 120
--warmup_epochs     Linear learning-rate warmup epochs, default: 10
--num_pids          Identities per batch for P x K sampling, default: 8
--num_imgs          Images per identity for P x K sampling, default: 8
--eval_bs           Evaluation batch size, default: 64
--lr                Base learning rate, default: 3.5e-4
--center_lr         Learning rate for center-loss centroids, default: 0.5
--center_weight     Center-loss weight, default: 5e-4
--img_h             Input image height, default: 256
--img_w             Input image width, default: 128
--eval_every        Evaluate every N epochs, default: 5
--use_keypoints     Fuse MediaPipe keypoints with RGB features
--use_ibn           Use ResNet50-IBN-a instead of standard ResNet50
--use_skeleton      Legacy mode using rendered skeleton images as input
--rerank            Apply k-reciprocal re-ranking during evaluation
--seed              Random seed, default: 42
```

Training checkpoints are saved in `checkpoints/` by default:

```text
checkpoints/epoch_005.pth
checkpoints/epoch_010.pth
checkpoints/best_model.pth
```

## Evaluation

Evaluate a saved checkpoint:

```powershell
python evaluate.py `
  --data_root market_1501_data/data `
  --checkpoint checkpoints/best_model.pth `
  --num_classes 751 `
  --use_keypoints
```

Use the same model flags that were used during training. For example, include
`--use_ibn` if the checkpoint was trained with `--use_ibn`, and include
`--use_skeleton` for legacy skeleton-image checkpoints.

Enable re-ranking:

```powershell
python evaluate.py `
  --data_root market_1501_data/data `
  --checkpoint checkpoints/best_model.pth `
  --num_classes 751 `
  --use_keypoints `
  --rerank
```

The script reports:

```text
Rank-1: ...
mAP   : ...
```

## Plot Training Curves

If training output is saved to a log file, plot the learning curves with:

```powershell
python plot_training_curves.py --log train.log --out training_curves.png
```

## Model Details

`GaitReIDNet` is built around a ResNet50 backbone with a BN neck and classifier.
It can optionally use:

- MediaPipe keypoint fusion through a small MLP branch.
- ResNet50-IBN-a for stronger cross-camera generalization.
- Rendered skeleton images in legacy mode.

The training objective combines:

- Cross-entropy with label smoothing.
- Batch-hard triplet loss.
- Center loss for tighter identity clusters.

The training loader uses P x K sampling, where each batch contains `P`
identities and `K` images per identity. This ensures triplet mining has
positive and negative examples in every batch.

## Notes

- `market_1501_data/readme.txt` contains the original Market-1501 dataset
  description and citation request.
- The dataset is intended for research use only.
- On Windows, if `train.py --help` has an encoding issue, run:

```powershell
$env:PYTHONIOENCODING='utf-8'
python train.py --help
```
