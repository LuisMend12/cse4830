# Preprocessing and Pose Contribution

This work adds two main contributions on top of the original Market-1501 dataset:

1. a preprocessing pipeline to clean and standardize the dataset
2. a pose-estimation pipeline to generate human pose features for each image

## Preprocessing Contribution

The preprocessing step was implemented in [preprocess.py](/Users/kylenguyen/Documents/cse4830project/cse4830/market_1501_data/preprocess.py).

Its purpose was to convert the raw Market-1501 image folders into a cleaner and more consistent format for downstream training and analysis.

The preprocessing pipeline does the following:

- reads images from `bounding_box_train/`, `bounding_box_test/`, and `query/`
- parses each filename to extract person ID, camera ID, sequence ID, frame number, and detection number
- removes invalid Market-1501 identities:
  - `pid = -1` for distractors
  - `pid = 0` for junk images
- keeps only valid person IDs
- resizes all valid images to `128 x 64`
- reorganizes images by split and person ID
- records image-level metadata in `data/metadata.csv`

The output of preprocessing is stored in:

- `data/processed/train/`
- `data/processed/test/`
- `data/processed/query/`
- `data/metadata.csv`

After preprocessing, the valid dataset contained:

- `12,936` training images
- `13,102` test images
- `3,362` query images
- `29,400` total valid images

## Pose Contribution

The pose-estimation step was implemented in [pose_estimation.py](/Users/kylenguyen/Documents/cse4830project/cse4830/market_1501_data/pose_estimation.py).

This contribution adds pose-based information for every preprocessed image using MediaPipe Pose Landmarker.

For each image, the pipeline generates:

- a `33 x 3` NumPy array of pose landmarks containing `x`, `y`, and `visibility`
- a rendered skeleton image showing the detected body structure

The pose outputs are stored in:

- `data/keypoints/`
- `data/skeleton_images/`
- `data/pose_stats.json`

Pose generation was run on all `29,400` preprocessed images. The final statistics were:

- pose detected on `20,805` images
- no pose detected on `8,595` images
- `0` processing errors
- detection rate: `70.8%`

When no pose is detected, the pipeline still saves:

- a zero-filled keypoint array
- a blank skeleton image

This keeps the dataset structure consistent and makes it easier to use in later experiments.

## Why This Contribution Matters

The preprocessing contribution makes the original dataset cleaner, standardized, and easier to load. The pose contribution adds a second representation of each person image through body structure, which can be useful for person re-identification experiments, multimodal training setups, or feature fusion with RGB images.

## Files Used

- [preprocess.py](/Users/kylenguyen/Documents/cse4830project/cse4830/market_1501_data/preprocess.py)
- [pose_estimation.py](/Users/kylenguyen/Documents/cse4830project/cse4830/market_1501_data/pose_estimation.py)
- [requirements.txt](/Users/kylenguyen/Documents/cse4830project/cse4830/market_1501_data/requirements.txt)
