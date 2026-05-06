"""
dataset.py — RGB + keypoint dataset for person ReID.

Now loads RGB crops as primary input by default, with optional MediaPipe
keypoints as a 99-d side input. Skeleton-image mode is preserved as an
option (`--use_skeleton` in train.py) but RGB+keypoints is the
recommended setup.
"""

import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


# ---------------------------------------------------------------------------
# Keypoint normalization
# ---------------------------------------------------------------------------

# MediaPipe Pose landmark indices we use for normalization
LEFT_HIP, RIGHT_HIP           = 23, 24
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12


def normalize_keypoints(kp: np.ndarray) -> np.ndarray:
    """
    Make keypoints translation- and scale-invariant.

      - Translation: center on hip midpoint
      - Scale:       divide by torso length (hip→shoulder distance)

    Body proportions are an identity signal; absolute pixel positions are
    not. This converts the raw (x, y) coords into a pose-relative frame
    where two crops of the same person at different walking phases or
    different positions in the bbox are much more comparable.

    Args:
        kp: (33, 3) array of (x, y, visibility)

    Returns:
        (33, 3) normalized array. If input is all-zero (no detection),
        returns input unchanged so the downstream model sees the
        no-detection signal as zeros.
    """
    if not np.any(kp):
        return kp

    out = kp.copy()
    hip_mid = (kp[LEFT_HIP, :2] + kp[RIGHT_HIP, :2]) / 2
    sh_mid  = (kp[LEFT_SHOULDER, :2] + kp[RIGHT_SHOULDER, :2]) / 2
    torso   = np.linalg.norm(sh_mid - hip_mid) + 1e-6
    out[:, :2] = (kp[:, :2] - hip_mid) / torso
    return out


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def build_transforms(split: str, img_size=(256, 128), use_rgb: bool = True):
    """
    Train/eval pipelines.

    For RGB input: ColorJitter regularizes against camera-specific color
    casts (Market-1501 has 6 cameras with noticeably different white
    balance), and RandomErasing works as intended.

    For skeleton input: ColorJitter is meaningless (no real color), and
    RandomErasing is dialed way down because erasing a chunk of a sparse
    1-pixel-wide skeleton wipes out most of the actual signal.
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        ops = [T.Resize(img_size)]
        if use_rgb:
            ops.append(T.ColorJitter(0.2, 0.2, 0.2, 0.1))
        ops += [
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(10),
            T.RandomCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean, std),
            T.RandomErasing(
                p     = 0.5  if use_rgb else 0.1,
                scale = (0.02, 0.25) if use_rgb else (0.02, 0.08),
            ),
        ]
        return T.Compose(ops)
    else:
        return T.Compose([
            T.Resize(img_size),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SkeletonReIDDataset(Dataset):
    """
    Loads RGB crops (default) or skeleton images, paired with identity
    labels and optional MediaPipe keypoint vectors.

    Directory layout (produced by preprocess.py and pose_estimation.py):
        processed/{split}/{pid:04d}/{filename}.jpg          <- RGB input (default)
        skeleton_images/{split}/{pid:04d}/{stem}.jpg        <- legacy skeleton input
        keypoints/{split}/{pid:04d}/{stem}.npy              <- (33,3) keypoints

    Args:
        data_root         : parent dir containing processed/, skeleton_images/, keypoints/
        split             : 'train', 'test', or 'query'
        transform         : torchvision transform pipeline
        use_rgb           : True → load RGB from processed/ (recommended)
                            False → load rendered skeleton image
        use_keypoints     : also load (33,3) MediaPipe arrays (normalized)
        skip_no_detection : exclude samples whose keypoint array is all-zero.
                            Default False — when use_rgb=True, RGB is
                            informative regardless of pose detection, so
                            keeping these samples gives a complete dataset.
    """

    def __init__(self,
                 data_root: str,
                 split: str,
                 transform=None,
                 use_rgb: bool = True,
                 use_keypoints: bool = False,
                 skip_no_detection: bool = False):

        self.data_root         = data_root
        self.split             = split
        self.transform         = transform
        self.use_rgb           = use_rgb
        self.use_keypoints     = use_keypoints
        self.skip_no_detection = skip_no_detection

        self.processed_dir = os.path.join(data_root, "processed",       split)
        self.skeleton_dir  = os.path.join(data_root, "skeleton_images", split)
        self.keypoint_dir  = os.path.join(data_root, "keypoints",       split)

        self.samples   = []   # list of (img_path, kp_path, label_idx, cam_id)
        self.pid2label = {}

        self._load_samples()

    def _load_samples(self):
        if not os.path.isdir(self.processed_dir):
            raise FileNotFoundError(
                f"Processed split not found: {self.processed_dir}\n"
                "Run preprocess.py first."
            )
        if not self.use_rgb and not os.path.isdir(self.skeleton_dir):
            raise FileNotFoundError(
                f"Skeleton directory not found: {self.skeleton_dir}\n"
                "Run pose_estimation.py first, or pass use_rgb=True."
            )

        skipped_no_det  = 0
        skipped_missing = 0

        pids = sorted(os.listdir(self.processed_dir))
        for label_idx, pid_str in enumerate(pids):
            pid_proc_dir = os.path.join(self.processed_dir, pid_str)
            if not os.path.isdir(pid_proc_dir):
                continue
            self.pid2label[pid_str] = label_idx

            for fname in sorted(os.listdir(pid_proc_dir)):
                if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
                    continue

                stem = os.path.splitext(fname)[0]

                # Resolve image path based on modality
                if self.use_rgb:
                    img_path = os.path.join(self.processed_dir, pid_str, fname)
                else:
                    img_path = os.path.join(self.skeleton_dir, pid_str, fname)
                    if not os.path.isfile(img_path):
                        img_path = os.path.join(self.skeleton_dir, pid_str, stem + ".png")

                if not os.path.isfile(img_path):
                    skipped_missing += 1
                    continue

                kp_path = os.path.join(self.keypoint_dir, pid_str, stem + ".npy")

                # Optional: drop blank-skeleton samples (mainly for legacy mode)
                if self.skip_no_detection and os.path.isfile(kp_path):
                    kp = np.load(kp_path)
                    if not np.any(kp):
                        skipped_no_det += 1
                        continue

                try:
                    cam_id = int(fname.split("_c")[1][0]) - 1   # 0-indexed
                except (IndexError, ValueError):
                    cam_id = 0

                self.samples.append((img_path, kp_path, label_idx, cam_id))

        modality = "RGB" if self.use_rgb else "skeleton"
        print(
            f"[{self.split}/{modality}] {len(self.samples)} samples | "
            f"{len(self.pid2label)} identities | "
            f"skipped (no pose): {skipped_no_det} | "
            f"skipped (missing file): {skipped_missing}"
        )

    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, kp_path, pid_label, cam_id = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        keypoints = torch.zeros(99, dtype=torch.float32)
        if self.use_keypoints and os.path.isfile(kp_path):
            kp = np.load(kp_path).astype(np.float32)   # (33, 3)
            kp = normalize_keypoints(kp)               # hip-centered, torso-scaled
            keypoints = torch.from_numpy(kp.flatten()) # (99,)

        return {
            "img":       img,
            "pid":       torch.tensor(pid_label, dtype=torch.long),
            "cam":       torch.tensor(cam_id,    dtype=torch.long),
            "keypoints": keypoints,
            "path":      img_path,
        }


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_dataloaders(data_root: str,
                      img_size:          tuple = (256, 128),
                      batch_size:        int   = 64,
                      num_workers:       int   = 4,
                      use_rgb:           bool  = True,
                      use_keypoints:     bool  = False,
                      skip_no_detection: bool  = False):
    """Returns (train_loader, test_loader, query_loader, num_train_classes)."""

    def make_ds(split):
        return SkeletonReIDDataset(
            data_root         = data_root,
            split             = split,
            transform         = build_transforms(split, img_size, use_rgb),
            use_rgb           = use_rgb,
            use_keypoints     = use_keypoints,
            skip_no_detection = skip_no_detection,
        )

    train_ds = make_ds("train")
    test_ds  = make_ds("test")
    query_ds = make_ds("query")

    def make_loader(ds, shuffle):
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True,
            drop_last=shuffle,
        )

    return (
        make_loader(train_ds, shuffle=True),
        make_loader(test_ds,  shuffle=False),
        make_loader(query_ds, shuffle=False),
        len(train_ds.pid2label),
    )