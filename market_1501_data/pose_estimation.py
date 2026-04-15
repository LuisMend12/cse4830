"""
pose_estimation.py - MediaPipe Pose Estimation on Market-1501

Runs MediaPipe PoseLandmarker on every preprocessed image and produces:
  - Raw keypoints: data/keypoints/{split}/{pid:04d}/{stem}.npy  (shape [33,3]: x,y,visibility)
  - Skeleton images: data/skeleton_images/{split}/{pid:04d}/{stem}.jpg

Run preprocess.py before this script.

Requires pose_landmarker_lite.task in the same directory as this script.
Download: curl -L -o pose_landmarker_lite.task \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"

Usage:
  python pose_estimation.py                # all splits
  python pose_estimation.py --split train  # one split
  python pose_estimation.py --limit 100    # quick test
"""

import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT             = Path(__file__).parent
SPLITS           = ["train", "test", "query"]
POSE_CONNECTIONS      = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS
PoseLandmarker        = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
BaseOptions           = mp.tasks.BaseOptions


def extract_keypoints(img_bgr, landmarker):
    """Returns [33, 3] float32 (x_norm, y_norm, visibility), or None if no person detected."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result  = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb))
    if not result.pose_landmarks:
        return None
    return np.array(
        [[lm.x, lm.y, getattr(lm, "visibility", 1.0)] for lm in result.pose_landmarks[0]],
        dtype=np.float32,
    )


def render_skeleton(keypoints, h=128, w=64):
    """Draws skeleton stick figure on a black canvas. Returns BGR image (h x w)."""
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    if keypoints is None:
        return canvas
    pts = (keypoints[:, :2] * [w, h]).astype(int)
    vis = keypoints[:, 2]
    for conn in POSE_CONNECTIONS:
        i, j = conn.start, conn.end
        if vis[i] > 0.3 and vis[j] > 0.3:
            cv2.line(canvas, tuple(pts[i]), tuple(pts[j]), (255, 255, 255), 1)
    for idx, (px, py) in enumerate(pts):
        if vis[idx] > 0.3:
            cv2.circle(canvas, (int(px), int(py)), 2, (0, 255, 0), -1)
    return canvas


def run(splits=None, limit=None):
    model_path    = ROOT / "pose_landmarker_lite.task"
    processed_dir = ROOT / "data" / "processed"
    keypoints_dir = ROOT / "data" / "keypoints"
    skeleton_dir  = ROOT / "data" / "skeleton_images"
    stats_file    = ROOT / "data" / "pose_stats.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}\nSee download instructions at top of this file.")

    if splits is None:
        splits = SPLITS

    stats = {"total": 0, "detected": 0, "no_detection": 0, "errors": 0}
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        for split in splits:
            src_root = processed_dir / split
            if not src_root.exists():
                print(f"[WARN] {src_root} not found — run preprocess.py first")
                continue

            images = sorted(src_root.rglob("*.jpg"))
            if limit:
                images = images[:limit]
            print(f"[{split}] {len(images)} images")

            for idx, src in enumerate(images):
                if idx % 1000 == 0 and idx > 0:
                    print(f"  {idx}/{len(images)} ({100*idx//len(images)}%)")
                stats["total"] += 1

                rel  = src.relative_to(src_root)
                stem = src.stem
                kp_dst = keypoints_dir / split / rel.parent / f"{stem}.npy"
                sk_dst = skeleton_dir  / split / rel.parent / f"{stem}.jpg"

                img = cv2.imread(str(src))
                if img is None:
                    stats["errors"] += 1
                    continue

                kp = extract_keypoints(img, landmarker)
                stats["detected" if kp is not None else "no_detection"] += 1

                kp_dst.parent.mkdir(parents=True, exist_ok=True)
                np.save(str(kp_dst), kp if kp is not None else np.zeros((33, 3), dtype=np.float32))

                sk_dst.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(sk_dst), render_skeleton(kp), [cv2.IMWRITE_JPEG_QUALITY, 95])

    det_rate = 100 * stats["detected"] / max(stats["total"], 1)
    print(f"\nDetected: {stats['detected']}/{stats['total']} ({det_rate:.1f}%)"
          + (f"  Errors: {stats['errors']}" if stats["errors"] else ""))
    print(f"Keypoints → {keypoints_dir}/")
    print(f"Skeletons → {skeleton_dir}/")

    stats_file.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=SPLITS, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run(splits=[args.split] if args.split else None, limit=args.limit)
