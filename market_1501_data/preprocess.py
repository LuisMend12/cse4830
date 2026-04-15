"""
preprocess.py - Market-1501 Data Preprocessing

Filters junk/distractor identities, resizes all images to 128x64,
and writes a metadata CSV.

Market-1501 filename: {pid:04d}_c{cam}s{seq}_{frame:06d}_{det:02d}.jpg
  pid = -1  → distractor (excluded)
  pid = 0   → junk       (excluded)
  pid 1-1501 → valid     (kept)

Outputs:
  data/processed/{split}/{pid:04d}/{filename}.jpg
  data/metadata.csv

Usage:
  python preprocess.py           # full run
  python preprocess.py --dry-run # report only, no writes
"""

import re
import csv
import argparse
from pathlib import Path
import cv2

ROOT    = Path(__file__).parent
TARGET  = (128, 64)  # (height, width)
SPLITS  = {"train": "bounding_box_train", "test": "bounding_box_test", "query": "query"}
NAME_RE = re.compile(r"^(-?\d+)_c(\d+)s(\d+)_(\d+)_(\d+)\.jpg$")


def parse(fname):
    m = NAME_RE.match(fname)
    return tuple(int(x) for x in m.groups()) if m else None


def preprocess(dry_run=False):
    out_dir  = ROOT / "data" / "processed"
    csv_path = ROOT / "data" / "metadata.csv"
    rows = []
    total = kept = skipped = errors = 0

    for split, folder in SPLITS.items():
        src_dir = ROOT / folder
        if not src_dir.exists():
            print(f"[WARN] Not found: {src_dir}")
            continue

        images = sorted(src_dir.glob("*.jpg"))
        print(f"[{split}] {len(images)} images")

        for src in images:
            total += 1
            parsed = parse(src.name)
            if not parsed:
                skipped += 1
                continue

            pid, cam, seq, frame, det = parsed
            if pid <= 0:
                skipped += 1
                continue

            dst = out_dir / split / f"{pid:04d}" / src.name
            if not dry_run:
                img = cv2.imread(str(src))
                if img is None:
                    errors += 1
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dst), cv2.resize(img, (TARGET[1], TARGET[0])),
                            [cv2.IMWRITE_JPEG_QUALITY, 95])

            rows.append({"split": split, "pid": pid, "camera": cam,
                         "sequence": seq, "frame": frame, "det": det,
                         "src_path": str(src.relative_to(ROOT)),
                         "dst_path": str(dst.relative_to(ROOT)) if not dry_run else ""})
            kept += 1

    if not dry_run and rows:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

    tag = "[DRY RUN] " if dry_run else ""
    print(f"\n{tag}Scanned: {total}  Kept: {kept}  Skipped: {skipped}"
          + (f"  Errors: {errors}" if errors else ""))
    if not dry_run:
        print(f"Output → {out_dir}/")
        print(f"CSV    → {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    preprocess(dry_run=parser.parse_args().dry_run)
