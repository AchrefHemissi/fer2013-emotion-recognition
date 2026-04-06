"""
Preprocessing pipeline for FER2013 — Week 2.
Steps:
  1. Dataset integrity check (corrupted images, invalid labels, MD5 duplicates)
  2. Gaussian denoising (kernel 3x3, sigma=0.8)
  3. CLAHE (clipLimit=2.0, tileGridSize=(4,4))
"""

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def apply_clahe(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE to a grayscale numpy image (H, W)."""
    if img.ndim == 3:
        img = img[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    return clahe.apply(img.astype(np.uint8))


def denoise_gaussian(img: np.ndarray) -> np.ndarray:
    """Apply Gaussian denoising (kernel 3x3, sigma=0.8)."""
    if img.ndim == 3:
        img = img[:, :, 0]
    return cv2.GaussianBlur(img.astype(np.uint8), (3, 3), sigmaX=0.8)


def verify_dataset(root_dir: str) -> dict:
    """
    Verify dataset integrity.
    Returns stats: total, corrupted, duplicates, valid, label_distribution.
    Supports CSV format and folder format.
    """
    root = Path(root_dir)
    stats = {
        "total": 0,
        "corrupted": 0,
        "duplicates": 0,
        "valid": 0,
        "label_distribution": defaultdict(int),
        "format": None,
        "errors": [],
    }

    csv_path = root / "fer2013.csv"
    if csv_path.exists():
        stats["format"] = "csv"
        print(f"[verify] CSV format detected: {csv_path}")
        df = pd.read_csv(csv_path)
        md5_set = set()

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Verifying"):
            stats["total"] += 1
            label = row["emotion"]
            if label not in range(7):
                stats["errors"].append(f"Row {idx}: invalid label {label}")
                stats["corrupted"] += 1
                continue
            try:
                pixels = np.array(row["pixels"].split(), dtype=np.uint8).reshape(48, 48)
            except Exception as e:
                stats["errors"].append(f"Row {idx}: {e}")
                stats["corrupted"] += 1
                continue
            md5 = hashlib.md5(pixels.tobytes()).hexdigest()
            if md5 in md5_set:
                stats["duplicates"] += 1
            else:
                md5_set.add(md5)
            stats["label_distribution"][int(label)] += 1
            stats["valid"] += 1

    else:
        stats["format"] = "folder"
        print(f"[verify] Folder format detected: {root}")
        md5_set = set()

        for split in ["train", "test"]:
            split_dir = root / split
            if not split_dir.exists():
                continue
            for class_dir in sorted(split_dir.iterdir()):
                if not class_dir.is_dir():
                    continue
                label_name = class_dir.name.lower()
                if label_name not in CLASS_NAMES:
                    continue
                label_idx = CLASS_NAMES.index(label_name)

                for img_path in class_dir.glob("*"):
                    if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                        continue
                    stats["total"] += 1
                    try:
                        img = Image.open(img_path).convert("L")
                        img_np = np.array(img)
                        md5 = hashlib.md5(img_np.tobytes()).hexdigest()
                        if md5 in md5_set:
                            stats["duplicates"] += 1
                        else:
                            md5_set.add(md5)
                        stats["label_distribution"][label_idx] += 1
                        stats["valid"] += 1
                    except Exception as e:
                        stats["errors"].append(f"{img_path}: {e}")
                        stats["corrupted"] += 1

    print(f"\n{'='*50}")
    print(f"Dataset Integrity Report")
    print(f"{'='*50}")
    print(f"Format    : {stats['format']}")
    print(f"Total     : {stats['total']}")
    print(f"Valid     : {stats['valid']}")
    print(f"Corrupted : {stats['corrupted']}")
    print(f"Duplicates: {stats['duplicates']}")
    print(f"\nClass distribution:")
    for idx, name in enumerate(CLASS_NAMES):
        count = stats["label_distribution"].get(idx, 0)
        pct = 100 * count / max(stats["total"], 1)
        print(f"  {name.capitalize():10s} ({idx}): {count:5d}  ({pct:.1f}%)")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FER2013 Preprocessing — CR2")
    parser.add_argument("--verify", type=str, metavar="DATA_DIR",
                        help="Verify dataset integrity")
    args = parser.parse_args()

    if args.verify:
        verify_dataset(args.verify)
