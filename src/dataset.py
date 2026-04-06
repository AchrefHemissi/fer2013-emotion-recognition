"""
FERDataset — supports both CSV format and folder format.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight


CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
SPLIT_MAP = {"Training": "train", "PublicTest": "val", "PrivateTest": "test"}


class FERDataset:
    """
    FER2013 Dataset supporting two formats:
      - CSV   : fer2013.csv with columns 'emotion', 'pixels', 'Usage'
      - Folder: train/<class>/*.png, test/<class>/*.png

    Args:
        root (str)  : Path to the dataset root directory.
        split (str) : 'train', 'val', or 'test'.
        transform   : Optional transform to apply to each image.
        fmt (str)   : 'csv' or 'folder'. If None, auto-detected.
    """

    def __init__(self, root: str, split: str = "train", transform=None, fmt: str = None):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.class_names = CLASS_NAMES

        if fmt is None:
            fmt = "csv" if (self.root / "fer2013.csv").exists() else "folder"
        self.fmt = fmt

        self.images = []  # list of np.ndarray (H, W) uint8
        self.labels = []  # list of int

        if fmt == "csv":
            self._load_csv()
        else:
            self._load_folder()

    # ── Loaders ─────────────────────────────────────────────────────────────

    def _load_csv(self):
        csv_path = self.root / "fer2013.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"fer2013.csv not found in {self.root}")

        df = pd.read_csv(csv_path)
        reverse_map = {v: k for k, v in SPLIT_MAP.items()}
        target_usage = reverse_map.get(self.split, self.split)

        usage_vals = df["Usage"].unique()
        if self.split in usage_vals:
            subset = df[df["Usage"] == self.split]
        elif target_usage in usage_vals:
            subset = df[df["Usage"] == target_usage]
        else:
            subset = df[df["Usage"].str.lower() == self.split.lower()]

        for _, row in subset.iterrows():
            pixels = np.array(row["pixels"].split(), dtype=np.uint8).reshape(48, 48)
            self.images.append(pixels)
            self.labels.append(int(row["emotion"]))

    def _load_folder(self):
        folder_map = {"train": "train", "val": "test", "test": "test"}
        split_folder = self.root / folder_map.get(self.split, self.split)

        if not split_folder.exists():
            alt = self.root / "test"
            if self.split == "val" and alt.exists():
                split_folder = alt
            else:
                raise FileNotFoundError(f"Split folder not found: {split_folder}")

        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = split_folder / class_name.lower()
            if not class_dir.exists():
                class_dir = split_folder / class_name
            if not class_dir.exists():
                continue
            for img_path in sorted(class_dir.glob("*")):
                if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                    continue
                try:
                    img = Image.open(img_path).convert("L")
                    self.images.append(np.array(img, dtype=np.uint8))
                    self.labels.append(class_idx)
                except Exception:
                    pass

    # ── Interface ────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_np = self.images[idx]
        label = self.labels[idx]
        img = Image.fromarray(img_np, mode="L")
        if self.transform:
            img = self.transform(img)
        return img, label

    def get_class_weights(self) -> np.ndarray:
        """Compute balanced class weights."""
        labels_np = np.array(self.labels)
        classes = np.arange(len(CLASS_NAMES))
        return compute_class_weight(class_weight="balanced", classes=classes, y=labels_np)

    def get_distribution(self) -> dict:
        """Return per-class sample count."""
        dist = {name: 0 for name in CLASS_NAMES}
        for label in self.labels:
            dist[CLASS_NAMES[label]] += 1
        return dist

    def __repr__(self):
        dist = self.get_distribution()
        lines = [f"FERDataset(split='{self.split}', format='{self.fmt}', total={len(self)})"]
        for name, count in dist.items():
            lines.append(f"  {name:10s}: {count}")
        return "\n".join(lines)
