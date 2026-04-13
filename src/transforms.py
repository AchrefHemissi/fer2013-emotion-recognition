"""
Torchvision-compatible transform classes for FER2013 preprocessing.

These wrap the OpenCV-based functions from preprocessing.py as callable objects
that operate on PIL Images, making them compatible with torchvision.transforms.Compose.
"""

import numpy as np
from PIL import Image

from preprocessing import apply_clahe, denoise_gaussian


class GaussianDenoise:
    """Apply Gaussian denoising (kernel 3×3, σ=0.8) to a PIL grayscale Image."""

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img)
        arr = denoise_gaussian(arr)
        return Image.fromarray(arr, mode="L")

    def __repr__(self) -> str:
        return "GaussianDenoise(kernel=(3,3), sigma=0.8)"


class CLAHE:
    """Apply CLAHE (clipLimit=2.0, tileGridSize=(4,4)) to a PIL grayscale Image."""

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img)
        arr = apply_clahe(arr)
        return Image.fromarray(arr, mode="L")

    def __repr__(self) -> str:
        return "CLAHE(clipLimit=2.0, tileGridSize=(4,4))"


class AddGaussianNoise:
    """
    Add Gaussian noise to a PIL grayscale Image.

    Args:
        std: standard deviation of the noise in pixel units (0–255 scale).
             Config default: 15
    """

    def __init__(self, std: float = 15.0):
        self.std = std

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, self.std, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="L")

    def __repr__(self) -> str:
        return f"AddGaussianNoise(std={self.std})"
