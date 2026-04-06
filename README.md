# CR2 — Image Preprocessing for FER2013

INSAT GL4 — Image Processing — Week 2 (06/04/2026)

Group: Rayen Chemlali, Mohamed Dhia Medini, Khalil Ghimaji, Mohamed Achref Hemissi

---

## Project Overview

This report covers the complete image preprocessing pipeline applied to the **FER2013** dataset for automatic facial emotion recognition.
The pipeline prepares raw images before any model training by cleaning, enhancing, and normalizing the data.

---

## Dataset — FER2013

| Property | Value |
| --- | --- |
| Total images | 35,887 |
| Image format | Grayscale, 48×48 pixels |
| Classes | 7 emotions |
| Source | Kaggle — `msambare/fer2013` |

### Class Distribution (verified results)

| Class | Count | Percentage |
| --- | --- | --- |
| Angry | 4,953 | 13.8% |
| Disgust | 547 | 1.5% |
| Fear | 5,121 | 14.3% |
| Happy | 8,989 | 25.0% |
| Sad | 6,077 | 16.9% |
| Surprise | 4,002 | 11.2% |
| Neutral | 6,198 | 17.3% |

**Critical imbalance:** Disgust (547) vs Happy (8,989) — ratio 1:16.
This will require class weights when training models (handled in later weeks).

### Integrity Check Results

| Metric | Result |
| --- | --- |
| Total images | 35,887 |
| Valid | 35,887 (100%) |
| Corrupted | 0 |
| Duplicates (MD5) | 1,853 |

No corrupted images were found. 1,853 duplicate images were detected via MD5 hashing — these are kept since they are part of the original dataset distribution.

---

## Preprocessing Pipeline

### Step 1 — Data Cleaning

Performed via `verify_dataset()` in `src/preprocessing.py`:

- Iterates over all images in `train/` and `test/` folders
- Detects corrupted files (unreadable images)
- Detects invalid labels (outside 0–6 range)
- Detects exact duplicates using **MD5 hashing** of pixel arrays

```bash
python src/preprocessing.py --verify data/fer2013/archive
```

### Step 2 — Resizing

All FER2013 images are standardized to **48×48 pixels** (grayscale).
This is already the native format of FER2013, so no resizing is applied.
The notebook illustrates the impact of different resolutions to justify this choice.

### Step 3a — Gaussian Denoising

Applied via `denoise_gaussian()`:

```python
cv2.GaussianBlur(img, ksize=(3, 3), sigmaX=0.8)
```

- **Kernel:** 3×3
- **Sigma (σ):** 0.8
- **Effect:** Reduces high-frequency noise while preserving facial edges
- The difference map (amplified ×4) shows the removed noise is minimal and uniform

### Step 3b — CLAHE (Contrast Enhancement)

Applied via `apply_clahe()`:

```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
clahe.apply(img)
```

- **clipLimit:** 2.0 — limits contrast amplification to avoid noise boost
- **tileGridSize:** (4, 4) — divides the 48×48 image into 16 local regions
- **Effect:** Enhances local contrast, making facial features more distinct
- Histogram comparison shows better pixel spread after CLAHE

### Step 4 — Normalization

Pixel values are normalized to zero mean and unit variance:

```text
normalized = (pixel / 255.0 - mean) / std
```

- **Mean:** 0.5070 (computed on train set only)
- **Std:** 0.2553 (computed on train set only)

**Important:** statistics are computed **only on the train set** to avoid data leakage into validation/test sets.

---

## Project Structure

```text
fer_emotions/
├── data/
│   └── fer2013/archive/
│       ├── train/          ← training images (per class folders)
│       └── test/           ← test images (per class folders)
├── src/
│   ├── preprocessing.py    ← verify_dataset, apply_clahe, denoise_gaussian
│   └── dataset.py          ← FERDataset class (loads images from folder or CSV)
├── notebooks/
│   └── cr2_preprocessing.ipynb  ← CR2 deliverable with all illustrations
├── results/                ← generated figures (after running notebook)
│   ├── cr2_cleaning.png
│   ├── cr2_resizing.png
│   ├── cr2_denoising.png
│   ├── cr2_clahe.png
│   ├── cr2_full_pipeline.png
│   └── cr2_normalization.png
├── configs/
│   └── config.yaml         ← all parameters centralized
└── requirements.txt
```

---

## How to Run

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Verify dataset integrity

```bash
python src/preprocessing.py --verify data/fer2013/archive
```

### 3. Generate all CR2 illustrations

```bash
jupyter notebook notebooks/cr2_preprocessing.ipynb
```

Run all cells in order. Figures are saved to `results/`.

---

## Configuration

All parameters are centralized in `configs/config.yaml`:

```yaml
data:
  root: "data/fer2013/archive"
  image_size: 48
  num_classes: 7
  class_names: ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
```

---

## Key Findings

1. **Dataset is clean** — 0 corrupted images out of 35,887
2. **Severe class imbalance** — Disgust is 16× less represented than Happy → class weights will be required during training
3. **1,853 duplicates** exist in the dataset — known characteristic of FER2013, kept intentionally
4. **CLAHE improves contrast** significantly for dark/underexposed face images
5. **Gaussian denoising** removes minor noise without blurring facial features at 48×48 resolution

---

## Dependencies

```text
opencv-python   — CLAHE, Gaussian blur
numpy           — array operations
pandas          — CSV loading
matplotlib      — visualizations
seaborn         — plots
scikit-learn    — class weight computation
Pillow          — image loading
tqdm            — progress bars
pyyaml          — config loading
jupyter         — notebook
```
