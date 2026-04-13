# FER2013 — Facial Emotion Recognition

INSAT GL4 — Image Processing Project (2026)

Group: Rayen Chemlali, Mohamed Dhia Medini, Khalil Ghimaji, Mohamed Achref Hemissi

---

## Project Roadmap

| Week | Date       | Topic                                                    | Deliverable        | Status      |
| ---- | ---------- | -------------------------------------------------------- | ------------------ | ----------- |
| 1    | 30/03/2026 | Project framing — problem, objectives, dataset selection | CR1 (PDF)          | Done        |
| 2    | 06/04/2026 | Image preprocessing pipeline                             | CR2 (notebook+PDF) | Done        |
| 3    | 13/04/2026 | Model design — architecture choice, global pipeline      | CR3                | In progress |

---

## Week 1 — Project Framing

Objective: Automatically classify facial expressions into 7 emotion categories from grayscale images.

### Dataset — FER2013

| Property | Value |
| --- | --- |
| Total | 35,887 images |
| Format | Grayscale, 48×48 pixels |
| Classes | 7 emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) |
| Source | Kaggle — `msambare/fer2013` |
| Structure | `train/` and `test/` folders, one sub-folder per class |

Critical observation: severe class imbalance — Disgust (547 samples) vs Happy (8,989 samples), ratio 1:16.
This must be compensated during training via class weights or oversampling.

Full problem statement: [docs/Compte-Rendu-1.pdf](docs/Compte-Rendu-1.pdf)

---

## Week 2 — Preprocessing Pipeline

Full report and illustrations: [docs/Compte-Rendu-2.pdf](docs/Compte-Rendu-2.pdf)

Executable notebook: [notebooks/cr2_preprocessing.ipynb](notebooks/cr2_preprocessing.ipynb)

### Pipeline steps

```text
Raw image (48×48 grayscale)
        │
        ▼
1. Dataset integrity check   ← MD5 duplicates, corrupted files, label validation
        │
        ▼
2. Gaussian denoising        ← kernel (3×3), σ=0.8
        │
        ▼
3. CLAHE                     ← clipLimit=2.0, tileGridSize=(4×4)
        │
        ▼
4. Normalization             ← (pixel/255 − 0.5070) / 0.2553
        │
        ▼
  Preprocessed image → ready for model input
```

### Integrity check results

| Metric | Result |
| --- | --- |
| Total | 35,887 |
| Valid | 35,887 (100%) |
| Corrupted | 0 |
| MD5 duplicates | 1,853 (kept - part of original distribution) |

### Key parameters (centralized in `configs/config.yaml`)

| Parameter | Value | Justification |
| --- | --- | --- |
| Image size | 48x48 | Native FER2013 format - no resizing needed |
| Gauss kernel | (3, 3) | Preserves edges at small resolution |
| Gauss sigma | 0.8 | Light denoising, no feature blur |
| CLAHE clipLimit | 2.0 | Limits noise amplification |
| CLAHE tileGridSize | (4, 4) | 16 local regions on 48×48 image |
| Mean (train) | 0.5070 | Computed on train set only — no leakage |
| Std (train) | 0.2553 | Computed on train set only — no leakage |

### Source files

- [src/preprocessing.py](src/preprocessing.py) — `apply_clahe()`, `denoise_gaussian()`, `verify_dataset()`
- [src/dataset.py](src/dataset.py) — `FERDataset` class (supports CSV and folder formats, computes class weights)

---

## Week 3 — Model Design (current)

Deliverable CR3: model description + global pipeline diagram.

### What is already in place

The codebase is structured to support model training without breaking Week 2 work:

- `FERDataset.__getitem__` returns `(PIL.Image, label)` — ready to wrap with any torchvision `transforms.Compose`
- `FERDataset.get_class_weights()` returns balanced weights — plug directly into `torch.nn.CrossEntropyLoss(weight=...)`
- `configs/config.yaml` already declares three candidate architectures:
  - `baseline_cnn` — custom lightweight CNN (to be implemented)
  - `resnet50` — transfer learning from ImageNet
  - `efficientnet_b0` — transfer learning, better accuracy/size tradeoff

### What needs to be added this week

1. **`src/model.py`** — implement the chosen architecture(s):
   - Baseline CNN: 3–4 conv blocks → GlobalAvgPool → FC(7)
   - Transfer learning variant: adapt pretrained backbone (1-channel input, 7-class head)

2. **`src/train.py`** — training loop:
   - `DataLoader` with `FERDataset` + preprocessing transforms
   - `CrossEntropyLoss` with class weights
   - Optimizer (Adam, lr=0.001), scheduler (ReduceLROnPlateau or CosineAnnealing)
   - Early stopping (patience=10 from config)
   - Checkpoint saving to `checkpoints/`

3. **`requirements.txt`** — add PyTorch and torchvision (currently missing):

   ```text
   torch>=2.0.0
   torchvision>=0.15.0
   ```

4. **Data augmentation** — apply during training only (already configured in `config.yaml`):
   - Horizontal flip (p=0.5), rotation ±10°, brightness/contrast jitter, random crop zoom [0.9, 1.1]

### Recommended global pipeline for CR3

```text
FER2013 dataset
      │
      ▼
FERDataset (src/dataset.py)
      │  CSV or folder auto-detection
      │  train / val / test splits
      ▼
transforms.Compose (training)          transforms.Compose (val/test)
  Gaussian denoise                       Gaussian denoise
  CLAHE                                  CLAHE
  Random horizontal flip                 Normalize(0.5070, 0.2553)
  Random rotation ±10°
  Color jitter
  ToTensor
  Normalize(0.5070, 0.2553)
      │                                        │
      └──────────────┬──────────────────────────┘
                     ▼
              Model (src/model.py)
              Baseline CNN  or  ResNet50  or  EfficientNet-B0
              Input: (B, 1, 48, 48)   Output: (B, 7)
                     │
                     ▼
              CrossEntropyLoss + class weights
                     │
                     ▼
              Adam optimizer + LR scheduler + early stopping
                     │
                     ▼
              Evaluation: accuracy, confusion matrix, per-class F1
```

---

## Project Structure

```text
fer_emotions/
├── configs/
│   └── config.yaml             ← all hyperparameters (data, training, augmentation, model)
├── data/
│   └── fer2013/archive/
│       ├── train/              ← 28,709 images, 7 class folders
│       └── test/               ← 7,178 images, 7 class folders
├── docs/
│   ├── Compte-Rendu-1.pdf      ← CR1: problem statement, objectives, dataset
│   └── Compte-Rendu-2.pdf      ← CR2: preprocessing report
├── notebooks/
│   └── cr2_preprocessing.ipynb ← preprocessing illustrations (CR2 deliverable)
├── results/
│   ├── cr2_cleaning.png
│   ├── cr2_resizing.png
│   ├── cr2_denoising.png
│   ├── cr2_clahe.png
│   ├── cr2_full_pipeline.png
│   └── cr2_normalization.png
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        ← apply_clahe, denoise_gaussian, verify_dataset
│   ├── dataset.py              ← FERDataset (CSV + folder, class weights)
│   ├── model.py                ← (Week 3) CNN / ResNet / EfficientNet
│   └── train.py                ← (Week 3) training loop
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Verify dataset integrity (Week 2)

```bash
python src/preprocessing.py --verify data/fer2013/archive
```

### 3. Reproduce CR2 illustrations

```bash
jupyter notebook notebooks/cr2_preprocessing.ipynb
```

Run all cells in order. Figures are saved to `results/`.

### 4. Train a model (Week 3 — coming)

```bash
python src/train.py --config configs/config.yaml --model baseline_cnn
```

---

## Dependencies

```text
opencv-python>=4.8.0    — CLAHE, Gaussian blur
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0     — class weight computation
Pillow>=9.5.0
tqdm>=4.65.0
pyyaml>=6.0
jupyter>=1.0.0
torch>=2.0.0            — (Week 3)
torchvision>=0.15.0     — (Week 3)
```
