# FER2013 — Facial Emotion Recognition

INSAT GL4 — Image Processing Project (2026)

Group: Rayen Chemlali, Mohamed Dhia Medini, Khalil Ghimaji, Mohamed Achref Hemissi

---

## Project Roadmap

| Week | Date       | Topic                                                    | Deliverable        | Status |
| ---- | ---------- | -------------------------------------------------------- | ------------------ | ------ |
| 1    | 30/03/2026 | Project framing — problem, objectives, dataset selection | CR1 (PDF)          | Done   |
| 2    | 06/04/2026 | Image preprocessing pipeline                             | CR2 (notebook+PDF) | Done   |
| 3    | 13/04/2026 | Model design — architecture choice, global pipeline      | CR3 (notebook+PDF) | Done   |
| 4    | 20/04/2026 | Model training — 3 architectures benchmark               | CR4 (notebook+PDF) | Done   |

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
This is compensated during training via class weights in `CrossEntropyLoss`.

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
- [src/dataset.py](src/dataset.py) — `FERDataset` class (folder format, class weights)

---

## Week 3 — Model Design

Full report: [docs/Compte-Rendu-3.pdf](docs/Compte-Rendu-3.pdf)

Notebook: [notebooks/cr3_model_design.ipynb](notebooks/cr3_model_design.ipynb)

Three architectures implemented in [src/model.py](src/model.py):

| Model | Params | Description |
| --- | --- | --- |
| `baseline_cnn` | ~390K | 4× ConvBlock (Conv→BN→ReLU→MaxPool), GAP, FC(7) |
| `deep_cnn` | ~1.5M | 4× DoubleConvBlock with SE attention, GAP, FC(256→512→7) |
| `efficientnet_b0` | ~4M | Pretrained EfficientNet-B0, 1-channel stem, fine-tuned head |

---

## Week 4 — Model Training

Full report: [docs/CR4.tex](docs/CR4.tex)

Notebook: [notebooks/cr4_training.ipynb](notebooks/cr4_training.ipynb)

### Training configuration

| Parameter | Value |
| --- | --- |
| Optimizer | AdamW (lr=0.001, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR (T_max=100, η_min=1e-6) |
| Loss | CrossEntropyLoss + class weights + label smoothing (0.1) |
| Batch size | 128 |
| Max epochs | 100 |
| Early stopping | patience=15 |
| Augmentation | HorizontalFlip, Rotation±10°, ColorJitter |
| Device | CUDA (AMP + cudnn.benchmark) |

### Results

| Model | Best Val Acc | Epochs | Train Acc @ Best | Overfit Gap |
| --- | --- | --- | --- | --- |
| Baseline CNN | 62.47% | 32/46 | 58.86% | −3.61% |
| **Deep CNN** | **68.32%** | **98/100** | **73.15%** | **+4.83%** |
| EfficientNet-B0 | 66.94% | 97/100 | 97.51% | +30.57% |

**Best model: Deep CNN** — reaches 68.32% validation accuracy with healthy generalization.

EfficientNet-B0 shows severe overfitting (train 97.5% vs val 66.9%) — ImageNet features are poorly suited to 48×48 grayscale images without backbone freezing.

### Training pipeline

```bash
# Train any model (set model.name in configs/config.yaml)
python src/train.py

# Or override directly:
python src/train.py --model deep_cnn
# Available: baseline_cnn | deep_cnn | efficientnet_b0
```

Checkpoints saved to `checkpoints/<model>_best.pth`. Training history saved to `logs/<model>_history.json`.

---

## Project Structure

```text
fer_emotions/
├── configs/
│   └── config.yaml                   ← all hyperparameters
├── data/
│   └── fer2013/archive/
│       ├── train/                    ← 28,709 images, 7 class folders
│       └── test/                     ← 7,178 images, 7 class folders
├── checkpoints/
│   ├── baseline_cnn_best.pth
│   ├── deep_cnn_best.pth
│   └── efficientnet_b0_best.pth
├── docs/
│   ├── Compte-Rendu-1.pdf
│   ├── Compte-Rendu-2.pdf
│   ├── Compte-Rendu-3.pdf
│   └── CR4.tex                       ← CR4 LaTeX report
├── logs/
│   ├── baseline_cnn_history.json
│   ├── deep_cnn_history.json
│   └── efficientnet_b0_history.json
├── notebooks/
│   ├── cr2_preprocessing.ipynb
│   ├── cr3_model_design.ipynb
│   ├── cr4_training.ipynb            ← 3-model benchmark
│   └── build_cr4.py                  ← script to rebuild cr4 notebook
├── results/
│   ├── cr2_*.png
│   ├── cr3_*.png
│   └── cr4_*.png                     ← training & evaluation figures
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── model.py                      ← BaselineCNN, DeepCNN, TransferModel
│   └── train.py                      ← training loop (AMP, early stopping)
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

### 2. Verify dataset integrity

```bash
python src/preprocessing.py --verify data/fer2013/archive
```

### 3. Train a model

```bash
python src/train.py --model deep_cnn
```

### 4. Run the CR4 comparison notebook

```bash
jupyter notebook notebooks/cr4_training.ipynb
# Kernel → Restart & Run All
```

### 5. Compile the LaTeX report

```bash
cd docs
pdflatex CR4.tex && pdflatex CR4.tex
```

---

## Dependencies

```text
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
Pillow>=9.5.0
tqdm>=4.65.0
pyyaml>=6.0
jupyter>=1.0.0
torch>=2.0.0
torchvision>=0.15.0
```
