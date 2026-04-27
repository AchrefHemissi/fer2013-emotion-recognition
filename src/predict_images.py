"""
Predict emotion on any folder of face images.

Usage:
    python src/predict_images.py --folder data/new_images
    python src/predict_images.py --folder data/new_images --model deep_cnn
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from model import BaselineCNN, DeepCNN, TransferModel
from transforms import GaussianDenoise, CLAHE

CLASS_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

VAL_TFM = transforms.Compose([
    GaussianDenoise(),
    CLAHE(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.563], std=[0.2627]),
])

MODEL_MAP = {
    'baseline_cnn'    : (BaselineCNN,  'baseline_cnn_best.pth',     {}),
    'deep_cnn'        : (DeepCNN,      'deep_cnn_best.pth',         {}),
    'efficientnet_b0' : (TransferModel,'efficientnet_b0_best.pth',  {'backbone': 'efficientnet_b0', 'pretrained': False}),
}


def load_model(model_name: str, device):
    cls, ckpt_file, kwargs = MODEL_MAP[model_name]
    model = cls(num_classes=7, **kwargs)
    ckpt  = torch.load(ROOT / 'checkpoints' / ckpt_file, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()
    return model


def preprocess_image(img_path: Path):
    """Load any image, convert to grayscale, resize to 48x48, return PIL Image."""
    from PIL import Image
    img = Image.open(img_path).convert('L').resize((48, 48), Image.LANCZOS)
    return img


def predict_one(img_pil, model, device):
    tensor = VAL_TFM(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze().cpu().numpy()
    pred_idx = int(probs.argmax())
    return pred_idx, CLASS_NAMES[pred_idx], probs


def run(folder: str, model_name: str = 'deep_cnn'):
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model   = load_model(model_name, device)
    folder  = Path(folder)

    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    images = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in extensions]

    if not images:
        print(f'Aucune image trouvee dans {folder}')
        return

    n      = len(images)
    n_cols = min(n, 5)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    axes = axes.flatten()

    print(f'Modele : {model_name}  |  Device : {device}')
    print(f'{"Fichier":30s}  {"Prediction":12s}  {"Confiance":>10}')
    print('-' * 58)

    for ax_idx, img_path in enumerate(images):
        try:
            img_pil = preprocess_image(img_path)
        except Exception as e:
            print(f'[ERREUR] {img_path.name}: {e}')
            axes[ax_idx].axis('off')
            continue

        pred_idx, pred_name, probs = predict_one(img_pil, model, device)
        conf = probs[pred_idx]

        print(f'{img_path.name:30s}  {pred_name:12s}  {conf*100:>9.1f}%')

        ax = axes[ax_idx]
        ax.imshow(np.array(img_pil), cmap='gray')
        ax.set_title(f'{pred_name}\n({conf*100:.1f}%)', fontweight='bold', fontsize=11)
        ax.axis('off')

        # Probability bar on the side via inset
        sub = ax.inset_axes([0.0, -0.35, 1.0, 0.30])
        colors = ['#2ecc71' if i == pred_idx else '#bdc3c7' for i in range(7)]
        y_pos = np.arange(len(CLASS_NAMES))
        sub.barh(y_pos, probs, color=colors, height=0.7)
        sub.set_yticks(y_pos)
        sub.set_yticklabels(CLASS_NAMES)
        sub.set_xlim(0, 1)
        sub.tick_params(labelsize=6)
        sub.set_xlabel('prob', fontsize=6)

    for ax_idx in range(len(images), len(axes)):
        axes[ax_idx].axis('off')

    plt.suptitle(f'Predictions — {model_name}  |  {folder.name}',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    out_path = ROOT / 'results' / f'cr5_new_images_{folder.name}.png'
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.show()
    print(f'\nFigure sauvegardee : {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', default='data/new_images', help='Dossier contenant les images')
    parser.add_argument('--model',  default='deep_cnn',
                        choices=list(MODEL_MAP), help='Modele a utiliser')
    args = parser.parse_args()
    run(args.folder, args.model)
