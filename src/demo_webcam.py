"""
Real-time emotion recognition from webcam.

Requires: OpenCV with face detection (Haar cascade).

Usage:
    python src/demo_webcam.py
    python src/demo_webcam.py --model deep_cnn --cam 0

Controls:
    Q / ESC  : quit
    S        : save current frame to results/
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from model import BaselineCNN, DeepCNN, TransferModel
from transforms import GaussianDenoise, CLAHE

CLASS_NAMES  = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
BAR_COLORS   = [
    (0,   0,  220),   # Angry      — red-ish (BGR)
    (0,  140, 255),   # Disgust    — orange
    (0,  200, 255),   # Fear       — yellow
    (0,  200,   0),   # Happy      — green
    (220, 100,  0),   # Sad        — blue
    (200,   0, 200),  # Surprise   — purple
    (160, 160, 160),  # Neutral    — gray
]

VAL_TFM = transforms.Compose([
    GaussianDenoise(),
    CLAHE(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.563], std=[0.2627]),
])

MODEL_MAP = {
    'baseline_cnn'    : (BaselineCNN,   'baseline_cnn_best.pth',    {}),
    'deep_cnn'        : (DeepCNN,       'deep_cnn_best.pth',        {}),
    'efficientnet_b0' : (TransferModel, 'efficientnet_b0_best.pth', {'backbone': 'efficientnet_b0', 'pretrained': False}),
}

CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'


def load_model(model_name, device):
    cls, ckpt_file, kwargs = MODEL_MAP[model_name]
    model = cls(num_classes=7, **kwargs)
    ckpt  = torch.load(ROOT / 'checkpoints' / ckpt_file, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_face(face_gray_48, model, device):
    """face_gray_48: np.ndarray (48,48) uint8 → (pred_idx, probs)"""
    from PIL import Image
    pil = Image.fromarray(face_gray_48, mode='L')
    tensor = VAL_TFM(pil).unsqueeze(0).to(device)
    probs  = torch.softmax(model(tensor), dim=1).squeeze().cpu().numpy()
    return int(probs.argmax()), probs


def draw_overlay(frame, x, y, w, h, pred_idx, probs):
    """Draw bounding box, label, and probability bars on frame."""
    label = CLASS_NAMES[pred_idx]
    conf  = probs[pred_idx]
    color = BAR_COLORS[pred_idx]

    # Bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Label background
    (tw, th), _ = cv2.getTextSize(f'{label} {conf*100:.0f}%', cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x, y - th - 10), (x + tw + 6, y), color, -1)
    cv2.putText(frame, f'{label} {conf*100:.0f}%',
                (x + 3, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Probability bars (right panel)
    bar_x = x + w + 10
    bar_w = 120
    bar_h = 14
    gap   = 4
    for i, (cls_name, p) in enumerate(zip(CLASS_NAMES, probs)):
        by = y + i * (bar_h + gap)
        filled = int(p * bar_w)
        cv2.rectangle(frame, (bar_x, by), (bar_x + bar_w, by + bar_h), (60, 60, 60), -1)
        cv2.rectangle(frame, (bar_x, by), (bar_x + filled, by + bar_h), BAR_COLORS[i], -1)
        cv2.putText(frame, f'{cls_name[:3]} {p*100:.0f}%',
                    (bar_x + bar_w + 4, by + bar_h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)


def run(model_name: str = 'deep_cnn', cam_idx: int = 0):
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model    = load_model(model_name, device)
    detector = cv2.CascadeClassifier(CASCADE_PATH)

    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f'Impossible d\'ouvrir la camera {cam_idx}')
        return

    print(f'Modele : {model_name}  |  Device : {device}')
    print('Appuyez sur Q/ESC pour quitter, S pour sauvegarder une capture.')

    fps_time  = time.time()
    fps_count = 0
    fps_val   = 0.0
    save_idx  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            face_crop = gray[y:y+h, x:x+w]
            face_48   = cv2.resize(face_crop, (48, 48))
            pred_idx, probs = predict_face(face_48, model, device)
            draw_overlay(frame, x, y, w, h, pred_idx, probs)

        # FPS counter
        fps_count += 1
        if time.time() - fps_time >= 1.0:
            fps_val   = fps_count / (time.time() - fps_time)
            fps_count = 0
            fps_time  = time.time()

        cv2.putText(frame, f'FPS: {fps_val:.1f}  |  {model_name}',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(frame, 'Q=quit  S=save',
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow('FER — Emotion Recognition', frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('s'):
            out = ROOT / 'results' / f'webcam_capture_{save_idx:03d}.png'
            cv2.imwrite(str(out), frame)
            print(f'Capture sauvegardee : {out}')
            save_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='deep_cnn', choices=list(MODEL_MAP))
    parser.add_argument('--cam',   default=0, type=int, help='Index camera (0=defaut)')
    args = parser.parse_args()
    run(args.model, args.cam)
