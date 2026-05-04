"""
Streamlit app — FER2013 Facial Emotion Recognition
INSAT GL4 · Image Processing Project · 2026

Run:
    streamlit run app.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from model import DeepCNN
from transforms import GaussianDenoise, CLAHE
from preprocessing import apply_clahe, denoise_gaussian

# ── Constants ──────────────────────────────────────────────────────────────

CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

EMOTION_EMOJI = {
    "Angry": "😠", "Disgust": "🤢", "Fear": "😨",
    "Happy": "😊", "Sad":    "😢", "Surprise": "😲", "Neutral": "😐",
}
EMOTION_COLOR = {
    "Angry": "#e74c3c", "Disgust": "#e67e22", "Fear": "#f39c12",
    "Happy": "#2ecc71", "Sad":    "#3498db", "Surprise": "#9b59b6",
    "Neutral": "#95a5a6",
}

VAL_TFM = transforms.Compose([
    GaussianDenoise(),
    CLAHE(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.563], std=[0.2627]),
])

CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ── Model loading ──────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = ROOT / "checkpoints" / "deep_cnn_best.pth"
    model = DeepCNN(num_classes=7)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, device

# ── Inference ──────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(img_pil: Image.Image, model, device) -> np.ndarray:
    tensor = VAL_TFM(img_pil).unsqueeze(0).to(device)
    return torch.softmax(model(tensor), dim=1).squeeze().cpu().numpy()

# ── Image preparation ──────────────────────────────────────────────────────

def to_grayscale_48(img_pil: Image.Image) -> Image.Image:
    return img_pil.convert("L").resize((48, 48), Image.LANCZOS)


def detect_face(img_pil: Image.Image):
    """Return (face PIL 48x48, bbox) or (None, None) if no face found."""
    gray = np.array(img_pil.convert("L"))
    faces = CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None, None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])  # largest face
    crop = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
    return Image.fromarray(crop, mode="L"), (x, y, w, h)

# ── Visualizations ─────────────────────────────────────────────────────────

def _prob_chart(probs: np.ndarray) -> plt.Figure:
    pred_idx = int(probs.argmax())
    colors = [
        EMOTION_COLOR[c] if i == pred_idx else "#dfe6e9"
        for i, c in enumerate(CLASS_NAMES)
    ]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    bars = ax.barh(CLASS_NAMES, probs * 100, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlim(0, 115)
    ax.set_xlabel("Confidence (%)", fontsize=9)
    ax.axvline(50, color="#b2bec3", linestyle="--", alpha=0.6, linewidth=1)
    for bar, p in zip(bars, probs):
        ax.text(p * 100 + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{p*100:.1f}%", va="center", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def _pipeline_chart(img_pil: Image.Image) -> plt.Figure:
    raw = np.array(img_pil.convert("L").resize((48, 48), Image.LANCZOS))
    denoised = denoise_gaussian(raw)
    enhanced = apply_clahe(denoised)
    normalized = (enhanced / 255.0 - 0.563) / 0.2627

    steps = [
        ("1. Raw (48×48)", raw, "gray", None),
        ("2. Gaussian denoising", denoised, "gray", None),
        ("3. CLAHE", enhanced, "gray", None),
        ("4. Normalized", normalized, "RdBu", True),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.8))
    for ax, (label, data, cmap, symmetric) in zip(axes, steps):
        kwargs = dict(cmap=cmap)
        if symmetric:
            vmax = max(abs(data.min()), abs(data.max()))
            kwargs.update(vmin=-vmax, vmax=vmax)
        ax.imshow(data, **kwargs)
        ax.set_title(label, fontsize=8.5, fontweight="bold")
        ax.axis("off")
    fig.tight_layout(pad=0.5)
    return fig

# ── Prediction block ───────────────────────────────────────────────────────

def prediction_block(img_pil: Image.Image, model, device,
                     auto_detect: bool, show_pipeline: bool):

    if auto_detect:
        face_img, _ = detect_face(img_pil)
        if face_img is None:
            st.info("No face detected — using full image resized to 48×48.")
            face_img = to_grayscale_48(img_pil)
        else:
            st.success("Face auto-detected and cropped.")
    else:
        face_img = to_grayscale_48(img_pil)

    probs     = predict(face_img, model, device)
    pred_idx  = int(probs.argmax())
    emotion   = CLASS_NAMES[pred_idx]
    conf      = probs[pred_idx]
    top3      = np.argsort(probs)[::-1][:3]

    col_img, col_result = st.columns([1, 2], gap="large")

    with col_img:
        st.image(face_img, caption="Model input (48×48)", width=180)

    with col_result:
        color = EMOTION_COLOR[emotion]
        emoji = EMOTION_EMOJI[emotion]
        st.markdown(
            f"<h2 style='color:{color}; margin:0'>{emoji} {emotion}</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(f"**Confidence:** {conf*100:.1f}%")

        if conf < 0.45:
            st.warning("⚠️ Low confidence — try a clearer, well-lit, front-facing photo.")

        st.markdown("**Top 3:**")
        for i in top3:
            bar = "█" * int(probs[i] * 20)
            st.markdown(
                f"&nbsp;&nbsp;{EMOTION_EMOJI[CLASS_NAMES[i]]} **{CLASS_NAMES[i]}** "
                f"— {probs[i]*100:.1f}% `{bar}`"
            )

    st.pyplot(_prob_chart(probs), use_container_width=True)

    if show_pipeline:
        st.subheader("Preprocessing pipeline")
        st.pyplot(_pipeline_chart(img_pil), use_container_width=True)

# ── App ────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FER2013 — Emotion Recognition",
    page_icon="😊",
    layout="wide",
)

st.title("😊 FER2013 — Facial Emotion Recognition")
st.caption("INSAT GL4 · Image Processing · Deep CNN · 65.2% test accuracy")

with st.spinner("Loading model…"):
    model, device = load_model()

with st.sidebar:
    st.header("⚙️ Settings")
    auto_detect   = st.toggle("Auto-detect face", value=True)
    show_pipeline = st.toggle("Show preprocessing steps", value=False)

    st.divider()
    st.markdown("**Model — Deep CNN**")
    st.markdown(
        "- 1.33M parameters\n"
        "- 65.2% test accuracy\n"
        "- Train/val gap: +4.83%\n"
        "- 7 emotions: Angry, Disgust,\n  Fear, Happy, Sad, Surprise, Neutral"
    )
    st.divider()
    st.caption(
        "Best results on front-facing, well-lit face crops. "
        "Model trained on FER2013 (48×48 grayscale)."
    )
    st.caption(f"Device: `{device}`")

tab_upload, tab_webcam = st.tabs(["📁 Upload Image", "📷 Capture Photo"])

with tab_upload:
    st.subheader("Upload a face image")
    uploaded = st.file_uploader(
        "Supported formats: JPG, PNG, BMP, WEBP",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )
    if uploaded:
        img_pil = Image.open(uploaded)
        st.image(img_pil, caption="Uploaded image", width=280)
        st.divider()
        prediction_block(img_pil, model, device, auto_detect, show_pipeline)

with tab_webcam:
    st.subheader("Capture a photo with your webcam")
    st.caption("Click **Take photo** then wait for the prediction to appear below.")
    captured = st.camera_input("Take photo")
    if captured:
        img_pil = Image.open(captured)
        st.divider()
        prediction_block(img_pil, model, device, auto_detect, show_pipeline)
