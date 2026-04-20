"""
Training pipeline for FER2013 — Week 4.

GPU optimisations activées automatiquement si CUDA est disponible :
  - Mixed precision (AMP) via torch.cuda.amp
  - cudnn.benchmark = True

Usage
-----
    python src/train.py
    python src/train.py --model resnet50
    python src/train.py --model efficientnet_b0
    python src/train.py --epochs 30 --lr 0.0005
    python src/train.py --resume
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import yaml

_src = Path(__file__).parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from dataset import FERDataset
from model import build_model
from transforms import AddGaussianNoise, CLAHE, GaussianDenoise


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def build_transforms(cfg: dict):
    aug = cfg["augmentation"]
    pre = cfg["preprocessing"]
    mean, std = pre["mean"], pre["std"]
    size = cfg["data"]["image_size"]

    use_custom = cfg.get("preprocessing", {}).get("use_custom_transforms", True)

    train_tfm = transforms.Compose([
        *([ GaussianDenoise(), CLAHE() ] if use_custom else []),
        transforms.RandomHorizontalFlip(p=aug["horizontal_flip"]),
        transforms.RandomRotation(degrees=aug["rotation_degrees"]),
        transforms.ColorJitter(
            brightness=aug["brightness"],
            contrast=aug["contrast"],
        ),
        transforms.RandomResizedCrop(
            size=size,
            scale=tuple(aug["zoom_scale"]),
        ),
        *([ AddGaussianNoise(std=aug.get("gaussian_noise_std", 15)) ] if use_custom else []),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std]),
    ])

    val_tfm = transforms.Compose([
        *([ GaussianDenoise(), CLAHE() ] if use_custom else []),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std]),
    ])

    return train_tfm, val_tfm


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def build_dataloaders(cfg: dict, train_tfm, val_tfm, num_workers: int = None):
    root = cfg["data"]["root"]
    batch = cfg["training"]["batch_size"]
    workers = num_workers if num_workers is not None else cfg["data"]["num_workers"]

    train_ds = FERDataset(root, split="train", transform=train_tfm)
    val_ds   = FERDataset(root, split="val",   transform=val_tfm)
    test_ds  = FERDataset(root, split="test",  transform=val_tfm)

    class_weights = torch.tensor(train_ds.get_class_weights(), dtype=torch.float32)

    loader_kwargs = dict(
        batch_size=batch,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(workers > 0),
    )

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, class_weights


# ---------------------------------------------------------------------------
# Epoch loop  (AMP-aware)
# ---------------------------------------------------------------------------

def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, desc=""):
    """
    Run one epoch. If optimizer is given → training mode. If scaler is given → AMP enabled.
    """
    training = optimizer is not None
    use_amp  = scaler is not None
    model.train() if training else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()

    with ctx:
        pbar = tqdm(loader, desc=desc, leave=False, ncols=95)
        for images, labels in pbar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                loss   = criterion(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += images.size(0)
            pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.4f}")

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: Path, model, optimizer=None, scheduler=None, scaler=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer  and "optimizer"  in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler  and "scheduler"  in ckpt: scheduler.load_state_dict(ckpt["scheduler"])
    if scaler     and "scaler"     in ckpt and ckpt["scaler"]: scaler.load_state_dict(ckpt["scaler"])
    return ckpt.get("epoch", 0), ckpt.get("best_val_acc", 0.0)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(cfg: dict, resume: bool = False, num_workers: int = None) -> dict:
    # ── Device & GPU setup ────────────────────────────────────────────────
    device_str = cfg["training"].get("device", "auto")
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_str == "auto"
        else torch.device(device_str)
    )

    use_amp = device.type == "cuda"
    if use_amp:
        torch.backends.cudnn.benchmark = True   # auto-tune convolution algorithms
        scaler = torch.amp.GradScaler("cuda")
    else:
        scaler = None

    print(f"\n{'='*62}")
    print(f"  FER2013 Training — {cfg['model']['name'].upper()}")
    print(f"  Device          : {device}" + (f"  [{torch.cuda.get_device_name(0)}]" if use_amp else ""))
    print(f"  Mixed precision : {'ENABLED (AMP)' if use_amp else 'disabled'}")
    print(f"  cudnn.benchmark : {'ENABLED' if use_amp else 'disabled'}")
    print(f"{'='*62}\n")

    # ── Data ─────────────────────────────────────────────────────────────
    train_tfm, val_tfm = build_transforms(cfg)
    train_loader, val_loader, test_loader, class_weights = build_dataloaders(
        cfg, train_tfm, val_tfm, num_workers=num_workers
    )
    print(f"Train : {len(train_loader.dataset):,} images")
    print(f"Val   : {len(val_loader.dataset):,} images")
    print(f"Test  : {len(test_loader.dataset):,} images\n")

    # ── Model ────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters : {total_params:,} total  |  {trainable:,} trainable\n")

    # ── Loss, optimizer, scheduler ───────────────────────────────────────
    label_smoothing = cfg["training"].get("label_smoothing", 0.1)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=label_smoothing,
    )
    lr        = cfg["training"]["learning_rate"]
    wd        = cfg["training"]["weight_decay"]
    opt_name   = cfg["training"].get("optimizer", "adamw").lower()
    sched_name = cfg["training"].get("scheduler", "cosine").lower()
    epochs     = cfg["training"]["epochs"]

    optimizer = (
        torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        if opt_name == "adamw"
        else torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    )

    if sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )

    # ── Paths ────────────────────────────────────────────────────────────
    model_name   = cfg["model"]["name"]
    ckpt_dir     = Path(cfg["paths"]["checkpoints"])
    log_dir      = Path(cfg["paths"]["logs"])
    best_ckpt    = ckpt_dir / f"{model_name}_best.pth"
    last_ckpt    = ckpt_dir / f"{model_name}_last.pth"
    history_path = log_dir  / f"{model_name}_history.json"
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── Resume ───────────────────────────────────────────────────────────
    start_epoch, best_val_acc = 0, 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    if resume and last_ckpt.exists():
        start_epoch, best_val_acc = load_checkpoint(
            last_ckpt, model, optimizer, scheduler, scaler
        )
        print(f"Resumed from epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)

    patience          = cfg["training"]["patience"]
    epochs_no_improve = 0

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler,
            desc=f"Epoch {epoch+1:03d}/{epochs} [train]",
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, device,
            desc=f"Epoch {epoch+1:03d}/{epochs} [val]  ",
        )

        current_lr = optimizer.param_groups[0]["lr"]
        if sched_name == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_acc)

        history["train_loss"].append(round(train_loss, 6))
        history["train_acc"].append(round(train_acc, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["val_acc"].append(round(val_acc, 6))
        history["lr"].append(current_lr)

        flag = "  ✓ best" if val_acc > best_val_acc else ""
        print(
            f"[{epoch+1:03d}/{epochs}]  "
            f"train loss={train_loss:.4f} acc={train_acc:.4f}  |  "
            f"val loss={val_loss:.4f} acc={val_acc:.4f}  |  "
            f"lr={current_lr:.2e}  ({time.time()-t0:.1f}s){flag}"
        )

        state = {
            "epoch":        epoch + 1,
            "model":        model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "scheduler":    scheduler.state_dict(),
            "scaler":       scaler.state_dict() if scaler else None,
            "best_val_acc": best_val_acc,
            "config":       cfg,
        }
        save_checkpoint(state, last_ckpt)

        if val_acc > best_val_acc:
            best_val_acc      = val_acc
            epochs_no_improve = 0
            save_checkpoint(state, best_ckpt)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping — no improvement for {patience} consecutive epochs.")
                break

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val accuracy : {best_val_acc:.4f}")
    print(f"Best checkpoint  : {best_ckpt}")
    print(f"History          : {history_path}\n")
    return history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train FER2013 emotion classifier")
    parser.add_argument("--config",  default="configs/config.yaml")
    parser.add_argument("--model",   default=None,
                        help="baseline_cnn | resnet50 | efficientnet_b0")
    parser.add_argument("--epochs",  type=int,   default=None)
    parser.add_argument("--lr",      type=float, default=None)
    parser.add_argument("--batch",   type=int,   default=None)
    parser.add_argument("--workers", type=int,   default=None)
    parser.add_argument("--resume",  action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.model:  cfg["model"]["name"]             = args.model
    if args.epochs: cfg["training"]["epochs"]        = args.epochs
    if args.lr:     cfg["training"]["learning_rate"] = args.lr
    if args.batch:  cfg["training"]["batch_size"]    = args.batch

    train(cfg, resume=args.resume, num_workers=args.workers)


if __name__ == "__main__":
    main()
