"""Build cr4_training.ipynb from scratch."""
import json
from pathlib import Path

def md(id_, src):
    return {"cell_type": "markdown", "id": id_, "metadata": {}, "source": [src]}

def code(id_, src):
    return {"cell_type": "code", "execution_count": None, "id": id_,
            "metadata": {}, "outputs": [], "source": [src]}

cells = [

md("c00", "# CR4 — Comparaison des 3 Modèles FER2013\n**baseline_cnn | deep_cnn | efficientnet_b0**"),

code("c01", """\
import sys, os, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import yaml
from sklearn.metrics import confusion_matrix, classification_report
from tqdm.auto import tqdm

project_root = Path.cwd().parent
os.chdir(project_root)
if str(project_root / 'src') not in sys.path:
    sys.path.insert(0, str(project_root / 'src'))

from dataset import FERDataset
from model import build_model
from train import build_transforms, load_checkpoint

plt.rcParams.update({'figure.dpi': 120})
RESULTS_DIR = project_root / 'results'
RESULTS_DIR.mkdir(exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device : {device}')
print(f'Root   : {project_root}')
"""),

code("c02", """\
with open('configs/config.yaml') as f:
    cfg = yaml.safe_load(f)

CLASS_NAMES = cfg['data']['class_names']
BATCH_SIZE  = cfg['training']['batch_size']
NUM_WORKERS = 0
MODELS = ['baseline_cnn', 'deep_cnn', 'efficientnet_b0']
COLORS = {'baseline_cnn': '#4C72B0', 'deep_cnn': '#DD8452', 'efficientnet_b0': '#55A868'}
LABELS = {'baseline_cnn': 'Baseline CNN', 'deep_cnn': 'Deep CNN', 'efficientnet_b0': 'EfficientNet-B0'}
print('MODELS :', MODELS)
print('Classes:', CLASS_NAMES)
"""),

md("c03", "## 1. Courbes d'apprentissage"),

code("c04", """\
# Charger les historiques
histories = {}
for m in MODELS:
    path = project_root / 'logs' / f'{m}_history.json'
    if path.exists():
        with open(path) as f:
            histories[m] = json.load(f)
        print(f"  {LABELS[m]:<20} best={max(histories[m]['val_acc']):.4f}  ({len(histories[m]['train_loss'])} epochs)")
    else:
        print(f"  {LABELS[m]:<20} pas encore entraine")
"""),

code("c05", """\
# Loss + Accuracy curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Courbes d'apprentissage — 3 modeles", fontsize=13, fontweight='bold')
for m, hist in histories.items():
    ep = list(range(1, len(hist['train_loss']) + 1))
    c  = COLORS[m]
    axes[0].plot(ep, hist['train_loss'], color=c, lw=1.2, alpha=0.4)
    axes[0].plot(ep, hist['val_loss'],   color=c, lw=2, label=LABELS[m])
    axes[1].plot(ep, [a*100 for a in hist['train_acc']], color=c, lw=1.2, alpha=0.4)
    axes[1].plot(ep, [a*100 for a in hist['val_acc']],   color=c, lw=2,
                 label=f"{LABELS[m]} ({max(hist['val_acc'])*100:.1f}%)")
axes[0].set_title('Loss'); axes[0].set_xlabel('Epoch'); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].axhline(70, color='red', ls='--', lw=1, label='70%')
axes[1].set_title('Accuracy (%)'); axes[1].set_xlabel('Epoch'); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'cr4_learning_curves.png', bbox_inches='tight')
plt.show()
"""),

code("c06", """\
# Learning rate
fig, ax = plt.subplots(figsize=(12, 3))
for m, hist in histories.items():
    ax.plot(range(1, len(hist['lr'])+1), hist['lr'], color=COLORS[m], lw=2, label=LABELS[m])
ax.set_title('Learning Rate'); ax.set_xlabel('Epoch'); ax.set_yscale('log')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'cr4_learning_rate.png', bbox_inches='tight')
plt.show()
"""),

md("c07", "## 2. Évaluation sur le jeu de test"),

code("c08", """\
_, val_tfm = build_transforms(cfg)
test_ds = FERDataset(cfg['data']['root'], split='test', transform=val_tfm)
test_loader = torch.utils.data.DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
print(f'Test : {len(test_ds):,} images')

results = {}
for m in MODELS:
    ckpt = project_root / 'checkpoints' / f'{m}_best.pth'
    if not ckpt.exists():
        print(f'  {LABELS[m]} : checkpoint introuvable'); continue
    cfg['model']['name'] = m
    model = build_model(cfg).to(device)
    load_checkpoint(ckpt, model)
    model.eval()
    preds, lbls = [], []
    with torch.no_grad():
        for imgs, lb in tqdm(test_loader, desc=LABELS[m], leave=False):
            preds.extend(model(imgs.to(device)).argmax(1).cpu().numpy())
            lbls.extend(lb.numpy())
    preds, lbls = np.array(preds), np.array(lbls)
    acc    = (preds == lbls).mean()
    report = classification_report(lbls, preds, target_names=CLASS_NAMES, output_dict=True)
    results[m] = {'preds': preds, 'labels': lbls, 'acc': acc, 'report': report}
    print(f'  {LABELS[m]:<20} test acc = {acc*100:.2f}%')
"""),

code("c09", """\
# Tableau comparatif
rows = []
for m, r in results.items():
    rep = r['report']
    rows.append({
        'Modele':       LABELS[m],
        'Test Acc (%)': round(r['acc']*100, 2),
        'Macro F1':     round(rep['macro avg']['f1-score'], 4),
        'Weighted F1':  round(rep['weighted avg']['f1-score'], 4),
        'Best class':   max(CLASS_NAMES, key=lambda c: rep[c]['f1-score']),
        'Worst class':  min(CLASS_NAMES, key=lambda c: rep[c]['f1-score']),
    })
df_cmp = pd.DataFrame(rows).set_index('Modele')
display(df_cmp.style.background_gradient(cmap='RdYlGn', subset=['Test Acc (%)','Macro F1']))
"""),

md("c10", "## 3. Matrice de confusion"),

code("c11", """\
for m, r in results.items():
    cm      = confusion_matrix(r['labels'], r['preds'])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Matrice de Confusion — {LABELS[m]}  (Test={r['acc']*100:.2f}%)",
                 fontsize=12, fontweight='bold')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0])
    axes[0].set_title('Absolues'); axes[0].tick_params(axis='x', rotation=30)
    axes[0].set_xlabel('Predit'); axes[0].set_ylabel('Reel')
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', vmin=0, vmax=1,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1])
    axes[1].set_title('Normalisee'); axes[1].tick_params(axis='x', rotation=30)
    axes[1].set_xlabel('Predit'); axes[1].set_ylabel('Reel')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'cr4_confusion_{m}.png', bbox_inches='tight')
    plt.show()
"""),

md("c12", "## 4. F1-score par classe"),

code("c13", """\
# F1 comparaison
fig, ax = plt.subplots(figsize=(13, 5))
x, width = np.arange(len(CLASS_NAMES)), 0.25
for i, (m, r) in enumerate(results.items()):
    f1_vals = [r['report'][c]['f1-score'] for c in CLASS_NAMES]
    ax.bar(x + i*width, f1_vals, width, label=LABELS[m], color=COLORS[m], alpha=0.85)
ax.set_xticks(x + width); ax.set_xticklabels(CLASS_NAMES, rotation=15)
ax.set_ylabel('F1-score'); ax.set_title('F1-score par classe — 3 modeles')
ax.set_ylim(0, 1.05); ax.legend(); ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'cr4_f1_comparison.png', bbox_inches='tight')
plt.show()

# Classification reports
for m, r in results.items():
    print(f"\\n{'='*52}\\n  {LABELS[m]}\\n{'='*52}")
    print(classification_report(r['labels'], r['preds'], target_names=CLASS_NAMES))
"""),

md("c14", "## 5. Résumé final"),

code("c15", """\
print('\\n' + '='*58)
print('  RESUME CR4 — COMPARAISON 3 MODELES')
print('='*58)
print(f"  {'Modele':<22} {'Test Acc':>10} {'Macro F1':>10} {'Weighted F1':>12}")
print('-'*58)
for m, r in results.items():
    rep = r['report']
    print(f"  {LABELS[m]:<22} {r['acc']*100:>9.2f}%"
          f" {rep['macro avg']['f1-score']:>10.4f}"
          f" {rep['weighted avg']['f1-score']:>12.4f}")
print('='*58)
best_m = max(results, key=lambda m: results[m]['acc'])
print(f"\\n  Meilleur : {LABELS[best_m]} ({results[best_m]['acc']*100:.2f}%)")
print(f"  70%      : {'ATTEINT' if results[best_m]['acc']>=0.70 else 'Non atteint'}")
"""),

code("c16", """\
# Figure resume
n = len(results)
fig, axes = plt.subplots(2, n, figsize=(6*n, 10))
fig.suptitle('CR4 — Resume Comparatif FER2013', fontsize=14, fontweight='bold')
for col, (m, r) in enumerate(results.items()):
    hist = histories.get(m, {})
    rep  = r['report']
    ax = axes[0, col]
    if hist:
        ep = list(range(1, len(hist['val_acc']) + 1))
        ax.plot(ep, [a*100 for a in hist['train_acc']], lw=1.5, alpha=0.4, color=COLORS[m])
        ax.plot(ep, [a*100 for a in hist['val_acc']],   lw=2,   color=COLORS[m],
                label=f"best {max(hist['val_acc'])*100:.1f}%")
        ax.axhline(70, color='red', ls='--', lw=1)
    ax.set_title(LABELS[m]); ax.set_xlabel('Epoch'); ax.set_ylabel('Acc (%)')
    ax.legend(); ax.grid(alpha=0.3)
    ax = axes[1, col]
    f1_vals   = [rep[c]['f1-score'] for c in CLASS_NAMES]
    bar_colors = ['#2ecc71' if v >= 0.65 else '#e74c3c' for v in f1_vals]
    ax.barh(CLASS_NAMES, f1_vals, color=bar_colors, alpha=0.85)
    ax.axvline(rep['macro avg']['f1-score'], color='navy', ls='--', lw=1.5,
               label=f"Macro F1={rep['macro avg']['f1-score']:.3f}")
    ax.set_title(f"F1 — acc={r['acc']*100:.2f}%")
    ax.set_xlim(0, 1); ax.legend(); ax.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'cr4_summary.png', bbox_inches='tight', dpi=150)
plt.show()
"""),

]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.9.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out = Path(__file__).parent / "cr4_training.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"Created: {out}")
print(f"Cells  : {len(cells)}")
