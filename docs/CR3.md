# Compte-Rendu 3 — Conception du Modèle

**Module :** Traitement d'Image  
**Filière :** GL4 — INSAT  
**Semaine :** 3 — 13/04/2026  
**Groupe :** Rayen Chemlali, Mohamed Dhia Medini, Khalil Ghimaji, Mohamed Achref Hemissi

---

## 1. Rappel du problème

L'objectif du projet est la **reconnaissance automatique des émotions faciales (FER)** : étant donné une image en niveaux de gris de résolution 48×48 pixels représentant un visage humain, prédire l'émotion parmi 7 classes : Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.

Le dataset utilisé est **FER2013** (Kaggle — `msambare/fer2013`) :

| Propriété | Valeur |
| --- | --- |
| Total images | 35 887 |
| Format | Niveaux de gris, 48×48 pixels |
| Ensemble train | 28 709 images |
| Ensemble test | 7 178 images |
| Classes | 7 émotions |

---

## 2. Choix de l'approche : Deep Learning vs ML classique

| Critère | ML classique (SVM, RF + HOG/LBP) | Deep Learning (CNN) |
| --- | --- | --- |
| Extraction de features | Manuelle (HOG, LBP, Gabor) | Apprise automatiquement |
| Précision sur FER2013 | 45–55 % | 60–88 % |
| Variation d'éclairage | Mal gérée (filtres fixes) | Gérée (CLAHE + features apprises) |
| Déséquilibre des classes | Difficile à ajuster | Sampler + loss pondérée |
| Passage à l'échelle | Non | Oui |

**Décision : Deep Learning.**  
FER2013 à 48×48 est suffisamment petit pour entraîner un CNN léger from scratch, et suffisamment grand (28 K images d'entraînement) pour bénéficier des représentations apprises. Les meilleures performances publiées sur ce dataset sont obtenues avec des CNN profonds (~70 % de précision), contre ~50 % pour les meilleures approches ML classiques.

---

## 3. Architectures du modèle

Trois architectures ont été conçues et comparées.

### 3.1 Baseline CNN (architecture personnalisée)

Architecture légère conçue spécifiquement pour des images 48×48 en niveaux de gris.

**Flux spatial :**

| Couche | Opération | Sortie |
| --- | --- | --- |
| Input | — | (B, 1, 48, 48) |
| Block 1 | Conv(1→32, 3×3) + BN + ReLU + MaxPool(2) | (B, 32, 24, 24) |
| Block 2 | Conv(32→64, 3×3) + BN + ReLU + MaxPool(2) | (B, 64, 12, 12) |
| Block 3 | Conv(64→128, 3×3) + BN + ReLU + MaxPool(2) | (B, 128, 6, 6) |
| Block 4 | Conv(128→256, 3×3) + BN + ReLU + MaxPool(2) | (B, 256, 3, 3) |
| GAP | GlobalAveragePool | (B, 256) |
| FC | Dropout(0.5) + Linear(256→7) | (B, 7) |

**Justification des choix de conception :**

- **Conv(3×3) + padding=1 :** conserve la résolution spatiale à chaque bloc avant le pooling. Un noyau 3×3 est le plus petit capable de capturer des relations spatiales entre pixels voisins.
- **BatchNorm après chaque Conv :** normalise les activations par batch, stabilise l'entraînement et permet des learning rates plus élevés. Indispensable car FER2013 contient des images issues de sources très variées (éclairage, contraste hétérogènes).
- **bias=False dans les Conv :** lorsque BatchNorm suit immédiatement, le biais de la convolution est annulé par le paramètre de décalage appris par BN (`beta`). Le supprimer évite une redondance et économise des paramètres.
- **MaxPool(2×2) :** divise la résolution par 2 à chaque bloc. Après 4 blocs, on passe de 48→24→12→6→3. Cela agrandit progressivement le champ récepteur.
- **GlobalAveragePool (GAP) :** remplace Flatten + FC large. Moyenne chaque carte de features en un scalaire → vecteur de 256 valeurs. Avantage : aucun risque d'overfitting lié aux positions spatiales, et invariant à la taille d'entrée.
- **Dropout(0.5) :** régularisation avant la couche finale. Force le modèle à ne pas dépendre d'un seul neurone. Particulièrement utile vu la taille réduite du dataset.

**Paramètres entraînables : 390 119**

### 3.2 ResNet-50 (Transfer Learning)

ResNet-50 pré-entraîné sur ImageNet, adapté pour FER2013.

**Adaptations nécessaires :**

1. **Entrée 1 canal :** la première couche Conv est remplacée par une Conv(1→64). Les poids pré-entraînés RGB sont sommés sur la dimension des canaux pour initialiser les poids grayscale : `W_gray = W_R + W_G + W_B`. Cette initialisation préserve les features ImageNet (équivalent à traiter l'image grayscale comme identique sur les 3 canaux).

2. **Suppression du MaxPool initial :** ResNet-50 standard applique stride=2 (Conv) puis MaxPool(2), ce qui réduirait 48×48 à 12×12 avant le premier bloc résiduel — trop agressif. Le MaxPool est remplacé par `nn.Identity()`.

**Flux spatial résultant :**

| Étape | Sortie |
| --- | --- |
| conv1 (stride=2) | (B, 64, 24, 24) |
| layer1 (pas de downsample) | (B, 256, 24, 24) |
| layer2 (stride=2) | (B, 512, 12, 12) |
| layer3 (stride=2) | (B, 1024, 6, 6) |
| layer4 (stride=2) | (B, 2048, 3, 3) |
| GAP | (B, 2048) |
| Dropout(0.5) + FC | (B, 7) |

**Paramètres totaux : 23 516 103**

### 3.3 EfficientNet-B0 (Transfer Learning)

EfficientNet-B0 pré-entraîné sur ImageNet, adapté de la même façon.

**Adaptation :** première Conv remplacée par Conv(1→32, stride=2), poids initialisés par somme des canaux RGB.

**Paramètres totaux : 4 015 939**

### 3.4 Comparaison des architectures

| Modèle | Paramètres | Entraîné depuis | Avantage principal |
| --- | --- | --- | --- |
| Baseline CNN | 390 119 | Scratch | Léger, interprétable, rapide |
| EfficientNet-B0 | 4 015 939 | ImageNet | Bon compromis taille/précision |
| ResNet-50 | 23 516 103 | ImageNet | Features riches, précision maximale |

---

## 5. Pipeline de transformations

### 5.1 Pipeline d'entraînement (avec augmentation)

```
GaussianDenoise(kernel=3×3, sigma=0.8)
CLAHE(clipLimit=2.0, tileGridSize=4×4)
RandomHorizontalFlip(p=0.5)
RandomRotation(degrees=±10°, fill=128)
ColorJitter(brightness=0.3, contrast=0.2)
RandomResizedCrop(size=48, scale=(0.9, 1.1), ratio=(1.0, 1.0))
ToTensor()
Normalize(mean=0.563, std=0.263)
```

### 5.2 Pipeline validation/test (sans augmentation)

```
GaussianDenoise(kernel=3×3, sigma=0.8)
CLAHE(clipLimit=2.0, tileGridSize=4×4)
ToTensor()
Normalize(mean=0.563, std=0.263)
```

### 5.3 Justification des augmentations

| Transformation | Justification | Paramètre |
| --- | --- | --- |
| RandomHorizontalFlip | Les visages sont symétriques, le miroir ne change pas l'émotion | p=0.5 |
| RandomRotation | Simule l'inclinaison naturelle de la tête | ±10° (conservateur) |
| fill=128 (gris moyen) | Remplace les coins noirs lors de la rotation — moins perturbateur que fill=0 | 128 |
| ColorJitter brightness | Simule les variations d'éclairage | ±30 % |
| ColorJitter contrast | Simule les différences de qualité d'image | ±20 % |
| RandomResizedCrop | Légère variation de zoom (±10 % de surface) | scale=(0.9, 1.1) |
| ratio=(1.0, 1.0) | Forçage de crops carrés — évite la distorsion des proportions du visage | — |

**Note :** `saturation` et `hue` dans ColorJitter sont désactivés (None) car les images sont en niveaux de gris.

---

## 6. Gestion du déséquilibre des classes

### 6.1 Analyse du déséquilibre

| Classe | Effectif | Poids équilibré |
| --- | --- | --- |
| Angry | 4 953 | 1.027 |
| Disgust | 547 | 9.407 |
| Fear | 5 121 | 1.001 |
| Happy | 8 989 | 0.568 |
| Sad | 6 077 | 0.849 |
| Surprise | 4 002 | 1.293 |
| Neutral | 6 198 | 0.826 |

Rapport maximal : Disgust / Happy = 1:16.

### 6.2 Stratégie retenue : WeightedRandomSampler uniquement

**Mécanisme :** à chaque epoch, les images sont tirées avec remplacement selon un poids proportionnel à l'inverse de la fréquence de leur classe. Disgust est tiré ~16× plus souvent que dans la distribution naturelle. Chaque batch de 64 images contient ainsi environ 9 images par classe.

**Pourquoi ne pas combiner avec une loss pondérée ?**

L'utilisation simultanée du sampler ET d'une `CrossEntropyLoss(weight=...)` produit une double correction :

```
Sampler seul   : Disgust apparaît 16/7 ≈ 2.3× plus souvent par batch
Loss pondérée  : gradient de Disgust multiplié par 9.4×
Combinaison    : 2.3 × 9.4 ≈ 21× — sur-correction sévère
```

Cela conduirait le modèle à se spécialiser excessivement sur Disgust au détriment des classes majoritaires. La **loss non pondérée est donc correcte** lorsque le sampler est actif, car les batches sont déjà équilibrés.

**Configuration retenue :**

```python
sampler   = WeightedRandomSampler(sample_weights, num_samples, replacement=True)
criterion = nn.CrossEntropyLoss()   # non pondérée
```

**Risque d'overfitting :** Disgust (547 images) est répété ~16× par epoch. L'augmentation des données (flip, rotation, jitter, crop) est essentielle pour introduire de la variété dans ces images répétées.

---

## 7. Configuration du DataLoader

| Paramètre | Valeur |
| --- | --- |
| Batch size | 64 |
| Train loader | 449 batches × 64 = 28 709 samples |
| Val/Test loader | 113 batches × 64 = 7 178 samples |
| Sampler (train) | WeightedRandomSampler |
| Sampler (val/test) | Séquentiel (distribution naturelle) |
| num_workers | 0 (configurable) |

---

## 8. Pipeline global

```
Dataset FER2013 (35 887 images, 7 classes)
        │
        ▼
  FERDataset (src/dataset.py)
  ├─ Détection automatique format CSV ou dossier
  ├─ Split train  : 28 709 images
  └─ Split val/test : 7 178 images
        │
        ├── ENTRAÎNEMENT ──────────────────────────────┐
        │   GaussianDenoise → CLAHE                   │
        │   RandomHorizontalFlip → RandomRotation      │
        │   ColorJitter → RandomResizedCrop            │
        │   ToTensor → Normalize(0.563, 0.263)         │
        │   WeightedRandomSampler                      │
        │                                              │
        └── VALIDATION / TEST ─────────────────────────┤
            GaussianDenoise → CLAHE                   │
            ToTensor → Normalize(0.563, 0.263)         │
                                                       ▼
                                          DataLoader (batch=64)
                                                       │
                                                       ▼
                                     Modèle (B, 1, 48, 48) → (B, 7)
                                     ├─ Baseline CNN   (390 K params)
                                     ├─ ResNet-50       (23.5 M params)
                                     └─ EfficientNet-B0  (4.0 M params)
                                                       │
                                                       ▼
                                       CrossEntropyLoss (non pondérée)
                                                       │
                                                       ▼
                                     Adam (lr=0.001, weight_decay=0.0001)
                                     Scheduler : ReduceLROnPlateau
                                     Early stopping (patience=10)
                                                       │
                                                       ▼
                                Évaluation : accuracy globale
                                             matrice de confusion
                                             F1-score par classe
```

---

## 9. Configuration d'entraînement (Semaine 4)

| Hyperparamètre | Valeur | Source |
| --- | --- | --- |
| Optimiseur | Adam | Standard pour FER |
| Learning rate | 0.001 | `config.yaml` |
| Weight decay | 0.0001 | `config.yaml` |
| Epochs max | 50 | `config.yaml` |
| Early stopping patience | 10 | `config.yaml` |
| Batch size | 64 | `config.yaml` |
| Device | CUDA (auto-détecté) | `config.yaml` |
| Scheduler | ReduceLROnPlateau | Semaine 4 |

---

## 10. Fichiers produits

| Fichier | Rôle |
| --- | --- |
| `src/model.py` | BaselineCNN, TransferModel, build_model() |
| `src/transforms.py` | GaussianDenoise, CLAHE, AddGaussianNoise (classes torchvision) |
| `notebooks/cr3_model.ipynb` | Notebook CR3 exécutable |
| `configs/config.yaml` | Mise à jour : section `preprocessing` avec mean=0.563, std=0.263 |
| `results/cr3_architecture.png` | Diagramme BaselineCNN |
| `results/cr3_model_comparison.png` | Comparaison du nombre de paramètres |
| `results/cr3_normalization.png` | Distribution avant/après CLAHE |
| `results/cr3_augmentation.png` | Exemples d'augmentation par classe |
| `results/cr3_sampling.png` | Distribution avant/après WeightedRandomSampler |
