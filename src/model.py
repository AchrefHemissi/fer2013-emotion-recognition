"""
Model architectures for FER2013 — Week 3.

Available models
----------------
BaselineCNN    : custom lightweight CNN designed for 48x48 grayscale input
TransferModel  : ResNet-50 or EfficientNet-B0 adapted for 1-channel input and 7 classes
build_model()  : factory function driven by configs/config.yaml
"""

import torch
import torch.nn as nn
from torchvision import models


# ---------------------------------------------------------------------------
# Shared building block
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv2d(bias=False) -> BatchNorm2d -> ReLU -> MaxPool2d(2x2)"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Baseline CNN
# ---------------------------------------------------------------------------

class BaselineCNN(nn.Module):
    """
    Lightweight custom CNN for 48x48 grayscale facial emotion recognition.

    Spatial flow:
        (B,   1, 48, 48)  --Block1-->  (B,  32, 24, 24)
                          --Block2-->  (B,  64, 12, 12)
                          --Block3-->  (B, 128,  6,  6)
                          --Block4-->  (B, 256,  3,  3)
                          --GAP---->   (B, 256)
                          --FC------>  (B,   7)

    Each ConvBlock: Conv(3x3, no bias) -> BN -> ReLU -> MaxPool(2x2)
    Total trainable parameters: ~390 K
    """

    def __init__(self, num_classes: int = 7, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1,   32),    # 48 -> 24
            ConvBlock(32,  64),    # 24 -> 12
            ConvBlock(64,  128),   # 12 ->  6
            ConvBlock(128, 256),   #  6 ->  3
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)       # (B, 256, 3, 3)
        x = self.gap(x)            # (B, 256, 1, 1)
        x = x.flatten(1)           # (B, 256)
        return self.classifier(x)  # (B, 7)


# ---------------------------------------------------------------------------
# Transfer learning model
# ---------------------------------------------------------------------------

class TransferModel(nn.Module):
    """
    Pretrained backbone adapted for FER2013 grayscale input.

    Two adaptations are made to the backbone:
      1. First conv layer is replaced to accept 1 input channel instead of 3.
         Pretrained RGB weights are summed across the channel dimension so that
         ImageNet features are preserved (sum is equivalent to treating the
         grayscale image as identical across R, G, B channels).
      2. The classification head is replaced with Dropout -> Linear(num_classes).

    For ResNet-50 specifically, the initial MaxPool is replaced with Identity
    so the 48x48 input is not downsampled too aggressively before the residual
    blocks begin.

    Spatial flow after stem (ResNet-50, no MaxPool):
        Input       : (B, 1, 48, 48)
        After conv1 : (B, 64, 24, 24)   stride-2 conv
        layer1      : (B, 256, 24, 24)  no downsampling
        layer2      : (B, 512, 12, 12)  stride-2
        layer3      : (B, 1024, 6, 6)   stride-2
        layer4      : (B, 2048, 3, 3)   stride-2
        GAP         : (B, 2048)
        FC          : (B, 7)

    Args:
        backbone    : 'resnet50' or 'efficientnet_b0'
        num_classes : number of output classes (7 for FER2013)
        dropout     : dropout rate applied before the final linear layer
        pretrained  : load ImageNet pretrained weights
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 7,
        dropout: float = 0.5,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base = models.resnet50(weights=weights)

            # --- 1-channel input ---
            old = base.conv1
            new = nn.Conv2d(
                1, old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                bias=False,
            )
            if pretrained:
                # sum RGB weights -> single grayscale channel
                new.weight.data = old.weight.data.sum(dim=1, keepdim=True)
            base.conv1 = new

            # --- Remove MaxPool (preserves spatial info at 48x48) ---
            base.maxpool = nn.Identity()

            # --- 7-class head ---
            in_features = base.fc.in_features
            base.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes),
            )
            self.model = base

        elif backbone == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base = models.efficientnet_b0(weights=weights)

            # --- 1-channel input ---
            old = base.features[0][0]
            new = nn.Conv2d(
                1, old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                bias=False,
            )
            if pretrained:
                new.weight.data = old.weight.data.sum(dim=1, keepdim=True)
            base.features[0][0] = new

            # --- 7-class head ---
            in_features = base.classifier[1].in_features
            base.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes),
            )
            self.model = base

        else:
            raise ValueError(
                f"Unknown backbone '{backbone}'. Choose 'resnet50' or 'efficientnet_b0'."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(config: dict) -> nn.Module:
    """
    Build and return a model from a config dict (loaded from config.yaml).

    Reads config['model']['name'] and dispatches to the correct class.
    Valid names: 'baseline_cnn', 'resnet50', 'efficientnet_b0'.

    Args:
        config: full config dict as returned by yaml.safe_load(config.yaml)

    Returns:
        nn.Module ready to be moved to a device and trained
    """
    cfg = config["model"]
    name        = cfg["name"]
    dropout     = cfg.get("dropout", 0.5)
    num_classes = config["data"]["num_classes"]
    pretrained  = cfg.get("pretrained", True)

    if name == "baseline_cnn":
        return BaselineCNN(num_classes=num_classes, dropout=dropout)
    elif name in ("resnet50", "efficientnet_b0"):
        return TransferModel(
            backbone=name,
            num_classes=num_classes,
            dropout=dropout,
            pretrained=pretrained,
        )
    else:
        raise ValueError(f"Unknown model name '{name}' in config.")


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dummy = torch.zeros(2, 1, 48, 48)

    for name in ("baseline_cnn", "resnet50", "efficientnet_b0"):
        cfg = {
            "model": {"name": name, "dropout": 0.5, "pretrained": False},
            "data":  {"num_classes": 7},
        }
        model = build_model(cfg)
        out   = model(dummy)
        total = sum(p.numel() for p in model.parameters())
        print(f"{name:<20}  output={tuple(out.shape)}  params={total:,}")
