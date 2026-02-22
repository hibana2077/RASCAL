"""timm-backed image encoder + heads.

This replaces the original handcrafted ResNet backbone. All timm models are
created with `pretrained=False` by design (per your requirement).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _require_timm():
    try:
        import timm  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Missing dependency: timm. Install with: pip install timm"
        ) from e


def get_timm_encoder(model_name: str) -> nn.Module:
    """Create a timm encoder that returns a pooled feature vector [B, D]."""
    _require_timm()
    import timm

    # num_classes=0 makes timm return features; global_pool='avg' ensures [B, D].
    encoder = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=0,
        global_pool="avg",
    )
    return encoder


def get_timm_feat_dim(model_name: str) -> int:
    encoder = get_timm_encoder(model_name)
    dim = getattr(encoder, "num_features", None)
    if dim is None:
        # timm models should expose num_features; keep a clear error if not.
        raise ValueError(f"timm model '{model_name}' does not expose num_features")
    return int(dim)


class SupConResNet(nn.Module):
    """Encoder (timm) + projection head (linear/mlp)."""

    def __init__(self, name: str = "resnet50", head: str = "mlp", feat_dim: int = 128):
        super().__init__()
        self.encoder = get_timm_encoder(name)
        dim_in = int(self.encoder.num_features)

        if head == "linear":
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == "mlp":
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim),
            )
        else:
            raise NotImplementedError(f"head not supported: {head}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat


class LinearClassifier(nn.Module):
    """Linear classifier for linear evaluation / CE head."""

    def __init__(self, name: str = "resnet50", num_classes: int = 10, in_dim: int | None = None):
        super().__init__()
        if in_dim is None:
            in_dim = get_timm_feat_dim(name)
        self.fc = nn.Linear(int(in_dim), int(num_classes))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)