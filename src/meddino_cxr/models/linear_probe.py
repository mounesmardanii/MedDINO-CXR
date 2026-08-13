from __future__ import annotations

import torch
import torch.nn as nn

from .dinov2 import DINOV2_FEATURE_DIM


NUM_CLASSES = 14


class DINOv2LinearProbe(nn.Module):
    def __init__(
        self,
        feature_dim: int = DINOV2_FEATURE_DIM,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes

        self.classifier = nn.Linear(
            feature_dim,
            num_classes,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        return self.classifier(
            embeddings
        )


def build_dinov2_linear_probe(
    feature_dim: int = DINOV2_FEATURE_DIM,
    num_classes: int = NUM_CLASSES,
) -> DINOv2LinearProbe:
    return DINOv2LinearProbe(
        feature_dim=feature_dim,
        num_classes=num_classes,
    )