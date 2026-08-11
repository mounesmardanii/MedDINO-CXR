from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


def compute_pos_weight(
    labels_path: str | Path,
    max_weight: float = 20.0,
) -> torch.Tensor:
    labels = np.load(
        Path(labels_path),
        mmap_mode="r",
    )

    positives = np.asarray(
        labels.sum(axis=0),
        dtype=np.float64,
    )

    negatives = len(labels) - positives

    weights = negatives / np.maximum(
        positives,
        1.0,
    )

    weights = np.clip(
        weights,
        a_min=1.0,
        a_max=max_weight,
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


def build_loss(
    pos_weight: torch.Tensor | None = None,
) -> nn.BCEWithLogitsLoss:
    return nn.BCEWithLogitsLoss(
        pos_weight=pos_weight,
    )