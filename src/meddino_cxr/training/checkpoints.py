from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    metrics: dict,
) -> Path:
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }

    torch.save(
        checkpoint,
        path,
    )

    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    device: str | torch.device = "cpu",
) -> dict:
    checkpoint = torch.load(
        Path(path),
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    return checkpoint


def is_better_metric(
    current: float,
    best: float | None,
) -> bool:
    if best is None:
        return True

    return current > best