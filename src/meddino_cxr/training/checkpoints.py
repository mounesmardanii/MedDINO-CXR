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
    device = torch.device(device)

    checkpoint = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)

    metadata = {
        "epoch": checkpoint["epoch"],
        "metrics": checkpoint.get(
            "metrics",
            {},
        ),
    }

    del checkpoint

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return metadata


def is_better_metric(
    current: float,
    best: float | None,
    min_delta: float = 0.0,
) -> bool:
    if best is None:
        return True

    return current > best + min_delta