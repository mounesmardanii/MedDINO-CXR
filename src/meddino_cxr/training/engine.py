from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.optim import Optimizer
from tqdm import tqdm


def train_one_epoch(
    model: nn.Module,
    loader: Iterable,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    max_batches: int | None = None,
    grad_clip: float | None = 5.0,
) -> float:
    model.train()

    running_loss = 0.0
    processed_batches = 0

    progress = tqdm(
        loader,
        desc="Training",
        leave=False,
    )

    for batch_index, (images, labels) in enumerate(progress):
        if (
            max_batches is not None
            and batch_index >= max_batches
        ):
            break

        images = images.to(
            device,
            non_blocking=device.type == "cuda",
        )

        labels = labels.to(
            device,
            non_blocking=device.type == "cuda",
        )

        optimizer.zero_grad(
            set_to_none=True,
        )

        logits = model(images)

        loss = criterion(
            logits,
            labels,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite training loss: {loss.item()}"
            )

        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip,
            )

        optimizer.step()

        running_loss += loss.item()
        processed_batches += 1

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    if processed_batches == 0:
        raise RuntimeError(
            "No training batches were processed."
        )

    return running_loss / processed_batches


@torch.inference_mode()
def evaluate_one_epoch(
    model: nn.Module,
    loader: Iterable,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    model.eval()

    running_loss = 0.0
    processed_batches = 0

    progress = tqdm(
        loader,
        desc="Validation",
        leave=False,
    )

    for batch_index, (images, labels) in enumerate(progress):
        if (
            max_batches is not None
            and batch_index >= max_batches
        ):
            break

        images = images.to(
            device,
            non_blocking=device.type == "cuda",
        )

        labels = labels.to(
            device,
            non_blocking=device.type == "cuda",
        )

        logits = model(images)

        loss = criterion(
            logits,
            labels,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite validation loss: {loss.item()}"
            )

        running_loss += loss.item()
        processed_batches += 1

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    if processed_batches == 0:
        raise RuntimeError(
            "No validation batches were processed."
        )

    return running_loss / processed_batches


@torch.inference_mode()
def predict(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()

    targets = []
    probabilities = []

    progress = tqdm(
        loader,
        desc="Prediction",
        leave=False,
    )

    for batch_index, (images, labels) in enumerate(progress):
        if (
            max_batches is not None
            and batch_index >= max_batches
        ):
            break

        images = images.to(
            device,
            non_blocking=device.type == "cuda",
        )

        logits = model(images)

        batch_probabilities = torch.sigmoid(
            logits
        )

        targets.append(
            labels.cpu()
        )

        probabilities.append(
            batch_probabilities.cpu()
        )

    if not targets:
        raise RuntimeError(
            "No prediction batches were processed."
        )

    return (
        torch.cat(targets, dim=0),
        torch.cat(probabilities, dim=0),
    )