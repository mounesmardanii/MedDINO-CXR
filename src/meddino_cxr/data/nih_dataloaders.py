from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader

from .nih_dataset import (
    DEFAULT_MANIFEST,
    NIHChestXray14Dataset,
)
from .transforms import (
    build_eval_transform,
    build_train_transform,
)


Split = Literal[
    "train",
    "validate",
    "test",
]


def build_nih_dataloader(
    split: Split,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    seed: int = 42,
    patient_ids: Iterable[str] | None = None,
) -> DataLoader:
    if split not in {
        "train",
        "validate",
        "test",
    }:
        raise ValueError(
            "split must be one of: "
            "'train', 'validate', or 'test'."
        )

    if (
        patient_ids is not None
        and split != "train"
    ):
        raise ValueError(
            "patient_ids can only be used "
            "for the train split."
        )

    transform = (
        build_train_transform()
        if split == "train"
        else build_eval_transform()
    )

    dataset = NIHChestXray14Dataset(
        split=split,
        manifest_path=manifest_path,
        transform=transform,
        patient_ids=patient_ids,
    )

    if pin_memory is None:
        pin_memory = (
            torch.cuda.is_available()
        )

    generator = torch.Generator()
    generator.manual_seed(
        seed
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        generator=generator,
        persistent_workers=(
            num_workers > 0
        ),
    )