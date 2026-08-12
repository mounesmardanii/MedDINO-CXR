from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader

from .dataset import ChestMNISTDataset
from .transforms import build_eval_transform, build_train_transform


Split = Literal["train", "val", "test"]


def build_dataloader(
    split: Split,
    data_dir: str | Path | None = None,
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool | None = None,
) -> DataLoader:
    if split not in {"train", "val", "test"}:
        raise ValueError(
            "split must be one of: 'train', 'val', or 'test'."
        )

    transform = (
        build_train_transform()
        if split == "train"
        else build_eval_transform()
    )

    dataset_kwargs = {
        "split": split,
        "transform": transform,
    }

    if data_dir is not None:
        dataset_kwargs["data_dir"] = data_dir

    dataset = ChestMNISTDataset(**dataset_kwargs)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def build_feature_dataloader(
    split: Split,
    data_dir: str | Path | None = None,
    batch_size: int = 16,
    num_workers: int = 0,
    pin_memory: bool | None = None,
) -> DataLoader:
    if split not in {"train", "val", "test"}:
        raise ValueError(
            "split must be one of: 'train', 'val', or 'test'."
        )

    dataset_kwargs = {
        "split": split,
        "transform": build_eval_transform(),
    }

    if data_dir is not None:
        dataset_kwargs["data_dir"] = data_dir

    dataset = ChestMNISTDataset(**dataset_kwargs)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )