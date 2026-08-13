from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset


Split = Literal["train", "val", "test"]

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_EMBEDDING_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dinov2_vits14"
)


class DINOv2EmbeddingDataset(Dataset):
    def __init__(
        self,
        split: Split,
        data_dir: str | Path | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(
                "split must be one of: 'train', 'val', or 'test'."
            )

        self.split = split

        self.data_dir = (
            Path(data_dir)
            if data_dir is not None
            else DEFAULT_EMBEDDING_DIR
        )

        embeddings_path = (
            self.data_dir
            / f"{split}_embeddings.npy"
        )

        labels_path = (
            self.data_dir
            / f"{split}_labels.npy"
        )

        if not embeddings_path.is_file():
            raise FileNotFoundError(
                embeddings_path
            )

        if not labels_path.is_file():
            raise FileNotFoundError(
                labels_path
            )

        self.embeddings = np.load(
            embeddings_path,
            mmap_mode="r",
        )

        self.labels = np.load(
            labels_path,
            mmap_mode="r",
        )

        if self.embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must have shape (N, D)."
            )

        if self.labels.ndim != 2:
            raise ValueError(
                "Labels must have shape (N, C)."
            )

        if (
            self.embeddings.shape[0]
            != self.labels.shape[0]
        ):
            raise ValueError(
                "Embedding and label counts do not match."
            )

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = torch.tensor(
            self.embeddings[index],
            dtype=torch.float32,
        )

        label = torch.tensor(
            self.labels[index],
            dtype=torch.float32,
        )

        return embedding, label