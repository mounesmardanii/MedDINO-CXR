from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


Split = Literal["train", "val", "test"]


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chestmnist_224"
)


class ChestMNISTDataset(Dataset):

    def __init__(
        self,
        split: Split,
        data_dir: str | Path = DEFAULT_DATA_DIR,
        transform: Callable | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(
                "split must be one of: "
                "'train', 'val', or 'test'."
            )

        self.split = split
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.transform = transform

        self.images_path = (
            self.data_dir
            / f"{split}_images.npy"
        )

        self.labels_path = (
            self.data_dir
            / f"{split}_labels.npy"
        )

        self._validate_files()

        self.images = np.load(
            self.images_path,
            mmap_mode="r",
        )

        self.labels = np.load(
            self.labels_path,
            mmap_mode="r",
        )

        self._validate_arrays()

    def _validate_files(self) -> None:
        if not self.images_path.is_file():
            raise FileNotFoundError(
                "Image array was not found:\n"
                f"{self.images_path}"
            )

        if not self.labels_path.is_file():
            raise FileNotFoundError(
                "Label array was not found:\n"
                f"{self.labels_path}"
            )

    def _validate_arrays(self) -> None:
        if len(self.images) != len(self.labels):
            raise ValueError(
                "Image and label counts do not match: "
                f"{len(self.images)} images vs "
                f"{len(self.labels)} labels."
            )

        if self.images.ndim != 3:
            raise ValueError(
                "Expected image array with shape "
                "(N, H, W), but received "
                f"{self.images.shape}."
            )

        if self.labels.ndim != 2:
            raise ValueError(
                "Expected label array with shape "
                "(N, C), but received "
                f"{self.labels.shape}."
            )

        if self.labels.shape[1] != 14:
            raise ValueError(
                "ChestMNIST should contain 14 labels "
                "per image, but received "
                f"{self.labels.shape[1]}."
            )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_array = np.asarray(
            self.images[index]
        )

        label_array = np.asarray(
            self.labels[index]
        ).copy()

        image = Image.fromarray(
            image_array,
            mode="L",
        ).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = to_tensor(image)

        labels = torch.tensor(
            label_array,
            dtype=torch.float32,
        )

        return image, labels

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"split='{self.split}', "
            f"samples={len(self)}, "
            f"data_dir='{self.data_dir}'"
            f")"
        )