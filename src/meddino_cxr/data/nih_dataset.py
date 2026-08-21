from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Callable, Iterable, Literal

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


Split = Literal["train", "validate", "test"]

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "nih_manifest_split.csv"
)

TARGET_COLUMNS = [
    "target_atelectasis",
    "target_cardiomegaly",
    "target_effusion",
    "target_infiltration",
    "target_mass",
    "target_nodule",
    "target_pneumonia",
    "target_pneumothorax",
    "target_consolidation",
    "target_edema",
    "target_emphysema",
    "target_fibrosis",
    "target_pleural_thickening",
    "target_hernia",
]

REQUIRED_COLUMNS = [
    "patient_id",
    "image_id",
    "split",
    "shard_path",
    "member_name",
    *TARGET_COLUMNS,
]


class NIHChestXray14Dataset(Dataset):

    def __init__(
        self,
        split: Split,
        manifest_path: str | Path = DEFAULT_MANIFEST,
        transform: Callable | None = None,
        patient_ids: Iterable[str] | None = None,
    ) -> None:
        if split not in {
            "train",
            "validate",
            "test",
        }:
            raise ValueError(
                "split must be one of: "
                "'train', 'validate', or 'test'."
            )

        self.split = split
        self.manifest_path = Path(
            manifest_path
        ).expanduser()

        if not self.manifest_path.is_absolute():
            self.manifest_path = (
                PROJECT_ROOT
                / self.manifest_path
            )

        self.manifest_path = (
            self.manifest_path.resolve()
        )

        self.transform = transform
        self._archives = {}

        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Manifest not found: "
                f"{self.manifest_path}"
            )

        frame = pd.read_csv(
            self.manifest_path,
            dtype={
                "patient_id": "string",
                "image_id": "string",
                "split": "string",
                "shard_path": "string",
                "member_name": "string",
            },
        )

        missing = [
            column
            for column in REQUIRED_COLUMNS
            if column not in frame.columns
        ]

        if missing:
            raise ValueError(
                f"Missing manifest columns: "
                f"{missing}"
            )

        frame = frame.loc[
            frame["split"].eq(split)
        ].copy()

        if patient_ids is not None:
            patient_set = {
                str(patient_id)
                for patient_id in patient_ids
            }

            if not patient_set:
                raise ValueError(
                    "patient_ids must not be empty."
                )

            frame = frame.loc[
                frame["patient_id"].isin(
                    patient_set
                )
            ].copy()

            found_patients = set(
                frame["patient_id"].astype(
                    str
                ).unique()
            )

            missing_patients = (
                patient_set
                - found_patients
            )

            if missing_patients:
                raise ValueError(
                    "Selected patients were not "
                    "found in the requested split: "
                    f"{len(missing_patients)}"
                )

        frame = frame.reset_index(
            drop=True
        )

        if frame.empty:
            raise ValueError(
                f"No NIH samples found for "
                f"split '{split}'."
            )

        if frame["image_id"].duplicated().any():
            raise ValueError(
                "Duplicate NIH image_id found."
            )

        if frame[
            "shard_path"
        ].isna().any():
            raise ValueError(
                "Missing shard_path found."
            )

        if frame[
            "member_name"
        ].isna().any():
            raise ValueError(
                "Missing member_name found."
            )

        labels = frame[
            TARGET_COLUMNS
        ].to_numpy(
            dtype=np.float32,
            copy=True,
        )

        if labels.shape != (
            len(frame),
            14,
        ):
            raise ValueError(
                f"Invalid label shape: "
                f"{labels.shape}"
            )

        if not np.isin(
            labels,
            [0.0, 1.0],
        ).all():
            raise ValueError(
                "NIH labels must be binary."
            )

        self.frame = frame
        self.labels = labels

        self.patient_ids = (
            frame["patient_id"]
            .astype(str)
            .to_numpy()
        )

        self.image_ids = (
            frame["image_id"]
            .astype(str)
            .to_numpy()
        )

        self.shard_paths = (
            frame["shard_path"]
            .astype(str)
            .to_numpy()
        )

        self.member_names = (
            frame["member_name"]
            .astype(str)
            .to_numpy()
        )

    def __len__(self) -> int:
        return len(
            self.frame
        )

    def _resolve_shard_path(
        self,
        value: str,
    ) -> Path:
        path = Path(value)

        if not path.is_absolute():
            path = (
                PROJECT_ROOT
                / path
            )

        return path.resolve()

    def _get_archive(
        self,
        shard_path: str,
    ) -> tarfile.TarFile:
        path = self._resolve_shard_path(
            shard_path
        )

        key = str(path)

        archive = self._archives.get(
            key
        )

        if archive is None:
            if not path.is_file():
                raise FileNotFoundError(
                    f"NIH TAR shard not found: "
                    f"{path}"
                )

            archive = tarfile.open(
                path,
                mode="r",
            )

            self._archives[
                key
            ] = archive

        return archive

    def _load_image(
        self,
        index: int,
    ) -> Image.Image:
        archive = self._get_archive(
            self.shard_paths[index]
        )

        member_name = (
            self.member_names[index]
        )

        extracted = archive.extractfile(
            member_name
        )

        if extracted is None:
            raise RuntimeError(
                f"Unable to extract "
                f"{member_name}"
            )

        with extracted:
            with Image.open(
                extracted
            ) as image:
                return image.convert(
                    "RGB"
                )

    def __getitem__(
        self,
        index: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        image = self._load_image(
            index
        )

        if self.transform is not None:
            image = self.transform(
                image
            )
        else:
            image = to_tensor(
                image
            )

        labels = torch.from_numpy(
            self.labels[index].copy()
        ).float()

        return (
            image,
            labels,
        )

    def close(self) -> None:
        for archive in (
            self._archives.values()
        ):
            archive.close()

        self._archives.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_archives"] = {}
        return state

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"split='{self.split}', "
            f"samples={len(self)}, "
            f"patients="
            f"{len(set(self.patient_ids))}, "
            f"manifest="
            f"'{self.manifest_path}')"
        )