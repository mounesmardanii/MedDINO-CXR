from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from meddino_cxr.data import build_feature_dataloader
from meddino_cxr.models import (
    DINOV2_FEATURE_DIM,
    DINOV2_MODEL,
    DINOV2_REPO,
    build_dinov2_vits14,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frozen DINOv2 embeddings for ChestMNIST."
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=["train", "val"],
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "processed"
        / "dinov2_vits14",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def save_metadata(
    path: Path,
    metadata: dict,
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )


@torch.inference_mode()
def extract_split(
    split: str,
    model: torch.nn.Module,
    device: torch.device,
    output_dir: Path,
    batch_size: int,
    num_workers: int,
    overwrite: bool,
) -> None:
    loader = build_feature_dataloader(
        split,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    sample_count = len(
        loader.dataset
    )

    embeddings_path = (
        output_dir
        / f"{split}_embeddings.npy"
    )

    labels_path = (
        output_dir
        / f"{split}_labels.npy"
    )

    metadata_path = (
        output_dir
        / f"{split}_metadata.json"
    )

    existing = [
        path
        for path in [
            embeddings_path,
            labels_path,
            metadata_path,
        ]
        if path.exists()
    ]

    if existing and not overwrite:
        names = ", ".join(
            path.name
            for path in existing
        )
        raise FileExistsError(
            f"Output files already exist: {names}"
        )

    embeddings = np.lib.format.open_memmap(
        embeddings_path,
        mode="w+",
        dtype=np.float32,
        shape=(
            sample_count,
            DINOV2_FEATURE_DIM,
        ),
    )

    labels = np.lib.format.open_memmap(
        labels_path,
        mode="w+",
        dtype=np.float32,
        shape=(
            sample_count,
            14,
        ),
    )

    offset = 0

    progress = tqdm(
        loader,
        desc=f"DINOv2 {split}",
    )

    for images, batch_labels in progress:
        images = images.to(
            device,
            non_blocking=device.type == "cuda",
        )

        features = model(
            images
        )

        features_np = (
            features
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

        labels_np = (
            batch_labels
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

        batch_count = (
            features_np.shape[0]
        )

        end = (
            offset
            + batch_count
        )

        embeddings[
            offset:end
        ] = features_np

        labels[
            offset:end
        ] = labels_np

        offset = end

    if offset != sample_count:
        raise RuntimeError(
            f"Expected {sample_count} samples, extracted {offset}."
        )

    embeddings.flush()
    labels.flush()

    del embeddings
    del labels

    metadata = {
        "split": split,
        "samples": sample_count,
        "feature_dim": DINOV2_FEATURE_DIM,
        "embedding_dtype": "float32",
        "label_dtype": "float32",
        "batch_size": batch_size,
        "model": DINOV2_MODEL,
        "repository": DINOV2_REPO,
        "backbone_frozen": True,
        "transform": "deterministic_eval",
    }

    save_metadata(
        metadata_path,
        metadata,
    )

    saved_embeddings = np.load(
        embeddings_path,
        mmap_mode="r",
    )

    saved_labels = np.load(
        labels_path,
        mmap_mode="r",
    )

    print()
    print(f"Split: {split}")
    print(
        "Embeddings: "
        f"{saved_embeddings.shape} "
        f"{saved_embeddings.dtype}"
    )
    print(
        "Labels: "
        f"{saved_labels.shape} "
        f"{saved_labels.dtype}"
    )
    print(
        f"Saved: {embeddings_path}"
    )
    print(
        f"Saved: {labels_path}"
    )
    print(
        f"Metadata: {metadata_path}"
    )


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = build_dinov2_vits14(
        freeze=True
    ).to(device)

    model.eval()

    print(f"Device: {device}")
    print(f"Model: {DINOV2_MODEL}")
    print(
        f"Feature dim: {DINOV2_FEATURE_DIM}"
    )
    print(
        f"Batch size: {args.batch_size}"
    )
    print(
        "Splits: "
        + ", ".join(args.splits)
    )

    for split in args.splits:
        extract_split(
            split=split,
            model=model,
            device=device,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            overwrite=args.overwrite,
        )

    print()
    print(
        "DINOv2 feature extraction completed."
    )


if __name__ == "__main__":
    main()