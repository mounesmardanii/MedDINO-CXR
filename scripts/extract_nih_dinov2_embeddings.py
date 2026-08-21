from __future__ import annotations

import argparse
import io
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from meddino_cxr.data.transforms import build_eval_transform
from meddino_cxr.models import build_dinov2_vits14


DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "nih_manifest_split.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nih_dinov2_embeddings"
)

NIH_HF_REPOSITORY = "yeigen/nih-chest-xray"
NIH_HF_REVISION = "c0b558ec72f1ce434f7355f0f5cf914e2d62c60a"

EXPECTED_IMAGES = 112120
EXPECTED_SHARDS = 113
FEATURE_DIM = 384

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
    "view_position",
    "shard_path",
    "member_name",
    *TARGET_COLUMNS,
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    return parser.parse_args()


def resolve_path(path):
    path = path.expanduser()

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def resolve_device(name):
    if name == "cpu":
        return torch.device("cpu")

    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available."
            )

        return torch.device("cuda")

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_manifest(path):
    require(
        path.is_file(),
        f"Manifest not found: {path}",
    )

    frame = pd.read_csv(
        path,
        dtype={
            "patient_id": "string",
            "image_id": "string",
            "split": "string",
            "view_position": "string",
            "shard_path": "string",
            "member_name": "string",
        },
    )

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in frame.columns
    ]

    require(
        not missing,
        f"Missing manifest columns: {missing}",
    )

    require(
        len(frame) == EXPECTED_IMAGES,
        (
            f"Expected {EXPECTED_IMAGES} manifest rows, "
            f"found {len(frame)}"
        ),
    )

    require(
        frame["image_id"].is_unique,
        "Manifest image_id is not unique.",
    )

    require(
        frame["shard_path"].notna().all(),
        "Manifest contains missing shard paths.",
    )

    require(
        frame["member_name"].notna().all(),
        "Manifest contains missing TAR member names.",
    )

    require(
        set(frame["split"].dropna().unique())
        <= {"train", "validate", "test"},
        "Unexpected split value found.",
    )

    return frame


def output_is_valid(path):
    if not path.is_file():
        return False

    try:
        with np.load(
            path,
            allow_pickle=False,
        ) as data:
            required = {
                "embeddings",
                "labels",
                "image_id",
                "patient_id",
                "split",
                "view_position",
            }

            if not required.issubset(
                set(data.files)
            ):
                return False

            n = len(
                data["image_id"]
            )

            return (
                data["embeddings"].shape
                == (n, FEATURE_DIM)
                and data["labels"].shape
                == (n, len(TARGET_COLUMNS))
            )

    except Exception:
        return False


def load_image(
    archive,
    member_name,
):
    member = archive.getmember(
        member_name
    )

    extracted = archive.extractfile(
        member
    )

    if extracted is None:
        raise RuntimeError(
            f"Could not extract TAR member: {member_name}"
        )

    raw = extracted.read()

    with Image.open(
        io.BytesIO(raw)
    ) as image:
        return image.convert(
            "RGB"
        )


def encode_batch(
    model,
    transform,
    images,
    device,
):
    batch = torch.stack(
        [
            transform(image)
            for image in images
        ],
        dim=0,
    ).to(
        device,
        non_blocking=True,
    )

    with torch.inference_mode():
        if device.type == "cuda":
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                features = model(
                    batch
                )
        else:
            features = model(
                batch
            )

    features = (
        features
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    require(
        features.ndim == 2,
        f"Unexpected embedding shape: {features.shape}",
    )

    require(
        features.shape[1] == FEATURE_DIM,
        (
            f"Expected feature dimension {FEATURE_DIM}, "
            f"found {features.shape[1]}"
        ),
    )

    return features


def process_shard(
    frame,
    shard_path,
    output_path,
    model,
    transform,
    device,
    batch_size,
):
    shard_file = resolve_path(
        Path(shard_path)
    )

    require(
        shard_file.is_file(),
        f"TAR shard not found: {shard_file}",
    )

    shard_rows = (
        frame.loc[
            frame["shard_path"].eq(
                shard_path
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    require(
        len(shard_rows) > 0,
        f"No manifest rows for shard: {shard_path}",
    )

    embeddings_parts = []

    image_ids = []
    patient_ids = []
    splits = []
    view_positions = []
    labels = []

    pending_images = []

    with tarfile.open(
        shard_file,
        "r",
    ) as archive:
        for row in shard_rows.itertuples(
            index=False
        ):
            image = load_image(
                archive,
                row.member_name,
            )

            pending_images.append(
                image
            )

            image_ids.append(
                str(row.image_id)
            )

            patient_ids.append(
                str(row.patient_id)
            )

            splits.append(
                str(row.split)
            )

            view_positions.append(
                str(row.view_position)
            )

            labels.append(
                [
                    int(
                        getattr(
                            row,
                            column,
                        )
                    )
                    for column in TARGET_COLUMNS
                ]
            )

            if len(
                pending_images
            ) >= batch_size:
                embeddings_parts.append(
                    encode_batch(
                        model,
                        transform,
                        pending_images,
                        device,
                    )
                )

                pending_images = []

        if pending_images:
            embeddings_parts.append(
                encode_batch(
                    model,
                    transform,
                    pending_images,
                    device,
                )
            )

    embeddings = np.concatenate(
        embeddings_parts,
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    labels_array = np.asarray(
        labels,
        dtype=np.uint8,
    )

    image_id_array = np.asarray(
        image_ids,
        dtype=str,
    )

    patient_id_array = np.asarray(
        patient_ids,
        dtype=str,
    )

    split_array = np.asarray(
        splits,
        dtype=str,
    )

    view_array = np.asarray(
        view_positions,
        dtype=str,
    )

    n = len(
        shard_rows
    )

    require(
        embeddings.shape
        == (n, FEATURE_DIM),
        (
            f"Bad embedding shape for {shard_path}: "
            f"{embeddings.shape}"
        ),
    )

    require(
        labels_array.shape
        == (n, len(TARGET_COLUMNS)),
        (
            f"Bad label shape for {shard_path}: "
            f"{labels_array.shape}"
        ),
    )

    temporary_path = (
        output_path.parent
        / f"{output_path.stem}.tmp.npz"
    )

    np.savez(
        temporary_path,
        embeddings=embeddings,
        labels=labels_array,
        image_id=image_id_array,
        patient_id=patient_id_array,
        split=split_array,
        view_position=view_array,
    )

    with np.load(
        temporary_path,
        allow_pickle=False,
    ) as data:
        require(
            data["embeddings"].shape
            == (n, FEATURE_DIM),
            "Temporary embedding audit failed.",
        )

        require(
            data["labels"].shape
            == (n, len(TARGET_COLUMNS)),
            "Temporary label audit failed.",
        )

    temporary_path.replace(
        output_path
    )

    return n


def audit_outputs(
    output_dir,
):
    files = sorted(
        output_dir.glob(
            "shard_*.npz"
        )
    )

    require(
        len(files) == EXPECTED_SHARDS,
        (
            f"Expected {EXPECTED_SHARDS} outputs, "
            f"found {len(files)}"
        ),
    )

    total = 0

    for path in files:
        require(
            output_is_valid(path),
            f"Invalid output: {path}",
        )

        with np.load(
            path,
            allow_pickle=False,
        ) as data:
            total += len(
                data["image_id"]
            )

    require(
        total == EXPECTED_IMAGES,
        (
            f"Expected {EXPECTED_IMAGES} saved images, "
            f"found {total}"
        ),
    )

    return total


def main():
    args = parse_args()

    manifest_path = resolve_path(
        args.manifest
    )

    output_dir = resolve_path(
        args.output_dir
    )

    frame = load_manifest(
        manifest_path
    )

    shard_paths = sorted(
        frame[
            "shard_path"
        ].unique().tolist()
    )

    require(
        len(shard_paths)
        == EXPECTED_SHARDS,
        (
            f"Expected {EXPECTED_SHARDS} manifest shards, "
            f"found {len(shard_paths)}"
        ),
    )

    print(
        "NIH HF repository:",
        NIH_HF_REPOSITORY,
    )

    print(
        "NIH HF revision:",
        NIH_HF_REVISION,
    )

    print(
        "Manifest:",
        manifest_path,
    )

    print(
        "Images:",
        len(frame),
    )

    print(
        "Shards:",
        len(shard_paths),
    )

    print(
        "Batch size:",
        args.batch_size,
    )

    if args.dry_run:
        print(
            "DRY RUN PASSED: True"
        )
        return

    device = resolve_device(
        args.device
    )

    print(
        "Device:",
        device,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = build_dinov2_vits14(
        freeze=True
    ).to(
        device
    )

    model.eval()

    transform = build_eval_transform(
        image_size=224
    )

    total_processed = 0
    started = time.perf_counter()

    for index, shard_path in enumerate(
        shard_paths
    ):
        output_path = (
            output_dir
            / f"shard_{index:03d}.npz"
        )

        if (
            not args.overwrite
            and output_is_valid(
                output_path
            )
        ):
            with np.load(
                output_path,
                allow_pickle=False,
            ) as data:
                saved = len(
                    data["image_id"]
                )

            total_processed += saved

            print(
                f"[{index + 1:03d}/{EXPECTED_SHARDS:03d}] "
                f"SKIP {output_path.name} "
                f"images={saved} "
                f"total={total_processed}"
            )

            continue

        shard_started = (
            time.perf_counter()
        )

        saved = process_shard(
            frame=frame,
            shard_path=shard_path,
            output_path=output_path,
            model=model,
            transform=transform,
            device=device,
            batch_size=args.batch_size,
        )

        total_processed += saved

        elapsed = (
            time.perf_counter()
            - shard_started
        )

        print(
            f"[{index + 1:03d}/{EXPECTED_SHARDS:03d}] "
            f"SAVED {output_path.name} "
            f"images={saved} "
            f"total={total_processed} "
            f"seconds={elapsed:.2f}"
        )

    total = audit_outputs(
        output_dir
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    print()

    print(
        "Shard outputs:",
        EXPECTED_SHARDS,
    )

    print(
        "Total saved images:",
        total,
    )

    print(
        "Feature dimension:",
        FEATURE_DIM,
    )

    print(
        "Elapsed seconds:",
        round(
            elapsed,
            2,
        ),
    )

    print(
        "AUDIT PASSED: True"
    )


if __name__ == "__main__":
    main()