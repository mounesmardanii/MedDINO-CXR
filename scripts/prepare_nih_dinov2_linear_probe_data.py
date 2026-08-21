from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nih_dinov2_embeddings"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nih_dinov2_vits14"
)

SUBSET_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "nih_label_efficiency_patients.csv"
)

EXPECTED_SHARDS = 113
EXPECTED_TOTAL = 112120
EXPECTED_TRAIN = 77911
EXPECTED_VAL = 8613
FEATURE_DIM = 384
NUM_CLASSES = 14


def combine(parts):
    if not parts:
        raise RuntimeError("No arrays collected.")
    return np.concatenate(parts, axis=0)


def main():
    files = sorted(INPUT_DIR.glob("shard_*.npz"))

    if len(files) != EXPECTED_SHARDS:
        raise RuntimeError(
            f"Expected {EXPECTED_SHARDS} shards, found {len(files)}"
        )

    train_embeddings_parts = []
    train_labels_parts = []
    train_image_parts = []
    train_patient_parts = []
    train_view_parts = []

    val_embeddings_parts = []
    val_labels_parts = []
    val_image_parts = []
    val_patient_parts = []
    val_view_parts = []

    total = 0

    for index, path in enumerate(files, start=1):
        with np.load(path, allow_pickle=True) as data:
            embeddings = np.asarray(
                data["embeddings"],
                dtype=np.float32,
            )

            labels = np.asarray(
                data["labels"],
                dtype=np.uint8,
            )

            image_id = data["image_id"].astype(str)
            patient_id = data["patient_id"].astype(str)
            split = data["split"].astype(str)
            view_position = data["view_position"].astype(str)

        n = len(image_id)

        if embeddings.shape != (n, FEATURE_DIM):
            raise RuntimeError(
                f"Bad embedding shape in {path.name}: {embeddings.shape}"
            )

        if labels.shape != (n, NUM_CLASSES):
            raise RuntimeError(
                f"Bad label shape in {path.name}: {labels.shape}"
            )

        train_mask = split == "train"
        val_mask = split == "validate"

        if train_mask.any():
            train_embeddings_parts.append(embeddings[train_mask])
            train_labels_parts.append(labels[train_mask])
            train_image_parts.append(image_id[train_mask])
            train_patient_parts.append(patient_id[train_mask])
            train_view_parts.append(view_position[train_mask])

        if val_mask.any():
            val_embeddings_parts.append(embeddings[val_mask])
            val_labels_parts.append(labels[val_mask])
            val_image_parts.append(image_id[val_mask])
            val_patient_parts.append(patient_id[val_mask])
            val_view_parts.append(view_position[val_mask])

        total += n

        if index % 20 == 0 or index == len(files):
            print(
                f"Processed shards: {index}/{len(files)} "
                f"images={total}"
            )

    train_embeddings = combine(train_embeddings_parts)
    train_labels = combine(train_labels_parts)
    train_image_id = combine(train_image_parts)
    train_patient_id = combine(train_patient_parts)
    train_view = combine(train_view_parts)

    val_embeddings = combine(val_embeddings_parts)
    val_labels = combine(val_labels_parts)
    val_image_id = combine(val_image_parts)
    val_patient_id = combine(val_patient_parts)
    val_view = combine(val_view_parts)

    if total != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL} total images, found {total}"
        )

    if len(train_embeddings) != EXPECTED_TRAIN:
        raise RuntimeError(
            f"Expected {EXPECTED_TRAIN} train images, found {len(train_embeddings)}"
        )

    if len(val_embeddings) != EXPECTED_VAL:
        raise RuntimeError(
            f"Expected {EXPECTED_VAL} validation images, found {len(val_embeddings)}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        OUTPUT_DIR / "train_embeddings.npy",
        train_embeddings,
    )

    np.save(
        OUTPUT_DIR / "train_labels.npy",
        train_labels,
    )

    np.save(
        OUTPUT_DIR / "val_embeddings.npy",
        val_embeddings,
    )

    np.save(
        OUTPUT_DIR / "val_labels.npy",
        val_labels,
    )

    train_index = pd.DataFrame(
        {
            "row_index": np.arange(
                len(train_image_id),
                dtype=np.int64,
            ),
            "image_id": train_image_id,
            "patient_id": train_patient_id,
            "split": "train",
            "view_position": train_view,
        }
    )

    val_index = pd.DataFrame(
        {
            "row_index": np.arange(
                len(val_image_id),
                dtype=np.int64,
            ),
            "image_id": val_image_id,
            "patient_id": val_patient_id,
            "split": "validate",
            "view_position": val_view,
        }
    )

    train_index.to_csv(
        OUTPUT_DIR / "train_index.csv",
        index=False,
    )

    val_index.to_csv(
        OUTPUT_DIR / "val_index.csv",
        index=False,
    )

    subset_frame = pd.read_csv(
        SUBSET_MANIFEST,
        dtype={"patient_id": "string"},
    )

    subset_frame["patient_id"] = (
        subset_frame["patient_id"].astype(str)
    )

    train_patients = set(
        train_patient_id.tolist()
    )

    known_patients = set(
        subset_frame["patient_id"].tolist()
    )

    missing = train_patients - known_patients

    if missing:
        raise RuntimeError(
            f"Train patients missing from subset manifest: {len(missing)}"
        )

    subset_arrays = {}

    subset_columns = [
        column
        for column in subset_frame.columns
        if column.startswith("seed_")
    ]

    for column in subset_columns:
        selected_patients = set(
            subset_frame.loc[
                subset_frame[column].eq(1),
                "patient_id",
            ].tolist()
        )

        indices = np.flatnonzero(
            np.isin(
                train_patient_id,
                list(selected_patients),
            )
        ).astype(np.int64)

        subset_arrays[column] = indices

        actual_patients = len(
            set(train_patient_id[indices].tolist())
        )

        print(
            f"{column}: "
            f"patients={actual_patients} "
            f"images={len(indices)}"
        )

    fractions = [
        "001",
        "005",
        "010",
        "025",
        "100",
    ]

    for seed in [42, 47, 52]:
        previous = set()

        for fraction in fractions:
            key = (
                f"seed_{seed}_fraction_{fraction}"
            )

            current = set(
                subset_arrays[key].tolist()
            )

            if not previous.issubset(current):
                raise RuntimeError(
                    f"Non-nested image subset: {key}"
                )

            previous = current

        if len(
            subset_arrays[
                f"seed_{seed}_fraction_100"
            ]
        ) != EXPECTED_TRAIN:
            raise RuntimeError(
                f"100% subset mismatch for seed {seed}"
            )

    np.savez(
        OUTPUT_DIR / "nih_train_subset_indices.npz",
        **subset_arrays,
    )

    print()
    print("Train embeddings:", train_embeddings.shape)
    print("Train labels:", train_labels.shape)
    print("Validation embeddings:", val_embeddings.shape)
    print("Validation labels:", val_labels.shape)
    print("Subset keys:", len(subset_arrays))
    print("AUDIT PASSED: True")


if __name__ == "__main__":
    main()