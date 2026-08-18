from pathlib import Path

import numpy as np
import pandas as pd

from meddino_cxr.data.label_efficiency import select_stratified_subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "manifests" / "nih_manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "manifests"

VALIDATION_FRACTION = 0.10
SPLIT_SEED = 2026

TARGET_COLUMNS = (
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
)

REQUIRED_COLUMNS = (
    "patient_id",
    "image_id",
    "source_split",
    "split",
    *TARGET_COLUMNS,
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    manifest = pd.read_csv(
        INPUT_PATH,
        dtype={
            "patient_id": "string",
            "image_id": "string",
        },
    )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in manifest.columns
    ]

    require(
        not missing_columns,
        f"Missing required columns: {missing_columns}",
    )

    require(
        manifest["image_id"].is_unique,
        "image_id is not unique",
    )

    require(
        set(manifest["source_split"].unique()) == {"train_val", "test"},
        f"Unexpected source_split values: {sorted(manifest['source_split'].unique())}",
    )

    for column in TARGET_COLUMNS:
        values = set(
            manifest[column]
            .dropna()
            .astype(int)
            .unique()
        )

        require(
            values <= {0, 1},
            f"Non-binary values in {column}: {sorted(values)}",
        )

    train_val = manifest.loc[
        manifest["source_split"].eq("train_val")
    ].copy()

    official_test = manifest.loc[
        manifest["source_split"].eq("test")
    ].copy()

    train_val_patients = set(
        train_val["patient_id"]
    )

    test_patients = set(
        official_test["patient_id"]
    )

    require(
        not train_val_patients.intersection(test_patients),
        "Patient overlap between official train_val and test",
    )

    patient_targets = (
        train_val[
            [
                "patient_id",
                *TARGET_COLUMNS,
            ]
        ]
        .groupby(
            "patient_id",
            sort=True,
        )[list(TARGET_COLUMNS)]
        .max()
    )

    patient_ids = patient_targets.index.to_numpy(
        dtype=str
    )

    labels = patient_targets.to_numpy(
        dtype=np.int8
    )

    total_train_val_patients = len(patient_ids)

    validation_size = int(
        round(
            total_train_val_patients
            * VALIDATION_FRACTION
        )
    )

    require(
        0 < validation_size < total_train_val_patients,
        f"Invalid validation size: {validation_size}",
    )

    candidate_indices = np.arange(
        total_train_val_patients,
        dtype=np.int64,
    )

    validation_indices = select_stratified_subset(
        candidate_indices=candidate_indices,
        labels=labels,
        target_size=validation_size,
        random_state=SPLIT_SEED,
    )

    validation_patients = set(
        patient_ids[
            validation_indices
        ]
    )

    training_patients = (
        train_val_patients
        - validation_patients
    )

    require(
        len(validation_patients) == validation_size,
        (
            "Validation patient count mismatch: "
            f"{len(validation_patients)} vs {validation_size}"
        ),
    )

    require(
        not training_patients.intersection(validation_patients),
        "Patient overlap between train and validation",
    )

    require(
        not training_patients.intersection(test_patients),
        "Patient overlap between train and test",
    )

    require(
        not validation_patients.intersection(test_patients),
        "Patient overlap between validation and test",
    )

    result = manifest.copy()
    result["split"] = ""

    result.loc[
        result["patient_id"].isin(training_patients),
        "split",
    ] = "train"

    result.loc[
        result["patient_id"].isin(validation_patients),
        "split",
    ] = "validate"

    result.loc[
        result["patient_id"].isin(test_patients),
        "split",
    ] = "test"

    require(
        result["split"].ne("").all(),
        "At least one manifest row was not assigned to a split",
    )

    require(
        set(result["split"].unique()) == {"train", "validate", "test"},
        f"Unexpected final split values: {sorted(result['split'].unique())}",
    )

    require(
        set(
            result.loc[
                result["source_split"].eq("test"),
                "image_id",
            ]
        )
        == set(official_test["image_id"]),
        "Official NIH test image membership changed",
    )

    require(
        result.loc[
            result["source_split"].eq("test"),
            "split",
        ].eq("test").all(),
        "Official NIH test row assigned outside test",
    )

    patient_split_counts = (
        result[
            [
                "patient_id",
                "split",
            ]
        ]
        .drop_duplicates()
        .groupby(
            "patient_id"
        )["split"]
        .nunique()
    )

    require(
        patient_split_counts.max() == 1,
        "At least one patient appears in multiple final splits",
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    full_path = OUTPUT_DIR / "nih_manifest_split.csv"
    train_path = OUTPUT_DIR / "nih_train.csv"
    validate_path = OUTPUT_DIR / "nih_validate.csv"
    test_path = OUTPUT_DIR / "nih_test.csv"
    patient_path = OUTPUT_DIR / "nih_split_patients.csv"

    result.to_csv(
        full_path,
        index=False,
    )

    for split_name, output_path in (
        ("train", train_path),
        ("validate", validate_path),
        ("test", test_path),
    ):
        result.loc[
            result["split"].eq(split_name)
        ].to_csv(
            output_path,
            index=False,
        )

    patient_assignments = (
        result[
            [
                "patient_id",
                "source_split",
                "split",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "split",
                "patient_id",
            ]
        )
    )

    patient_assignments.to_csv(
        patient_path,
        index=False,
    )

    image_counts = (
        result["split"]
        .value_counts()
        .reindex(
            [
                "train",
                "validate",
                "test",
            ]
        )
    )

    patient_counts = (
        patient_assignments["split"]
        .value_counts()
        .reindex(
            [
                "train",
                "validate",
                "test",
            ]
        )
    )

    print("NIH patient-level split built successfully.")
    print()
    print(f"Validation fraction: {VALIDATION_FRACTION:.2f}")
    print(f"Split seed: {SPLIT_SEED}")
    print(f"Official train_val patients: {total_train_val_patients}")
    print(f"Train patients: {int(patient_counts['train'])}")
    print(f"Validation patients: {int(patient_counts['validate'])}")
    print(f"Test patients: {int(patient_counts['test'])}")
    print()
    print(f"Train images: {int(image_counts['train'])}")
    print(f"Validation images: {int(image_counts['validate'])}")
    print(f"Test images: {int(image_counts['test'])}")
    print()
    print(f"Full manifest: {full_path}")
    print(f"Train manifest: {train_path}")
    print(f"Validation manifest: {validate_path}")
    print(f"Test manifest: {test_path}")
    print(f"Patient assignments: {patient_path}")


if __name__ == "__main__":
    main()