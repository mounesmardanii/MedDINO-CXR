from pathlib import Path

import numpy as np
import pandas as pd

from meddino_cxr.data.label_efficiency import (
    build_nested_multilabel_subsets,
    compute_class_statistics,
    validate_nested_subsets,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "manifests" / "nih_train.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "manifests" / "nih_label_efficiency_patients.csv"

FRACTIONS = (0.01, 0.05, 0.10, 0.25, 1.00)
SEEDS = (42, 47, 52)

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

manifest = pd.read_csv(
    INPUT_PATH,
    dtype={
        "patient_id": "string",
        "image_id": "string",
    },
)

patient_targets = (
    manifest[
        [
            "patient_id",
            *TARGET_COLUMNS,
        ]
    ]
    .groupby("patient_id", sort=True)[list(TARGET_COLUMNS)]
    .max()
)

patient_ids = patient_targets.index.to_numpy(dtype=str)
labels = patient_targets.to_numpy(dtype=np.int8)

result = pd.DataFrame({"patient_id": patient_ids})

print("Training patients:", len(patient_ids))

for seed in SEEDS:
    subsets = build_nested_multilabel_subsets(
        labels=labels,
        fractions=FRACTIONS,
        seed=seed,
    )

    validate_nested_subsets(subsets)

    print()
    print("Seed:", seed)

    for fraction in FRACTIONS:
        indices = subsets[fraction]
        column = f"seed_{seed}_fraction_{int(fraction * 100):03d}"
        result[column] = 0
        result.loc[indices, column] = 1

        positives, _ = compute_class_statistics(
            labels=labels,
            indices=indices,
        )

        print(
            f"{int(fraction * 100):3d}% "
            f"patients={len(indices):5d} "
            f"min_positive={int(positives.min()):3d}"
        )

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT_PATH, index=False)

assert result["patient_id"].is_unique
assert len(result) == 25207

print()
print("Output:", OUTPUT_PATH)
print("Rows:", len(result))
print("AUDIT PASSED")
