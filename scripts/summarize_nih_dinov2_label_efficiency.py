from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "nih_dinov2_linear_probe"
)

CHECKPOINT_ROOT = (
    PROJECT_ROOT
    / "checkpoints"
    / "nih_dinov2_linear_probe"
)

SEEDS = (
    42,
    47,
    52,
)

FRACTIONS = (
    1,
    5,
    10,
    25,
    100,
)

FRACTION_CODES = {
    1: "001",
    5: "005",
    10: "010",
    25: "025",
    100: "100",
}

CLASS_NAMES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

METRIC_KEYS = [
    "macro_roc_auc",
    "micro_roc_auc",
    "macro_average_precision",
    "micro_average_precision",
]


def run_name(seed, fraction):
    return (
        f"seed_{seed}_"
        f"fraction_{FRACTION_CODES[fraction]}"
    )


def load_run(seed, fraction):
    name = run_name(
        seed,
        fraction,
    )

    checkpoint_path = (
        CHECKPOINT_ROOT
        / f"{name}.pt"
    )

    config_path = (
        OUTPUT_ROOT
        / name
        / "run_config.json"
    )

    history_path = (
        OUTPUT_ROOT
        / name
        / "history.json"
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            checkpoint_path
        )

    if not config_path.is_file():
        raise FileNotFoundError(
            config_path
        )

    if not history_path.is_file():
        raise FileNotFoundError(
            history_path
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    with history_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        history = json.load(file)

    if not history:
        raise RuntimeError(
            f"{name}: empty history"
        )

    metrics = checkpoint.get(
        "metrics",
        {},
    )

    for key in METRIC_KEYS:
        if key not in metrics:
            raise KeyError(
                f"{name}: missing {key}"
            )

    per_class_roc_auc = metrics.get(
        "per_class_roc_auc"
    )

    per_class_ap = metrics.get(
        "per_class_average_precision"
    )

    if (
        per_class_roc_auc is None
        or len(per_class_roc_auc) != 14
    ):
        raise RuntimeError(
            f"{name}: invalid per-class ROC-AUC"
        )

    if (
        per_class_ap is None
        or len(per_class_ap) != 14
    ):
        raise RuntimeError(
            f"{name}: invalid per-class AP"
        )

    return {
        "run_name": name,
        "fraction": fraction,
        "seed": seed,
        "train_images": int(
            config["train_images"]
        ),
        "validation_images": int(
            config["validation_images"]
        ),
        "best_epoch": int(
            checkpoint["epoch"]
        ),
        "epochs_executed": int(
            history[-1]["epoch"]
        ),
        "macro_roc_auc": float(
            metrics["macro_roc_auc"]
        ),
        "micro_roc_auc": float(
            metrics["micro_roc_auc"]
        ),
        "macro_average_precision": float(
            metrics["macro_average_precision"]
        ),
        "micro_average_precision": float(
            metrics["micro_average_precision"]
        ),
        "per_class_roc_auc": [
            float(value)
            for value in per_class_roc_auc
        ],
        "per_class_average_precision": [
            float(value)
            for value in per_class_ap
        ],
    }


def write_csv(
    path,
    rows,
    fieldnames,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def mean_sd(values):
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    return (
        float(array.mean()),
        float(
            array.std(
                ddof=1
            )
        ),
    )


def main():
    runs = []

    for fraction in FRACTIONS:
        for seed in SEEDS:
            runs.append(
                load_run(
                    seed,
                    fraction,
                )
            )

    if len(runs) != 15:
        raise RuntimeError(
            f"Expected 15 runs, found {len(runs)}"
        )

    run_rows = []

    for run in runs:
        run_rows.append(
            {
                "run_name": run["run_name"],
                "fraction_percent": run["fraction"],
                "seed": run["seed"],
                "train_images": run["train_images"],
                "validation_images": run[
                    "validation_images"
                ],
                "best_epoch": run["best_epoch"],
                "epochs_executed": run[
                    "epochs_executed"
                ],
                "macro_roc_auc": run[
                    "macro_roc_auc"
                ],
                "micro_roc_auc": run[
                    "micro_roc_auc"
                ],
                "macro_average_precision": run[
                    "macro_average_precision"
                ],
                "micro_average_precision": run[
                    "micro_average_precision"
                ],
            }
        )

    run_csv = (
        OUTPUT_ROOT
        / "validation_runs.csv"
    )

    write_csv(
        run_csv,
        run_rows,
        [
            "run_name",
            "fraction_percent",
            "seed",
            "train_images",
            "validation_images",
            "best_epoch",
            "epochs_executed",
            "macro_roc_auc",
            "micro_roc_auc",
            "macro_average_precision",
            "micro_average_precision",
        ],
    )

    summary_rows = []

    for fraction in FRACTIONS:
        selected = [
            run
            for run in runs
            if run["fraction"] == fraction
        ]

        row = {
            "fraction_percent": fraction,
            "seeds": len(selected),
            "train_images_min": min(
                run["train_images"]
                for run in selected
            ),
            "train_images_max": max(
                run["train_images"]
                for run in selected
            ),
        }

        for key in METRIC_KEYS:
            mean, sd = mean_sd(
                [
                    run[key]
                    for run in selected
                ]
            )

            row[f"{key}_mean"] = mean
            row[f"{key}_sd"] = sd

        summary_rows.append(
            row
        )

    summary_csv = (
        OUTPUT_ROOT
        / "validation_summary.csv"
    )

    write_csv(
        summary_csv,
        summary_rows,
        list(
            summary_rows[0].keys()
        ),
    )

    per_class_rows = []

    for fraction in FRACTIONS:
        selected = [
            run
            for run in runs
            if run["fraction"] == fraction
        ]

        for class_index, class_name in enumerate(
            CLASS_NAMES
        ):
            roc_values = [
                run[
                    "per_class_roc_auc"
                ][class_index]
                for run in selected
            ]

            ap_values = [
                run[
                    "per_class_average_precision"
                ][class_index]
                for run in selected
            ]

            roc_mean, roc_sd = mean_sd(
                roc_values
            )

            ap_mean, ap_sd = mean_sd(
                ap_values
            )

            per_class_rows.append(
                {
                    "fraction_percent": fraction,
                    "class": class_name,
                    "roc_auc_mean": roc_mean,
                    "roc_auc_sd": roc_sd,
                    "average_precision_mean": ap_mean,
                    "average_precision_sd": ap_sd,
                }
            )

    per_class_csv = (
        OUTPUT_ROOT
        / "validation_per_class_summary.csv"
    )

    write_csv(
        per_class_csv,
        per_class_rows,
        list(
            per_class_rows[0].keys()
        ),
    )

    print(
        "Runs audited:",
        len(runs),
    )

    print()

    print(
        "Fraction | Macro ROC-AUC mean +/- SD | Macro AP mean +/- SD"
    )

    print(
        "-" * 75
    )

    for row in summary_rows:
        print(
            f"{row['fraction_percent']:>3}%"
            " | "
            f"{row['macro_roc_auc_mean']:.6f}"
            " +/- "
            f"{row['macro_roc_auc_sd']:.6f}"
            " | "
            f"{row['macro_average_precision_mean']:.6f}"
            " +/- "
            f"{row['macro_average_precision_sd']:.6f}"
        )

    print()

    print(
        "Runs CSV:",
        run_csv,
    )

    print(
        "Summary CSV:",
        summary_csv,
    )

    print(
        "Per-class CSV:",
        per_class_csv,
    )

    print(
        "AUDIT PASSED: True"
    )


if __name__ == "__main__":
    main()