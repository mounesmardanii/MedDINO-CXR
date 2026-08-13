import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "label_efficiency"
RUN_ROOT = OUTPUT_ROOT / "dinov2"
SUBSET_ROOT = OUTPUT_ROOT / "subsets"
RESULTS_ROOT = PROJECT_ROOT / "results" / "label_efficiency"

FRACTIONS = (5, 10, 25, 50, 100)
SEEDS = (42, 47, 52)
MIN_DELTA = 1e-4
EXPECTED_CLASSES = 14

CLASS_NAMES = (
    "atelectasis",
    "cardiomegaly",
    "effusion",
    "infiltration",
    "mass",
    "nodule",
    "pneumonia",
    "pneumothorax",
    "consolidation",
    "edema",
    "emphysema",
    "fibrosis",
    "pleural",
    "hernia",
)

SCALAR_METRICS = (
    "train_loss",
    "val_loss",
    "macro_roc_auc",
    "micro_roc_auc",
    "macro_average_precision",
    "micro_average_precision",
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def values_match(a, b, tolerance=1e-10):
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tolerance)


def reconstruct_selected_epoch(history):
    best = -math.inf
    selected_epoch = None

    for record in history:
        value = float(record["macro_roc_auc"])
        if value > best + MIN_DELTA:
            best = value
            selected_epoch = int(record["epoch"])

    return selected_epoch, best


def load_subset_size(seed, fraction):
    archive_path = SUBSET_ROOT / f"seed_{seed}.npz"
    require(archive_path.exists(), f"Missing subset archive: {archive_path}")

    key = f"fraction_{fraction:03d}"

    with np.load(archive_path) as archive:
        require(key in archive.files, f"Missing subset key {key}: {archive_path}")
        return int(len(archive[key]))


def audit_run(seed, fraction):
    run_dir = RUN_ROOT / f"fraction_{fraction:03d}" / f"seed_{seed}"
    checkpoint_path = run_dir / "best.pt"
    history_path = run_dir / "history.json"

    require(checkpoint_path.exists(), f"Missing checkpoint: {checkpoint_path}")
    require(history_path.exists(), f"Missing history: {history_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    history = json.loads(history_path.read_text(encoding="utf-8"))

    require(isinstance(checkpoint, dict), f"Invalid checkpoint: {checkpoint_path}")
    require(isinstance(history, list), f"Invalid history: {history_path}")
    require(len(history) > 0, f"Empty history: {history_path}")

    selected_epoch = int(checkpoint["epoch"])
    stopped_epoch = int(history[-1]["epoch"])
    metrics = checkpoint["metrics"]

    epochs = [int(record["epoch"]) for record in history]
    require(
        epochs == list(range(1, stopped_epoch + 1)),
        f"Non-consecutive epochs: fraction={fraction}, seed={seed}",
    )

    require(
        selected_epoch <= stopped_epoch,
        f"Selected epoch after stopped epoch: fraction={fraction}, seed={seed}",
    )

    selected_records = [
        record for record in history if int(record["epoch"]) == selected_epoch
    ]

    require(
        len(selected_records) == 1,
        f"Selected epoch missing from history: fraction={fraction}, seed={seed}",
    )

    selected_record = selected_records[0]

    for metric in SCALAR_METRICS:
        require(metric in metrics, f"Missing metric {metric}: {checkpoint_path}")
        require(metric in selected_record, f"Missing history metric {metric}")
        require(
            values_match(metrics[metric], selected_record[metric]),
            f"Checkpoint/history mismatch for {metric}: fraction={fraction}, seed={seed}",
        )

    per_class_roc = metrics["per_class_roc_auc"]
    per_class_ap = metrics["per_class_average_precision"]

    require(
        len(per_class_roc) == EXPECTED_CLASSES,
        f"Invalid ROC-AUC class count: fraction={fraction}, seed={seed}",
    )
    require(
        len(per_class_ap) == EXPECTED_CLASSES,
        f"Invalid AP class count: fraction={fraction}, seed={seed}",
    )

    reconstructed_epoch, reconstructed_score = reconstruct_selected_epoch(history)

    require(
        reconstructed_epoch == selected_epoch,
        f"Checkpoint selection mismatch: fraction={fraction}, seed={seed}, checkpoint={selected_epoch}, reconstructed={reconstructed_epoch}",
    )

    require(
        values_match(reconstructed_score, metrics["macro_roc_auc"]),
        f"Best score mismatch: fraction={fraction}, seed={seed}",
    )

    row = {
        "model": "dinov2_linear_probe",
        "fraction_percent": fraction,
        "seed": seed,
        "train_samples": load_subset_size(seed, fraction),
        "selected_epoch": selected_epoch,
        "stopped_epoch": stopped_epoch,
    }

    for metric in SCALAR_METRICS:
        row[metric] = float(metrics[metric])

    for index, class_name in enumerate(CLASS_NAMES):
        row[f"{class_name}_roc_auc"] = float(per_class_roc[index])
        row[f"{class_name}_average_precision"] = float(per_class_ap[index])

    return row


def build_summary(runs):
    metrics = (
        "macro_roc_auc",
        "micro_roc_auc",
        "macro_average_precision",
        "micro_average_precision",
    )

    rows = []

    for fraction in FRACTIONS:
        group = runs[runs["fraction_percent"] == fraction]

        require(
            len(group) == len(SEEDS),
            f"Unexpected number of runs for fraction {fraction}",
        )

        row = {
            "model": "dinov2_linear_probe",
            "fraction_percent": fraction,
            "train_samples": int(group["train_samples"].iloc[0]),
            "n_runs": int(len(group)),
            "selected_epoch_mean": float(group["selected_epoch"].mean()),
            "selected_epoch_std": float(group["selected_epoch"].std(ddof=1)),
            "stopped_epoch_mean": float(group["stopped_epoch"].mean()),
            "stopped_epoch_std": float(group["stopped_epoch"].std(ddof=1)),
        }

        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    rows = []

    for fraction in FRACTIONS:
        for seed in SEEDS:
            rows.append(audit_run(seed, fraction))

    require(len(rows) == 15, "Expected exactly 15 runs")

    runs = pd.DataFrame(rows).sort_values(
        ["fraction_percent", "seed"]
    ).reset_index(drop=True)

    summary = build_summary(runs)

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    runs_path = RESULTS_ROOT / "dinov2_runs.csv"
    summary_path = RESULTS_ROOT / "dinov2_summary.csv"

    runs.to_csv(runs_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("Audit passed: 15/15 runs")
    print()
    print(
        summary[
            [
                "fraction_percent",
                "train_samples",
                "macro_roc_auc_mean",
                "macro_roc_auc_std",
            ]
        ].to_string(index=False)
    )
    print()
    print(f"Runs: {runs_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()