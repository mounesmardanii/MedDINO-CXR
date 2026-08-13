from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize and visualize DINOv2 linear probe results."
    )

    parser.add_argument(
        "--history",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "dinov2_linear_probe"
        / "history.json",
    )

    parser.add_argument(
        "--test-metrics",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "dinov2_linear_probe_eval"
        / "test_metrics.json",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "results"
        / "dinov2_linear_probe",
    )

    return parser.parse_args()


def load_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_csv(
    path: Path,
    rows: list[dict],
    fieldnames: list[str],
) -> None:
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
        writer.writerows(rows)


def build_overall_rows(
    test_metrics: dict,
) -> list[dict]:
    return [
        {
            "model": test_metrics["model"],
            "checkpoint_epoch": test_metrics[
                "checkpoint_epoch"
            ],
            "samples": test_metrics["samples"],
            "macro_roc_auc": test_metrics[
                "macro_roc_auc"
            ],
            "micro_roc_auc": test_metrics[
                "micro_roc_auc"
            ],
            "macro_average_precision": test_metrics[
                "macro_average_precision"
            ],
            "micro_average_precision": test_metrics[
                "micro_average_precision"
            ],
        }
    ]


def build_per_class_rows(
    test_metrics: dict,
) -> list[dict]:
    rows = []

    for item in test_metrics["per_class"]:
        rows.append(
            {
                "class_index": item[
                    "class_index"
                ],
                "class_name": item[
                    "class_name"
                ],
                "roc_auc": item[
                    "roc_auc"
                ],
                "average_precision": item[
                    "average_precision"
                ],
            }
        )

    return rows


def build_history_rows(
    history: list[dict],
) -> list[dict]:
    rows = []

    for item in history:
        rows.append(
            {
                "epoch": item["epoch"],
                "train_loss": item[
                    "train_loss"
                ],
                "val_loss": item[
                    "val_loss"
                ],
                "macro_roc_auc": item[
                    "macro_roc_auc"
                ],
                "micro_roc_auc": item[
                    "micro_roc_auc"
                ],
                "macro_average_precision": item[
                    "macro_average_precision"
                ],
                "micro_average_precision": item[
                    "micro_average_precision"
                ],
            }
        )

    return rows


def plot_training_loss(
    history: list[dict],
    selected_epoch: int,
    output_path: Path,
) -> None:
    epochs = [
        item["epoch"]
        for item in history
    ]

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        [
            item["train_loss"]
            for item in history
        ],
        label="Train",
    )

    plt.plot(
        epochs,
        [
            item["val_loss"]
            for item in history
        ],
        label="Validation",
    )

    plt.axvline(
        selected_epoch,
        linestyle="--",
        label=f"Selected epoch ({selected_epoch})",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        "DINOv2 Linear Probe Training and Validation Loss"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=200,
    )
    plt.close()


def plot_validation_macro_roc_auc(
    history: list[dict],
    selected_epoch: int,
    output_path: Path,
) -> None:
    epochs = [
        item["epoch"]
        for item in history
    ]

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        [
            item["macro_roc_auc"]
            for item in history
        ],
    )

    plt.axvline(
        selected_epoch,
        linestyle="--",
        label=f"Selected epoch ({selected_epoch})",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Macro ROC-AUC")
    plt.title(
        "DINOv2 Linear Probe Validation Macro ROC-AUC"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=200,
    )
    plt.close()


def plot_per_class_roc_auc(
    rows: list[dict],
    output_path: Path,
) -> None:
    class_names = [
        row["class_name"]
        for row in rows
    ]

    values = [
        row["roc_auc"]
        for row in rows
    ]

    positions = list(
        range(len(rows))
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        positions,
        values,
    )

    plt.xticks(
        positions,
        class_names,
        rotation=45,
        ha="right",
    )
    plt.ylabel("ROC-AUC")
    plt.title(
        "DINOv2 Linear Probe Test ROC-AUC by ChestMNIST Class"
    )
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=200,
    )
    plt.close()


def plot_per_class_average_precision(
    rows: list[dict],
    output_path: Path,
) -> None:
    class_names = [
        row["class_name"]
        for row in rows
    ]

    values = [
        row["average_precision"]
        for row in rows
    ]

    positions = list(
        range(len(rows))
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        positions,
        values,
    )

    plt.xticks(
        positions,
        class_names,
        rotation=45,
        ha="right",
    )
    plt.ylabel("Average Precision")
    plt.title(
        "DINOv2 Linear Probe Test Average Precision by ChestMNIST Class"
    )
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=200,
    )
    plt.close()


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    history = load_json(
        args.history
    )

    test_metrics = load_json(
        args.test_metrics
    )

    overall_rows = build_overall_rows(
        test_metrics
    )

    per_class_rows = build_per_class_rows(
        test_metrics
    )

    history_rows = build_history_rows(
        history
    )

    selected_epoch = int(
        test_metrics["checkpoint_epoch"]
    )

    save_csv(
        args.output_dir
        / "overall_metrics.csv",
        overall_rows,
        [
            "model",
            "checkpoint_epoch",
            "samples",
            "macro_roc_auc",
            "micro_roc_auc",
            "macro_average_precision",
            "micro_average_precision",
        ],
    )

    save_csv(
        args.output_dir
        / "per_class_metrics.csv",
        per_class_rows,
        [
            "class_index",
            "class_name",
            "roc_auc",
            "average_precision",
        ],
    )

    save_csv(
        args.output_dir
        / "training_history.csv",
        history_rows,
        [
            "epoch",
            "train_loss",
            "val_loss",
            "macro_roc_auc",
            "micro_roc_auc",
            "macro_average_precision",
            "micro_average_precision",
        ],
    )

    plot_training_loss(
        history,
        selected_epoch,
        args.output_dir
        / "training_loss.png",
    )

    plot_validation_macro_roc_auc(
        history,
        selected_epoch,
        args.output_dir
        / "validation_macro_roc_auc.png",
    )

    plot_per_class_roc_auc(
        per_class_rows,
        args.output_dir
        / "test_per_class_roc_auc.png",
    )

    plot_per_class_average_precision(
        per_class_rows,
        args.output_dir
        / "test_per_class_average_precision.png",
    )

    print(
        "Results directory: "
        f"{args.output_dir}"
    )
    print(
        "Created: overall_metrics.csv"
    )
    print(
        "Created: per_class_metrics.csv"
    )
    print(
        "Created: training_history.csv"
    )
    print(
        "Created: training_loss.png"
    )
    print(
        "Created: validation_macro_roc_auc.png"
    )
    print(
        "Created: test_per_class_roc_auc.png"
    )
    print(
        "Created: test_per_class_average_precision.png"
    )


if __name__ == "__main__":
    main()