from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize and visualize ResNet18 ChestMNIST experiments."
    )

    parser.add_argument(
        "--unweighted-history",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "resnet18_unweighted_final"
        / "history.json",
    )

    parser.add_argument(
        "--weighted-history",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "resnet18_weighted_final"
        / "history.json",
    )

    parser.add_argument(
        "--unweighted-test",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "resnet18_unweighted_final_eval"
        / "test_metrics.json",
    )

    parser.add_argument(
        "--weighted-test",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "resnet18_weighted_final_eval"
        / "test_metrics.json",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "resnet18_comparison",
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
    unweighted_test: dict,
    weighted_test: dict,
) -> list[dict]:
    return [
        {
            "model": "resnet18_unweighted",
            "checkpoint_epoch": unweighted_test[
                "checkpoint_epoch"
            ],
            "macro_roc_auc": unweighted_test[
                "macro_roc_auc"
            ],
            "micro_roc_auc": unweighted_test[
                "micro_roc_auc"
            ],
            "macro_average_precision": unweighted_test[
                "macro_average_precision"
            ],
            "micro_average_precision": unweighted_test[
                "micro_average_precision"
            ],
        },
        {
            "model": "resnet18_weighted",
            "checkpoint_epoch": weighted_test[
                "checkpoint_epoch"
            ],
            "macro_roc_auc": weighted_test[
                "macro_roc_auc"
            ],
            "micro_roc_auc": weighted_test[
                "micro_roc_auc"
            ],
            "macro_average_precision": weighted_test[
                "macro_average_precision"
            ],
            "micro_average_precision": weighted_test[
                "micro_average_precision"
            ],
        },
    ]


def build_per_class_rows(
    unweighted_test: dict,
    weighted_test: dict,
) -> list[dict]:
    rows = []

    for unweighted, weighted in zip(
        unweighted_test["per_class"],
        weighted_test["per_class"],
        strict=True,
    ):
        if (
            unweighted["class_index"]
            != weighted["class_index"]
        ):
            raise ValueError(
                "Class index mismatch."
            )

        if (
            unweighted["class_name"]
            != weighted["class_name"]
        ):
            raise ValueError(
                "Class name mismatch."
            )

        unweighted_roc = unweighted[
            "roc_auc"
        ]
        weighted_roc = weighted[
            "roc_auc"
        ]
        unweighted_ap = unweighted[
            "average_precision"
        ]
        weighted_ap = weighted[
            "average_precision"
        ]

        rows.append(
            {
                "class_index": unweighted[
                    "class_index"
                ],
                "class_name": unweighted[
                    "class_name"
                ],
                "unweighted_roc_auc": unweighted_roc,
                "weighted_roc_auc": weighted_roc,
                "roc_auc_delta_weighted_minus_unweighted": (
                    weighted_roc
                    - unweighted_roc
                ),
                "unweighted_average_precision": unweighted_ap,
                "weighted_average_precision": weighted_ap,
                "average_precision_delta_weighted_minus_unweighted": (
                    weighted_ap
                    - unweighted_ap
                ),
            }
        )

    return rows


def build_history_rows(
    unweighted_history: list[dict],
    weighted_history: list[dict],
) -> list[dict]:
    rows = []

    for model_name, history in [
        (
            "resnet18_unweighted",
            unweighted_history,
        ),
        (
            "resnet18_weighted",
            weighted_history,
        ),
    ]:
        for item in history:
            rows.append(
                {
                    "model": model_name,
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
    unweighted_history: list[dict],
    weighted_history: list[dict],
    output_path: Path,
) -> None:
    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        [
            item["epoch"]
            for item in unweighted_history
        ],
        [
            item["train_loss"]
            for item in unweighted_history
        ],
        marker="o",
        label="Unweighted Train",
    )

    plt.plot(
        [
            item["epoch"]
            for item in unweighted_history
        ],
        [
            item["val_loss"]
            for item in unweighted_history
        ],
        marker="o",
        label="Unweighted Validation",
    )

    plt.plot(
        [
            item["epoch"]
            for item in weighted_history
        ],
        [
            item["train_loss"]
            for item in weighted_history
        ],
        marker="o",
        label="Weighted Train",
    )

    plt.plot(
        [
            item["epoch"]
            for item in weighted_history
        ],
        [
            item["val_loss"]
            for item in weighted_history
        ],
        marker="o",
        label="Weighted Validation",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ResNet18 Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=200,
    )
    plt.close()


def plot_validation_macro_roc_auc(
    unweighted_history: list[dict],
    weighted_history: list[dict],
    output_path: Path,
) -> None:
    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        [
            item["epoch"]
            for item in unweighted_history
        ],
        [
            item["macro_roc_auc"]
            for item in unweighted_history
        ],
        marker="o",
        label="Unweighted",
    )

    plt.plot(
        [
            item["epoch"]
            for item in weighted_history
        ],
        [
            item["macro_roc_auc"]
            for item in weighted_history
        ],
        marker="o",
        label="Weighted",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Macro ROC-AUC")
    plt.title(
        "Validation Macro ROC-AUC"
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

    positions = list(
        range(len(rows))
    )

    width = 0.38

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        [
            position - width / 2
            for position in positions
        ],
        [
            row["unweighted_roc_auc"]
            for row in rows
        ],
        width=width,
        label="Unweighted",
    )

    plt.bar(
        [
            position + width / 2
            for position in positions
        ],
        [
            row["weighted_roc_auc"]
            for row in rows
        ],
        width=width,
        label="Weighted",
    )

    plt.xticks(
        positions,
        class_names,
        rotation=45,
        ha="right",
    )
    plt.ylabel("ROC-AUC")
    plt.title(
        "Test ROC-AUC by ChestMNIST Class"
    )
    plt.legend()
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

    positions = list(
        range(len(rows))
    )

    width = 0.38

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        [
            position - width / 2
            for position in positions
        ],
        [
            row[
                "unweighted_average_precision"
            ]
            for row in rows
        ],
        width=width,
        label="Unweighted",
    )

    plt.bar(
        [
            position + width / 2
            for position in positions
        ],
        [
            row[
                "weighted_average_precision"
            ]
            for row in rows
        ],
        width=width,
        label="Weighted",
    )

    plt.xticks(
        positions,
        class_names,
        rotation=45,
        ha="right",
    )
    plt.ylabel("Average Precision")
    plt.title(
        "Test Average Precision by ChestMNIST Class"
    )
    plt.legend()
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

    unweighted_history = load_json(
        args.unweighted_history
    )

    weighted_history = load_json(
        args.weighted_history
    )

    unweighted_test = load_json(
        args.unweighted_test
    )

    weighted_test = load_json(
        args.weighted_test
    )

    overall_rows = build_overall_rows(
        unweighted_test,
        weighted_test,
    )

    per_class_rows = build_per_class_rows(
        unweighted_test,
        weighted_test,
    )

    history_rows = build_history_rows(
        unweighted_history,
        weighted_history,
    )

    save_csv(
        args.output_dir
        / "overall_metrics.csv",
        overall_rows,
        [
            "model",
            "checkpoint_epoch",
            "macro_roc_auc",
            "micro_roc_auc",
            "macro_average_precision",
            "micro_average_precision",
        ],
    )

    save_csv(
        args.output_dir
        / "per_class_comparison.csv",
        per_class_rows,
        [
            "class_index",
            "class_name",
            "unweighted_roc_auc",
            "weighted_roc_auc",
            "roc_auc_delta_weighted_minus_unweighted",
            "unweighted_average_precision",
            "weighted_average_precision",
            "average_precision_delta_weighted_minus_unweighted",
        ],
    )

    save_csv(
        args.output_dir
        / "training_history.csv",
        history_rows,
        [
            "model",
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
        unweighted_history,
        weighted_history,
        args.output_dir
        / "training_loss.png",
    )

    plot_validation_macro_roc_auc(
        unweighted_history,
        weighted_history,
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
        "Created: per_class_comparison.csv"
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