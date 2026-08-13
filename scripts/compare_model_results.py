from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare ResNet18 and DINOv2 ChestMNIST results."
    )

    parser.add_argument(
        "--resnet-overall",
        type=Path,
        default=PROJECT_ROOT
        / "results"
        / "resnet18"
        / "overall_metrics.csv",
    )

    parser.add_argument(
        "--resnet-per-class",
        type=Path,
        default=PROJECT_ROOT
        / "results"
        / "resnet18"
        / "per_class_comparison.csv",
    )

    parser.add_argument(
        "--dinov2-overall",
        type=Path,
        default=PROJECT_ROOT
        / "results"
        / "dinov2_linear_probe"
        / "overall_metrics.csv",
    )

    parser.add_argument(
        "--dinov2-per-class",
        type=Path,
        default=PROJECT_ROOT
        / "results"
        / "dinov2_linear_probe"
        / "per_class_metrics.csv",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "results"
        / "model_comparison",
    )

    return parser.parse_args()


def load_csv(
    path: Path,
) -> list[dict]:
    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        return list(
            csv.DictReader(file)
        )


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
    resnet_rows: list[dict],
    dinov2_rows: list[dict],
) -> list[dict]:
    if len(resnet_rows) != 2:
        raise ValueError(
            "Expected two ResNet result rows."
        )

    if len(dinov2_rows) != 1:
        raise ValueError(
            "Expected one DINOv2 result row."
        )

    rows = []

    for item in resnet_rows:
        model = item["model"]

        if model == "resnet18_unweighted":
            regime = "end_to_end_finetuning"
            role = "primary_baseline"
        elif model == "resnet18_weighted":
            regime = "end_to_end_finetuning"
            role = "imbalance_ablation"
        else:
            raise ValueError(
                f"Unexpected ResNet model: {model}"
            )

        rows.append(
            {
                "model": model,
                "adaptation_regime": regime,
                "comparison_role": role,
                "checkpoint_epoch": item[
                    "checkpoint_epoch"
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

    dino = dinov2_rows[0]

    rows.append(
        {
            "model": dino["model"],
            "adaptation_regime": "frozen_linear_probe",
            "comparison_role": "foundation_model_probe",
            "checkpoint_epoch": dino[
                "checkpoint_epoch"
            ],
            "macro_roc_auc": dino[
                "macro_roc_auc"
            ],
            "micro_roc_auc": dino[
                "micro_roc_auc"
            ],
            "macro_average_precision": dino[
                "macro_average_precision"
            ],
            "micro_average_precision": dino[
                "micro_average_precision"
            ],
        }
    )

    return rows


def build_primary_summary(
    overall_rows: list[dict],
) -> list[dict]:
    resnet = next(
        row
        for row in overall_rows
        if row["comparison_role"]
        == "primary_baseline"
    )

    dinov2 = next(
        row
        for row in overall_rows
        if row["comparison_role"]
        == "foundation_model_probe"
    )

    metric_names = [
        "macro_roc_auc",
        "micro_roc_auc",
        "macro_average_precision",
        "micro_average_precision",
    ]

    rows = []

    for metric in metric_names:
        resnet_value = float(
            resnet[metric]
        )
        dinov2_value = float(
            dinov2[metric]
        )

        rows.append(
            {
                "metric": metric,
                "resnet18_unweighted": resnet_value,
                "dinov2_linear_probe": dinov2_value,
                "delta_dinov2_minus_resnet18": (
                    dinov2_value
                    - resnet_value
                ),
            }
        )

    return rows


def build_per_class_rows(
    resnet_rows: list[dict],
    dinov2_rows: list[dict],
) -> list[dict]:
    if len(resnet_rows) != len(
        dinov2_rows
    ):
        raise ValueError(
            "Per-class result lengths differ."
        )

    dino_by_index = {
        int(row["class_index"]): row
        for row in dinov2_rows
    }

    rows = []

    for resnet in resnet_rows:
        index = int(
            resnet["class_index"]
        )

        if index not in dino_by_index:
            raise ValueError(
                f"Missing DINOv2 class index {index}."
            )

        dinov2 = dino_by_index[
            index
        ]

        if (
            resnet["class_name"]
            != dinov2["class_name"]
        ):
            raise ValueError(
                f"Class name mismatch at index {index}."
            )

        resnet_roc = float(
            resnet[
                "unweighted_roc_auc"
            ]
        )

        dinov2_roc = float(
            dinov2[
                "roc_auc"
            ]
        )

        resnet_ap = float(
            resnet[
                "unweighted_average_precision"
            ]
        )

        dinov2_ap = float(
            dinov2[
                "average_precision"
            ]
        )

        rows.append(
            {
                "class_index": index,
                "class_name": resnet[
                    "class_name"
                ],
                "resnet18_roc_auc": resnet_roc,
                "dinov2_roc_auc": dinov2_roc,
                "roc_auc_delta_dinov2_minus_resnet18": (
                    dinov2_roc
                    - resnet_roc
                ),
                "resnet18_average_precision": resnet_ap,
                "dinov2_average_precision": dinov2_ap,
                "average_precision_delta_dinov2_minus_resnet18": (
                    dinov2_ap
                    - resnet_ap
                ),
            }
        )

    return rows


def plot_overall_metrics(
    overall_rows: list[dict],
    output_path: Path,
) -> None:
    primary = [
        row
        for row in overall_rows
        if row["comparison_role"]
        != "imbalance_ablation"
    ]

    labels = [
        "ResNet18 fine-tuned",
        "DINOv2 frozen probe",
    ]

    metrics = [
        "macro_roc_auc",
        "micro_roc_auc",
        "macro_average_precision",
        "micro_average_precision",
    ]

    metric_labels = [
        "Macro ROC-AUC",
        "Micro ROC-AUC",
        "Macro AP",
        "Micro AP",
    ]

    positions = list(
        range(len(metrics))
    )

    width = 0.36

    plt.figure(
        figsize=(9, 5)
    )

    for model_index, row in enumerate(
        primary
    ):
        offsets = [
            position
            + (
                model_index
                - 0.5
            )
            * width
            for position in positions
        ]

        plt.bar(
            offsets,
            [
                float(row[metric])
                for metric in metrics
            ],
            width=width,
            label=labels[
                model_index
            ],
        )

    plt.xticks(
        positions,
        metric_labels,
    )
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.title(
        "ChestMNIST Full-Label Test Performance"
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
    positions = list(
        range(len(rows))
    )

    width = 0.36

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        [
            position
            - width / 2
            for position in positions
        ],
        [
            row["resnet18_roc_auc"]
            for row in rows
        ],
        width=width,
        label="ResNet18 fine-tuned",
    )

    plt.bar(
        [
            position
            + width / 2
            for position in positions
        ],
        [
            row["dinov2_roc_auc"]
            for row in rows
        ],
        width=width,
        label="DINOv2 frozen probe",
    )

    plt.xticks(
        positions,
        [
            row["class_name"]
            for row in rows
        ],
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
    positions = list(
        range(len(rows))
    )

    width = 0.36

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        [
            position
            - width / 2
            for position in positions
        ],
        [
            row[
                "resnet18_average_precision"
            ]
            for row in rows
        ],
        width=width,
        label="ResNet18 fine-tuned",
    )

    plt.bar(
        [
            position
            + width / 2
            for position in positions
        ],
        [
            row[
                "dinov2_average_precision"
            ]
            for row in rows
        ],
        width=width,
        label="DINOv2 frozen probe",
    )

    plt.xticks(
        positions,
        [
            row["class_name"]
            for row in rows
        ],
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

    resnet_overall = load_csv(
        args.resnet_overall
    )

    resnet_per_class = load_csv(
        args.resnet_per_class
    )

    dinov2_overall = load_csv(
        args.dinov2_overall
    )

    dinov2_per_class = load_csv(
        args.dinov2_per_class
    )

    overall_rows = build_overall_rows(
        resnet_overall,
        dinov2_overall,
    )

    primary_summary = build_primary_summary(
        overall_rows
    )

    per_class_rows = build_per_class_rows(
        resnet_per_class,
        dinov2_per_class,
    )

    save_csv(
        args.output_dir
        / "overall_metrics.csv",
        overall_rows,
        [
            "model",
            "adaptation_regime",
            "comparison_role",
            "checkpoint_epoch",
            "macro_roc_auc",
            "micro_roc_auc",
            "macro_average_precision",
            "micro_average_precision",
        ],
    )

    save_csv(
        args.output_dir
        / "primary_comparison.csv",
        primary_summary,
        [
            "metric",
            "resnet18_unweighted",
            "dinov2_linear_probe",
            "delta_dinov2_minus_resnet18",
        ],
    )

    save_csv(
        args.output_dir
        / "per_class_comparison.csv",
        per_class_rows,
        [
            "class_index",
            "class_name",
            "resnet18_roc_auc",
            "dinov2_roc_auc",
            "roc_auc_delta_dinov2_minus_resnet18",
            "resnet18_average_precision",
            "dinov2_average_precision",
            "average_precision_delta_dinov2_minus_resnet18",
        ],
    )

    plot_overall_metrics(
        overall_rows,
        args.output_dir
        / "overall_metrics.png",
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
        "Created: primary_comparison.csv"
    )
    print(
        "Created: per_class_comparison.csv"
    )
    print(
        "Created: overall_metrics.png"
    )
    print(
        "Created: test_per_class_roc_auc.png"
    )
    print(
        "Created: test_per_class_average_precision.png"
    )


if __name__ == "__main__":
    main()