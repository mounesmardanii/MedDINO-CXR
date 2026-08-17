import csv
import json
import math
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "final_test"
LABEL_RESULTS_ROOT = PROJECT_ROOT / "results" / "label_efficiency"
RESULTS_ROOT = PROJECT_ROOT / "results" / "final_test"

FRACTIONS = (5, 10, 25, 50, 100)
SEEDS = (42, 47, 52)
EXPECTED_SAMPLES = 22433
EXPECTED_CLASSES = 14

MODELS = {
    "dinov2": {
        "output_dir": "dinov2",
        "validation_runs": LABEL_RESULTS_ROOT / "dinov2_runs.csv",
    },
    "resnet18": {
        "output_dir": "resnet18",
        "validation_runs": LABEL_RESULTS_ROOT / "resnet18_runs.csv",
    },
}

METRICS = (
    "macro_roc_auc",
    "micro_roc_auc",
    "macro_average_precision",
    "micro_average_precision",
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def values_match(a, b, tolerance=1e-10):
    return math.isclose(
        float(a),
        float(b),
        rel_tol=0.0,
        abs_tol=tolerance,
    )


def load_validation_runs(path):
    require(
        path.is_file(),
        f"Missing validation runs: {path}",
    )

    data = pd.read_csv(path)

    require(
        len(data) == 15,
        f"Expected 15 validation runs: {path}",
    )

    return data


def audit_run(
    model,
    config,
    validation_runs,
    fraction,
    seed,
):
    run_dir = (
        OUTPUT_ROOT
        / config["output_dir"]
        / f"fraction_{fraction:03d}"
        / f"seed_{seed}"
    )

    metrics_path = run_dir / "test_metrics.json"
    per_class_path = run_dir / "test_per_class_metrics.csv"

    require(
        metrics_path.is_file(),
        f"Missing metrics: {metrics_path}",
    )

    require(
        per_class_path.is_file(),
        f"Missing per-class CSV: {per_class_path}",
    )

    with metrics_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metrics = json.load(file)

    with per_class_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        per_class_rows = list(
            csv.DictReader(file)
        )

    require(
        metrics["split"] == "test",
        f"Invalid split: {metrics_path}",
    )

    require(
        int(metrics["samples"]) == EXPECTED_SAMPLES,
        f"Invalid sample count: {metrics_path}",
    )

    require(
        len(metrics["per_class"]) == EXPECTED_CLASSES,
        f"Invalid JSON class count: {metrics_path}",
    )

    require(
        len(per_class_rows) == EXPECTED_CLASSES,
        f"Invalid CSV class count: {per_class_path}",
    )

    validation_match = validation_runs[
        (
            validation_runs["fraction_percent"]
            == fraction
        )
        & (
            validation_runs["seed"]
            == seed
        )
    ]

    require(
        len(validation_match) == 1,
        f"Validation run missing: model={model}, fraction={fraction}, seed={seed}",
    )

    expected_epoch = int(
        validation_match.iloc[0]["selected_epoch"]
    )

    require(
        int(metrics["checkpoint_epoch"]) == expected_epoch,
        f"Checkpoint epoch mismatch: model={model}, fraction={fraction}, seed={seed}",
    )

    for metric in METRICS:
        require(
            metric in metrics,
            f"Missing metric {metric}: {metrics_path}",
        )

    row = {
        "model": model,
        "fraction_percent": fraction,
        "seed": seed,
        "samples": int(metrics["samples"]),
        "checkpoint_epoch": int(
            metrics["checkpoint_epoch"]
        ),
    }

    for metric in METRICS:
        row[metric] = float(
            metrics[metric]
        )

    per_class_output = []

    for index in range(EXPECTED_CLASSES):
        json_row = metrics["per_class"][index]
        csv_row = per_class_rows[index]

        require(
            int(json_row["class_index"]) == index,
            f"Invalid JSON class index: {metrics_path}",
        )

        require(
            int(csv_row["class_index"]) == index,
            f"Invalid CSV class index: {per_class_path}",
        )

        require(
            json_row["class_name"]
            == csv_row["class_name"],
            f"Class name mismatch: {per_class_path}",
        )

        require(
            values_match(
                json_row["roc_auc"],
                csv_row["roc_auc"],
            ),
            f"ROC-AUC mismatch: {per_class_path}",
        )

        require(
            values_match(
                json_row["average_precision"],
                csv_row["average_precision"],
            ),
            f"AP mismatch: {per_class_path}",
        )

        per_class_output.append(
            {
                "model": model,
                "fraction_percent": fraction,
                "seed": seed,
                "class_index": index,
                "class_name": json_row["class_name"],
                "roc_auc": float(
                    json_row["roc_auc"]
                ),
                "average_precision": float(
                    json_row["average_precision"]
                ),
            }
        )

    return row, per_class_output


def build_summary(runs):
    rows = []

    for model in MODELS:
        for fraction in FRACTIONS:
            group = runs[
                (runs["model"] == model)
                & (
                    runs["fraction_percent"]
                    == fraction
                )
            ]

            require(
                len(group) == 3,
                f"Expected 3 test runs: model={model}, fraction={fraction}",
            )

            row = {
                "model": model,
                "fraction_percent": fraction,
                "samples": EXPECTED_SAMPLES,
                "n_runs": len(group),
            }

            for metric in METRICS:
                row[
                    f"{metric}_mean"
                ] = float(
                    group[metric].mean()
                )

                row[
                    f"{metric}_std"
                ] = float(
                    group[metric].std(
                        ddof=1
                    )
                )

            rows.append(row)

    return pd.DataFrame(rows)


def build_comparison(runs):
    dino = runs[
        runs["model"] == "dinov2"
    ].copy()

    resnet = runs[
        runs["model"] == "resnet18"
    ].copy()

    columns = [
        "fraction_percent",
        "seed",
        *METRICS,
    ]

    paired = dino[columns].merge(
        resnet[columns],
        on=[
            "fraction_percent",
            "seed",
        ],
        suffixes=(
            "_dinov2",
            "_resnet18",
        ),
        validate="one_to_one",
    )

    rows = []

    for fraction in FRACTIONS:
        group = paired[
            paired["fraction_percent"]
            == fraction
        ]

        row = {
            "fraction_percent": fraction,
            "n_pairs": len(group),
        }

        for metric in METRICS:
            gap = (
                group[
                    f"{metric}_resnet18"
                ]
                - group[
                    f"{metric}_dinov2"
                ]
            )

            row[
                f"{metric}_dinov2_mean"
            ] = float(
                group[
                    f"{metric}_dinov2"
                ].mean()
            )

            row[
                f"{metric}_resnet18_mean"
            ] = float(
                group[
                    f"{metric}_resnet18"
                ].mean()
            )

            row[
                f"{metric}_gap_mean"
            ] = float(
                gap.mean()
            )

            row[
                f"{metric}_gap_std"
            ] = float(
                gap.std(ddof=1)
            )

            row[
                f"{metric}_resnet18_wins"
            ] = int(
                (gap > 0).sum()
            )

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    run_rows = []
    per_class_rows = []

    validation_data = {
        model: load_validation_runs(
            config["validation_runs"]
        )
        for model, config in MODELS.items()
    }

    for model, config in MODELS.items():
        for fraction in FRACTIONS:
            for seed in SEEDS:
                run_row, class_rows = audit_run(
                    model,
                    config,
                    validation_data[model],
                    fraction,
                    seed,
                )

                run_rows.append(
                    run_row
                )

                per_class_rows.extend(
                    class_rows
                )

    require(
        len(run_rows) == 30,
        "Expected exactly 30 audited test runs",
    )

    require(
        len(per_class_rows) == 420,
        "Expected exactly 420 per-class rows",
    )

    runs = pd.DataFrame(
        run_rows
    ).sort_values(
        [
            "model",
            "fraction_percent",
            "seed",
        ]
    ).reset_index(
        drop=True
    )

    per_class = pd.DataFrame(
        per_class_rows
    )

    summary = build_summary(
        runs
    )

    comparison = build_comparison(
        runs
    )

    RESULTS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    runs_path = (
        RESULTS_ROOT
        / "final_test_runs.csv"
    )

    summary_path = (
        RESULTS_ROOT
        / "final_test_summary.csv"
    )

    comparison_path = (
        RESULTS_ROOT
        / "final_test_comparison.csv"
    )

    per_class_path = (
        RESULTS_ROOT
        / "final_test_per_class.csv"
    )

    runs.to_csv(
        runs_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    per_class.to_csv(
        per_class_path,
        index=False,
    )

    print(
        "Final test audit passed: 30/30 evaluations"
    )

    print(
        "Per-class audit passed: 420/420 rows"
    )

    print()

    print(
        comparison[
            [
                "fraction_percent",
                "macro_roc_auc_dinov2_mean",
                "macro_roc_auc_resnet18_mean",
                "macro_roc_auc_gap_mean",
                "macro_roc_auc_gap_std",
                "macro_roc_auc_resnet18_wins",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print(
        f"Runs: {runs_path}"
    )

    print(
        f"Summary: {summary_path}"
    )

    print(
        f"Comparison: {comparison_path}"
    )

    print(
        f"Per-class: {per_class_path}"
    )


if __name__ == "__main__":
    main()