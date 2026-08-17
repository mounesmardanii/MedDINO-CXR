from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results" / "label_efficiency"

DINO_SUMMARY_PATH = RESULTS_ROOT / "dinov2_summary.csv"
RESNET_SUMMARY_PATH = RESULTS_ROOT / "resnet18_summary.csv"
DINO_RUNS_PATH = RESULTS_ROOT / "dinov2_runs.csv"
RESNET_RUNS_PATH = RESULTS_ROOT / "resnet18_runs.csv"

SUMMARY_OUTPUT_PATH = RESULTS_ROOT / "model_comparison_summary.csv"
RUNS_OUTPUT_PATH = RESULTS_ROOT / "model_comparison_runs.csv"

FRACTIONS = (5, 10, 25, 50, 100)
SEEDS = (42, 47, 52)

METRICS = (
    "macro_roc_auc",
    "micro_roc_auc",
    "macro_average_precision",
    "micro_average_precision",
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_csv(path):
    require(path.exists(), f"Missing file: {path}")
    return pd.read_csv(path)


def validate_summary(dino, resnet):
    require(len(dino) == 5, "DINOv2 summary must contain 5 rows")
    require(len(resnet) == 5, "ResNet summary must contain 5 rows")

    require(
        tuple(dino["fraction_percent"].tolist()) == FRACTIONS,
        "Unexpected DINOv2 fractions",
    )

    require(
        tuple(resnet["fraction_percent"].tolist()) == FRACTIONS,
        "Unexpected ResNet fractions",
    )

    require(
        dino["train_samples"].tolist()
        == resnet["train_samples"].tolist(),
        "Summary train sample counts do not match",
    )

    require(
        (dino["n_runs"] == 3).all(),
        "Unexpected DINOv2 run count",
    )

    require(
        (resnet["n_runs"] == 3).all(),
        "Unexpected ResNet run count",
    )


def validate_runs(dino, resnet):
    require(len(dino) == 15, "DINOv2 runs must contain 15 rows")
    require(len(resnet) == 15, "ResNet runs must contain 15 rows")

    expected_pairs = {
        (fraction, seed)
        for fraction in FRACTIONS
        for seed in SEEDS
    }

    dino_pairs = set(
        zip(
            dino["fraction_percent"],
            dino["seed"],
        )
    )

    resnet_pairs = set(
        zip(
            resnet["fraction_percent"],
            resnet["seed"],
        )
    )

    require(
        dino_pairs == expected_pairs,
        "Unexpected DINOv2 fraction/seed pairs",
    )

    require(
        resnet_pairs == expected_pairs,
        "Unexpected ResNet fraction/seed pairs",
    )


def build_summary_comparison(dino, resnet):
    dino_columns = [
        "fraction_percent",
        "train_samples",
        "n_runs",
    ]

    resnet_columns = [
        "fraction_percent",
        "train_samples",
        "n_runs",
    ]

    for metric in METRICS:
        dino_columns.extend(
            [
                f"{metric}_mean",
                f"{metric}_std",
            ]
        )

        resnet_columns.extend(
            [
                f"{metric}_mean",
                f"{metric}_std",
            ]
        )

    merged = dino[dino_columns].merge(
        resnet[resnet_columns],
        on=[
            "fraction_percent",
            "train_samples",
        ],
        suffixes=(
            "_dinov2",
            "_resnet18",
        ),
        validate="one_to_one",
    )

    for metric in METRICS:
        merged[
            f"{metric}_difference_resnet_minus_dinov2"
        ] = (
            merged[f"{metric}_mean_resnet18"]
            - merged[f"{metric}_mean_dinov2"]
        )

    return merged


def build_run_comparison(dino, resnet):
    columns = [
        "fraction_percent",
        "seed",
        "train_samples",
        *METRICS,
    ]

    merged = dino[columns].merge(
        resnet[columns],
        on=[
            "fraction_percent",
            "seed",
            "train_samples",
        ],
        suffixes=(
            "_dinov2",
            "_resnet18",
        ),
        validate="one_to_one",
    )

    require(
        len(merged) == 15,
        "Expected exactly 15 paired runs",
    )

    for metric in METRICS:
        merged[
            f"{metric}_difference_resnet_minus_dinov2"
        ] = (
            merged[f"{metric}_resnet18"]
            - merged[f"{metric}_dinov2"]
        )

    return merged.sort_values(
        [
            "fraction_percent",
            "seed",
        ]
    ).reset_index(drop=True)


def main():
    dino_summary = load_csv(
        DINO_SUMMARY_PATH
    )

    resnet_summary = load_csv(
        RESNET_SUMMARY_PATH
    )

    dino_runs = load_csv(
        DINO_RUNS_PATH
    )

    resnet_runs = load_csv(
        RESNET_RUNS_PATH
    )

    validate_summary(
        dino_summary,
        resnet_summary,
    )

    validate_runs(
        dino_runs,
        resnet_runs,
    )

    summary = build_summary_comparison(
        dino_summary,
        resnet_summary,
    )

    runs = build_run_comparison(
        dino_runs,
        resnet_runs,
    )

    summary.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False,
    )

    runs.to_csv(
        RUNS_OUTPUT_PATH,
        index=False,
    )

    display = summary[
        [
            "fraction_percent",
            "train_samples",
            "macro_roc_auc_mean_dinov2",
            "macro_roc_auc_std_dinov2",
            "macro_roc_auc_mean_resnet18",
            "macro_roc_auc_std_resnet18",
            "macro_roc_auc_difference_resnet_minus_dinov2",
        ]
    ]

    print(
        "Comparison validation passed"
    )

    print(
        "Summary rows: "
        f"{len(summary)}"
    )

    print(
        "Paired runs: "
        f"{len(runs)}"
    )

    print()

    print(
        display.to_string(
            index=False
        )
    )

    print()

    print(
        f"Summary: {SUMMARY_OUTPUT_PATH}"
    )

    print(
        f"Paired runs: {RUNS_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()