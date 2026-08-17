from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results" / "label_efficiency"

INPUT_PATH = RESULTS_ROOT / "model_comparison_runs.csv"
OUTPUT_PATH = RESULTS_ROOT / "model_comparison_gaps.csv"

FRACTIONS = (5, 10, 25, 50, 100)

METRICS = (
    "macro_roc_auc",
    "micro_roc_auc",
    "macro_average_precision",
    "micro_average_precision",
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    data = pd.read_csv(INPUT_PATH)

    require(
        len(data) == 15,
        "Expected exactly 15 paired runs",
    )

    rows = []

    for fraction in FRACTIONS:
        group = data[
            data["fraction_percent"] == fraction
        ]

        require(
            len(group) == 3,
            f"Expected 3 paired runs for fraction {fraction}",
        )

        for metric in METRICS:
            dino_column = f"{metric}_dinov2"
            resnet_column = f"{metric}_resnet18"
            gap_column = (
                f"{metric}_difference_resnet_minus_dinov2"
            )

            require(
                dino_column in group.columns,
                f"Missing column: {dino_column}",
            )

            require(
                resnet_column in group.columns,
                f"Missing column: {resnet_column}",
            )

            require(
                gap_column in group.columns,
                f"Missing column: {gap_column}",
            )

            gaps = group[gap_column]

            rows.append(
                {
                    "fraction_percent": fraction,
                    "train_samples": int(
                        group["train_samples"].iloc[0]
                    ),
                    "metric": metric,
                    "dinov2_mean": float(
                        group[dino_column].mean()
                    ),
                    "resnet18_mean": float(
                        group[resnet_column].mean()
                    ),
                    "paired_gap_mean": float(
                        gaps.mean()
                    ),
                    "paired_gap_std": float(
                        gaps.std(ddof=1)
                    ),
                    "resnet18_wins": int(
                        (gaps > 0).sum()
                    ),
                    "dinov2_wins": int(
                        (gaps < 0).sum()
                    ),
                    "ties": int(
                        (gaps == 0).sum()
                    ),
                }
            )

    result = pd.DataFrame(rows)

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    macro = result[
        result["metric"] == "macro_roc_auc"
    ]

    print(
        "Paired gap analysis passed"
    )

    print(
        f"Rows: {len(result)}"
    )

    print()

    print(
        macro[
            [
                "fraction_percent",
                "dinov2_mean",
                "resnet18_mean",
                "paired_gap_mean",
                "paired_gap_std",
                "resnet18_wins",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print(
        f"Output: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()