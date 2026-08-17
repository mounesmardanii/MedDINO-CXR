from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results" / "label_efficiency"

DINO_PATH = RESULTS_ROOT / "dinov2_runs.csv"
RESNET_PATH = RESULTS_ROOT / "resnet18_runs.csv"
OUTPUT_PATH = RESULTS_ROOT / "model_comparison_per_class.csv"

FRACTIONS = (5, 10, 25, 50, 100)
SEEDS = (42, 47, 52)

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


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    dino = pd.read_csv(DINO_PATH)
    resnet = pd.read_csv(RESNET_PATH)

    require(
        len(dino) == 15,
        "Expected 15 DINOv2 runs",
    )

    require(
        len(resnet) == 15,
        "Expected 15 ResNet18 runs",
    )

    expected_pairs = {
        (fraction, seed)
        for fraction in FRACTIONS
        for seed in SEEDS
    }

    require(
        set(
            zip(
                dino["fraction_percent"],
                dino["seed"],
            )
        )
        == expected_pairs,
        "Unexpected DINOv2 fraction/seed pairs",
    )

    require(
        set(
            zip(
                resnet["fraction_percent"],
                resnet["seed"],
            )
        )
        == expected_pairs,
        "Unexpected ResNet18 fraction/seed pairs",
    )

    rows = []

    for fraction in FRACTIONS:
        for class_name in CLASS_NAMES:
            roc_column = f"{class_name}_roc_auc"
            ap_column = f"{class_name}_average_precision"

            dino_group = dino[
                dino["fraction_percent"] == fraction
            ].sort_values("seed")

            resnet_group = resnet[
                resnet["fraction_percent"] == fraction
            ].sort_values("seed")

            require(
                dino_group["seed"].tolist()
                == resnet_group["seed"].tolist(),
                f"Seed mismatch: fraction={fraction}",
            )

            roc_gap = (
                resnet_group[roc_column].to_numpy()
                - dino_group[roc_column].to_numpy()
            )

            ap_gap = (
                resnet_group[ap_column].to_numpy()
                - dino_group[ap_column].to_numpy()
            )

            rows.append(
                {
                    "fraction_percent": fraction,
                    "class_name": class_name,
                    "dinov2_roc_auc_mean": float(
                        dino_group[roc_column].mean()
                    ),
                    "dinov2_roc_auc_std": float(
                        dino_group[roc_column].std(ddof=1)
                    ),
                    "resnet18_roc_auc_mean": float(
                        resnet_group[roc_column].mean()
                    ),
                    "resnet18_roc_auc_std": float(
                        resnet_group[roc_column].std(ddof=1)
                    ),
                    "roc_auc_gap_mean": float(
                        roc_gap.mean()
                    ),
                    "roc_auc_gap_std": float(
                        roc_gap.std(ddof=1)
                    ),
                    "resnet18_roc_auc_wins": int(
                        (roc_gap > 0).sum()
                    ),
                    "dinov2_roc_auc_wins": int(
                        (roc_gap < 0).sum()
                    ),
                    "dinov2_ap_mean": float(
                        dino_group[ap_column].mean()
                    ),
                    "dinov2_ap_std": float(
                        dino_group[ap_column].std(ddof=1)
                    ),
                    "resnet18_ap_mean": float(
                        resnet_group[ap_column].mean()
                    ),
                    "resnet18_ap_std": float(
                        resnet_group[ap_column].std(ddof=1)
                    ),
                    "ap_gap_mean": float(
                        ap_gap.mean()
                    ),
                    "ap_gap_std": float(
                        ap_gap.std(ddof=1)
                    ),
                    "resnet18_ap_wins": int(
                        (ap_gap > 0).sum()
                    ),
                    "dinov2_ap_wins": int(
                        (ap_gap < 0).sum()
                    ),
                }
            )

    result = pd.DataFrame(rows)

    require(
        len(result) == 70,
        "Expected exactly 70 class/fraction rows",
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    full = result[
        result["fraction_percent"] == 100
    ].sort_values(
        "roc_auc_gap_mean",
        ascending=False,
    )

    print(
        "Per-class comparison passed"
    )

    print(
        f"Rows: {len(result)}"
    )

    print()

    print(
        full[
            [
                "class_name",
                "dinov2_roc_auc_mean",
                "resnet18_roc_auc_mean",
                "roc_auc_gap_mean",
                "resnet18_roc_auc_wins",
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