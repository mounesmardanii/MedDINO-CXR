from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results" / "final_test"
FIGURES_ROOT = PROJECT_ROOT / "results" / "figures"

INPUT_PATH = RESULTS_ROOT / "final_test_per_class.csv"

PNG_PATH = (
    FIGURES_ROOT
    / "final_test_per_class_roc_auc_100_percent.png"
)

PDF_PATH = (
    FIGURES_ROOT
    / "final_test_per_class_roc_auc_100_percent.pdf"
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    data = pd.read_csv(INPUT_PATH)

    data = data[
        data["fraction_percent"] == 100
    ].copy()

    require(
        len(data) == 84,
        "Expected 84 per-class rows at 100% labels",
    )

    summary = (
        data.groupby(
            [
                "model",
                "class_name",
            ],
            as_index=False,
        )
        .agg(
            roc_auc_mean=(
                "roc_auc",
                "mean",
            ),
            roc_auc_std=(
                "roc_auc",
                lambda values: values.std(
                    ddof=1
                ),
            ),
        )
    )

    dino = summary[
        summary["model"] == "dinov2"
    ].copy()

    resnet = summary[
        summary["model"] == "resnet18"
    ].copy()

    require(
        len(dino) == 14,
        "Expected 14 DINOv2 classes",
    )

    require(
        len(resnet) == 14,
        "Expected 14 ResNet18 classes",
    )

    merged = dino.merge(
        resnet,
        on="class_name",
        suffixes=(
            "_dinov2",
            "_resnet18",
        ),
        validate="one_to_one",
    )

    merged = merged.sort_values(
        "roc_auc_mean_resnet18",
        ascending=True,
    )

    FIGURES_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    y = np.arange(
        len(merged)
    )

    height = 0.36

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    ax.barh(
        y - height / 2,
        merged[
            "roc_auc_mean_dinov2"
        ],
        height=height,
        xerr=merged[
            "roc_auc_std_dinov2"
        ],
        capsize=3,
        label="DINOv2 frozen linear probe",
    )

    ax.barh(
        y + height / 2,
        merged[
            "roc_auc_mean_resnet18"
        ],
        height=height,
        xerr=merged[
            "roc_auc_std_resnet18"
        ],
        capsize=3,
        label="ResNet18 end-to-end fine-tuning",
    )

    ax.set_yticks(
        y
    )

    ax.set_yticklabels(
        merged["class_name"]
    )

    ax.set_xlabel(
        "Test ROC-AUC"
    )

    ax.set_ylabel(
        "Finding"
    )

    ax.set_title(
        "Final Test Per-Class Performance with 100% Labeled Data"
    )

    ax.set_xlim(
        0.5,
        0.95,
    )

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    ax.legend(
        loc="lower right"
    )

    fig.tight_layout()

    fig.savefig(
        PNG_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        PDF_PATH,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"PNG: {PNG_PATH}"
    )

    print(
        f"PDF: {PDF_PATH}"
    )


if __name__ == "__main__":
    main()