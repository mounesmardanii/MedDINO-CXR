from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results" / "label_efficiency"
FIGURES_ROOT = PROJECT_ROOT / "results" / "figures"

INPUT_PATH = RESULTS_ROOT / "model_comparison_per_class.csv"

PNG_PATH = (
    FIGURES_ROOT
    / "per_class_roc_auc_100_percent.png"
)

PDF_PATH = (
    FIGURES_ROOT
    / "per_class_roc_auc_100_percent.pdf"
)


def main():
    data = pd.read_csv(INPUT_PATH)

    data = data[
        data["fraction_percent"] == 100
    ].copy()

    data = data.sort_values(
        "resnet18_roc_auc_mean",
        ascending=True,
    )

    FIGURES_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    y = np.arange(len(data))
    height = 0.36

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    ax.barh(
        y - height / 2,
        data["dinov2_roc_auc_mean"],
        height=height,
        xerr=data["dinov2_roc_auc_std"],
        capsize=3,
        label="DINOv2 frozen linear probe",
    )

    ax.barh(
        y + height / 2,
        data["resnet18_roc_auc_mean"],
        height=height,
        xerr=data["resnet18_roc_auc_std"],
        capsize=3,
        label="ResNet18 end-to-end fine-tuning",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(
        data["class_name"]
    )

    ax.set_xlabel(
        "Validation ROC-AUC"
    )

    ax.set_ylabel(
        "Finding"
    )

    ax.set_title(
        "Per-Class Performance with 100% Labeled Data"
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