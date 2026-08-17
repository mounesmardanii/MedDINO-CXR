from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results" / "label_efficiency"
FIGURES_ROOT = PROJECT_ROOT / "results" / "figures"

INPUT_PATH = RESULTS_ROOT / "model_comparison_summary.csv"
PNG_PATH = FIGURES_ROOT / "label_efficiency_macro_roc_auc.png"
PDF_PATH = FIGURES_ROOT / "label_efficiency_macro_roc_auc.pdf"


def main():
    data = pd.read_csv(INPUT_PATH)

    FIGURES_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    x = data["fraction_percent"]

    fig, ax = plt.subplots(
        figsize=(8, 5.5)
    )

    ax.errorbar(
        x,
        data["macro_roc_auc_mean_dinov2"],
        yerr=data["macro_roc_auc_std_dinov2"],
        marker="o",
        capsize=4,
        linewidth=2,
        label="DINOv2 frozen linear probe",
    )

    ax.errorbar(
        x,
        data["macro_roc_auc_mean_resnet18"],
        yerr=data["macro_roc_auc_std_resnet18"],
        marker="o",
        capsize=4,
        linewidth=2,
        label="ResNet18 end-to-end fine-tuning",
    )

    ax.set_xlabel(
        "Labeled training data (%)"
    )

    ax.set_ylabel(
        "Validation Macro ROC-AUC"
    )

    ax.set_title(
        "Label Efficiency on ChestMNIST"
    )

    ax.set_xticks(
        x
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

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