from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results" / "final_test"
FIGURES_ROOT = PROJECT_ROOT / "results" / "figures"

INPUT_PATH = RESULTS_ROOT / "final_test_summary.csv"

PNG_PATH = (
    FIGURES_ROOT
    / "final_test_label_efficiency_macro_roc_auc.png"
)

PDF_PATH = (
    FIGURES_ROOT
    / "final_test_label_efficiency_macro_roc_auc.pdf"
)


def main():
    data = pd.read_csv(INPUT_PATH)

    dino = data[
        data["model"] == "dinov2"
    ].sort_values(
        "fraction_percent"
    )

    resnet = data[
        data["model"] == "resnet18"
    ].sort_values(
        "fraction_percent"
    )

    if len(dino) != 5:
        raise RuntimeError(
            "Expected 5 DINOv2 rows"
        )

    if len(resnet) != 5:
        raise RuntimeError(
            "Expected 5 ResNet18 rows"
        )

    if (
        dino["fraction_percent"].tolist()
        != resnet["fraction_percent"].tolist()
    ):
        raise RuntimeError(
            "Fraction mismatch"
        )

    FIGURES_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    x = dino[
        "fraction_percent"
    ]

    fig, ax = plt.subplots(
        figsize=(8, 5.5)
    )

    ax.errorbar(
        x,
        dino[
            "macro_roc_auc_mean"
        ],
        yerr=dino[
            "macro_roc_auc_std"
        ],
        marker="o",
        capsize=4,
        linewidth=2,
        label="DINOv2 frozen linear probe",
    )

    ax.errorbar(
        x,
        resnet[
            "macro_roc_auc_mean"
        ],
        yerr=resnet[
            "macro_roc_auc_std"
        ],
        marker="o",
        capsize=4,
        linewidth=2,
        label="ResNet18 end-to-end fine-tuning",
    )

    ax.set_xlabel(
        "Labeled training data (%)"
    )

    ax.set_ylabel(
        "Test Macro ROC-AUC"
    )

    ax.set_title(
        "Final Test Label Efficiency on ChestMNIST"
    )

    ax.set_xticks(
        x
    )

    ax.set_ylim(
        0.5,
        0.9,
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