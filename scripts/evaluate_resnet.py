from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from meddino_cxr.data import build_dataloader
from meddino_cxr.models import build_resnet18
from meddino_cxr.training import (
    compute_multilabel_metrics,
    load_checkpoint,
    predict,
)


CLASS_NAMES = [
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained ResNet18 checkpoint on ChestMNIST."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--split",
        type=str,
        choices=["val", "test"],
        default="test",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--max-test-batches",
        type=int,
        default=None,
    )

    return parser.parse_args()


def save_json(
    path: Path,
    data: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
        )


def save_per_class_csv(
    path: Path,
    per_class_metrics: list[dict],
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
            fieldnames=[
                "class_index",
                "class_name",
                "roc_auc",
                "average_precision",
            ],
        )

        writer.writeheader()
        writer.writerows(
            per_class_metrics
        )


def main() -> None:
    args = parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    loader = build_dataloader(
        args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_resnet18(
        pretrained=False
    ).to(device)

    checkpoint = load_checkpoint(
        path=args.checkpoint,
        model=model,
        optimizer=None,
        device=device,
    )

    targets, probabilities = predict(
        model=model,
        loader=loader,
        device=device,
        max_batches=args.max_test_batches,
    )

    metrics = compute_multilabel_metrics(
        targets,
        probabilities,
    )

    per_class_metrics = []

    for index, class_name in enumerate(
        CLASS_NAMES
    ):
        per_class_metrics.append(
            {
                "class_index": index,
                "class_name": class_name,
                "roc_auc": metrics[
                    "per_class_roc_auc"
                ][index],
                "average_precision": metrics[
                    "per_class_average_precision"
                ][index],
            }
        )

    result = {
        "checkpoint": str(
            args.checkpoint
        ),
        "checkpoint_epoch": checkpoint[
            "epoch"
        ],
        "split": args.split,
        "samples": int(
            targets.shape[0]
        ),
        "macro_roc_auc": metrics[
            "macro_roc_auc"
        ],
        "micro_roc_auc": metrics[
            "micro_roc_auc"
        ],
        "macro_average_precision": metrics[
            "macro_average_precision"
        ],
        "micro_average_precision": metrics[
            "micro_average_precision"
        ],
        "per_class": per_class_metrics,
    }

    metrics_path = (
        args.output_dir
        / f"{args.split}_metrics.json"
    )

    csv_path = (
        args.output_dir
        / f"{args.split}_per_class_metrics.csv"
    )

    save_json(
        metrics_path,
        result,
    )

    save_per_class_csv(
        csv_path,
        per_class_metrics,
    )

    print(f"Device: {device}")
    print(
        f"Checkpoint: {args.checkpoint}"
    )
    print(
        "Checkpoint epoch: "
        f"{checkpoint['epoch']}"
    )
    print(
        f"Split: {args.split}"
    )
    print(
        f"Samples: {targets.shape[0]}"
    )
    print(
        "Macro ROC-AUC: "
        f"{metrics['macro_roc_auc']:.6f}"
    )
    print(
        "Micro ROC-AUC: "
        f"{metrics['micro_roc_auc']:.6f}"
    )
    print(
        "Macro AP: "
        f"{metrics['macro_average_precision']:.6f}"
    )
    print(
        "Micro AP: "
        f"{metrics['micro_average_precision']:.6f}"
    )
    print(
        f"Metrics: {metrics_path}"
    )
    print(
        f"Per-class CSV: {csv_path}"
    )


if __name__ == "__main__":
    main()