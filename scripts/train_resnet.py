from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from meddino_cxr.data import build_dataloader
from meddino_cxr.models import build_resnet18
from meddino_cxr.training import (
    build_loss,
    build_optimizer,
    compute_multilabel_metrics,
    compute_pos_weight,
    is_better_metric,
    predict,
    save_checkpoint,
    evaluate_one_epoch,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a ResNet18 baseline on ChestMNIST."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--weighted-loss",
        action="store_true",
    )

    parser.add_argument(
        "--max-pos-weight",
        type=float,
        default=20.0,
    )

    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=PROJECT_ROOT
        / "checkpoints"
        / "resnet18_best.pt",
    )

    parser.add_argument(
        "--history-path",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "resnet18"
        / "history.json",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_history(
    path: Path,
    history: list[dict],
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
            history,
            file,
            indent=2,
        )


def main() -> None:
    args = parse_args()

    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    train_loader = build_dataloader(
        "train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    val_loader = build_dataloader(
        "val",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_resnet18().to(device)

    pos_weight = None

    if args.weighted_loss:
        labels_path = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "chestmnist_224"
            / "train_labels.npy"
        )

        pos_weight = compute_pos_weight(
            labels_path,
            max_weight=args.max_pos_weight,
        ).to(device)

    criterion = build_loss(
        pos_weight=pos_weight,
    )

    optimizer = build_optimizer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_macro_roc_auc = None
    history = []

    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Weighted loss: {args.weighted_loss}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    for epoch in range(1, args.epochs + 1):
        print()
        print(f"Epoch {epoch}/{args.epochs}")

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            max_batches=args.max_train_batches,
        )

        val_loss = evaluate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            max_batches=args.max_val_batches,
        )

        targets, probabilities = predict(
            model=model,
            loader=val_loader,
            device=device,
            max_batches=args.max_val_batches,
        )

        metrics = compute_multilabel_metrics(
            targets,
            probabilities,
        )

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **metrics,
        }

        history.append(
            epoch_result
        )

        save_history(
            args.history_path,
            history,
        )

        macro_roc_auc = metrics[
            "macro_roc_auc"
        ]

        print(
            f"Train loss: {train_loss:.6f}"
        )

        print(
            f"Validation loss: {val_loss:.6f}"
        )

        print(
            "Macro ROC-AUC: "
            f"{macro_roc_auc:.6f}"
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

        if is_better_metric(
            macro_roc_auc,
            best_macro_roc_auc,
        ):
            best_macro_roc_auc = macro_roc_auc

            checkpoint_metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                **metrics,
            }

            save_checkpoint(
                path=args.checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=checkpoint_metrics,
            )

            print(
                "Best checkpoint saved: "
                f"{args.checkpoint_path}"
            )

    print()
    print("Training completed.")

    if best_macro_roc_auc is not None:
        print(
            "Best validation Macro ROC-AUC: "
            f"{best_macro_roc_auc:.6f}"
        )

    print(
        f"History: {args.history_path}"
    )


if __name__ == "__main__":
    main()