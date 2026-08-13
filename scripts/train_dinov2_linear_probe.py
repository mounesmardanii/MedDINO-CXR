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

from meddino_cxr.data import build_embedding_dataloader
from meddino_cxr.models import build_dinov2_linear_probe
from meddino_cxr.training import (
    build_loss,
    build_optimizer,
    compute_multilabel_metrics,
    evaluate_one_epoch,
    is_better_metric,
    predict,
    save_checkpoint,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a linear probe on frozen DINOv2 ChestMNIST embeddings."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
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
        "--patience",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
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
        / "dinov2_vits14_linear_probe_best.pt",
    )

    parser.add_argument(
        "--history-path",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "dinov2_linear_probe"
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

    train_loader = build_embedding_dataloader(
        "train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    val_loader = build_embedding_dataloader(
        "val",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = build_dinov2_linear_probe().to(
        device
    )

    criterion = build_loss()

    optimizer = build_optimizer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history = []

    best_macro_roc_auc = None
    epochs_without_improvement = 0

    print(f"Device: {device}")
    print(f"Max epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Seed: {args.seed}")
    print(f"Patience: {args.patience}")
    print(f"Min delta: {args.min_delta}")
    print(
        f"Train samples: {len(train_loader.dataset)}"
    )
    print(
        f"Validation samples: {len(val_loader.dataset)}"
    )
    print(
        "Trainable parameters: "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        print()
        print(
            f"Epoch {epoch}/{args.epochs}"
        )

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

        improved = is_better_metric(
            current=macro_roc_auc,
            best=best_macro_roc_auc,
            min_delta=args.min_delta,
        )

        if improved:
            best_macro_roc_auc = macro_roc_auc
            epochs_without_improvement = 0

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

        else:
            epochs_without_improvement += 1

            print(
                "No improvement: "
                f"{epochs_without_improvement}/"
                f"{args.patience}"
            )

        if (
            epochs_without_improvement
            >= args.patience
        ):
            print()
            print(
                "Early stopping triggered."
            )
            break

    print()
    print(
        "Linear probe training completed."
    )

    if best_macro_roc_auc is not None:
        print(
            "Best validation Macro ROC-AUC: "
            f"{best_macro_roc_auc:.6f}"
        )

    print(
        f"History: {args.history_path}"
    )

    print(
        f"Checkpoint: {args.checkpoint_path}"
    )


if __name__ == "__main__":
    main()