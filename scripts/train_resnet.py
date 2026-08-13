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
    evaluate_one_epoch,
    is_better_metric,
    load_checkpoint,
    predict,
    save_checkpoint,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a ResNet18 baseline on ChestMNIST."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
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
        "--patience",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--subset-archive",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--subset-key",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--resume-from",
        type=Path,
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


def load_history(
    path: Path,
) -> list[dict]:
    if not path.is_file():
        return []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


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


def resolve_subset(
    archive_path: Path | None,
    subset_key: str | None,
) -> tuple[Path | None, np.ndarray | None]:
    if (archive_path is None) != (subset_key is None):
        raise ValueError(
            "--subset-archive and --subset-key must be provided together."
        )

    if archive_path is None:
        return None, None

    if not archive_path.is_absolute():
        archive_path = PROJECT_ROOT / archive_path

    archive_path = archive_path.resolve()

    if not archive_path.is_file():
        raise FileNotFoundError(
            f"Subset archive not found: {archive_path}"
        )

    with np.load(archive_path) as archive:
        if subset_key not in archive.files:
            available = ", ".join(archive.files)
            raise KeyError(
                f"Subset key '{subset_key}' not found. "
                f"Available keys: {available}"
            )

        indices = np.asarray(
            archive[subset_key],
            dtype=np.int64,
        )

    if indices.ndim != 1:
        raise ValueError(
            "Subset indices must be one-dimensional."
        )

    if indices.size == 0:
        raise ValueError(
            "Subset indices must not be empty."
        )

    if np.unique(indices).size != indices.size:
        raise ValueError(
            "Subset indices must be unique."
        )

    return archive_path, indices


def main() -> None:
    args = parse_args()

    set_seed(args.seed)

    subset_archive, subset_indices = resolve_subset(
        args.subset_archive,
        args.subset_key,
    )

    if args.weighted_loss and subset_indices is not None:
        raise ValueError(
            "Weighted loss is not supported for subset training."
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    loader_seed = (
        args.seed
        if subset_indices is not None
        else None
    )

    train_loader = build_dataloader(
        "train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=loader_seed,
        indices=subset_indices,
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

    history = load_history(
        args.history_path
    )

    start_epoch = 1
    best_macro_roc_auc = None
    epochs_without_improvement = 0

    if args.resume_from is not None:
        checkpoint = load_checkpoint(
            path=args.resume_from,
            model=model,
            optimizer=optimizer,
            device=device,
        )

        start_epoch = (
            int(checkpoint["epoch"]) + 1
        )

        checkpoint_metrics = checkpoint.get(
            "metrics",
            {},
        )

        if "macro_roc_auc" in checkpoint_metrics:
            best_macro_roc_auc = float(
                checkpoint_metrics[
                    "macro_roc_auc"
                ]
            )

        history = [
            item
            for item in history
            if int(item["epoch"]) < start_epoch
        ]

        print(
            "Resumed from: "
            f"{args.resume_from}"
        )

        print(
            "Checkpoint epoch: "
            f"{checkpoint['epoch']}"
        )

        print(
            "Best Macro ROC-AUC: "
            f"{best_macro_roc_auc}"
        )

    print(f"Device: {device}")
    print(f"Max epochs: {args.epochs}")
    print(f"Start epoch: {start_epoch}")
    print(f"Batch size: {args.batch_size}")
    print(f"Weighted loss: {args.weighted_loss}")
    print(f"Seed: {args.seed}")
    print(f"Patience: {args.patience}")
    print(f"Min delta: {args.min_delta}")

    if subset_archive is not None:
        print(f"Subset archive: {subset_archive}")
        print(f"Subset key: {args.subset_key}")

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    if start_epoch > args.epochs:
        print(
            "Training already reached "
            "the requested epoch limit."
        )
        return

    for epoch in range(
        start_epoch,
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