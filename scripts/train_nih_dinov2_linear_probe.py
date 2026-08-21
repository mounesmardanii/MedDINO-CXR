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


DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nih_dinov2_vits14"
)

SUBSET_ARCHIVE = (
    DATA_DIR
    / "nih_train_subset_indices.npz"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "nih_dinov2_linear_probe"
)

CHECKPOINT_ROOT = (
    PROJECT_ROOT
    / "checkpoints"
    / "nih_dinov2_linear_probe"
)

FRACTION_CODES = {
    1: "001",
    5: "005",
    10: "010",
    25: "025",
    100: "100",
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fraction",
        type=int,
        required=True,
        choices=sorted(FRACTION_CODES),
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        choices=[42, 47, 52],
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
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
        "--max-pos-weight",
        type=float,
        default=20.0,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "auto"],
    )

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name):
    if name == "cpu":
        return torch.device("cpu")

    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")

        return torch.device("cuda")

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def compute_subset_pos_weight(
    labels,
    indices,
    max_weight,
):
    subset_labels = np.asarray(
        labels[indices],
        dtype=np.float64,
    )

    positives = subset_labels.sum(
        axis=0
    )

    negatives = (
        len(subset_labels)
        - positives
    )

    weights = negatives / np.maximum(
        positives,
        1.0,
    )

    weights = np.clip(
        weights,
        1.0,
        max_weight,
    )

    return (
        positives.astype(int),
        torch.tensor(
            weights,
            dtype=torch.float32,
        ),
    )


def save_json(path, value):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            value,
            file,
            indent=2,
        )


def main():
    args = parse_args()

    set_seed(args.seed)

    fraction_code = FRACTION_CODES[
        args.fraction
    ]

    subset_key = (
        f"seed_{args.seed}_"
        f"fraction_{fraction_code}"
    )

    run_name = subset_key

    output_dir = (
        OUTPUT_ROOT
        / run_name
    )

    checkpoint_path = (
        CHECKPOINT_ROOT
        / f"{run_name}.pt"
    )

    history_path = (
        output_dir
        / "history.json"
    )

    status_path = (
        output_dir
        / "run_status.json"
    )

    config_path = (
        output_dir
        / "run_config.json"
    )

    if not SUBSET_ARCHIVE.is_file():
        raise FileNotFoundError(
            SUBSET_ARCHIVE
        )

    with np.load(
        SUBSET_ARCHIVE
    ) as archive:
        if subset_key not in archive:
            raise KeyError(
                subset_key
            )

        train_indices = np.asarray(
            archive[subset_key],
            dtype=np.int64,
        ).copy()

    labels = np.load(
        DATA_DIR / "train_labels.npy",
        mmap_mode="r",
    )

    positive_counts, pos_weight = (
        compute_subset_pos_weight(
            labels,
            train_indices,
            args.max_pos_weight,
        )
    )

    device = resolve_device(
        args.device
    )

    pos_weight = pos_weight.to(
        device
    )

    train_loader = build_embedding_dataloader(
        "train",
        data_dir=DATA_DIR,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=False,
        seed=args.seed,
        indices=train_indices,
    )

    val_loader = build_embedding_dataloader(
        "val",
        data_dir=DATA_DIR,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=False,
        seed=args.seed,
    )

    model = build_dinov2_linear_probe().to(
        device
    )

    criterion = build_loss(
        pos_weight=pos_weight
    )

    optimizer = build_optimizer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    config = {
        "model": "dinov2_vits14_frozen_linear_probe",
        "dataset": "nih_chestxray14",
        "fraction_percent": args.fraction,
        "fraction_code": fraction_code,
        "seed": args.seed,
        "subset_key": subset_key,
        "train_images": len(train_loader.dataset),
        "validation_images": len(val_loader.dataset),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "max_pos_weight": args.max_pos_weight,
        "positive_counts": positive_counts.tolist(),
        "pos_weight": pos_weight.detach().cpu().tolist(),
        "device": str(device),
        "selection_metric": "validation_macro_roc_auc",
        "test_used": False,
    }

    save_json(
        config_path,
        config,
    )

    if status_path.is_file():
        status_path.unlink()

    print(
        "Run:",
        run_name,
    )

    print(
        "Device:",
        device,
    )

    print(
        "Train samples:",
        len(train_loader.dataset),
    )

    print(
        "Validation samples:",
        len(val_loader.dataset),
    )

    print(
        "Positive counts:",
        positive_counts.tolist(),
    )

    print(
        "Pos weights:",
        [
            round(value, 4)
            for value
            in pos_weight.detach().cpu().tolist()
        ],
    )

    history = []
    best_macro_roc_auc = None
    epochs_without_improvement = 0
    stop_reason = "max_epochs_reached"

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss = evaluate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        targets, probabilities = predict(
            model=model,
            loader=val_loader,
            device=device,
        )

        metrics = compute_multilabel_metrics(
            targets,
            probabilities,
        )

        result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **metrics,
        }

        history.append(
            result
        )

        save_json(
            history_path,
            history,
        )

        macro_roc_auc = metrics[
            "macro_roc_auc"
        ]

        print()

        print(
            f"Epoch {epoch}/{args.epochs}"
        )

        print(
            f"Train loss: {train_loss:.6f}"
        )

        print(
            f"Validation loss: {val_loss:.6f}"
        )

        print(
            f"Macro ROC-AUC: {macro_roc_auc:.6f}"
        )

        print(
            "Macro AP: "
            f"{metrics['macro_average_precision']:.6f}"
        )

        improved = is_better_metric(
            current=macro_roc_auc,
            best=best_macro_roc_auc,
            min_delta=args.min_delta,
        )

        if improved:
            best_macro_roc_auc = (
                macro_roc_auc
            )

            epochs_without_improvement = 0

            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    **metrics,
                },
            )

            print(
                "Best checkpoint saved."
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
            stop_reason = "early_stopping"

            print()

            print(
                "Early stopping triggered."
            )

            break

    if not history:
        raise RuntimeError(
            "Training produced no history."
        )

    if best_macro_roc_auc is None:
        raise RuntimeError(
            "No valid best validation metric was recorded."
        )

    if not checkpoint_path.is_file():
        raise RuntimeError(
            f"Checkpoint was not created: {checkpoint_path}"
        )

    last_epoch = int(
        history[-1]["epoch"]
    )

    save_json(
        status_path,
        {
            "completed": True,
            "run_name": run_name,
            "fraction_percent": args.fraction,
            "seed": args.seed,
            "last_epoch": last_epoch,
            "max_epochs": args.epochs,
            "stop_reason": stop_reason,
            "best_validation_macro_roc_auc": float(
                best_macro_roc_auc
            ),
            "checkpoint_path": str(
                checkpoint_path
            ),
            "history_path": str(
                history_path
            ),
            "test_used": False,
        },
    )

    print()

    print(
        "Best validation Macro ROC-AUC:",
        f"{best_macro_roc_auc:.6f}",
    )

    print(
        "History:",
        history_path,
    )

    print(
        "Checkpoint:",
        checkpoint_path,
    )

    print(
        "Run status:",
        status_path,
    )


if __name__ == "__main__":
    main()