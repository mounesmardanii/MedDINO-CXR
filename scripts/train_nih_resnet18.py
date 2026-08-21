from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from meddino_cxr.data.nih_dataloaders import build_nih_dataloader
from meddino_cxr.models import build_resnet18
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


SUBSET_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "nih_label_efficiency_patients.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "nih_resnet18"
)

CHECKPOINT_ROOT = (
    PROJECT_ROOT
    / "checkpoints"
    / "nih_resnet18"
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
        default=30,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
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
        default="auto",
        choices=["cpu", "cuda", "auto"],
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
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
            raise RuntimeError(
                "CUDA is not available."
            )

        return torch.device("cuda")

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
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


def load_selected_patients(
    fraction,
    seed,
):
    if not SUBSET_MANIFEST.is_file():
        raise FileNotFoundError(
            SUBSET_MANIFEST
        )

    frame = pd.read_csv(
        SUBSET_MANIFEST,
        dtype={
            "patient_id": str,
        },
    )

    column = (
        f"seed_{seed}_"
        f"fraction_{FRACTION_CODES[fraction]}"
    )

    if column not in frame.columns:
        raise KeyError(
            column
        )

    selected = (
        frame.loc[
            frame[column].eq(1),
            "patient_id",
        ]
        .astype(str)
        .tolist()
    )

    if not selected:
        raise RuntimeError(
            "No patients were selected."
        )

    if len(selected) != len(
        set(selected)
    ):
        raise RuntimeError(
            "Duplicate selected patient IDs."
        )

    return selected


def compute_pos_weight(
    labels,
    max_weight,
):
    labels = np.asarray(
        labels,
        dtype=np.float64,
    )

    positives = labels.sum(
        axis=0
    )

    negatives = (
        len(labels)
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


def main():
    args = parse_args()

    set_seed(
        args.seed
    )

    selected_patients = (
        load_selected_patients(
            fraction=args.fraction,
            seed=args.seed,
        )
    )

    fraction_code = (
        FRACTION_CODES[
            args.fraction
        ]
    )

    run_name = (
        f"seed_{args.seed}_"
        f"fraction_{fraction_code}"
    )

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

    config_path = (
        output_dir
        / "run_config.json"
    )

    status_path = (
        output_dir
        / "run_status.json"
    )

    train_loader = build_nih_dataloader(
        split="train",
        batch_size=args.batch_size,
        num_workers=(
            0
            if args.dry_run
            else args.num_workers
        ),
        seed=args.seed,
        patient_ids=selected_patients,
    )

    val_loader = build_nih_dataloader(
        split="validate",
        batch_size=args.batch_size,
        num_workers=(
            0
            if args.dry_run
            else args.num_workers
        ),
        seed=args.seed,
    )

    train_dataset = (
        train_loader.dataset
    )

    positive_counts, pos_weight = (
        compute_pos_weight(
            train_dataset.labels,
            args.max_pos_weight,
        )
    )

    print(
        "Run:",
        run_name,
    )

    print(
        "Fraction:",
        f"{args.fraction}%",
    )

    print(
        "Seed:",
        args.seed,
    )

    print(
        "Selected patients:",
        len(selected_patients),
    )

    print(
        "Train images:",
        len(train_loader.dataset),
    )

    print(
        "Validation images:",
        len(val_loader.dataset),
    )

    print(
        "Positive counts:",
        positive_counts.tolist(),
    )

    print(
        "Pos weights:",
        [
            round(
                value,
                4,
            )
            for value in pos_weight.tolist()
        ],
    )

    print(
        "Max epochs:",
        args.epochs,
    )

    print(
        "Batch size:",
        args.batch_size,
    )

    print(
        "Learning rate:",
        args.learning_rate,
    )

    print(
        "Weight decay:",
        args.weight_decay,
    )

    print(
        "Patience:",
        args.patience,
    )

    print(
        "Min delta:",
        args.min_delta,
    )

    print(
        "Test used:",
        False,
    )

    if args.dry_run:
        print(
            "DRY RUN PASSED: True"
        )

        train_dataset.close()
        val_loader.dataset.close()

        return

    device = resolve_device(
        args.device
    )

    print(
        "Device:",
        device,
    )

    pos_weight = pos_weight.to(
        device
    )

    model = build_resnet18(
        pretrained=True
    ).to(
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
        "model": "resnet18_imagenet_end_to_end",
        "dataset": "nih_chestxray14",
        "fraction_percent": args.fraction,
        "fraction_code": fraction_code,
        "seed": args.seed,
        "run_name": run_name,
        "selected_patients": len(
            selected_patients
        ),
        "train_images": len(
            train_loader.dataset
        ),
        "validation_images": len(
            val_loader.dataset
        ),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "max_pos_weight": args.max_pos_weight,
        "positive_counts": positive_counts.tolist(),
        "pos_weight": pos_weight.detach().cpu().tolist(),
        "pretrained": True,
        "pretrained_weights": "ResNet18_Weights.DEFAULT",
        "device": str(device),
        "selection_metric": "validation_macro_roc_auc",
        "test_used": False,
        "external_used": False,
    }

    save_json(
        config_path,
        config,
    )

    if status_path.is_file():
        status_path.unlink()

    history = []
    best_macro_roc_auc = None
    epochs_without_improvement = 0
    stop_reason = (
        "max_epochs_reached"
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        train_loss = (
            train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )
        )

        val_loss = (
            evaluate_one_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
            )
        )

        targets, probabilities = (
            predict(
                model=model,
                loader=val_loader,
                device=device,
            )
        )

        metrics = (
            compute_multilabel_metrics(
                targets,
                probabilities,
            )
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

        macro_roc_auc = (
            metrics[
                "macro_roc_auc"
            ]
        )

        print()

        print(
            f"Epoch {epoch}/{args.epochs}"
        )

        print(
            f"Train loss: "
            f"{train_loss:.6f}"
        )

        print(
            f"Validation loss: "
            f"{val_loss:.6f}"
        )

        print(
            f"Macro ROC-AUC: "
            f"{macro_roc_auc:.6f}"
        )

        print(
            "Macro AP: "
            f"{metrics['macro_average_precision']:.6f}"
        )

        improved = (
            is_better_metric(
                current=macro_roc_auc,
                best=best_macro_roc_auc,
                min_delta=args.min_delta,
            )
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
            stop_reason = (
                "early_stopping"
            )

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
            "No valid best metric."
        )

    if not checkpoint_path.is_file():
        raise RuntimeError(
            f"Checkpoint missing: "
            f"{checkpoint_path}"
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
            "external_used": False,
        },
    )

    train_dataset.close()
    val_loader.dataset.close()

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