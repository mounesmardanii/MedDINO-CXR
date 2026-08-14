from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SEEDS = (
    42,
    47,
    52,
)

DEFAULT_FRACTIONS = (
    5,
    10,
    25,
    50,
    100,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run systematic ResNet18 "
            "label-efficiency experiments."
        )
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
    )

    parser.add_argument(
        "--fractions",
        type=int,
        nargs="+",
        default=list(DEFAULT_FRACTIONS),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
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
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
    )

    return parser.parse_args()


def fraction_key(
    fraction: int,
) -> str:
    return f"fraction_{fraction:03d}"


def get_run_paths(
    seed: int,
    fraction: int,
) -> tuple[Path, Path, Path]:
    run_dir = (
        PROJECT_ROOT
        / "outputs"
        / "label_efficiency"
        / "resnet"
        / f"fraction_{fraction:03d}"
        / f"seed_{seed}"
    )

    checkpoint_path = (
        run_dir
        / "best.pt"
    )

    history_path = (
        run_dir
        / "history.json"
    )

    return (
        run_dir,
        checkpoint_path,
        history_path,
    )


def build_command(
    seed: int,
    fraction: int,
    args: argparse.Namespace,
) -> list[str]:
    archive = (
        PROJECT_ROOT
        / "outputs"
        / "label_efficiency"
        / "subsets"
        / f"seed_{seed}.npz"
    )

    _, checkpoint_path, history_path = (
        get_run_paths(
            seed=seed,
            fraction=fraction,
        )
    )

    return [
        sys.executable,
        str(
            PROJECT_ROOT
            / "scripts"
            / "train_resnet.py"
        ),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--seed",
        str(seed),
        "--patience",
        str(args.patience),
        "--min-delta",
        str(args.min_delta),
        "--subset-archive",
        str(archive),
        "--subset-key",
        fraction_key(
            fraction
        ),
        "--checkpoint-path",
        str(checkpoint_path),
        "--history-path",
        str(history_path),
    ]


def validate_inputs(
    seeds: tuple[int, ...],
    fractions: tuple[int, ...],
) -> None:
    allowed_fractions = {
        5,
        10,
        25,
        50,
        100,
    }

    if not seeds:
        raise ValueError(
            "At least one seed is required."
        )

    if not fractions:
        raise ValueError(
            "At least one fraction is required."
        )

    invalid = (
        set(fractions)
        - allowed_fractions
    )

    if invalid:
        raise ValueError(
            f"Unsupported fractions: {sorted(invalid)}"
        )

    for seed in seeds:
        archive = (
            PROJECT_ROOT
            / "outputs"
            / "label_efficiency"
            / "subsets"
            / f"seed_{seed}.npz"
        )

        if not archive.is_file():
            raise FileNotFoundError(
                f"Subset archive not found: {archive}"
            )


def is_run_complete(
    history_path: Path,
    max_epochs: int,
    patience: int,
    min_delta: float,
) -> bool:
    if not history_path.is_file():
        return False

    with history_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        history = json.load(file)

    if not history:
        return False

    expected_epochs = list(
        range(
            1,
            len(history) + 1,
        )
    )

    actual_epochs = [
        int(item["epoch"])
        for item in history
    ]

    if actual_epochs != expected_epochs:
        return False

    best = None
    epochs_without_improvement = 0

    for item in history:
        value = float(
            item["macro_roc_auc"]
        )

        if (
            best is None
            or value > best + min_delta
        ):
            best = value
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            return True

    return actual_epochs[-1] >= max_epochs


def main() -> None:
    args = parse_args()

    seeds = tuple(
        int(seed)
        for seed in args.seeds
    )

    fractions = tuple(
        int(fraction)
        for fraction in args.fractions
    )

    validate_inputs(
        seeds,
        fractions,
    )

    total_runs = (
        len(seeds)
        * len(fractions)
    )

    print(
        f"Total runs: {total_runs}"
    )

    print(
        f"Seeds: {seeds}"
    )

    print(
        f"Fractions: {fractions}"
    )

    print(
        f"Max epochs: {args.epochs}"
    )

    print(
        f"Batch size: {args.batch_size}"
    )

    print(
        f"Learning rate: {args.learning_rate}"
    )

    print(
        f"Weight decay: {args.weight_decay}"
    )

    print(
        f"Patience: {args.patience}"
    )

    print(
        f"Min delta: {args.min_delta}"
    )

    print(
        f"Dry run: {args.dry_run}"
    )

    print(
        f"Skip existing: {args.skip_existing}"
    )

    run_number = 0
    completed = 0
    skipped = 0

    for fraction in fractions:
        for seed in seeds:
            run_number += 1

            _, checkpoint_path, history_path = (
                get_run_paths(
                    seed=seed,
                    fraction=fraction,
                )
            )

            checkpoint_exists = (
                checkpoint_path.is_file()
            )

            history_exists = (
                history_path.is_file()
            )

            print()
            print(
                "=" * 80
            )

            print(
                f"Run {run_number}/{total_runs}"
            )

            print(
                f"Fraction: {fraction}%"
            )

            print(
                f"Seed: {seed}"
            )

            run_complete = (
                checkpoint_exists
                and history_exists
                and is_run_complete(
                    history_path=history_path,
                    max_epochs=args.epochs,
                    patience=args.patience,
                    min_delta=args.min_delta,
                )
            )

            if (
                args.skip_existing
                and run_complete
            ):
                skipped += 1

                print(
                    "Completed existing run skipped."
                )

                continue

            incomplete_existing = (
                checkpoint_exists
                or history_exists
            )

            if incomplete_existing:
                print(
                    "Incomplete existing run detected. "
                    "Restarting from scratch."
                )

                if not args.dry_run:
                    if checkpoint_exists:
                        checkpoint_path.unlink()

                    if history_exists:
                        history_path.unlink()

            command = build_command(
                seed=seed,
                fraction=fraction,
                args=args,
            )

            print(
                "Command:"
            )

            print(
                subprocess.list2cmdline(
                    command
                )
            )

            if args.dry_run:
                continue

            subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                check=True,
            )

            completed += 1

    print()
    print(
        "Experiment runner completed."
    )

    print(
        f"Completed runs: {completed}"
    )

    print(
        f"Skipped runs: {skipped}"
    )


if __name__ == "__main__":
    main()