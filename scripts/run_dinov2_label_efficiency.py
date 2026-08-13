from __future__ import annotations

import argparse
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
            "Run systematic DINOv2 linear-probe "
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
        default=1024,
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
        "--dry-run",
        action="store_true",
    )

    return parser.parse_args()


def fraction_key(
    fraction: int,
) -> str:
    return f"fraction_{fraction:03d}"


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

    run_dir = (
        PROJECT_ROOT
        / "outputs"
        / "label_efficiency"
        / "dinov2"
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

    return [
        sys.executable,
        str(
            PROJECT_ROOT
            / "scripts"
            / "train_dinov2_linear_probe.py"
        ),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
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
        f"Dry run: {args.dry_run}"
    )

    run_number = 0

    for fraction in fractions:
        for seed in seeds:
            run_number += 1

            command = build_command(
                seed=seed,
                fraction=fraction,
                args=args,
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

    print()
    print(
        "Experiment runner completed."
    )


if __name__ == "__main__":
    main()