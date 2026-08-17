from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FRACTIONS = (
    5,
    10,
    25,
    50,
    100,
)

SEEDS = (
    42,
    47,
    52,
)

MODELS = (
    "dinov2",
    "resnet18",
)

EXPECTED_TEST_SAMPLES = 22433
EXPECTED_CLASSES = 14


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run final frozen test evaluation."
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        default=list(MODELS),
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


def get_checkpoint_path(
    model,
    fraction,
    seed,
):
    model_dir = (
        "dinov2"
        if model == "dinov2"
        else "resnet"
    )

    return (
        PROJECT_ROOT
        / "outputs"
        / "label_efficiency"
        / model_dir
        / f"fraction_{fraction:03d}"
        / f"seed_{seed}"
        / "best.pt"
    )


def get_output_dir(
    model,
    fraction,
    seed,
):
    return (
        PROJECT_ROOT
        / "outputs"
        / "final_test"
        / model
        / f"fraction_{fraction:03d}"
        / f"seed_{seed}"
    )


def get_output_paths(
    model,
    fraction,
    seed,
):
    output_dir = get_output_dir(
        model,
        fraction,
        seed,
    )

    return (
        output_dir,
        output_dir / "test_metrics.json",
        output_dir / "test_per_class_metrics.csv",
    )


def validate_checkpoint_paths(models):
    for model in models:
        for fraction in FRACTIONS:
            for seed in SEEDS:
                path = get_checkpoint_path(
                    model,
                    fraction,
                    seed,
                )

                if not path.is_file():
                    raise FileNotFoundError(
                        f"Missing checkpoint: {path}"
                    )


def evaluation_complete(
    model,
    fraction,
    seed,
):
    checkpoint_path = get_checkpoint_path(
        model,
        fraction,
        seed,
    )

    (
        _,
        metrics_path,
        per_class_path,
    ) = get_output_paths(
        model,
        fraction,
        seed,
    )

    if not metrics_path.is_file():
        return False

    if not per_class_path.is_file():
        return False

    try:
        with metrics_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            metrics = json.load(file)

        if metrics.get("split") != "test":
            return False

        if int(metrics.get("samples", -1)) != EXPECTED_TEST_SAMPLES:
            return False

        stored_checkpoint = Path(
            metrics.get("checkpoint", "")
        ).resolve()

        if stored_checkpoint != checkpoint_path.resolve():
            return False

        required_metrics = (
            "macro_roc_auc",
            "micro_roc_auc",
            "macro_average_precision",
            "micro_average_precision",
        )

        for metric in required_metrics:
            if metric not in metrics:
                return False

        per_class = metrics.get(
            "per_class",
            []
        )

        if len(per_class) != EXPECTED_CLASSES:
            return False

        with per_class_path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as file:
            rows = list(
                csv.DictReader(file)
            )

        if len(rows) != EXPECTED_CLASSES:
            return False

    except (
        OSError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
    ):
        return False

    return True


def build_command(
    model,
    fraction,
    seed,
):
    checkpoint_path = get_checkpoint_path(
        model,
        fraction,
        seed,
    )

    output_dir = get_output_dir(
        model,
        fraction,
        seed,
    )

    if model == "dinov2":
        script = (
            PROJECT_ROOT
            / "scripts"
            / "evaluate_dinov2_linear_probe.py"
        )

        return [
            sys.executable,
            str(script),
            "--checkpoint",
            str(checkpoint_path),
            "--output-dir",
            str(output_dir),
            "--split",
            "test",
            "--batch-size",
            "1024",
            "--num-workers",
            "0",
        ]

    script = (
        PROJECT_ROOT
        / "scripts"
        / "evaluate_resnet.py"
    )

    return [
        sys.executable,
        str(script),
        "--checkpoint",
        str(checkpoint_path),
        "--output-dir",
        str(output_dir),
        "--split",
        "test",
        "--batch-size",
        "16",
        "--num-workers",
        "0",
    ]


def main():
    args = parse_args()

    models = tuple(args.models)

    validate_checkpoint_paths(
        models
    )

    total_runs = (
        len(models)
        * len(FRACTIONS)
        * len(SEEDS)
    )

    print(
        f"Total evaluations: {total_runs}"
    )

    print(
        f"Models: {models}"
    )

    print(
        f"Fractions: {FRACTIONS}"
    )

    print(
        f"Seeds: {SEEDS}"
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

    for model in models:
        for fraction in FRACTIONS:
            for seed in SEEDS:
                run_number += 1

                print()
                print(
                    "=" * 80
                )

                print(
                    f"Evaluation {run_number}/{total_runs}"
                )

                print(
                    f"Model: {model}"
                )

                print(
                    f"Fraction: {fraction}%"
                )

                print(
                    f"Seed: {seed}"
                )

                complete = evaluation_complete(
                    model,
                    fraction,
                    seed,
                )

                if (
                    args.skip_existing
                    and complete
                ):
                    skipped += 1

                    print(
                        "Completed existing evaluation skipped."
                    )

                    continue

                (
                    output_dir,
                    metrics_path,
                    per_class_path,
                ) = get_output_paths(
                    model,
                    fraction,
                    seed,
                )

                if (
                    metrics_path.exists()
                    or per_class_path.exists()
                ):
                    print(
                        "Incomplete existing evaluation detected."
                    )

                    if not args.dry_run:
                        if metrics_path.exists():
                            metrics_path.unlink()

                        if per_class_path.exists():
                            per_class_path.unlink()

                command = build_command(
                    model,
                    fraction,
                    seed,
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

                output_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    check=True,
                )

                if not evaluation_complete(
                    model,
                    fraction,
                    seed,
                ):
                    raise RuntimeError(
                        "Evaluation completed but output validation failed: "
                        f"model={model}, fraction={fraction}, seed={seed}"
                    )

                completed += 1

    print()
    print(
        "Final test evaluation runner completed."
    )

    print(
        f"Completed evaluations: {completed}"
    )

    print(
        f"Skipped evaluations: {skipped}"
    )


if __name__ == "__main__":
    main()