from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEEDS = (
    42,
    47,
    52,
)

FRACTIONS = (
    1,
    5,
    10,
    25,
    100,
)

EPOCHS = 100

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
        "--dry-run",
        action="store_true",
    )

    return parser.parse_args()


def get_run_name(
    seed,
    fraction,
):
    return (
        f"seed_{seed}_"
        f"fraction_{FRACTION_CODES[fraction]}"
    )


def get_paths(
    seed,
    fraction,
):
    name = get_run_name(
        seed,
        fraction,
    )

    output_dir = (
        PROJECT_ROOT
        / "outputs"
        / "nih_dinov2_linear_probe"
        / name
    )

    status_path = (
        output_dir
        / "run_status.json"
    )

    history_path = (
        output_dir
        / "history.json"
    )

    checkpoint_path = (
        PROJECT_ROOT
        / "checkpoints"
        / "nih_dinov2_linear_probe"
        / f"{name}.pt"
    )

    return (
        name,
        status_path,
        history_path,
        checkpoint_path,
    )


def is_complete(
    seed,
    fraction,
):
    (
        name,
        status_path,
        history_path,
        checkpoint_path,
    ) = get_paths(
        seed,
        fraction,
    )

    if status_path.is_file():
        try:
            with status_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                status = json.load(file)

            if (
                status.get("completed") is True
                and checkpoint_path.is_file()
            ):
                return True

        except Exception:
            pass

    if (
        history_path.is_file()
        and checkpoint_path.is_file()
    ):
        try:
            with history_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                history = json.load(file)

            if (
                history
                and int(
                    history[-1]["epoch"]
                ) >= EPOCHS
            ):
                return True

        except Exception:
            pass

    return False


def build_command(
    seed,
    fraction,
):
    return [
        sys.executable,
        str(
            PROJECT_ROOT
            / "scripts"
            / "train_nih_dinov2_linear_probe.py"
        ),
        "--fraction",
        str(fraction),
        "--seed",
        str(seed),
        "--epochs",
        str(EPOCHS),
        "--device",
        "cpu",
    ]


def main():
    args = parse_args()

    jobs = [
        (
            fraction,
            seed,
        )
        for fraction in FRACTIONS
        for seed in SEEDS
    ]

    print(
        "Total experiments:",
        len(jobs),
    )

    print(
        "Max epochs:",
        EPOCHS,
    )

    print(
        "Dry run:",
        args.dry_run,
    )

    completed_before = 0
    executed = 0

    for index, (
        fraction,
        seed,
    ) in enumerate(
        jobs,
        start=1,
    ):
        name = get_run_name(
            seed,
            fraction,
        )

        print()
        print(
            "=" * 70
        )

        print(
            f"Experiment {index}/"
            f"{len(jobs)}"
        )

        print(
            "Run:",
            name,
        )

        if is_complete(
            seed,
            fraction,
        ):
            print(
                "Status: SKIP"
            )

            completed_before += 1
            continue

        print(
            "Status: RUN"
        )

        command = build_command(
            seed,
            fraction,
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

        if not is_complete(
            seed,
            fraction,
        ):
            raise RuntimeError(
                f"Run did not complete correctly: {name}"
            )

        executed += 1

    print()
    print(
        "=" * 70
    )

    print(
        "Runner completed."
    )

    print(
        "Already completed:",
        completed_before,
    )

    print(
        "Executed now:",
        executed,
    )


if __name__ == "__main__":
    main()