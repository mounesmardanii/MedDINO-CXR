from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chestmnist_224"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract ChestMNIST 224x224 arrays from "
            "the original .npz archive."
        )
    )

    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to chestmnist_224.npz",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory where extracted .npy files will "
            "be stored."
        ),
    )

    return parser.parse_args()


def extract_archive(
    source_file: Path,
    output_dir: Path,
) -> None:
    source_file = source_file.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    if not source_file.is_file():
        raise FileNotFoundError(
            f"Dataset archive was not found:\n{source_file}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with zipfile.ZipFile(source_file, "r") as archive:
        members = [
            member
            for member in archive.infolist()
            if not member.is_dir()
        ]

        required_bytes = sum(
            member.file_size
            for member in members
        )

        free_bytes = shutil.disk_usage(
            output_dir
        ).free

        print("=" * 70)
        print("CHESTMNIST EXTRACTION")
        print("=" * 70)

        print(f"Source: {source_file}")
        print(f"Output: {output_dir}")

        print(
            "Required space: "
            f"{required_bytes / 1024**3:.2f} GB"
        )

        print(
            "Available space: "
            f"{free_bytes / 1024**3:.2f} GB"
        )

        if free_bytes < required_bytes + 1024**3:
            raise RuntimeError(
                "Not enough free disk space. "
                "At least 1 GB of additional free "
                "space is required."
            )

        for member in members:
            target_file = (
                output_dir
                / Path(member.filename).name
            )

            temporary_file = target_file.with_suffix(
                target_file.suffix + ".part"
            )

            if (
                target_file.exists()
                and target_file.stat().st_size
                == member.file_size
            ):
                print(
                    "Skipping existing file: "
                    f"{target_file.name}"
                )
                continue

            print(
                f"Extracting: {target_file.name}"
            )

            with archive.open(
                member,
                "r",
            ) as source:
                with temporary_file.open(
                    "wb"
                ) as destination:
                    shutil.copyfileobj(
                        source,
                        destination,
                        length=16 * 1024 * 1024,
                    )

            if (
                temporary_file.stat().st_size
                != member.file_size
            ):
                temporary_file.unlink(
                    missing_ok=True
                )

                raise IOError(
                    "Incomplete extraction: "
                    f"{target_file.name}"
                )

            temporary_file.replace(
                target_file
            )

    print("\nExtraction completed successfully.")

    print("\nExtracted files:")

    for file_path in sorted(
        output_dir.glob("*.npy")
    ):
        size_gb = (
            file_path.stat().st_size
            / 1024**3
        )

        print(
            f"{file_path.name:20s} "
            f"{size_gb:.3f} GB"
        )


def main() -> None:
    args = parse_args()

    extract_archive(
        source_file=args.source,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()