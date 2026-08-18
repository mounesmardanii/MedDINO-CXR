import tarfile
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "nih"
SHARD_ROOT = DATA_ROOT / "hf_shards" / "data"
METADATA_PATH = DATA_ROOT / "Data_Entry_2017_v2020.csv"

EXPECTED_SHARDS = 113
EXPECTED_IMAGES = 112120


def main():
    metadata = pd.read_csv(
        METADATA_PATH,
        usecols=["Image Index"],
    )

    expected = set(
        metadata["Image Index"].astype(str)
    )

    shards = sorted(
        SHARD_ROOT.glob("*.tar")
    )

    if len(shards) != EXPECTED_SHARDS:
        raise RuntimeError(
            f"Expected {EXPECTED_SHARDS} shards, found {len(shards)}"
        )

    seen = set()
    total = 0
    duplicate_count = 0

    for index, shard in enumerate(
        shards,
        start=1,
    ):
        print(
            f"Checking {index:03d}/{len(shards):03d}: {shard.name}"
        )

        with tarfile.open(
            shard,
            "r",
        ) as archive:
            members = [
                member
                for member in archive
                if member.isfile()
            ]

        for member in members:
            name = Path(
                member.name
            ).name

            total += 1

            if name in seen:
                duplicate_count += 1

            seen.add(name)

    missing = expected - seen
    extra = seen - expected

    print()
    print(f"Total archive images: {total}")
    print(f"Unique archive images: {len(seen)}")
    print(f"Expected metadata images: {len(expected)}")
    print(f"Duplicate images: {duplicate_count}")
    print(f"Missing images: {len(missing)}")
    print(f"Extra images: {len(extra)}")

    if total != EXPECTED_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_IMAGES} total images, found {total}"
        )

    if len(seen) != EXPECTED_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_IMAGES} unique images, found {len(seen)}"
        )

    if duplicate_count != 0:
        raise RuntimeError(
            f"Found {duplicate_count} duplicate images"
        )

    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} metadata images"
        )

    if extra:
        raise RuntimeError(
            f"Found {len(extra)} unexpected images"
        )

    print()
    print("NIH shard integrity audit passed.")


if __name__ == "__main__":
    main()