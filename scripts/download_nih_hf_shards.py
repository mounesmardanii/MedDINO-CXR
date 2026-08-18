import argparse
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "yeigen/nih-chest-xray"
REPO_TYPE = "dataset"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "raw" / "nih" / "hf_shards"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--end",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=5,
    )

    return parser.parse_args()


def get_shards():
    files = HfApi().list_repo_files(
        REPO_ID,
        repo_type=REPO_TYPE,
    )

    shards = sorted(
        path
        for path in files
        if path.startswith("data/")
        and path.endswith(".tar")
    )

    if len(shards) != 113:
        raise RuntimeError(
            f"Expected 113 shards, found {len(shards)}"
        )

    return shards


def download_shard(filename, retries):
    target = OUTPUT_ROOT / filename

    if target.is_file() and target.stat().st_size > 0:
        print(
            f"Already present: {filename} "
            f"({target.stat().st_size / 1024**2:.1f} MB)"
        )
        return "skipped"

    for attempt in range(
        1,
        retries + 1,
    ):
        try:
            print()
            print(
                f"Downloading {filename} "
                f"(attempt {attempt}/{retries})"
            )

            path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type=REPO_TYPE,
                local_dir=OUTPUT_ROOT,
            )

            path = Path(path)

            if not path.is_file():
                raise RuntimeError(
                    f"Downloaded file missing: {path}"
                )

            if path.stat().st_size == 0:
                raise RuntimeError(
                    f"Downloaded file is empty: {path}"
                )

            print(
                f"Completed: {filename} "
                f"({path.stat().st_size / 1024**2:.1f} MB)"
            )

            return "completed"

        except Exception as error:
            print(
                f"Failed: {filename}: {error}"
            )

            if attempt == retries:
                raise

            time.sleep(
                min(
                    30,
                    attempt * 5,
                )
            )

    raise RuntimeError(
        f"Unable to download {filename}"
    )


def main():
    args = parse_args()
    shards = get_shards()

    start = args.start

    end = (
        len(shards) - 1
        if args.end is None
        else args.end
    )

    if start < 0:
        raise RuntimeError(
            "--start must be >= 0"
        )

    if end >= len(shards):
        raise RuntimeError(
            f"--end must be <= {len(shards) - 1}"
        )

    if start > end:
        raise RuntimeError(
            "--start cannot exceed --end"
        )

    selected = shards[
        start:end + 1
    ]

    print(
        f"Repository shards: {len(shards)}"
    )
    print(
        f"Selected shards: {len(selected)}"
    )
    print(
        f"Range: {start} to {end}"
    )
    print(
        f"Output: {OUTPUT_ROOT}"
    )

    completed = 0
    skipped = 0

    for index, filename in enumerate(
        selected,
        start=start,
    ):
        print()
        print(
            "=" * 70
        )
        print(
            f"Shard {index:03d}/112"
        )

        result = download_shard(
            filename,
            args.retries,
        )

        if result == "completed":
            completed += 1
        else:
            skipped += 1

    print()
    print(
        "NIH shard downloader completed."
    )
    print(
        f"Completed downloads: {completed}"
    )
    print(
        f"Skipped existing: {skipped}"
    )


if __name__ == "__main__":
    main()