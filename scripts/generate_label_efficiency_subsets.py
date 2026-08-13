from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from meddino_cxr.data import (
    DEFAULT_FRACTIONS,
    build_nested_multilabel_subsets,
    compute_class_statistics,
    hash_indices,
    validate_nested_subsets,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLASS_NAMES = (
    "atelectasis",
    "cardiomegaly",
    "effusion",
    "infiltration",
    "mass",
    "nodule",
    "pneumonia",
    "pneumothorax",
    "consolidation",
    "edema",
    "emphysema",
    "fibrosis",
    "pleural",
    "hernia",
)

DEFAULT_SEEDS = (
    42,
    47,
    52,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate reproducible nested multi-label "
            "ChestMNIST training subsets."
        )
    )

    parser.add_argument(
        "--labels-path",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "processed"
            / "chestmnist_224"
            / "train_labels.npy"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "label_efficiency"
            / "subsets"
        ),
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results"
            / "label_efficiency"
        ),
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
    )

    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=list(DEFAULT_FRACTIONS),
    )

    return parser.parse_args()


def hash_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:
        while True:
            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def fraction_key(
    fraction: float,
) -> str:
    return (
        f"fraction_"
        f"{int(round(fraction * 100)):03d}"
    )


def save_csv(
    path: Path,
    rows: list[dict],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def main() -> None:
    args = parse_args()

    labels_path = (
        args.labels_path
        .expanduser()
        .resolve()
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    results_dir = (
        args.results_dir
        .expanduser()
        .resolve()
    )

    if not labels_path.is_file():
        raise FileNotFoundError(
            f"Labels file not found: {labels_path}"
        )

    labels = np.load(
        labels_path,
        mmap_mode="r",
    )

    if labels.ndim != 2:
        raise ValueError(
            "Expected labels with shape (N, C)."
        )

    if labels.shape[1] != len(
        CLASS_NAMES
    ):
        raise ValueError(
            "Unexpected number of ChestMNIST classes."
        )

    fractions = tuple(
        float(value)
        for value in args.fractions
    )

    seeds = tuple(
        int(value)
        for value in args.seeds
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    full_indices = np.arange(
        len(labels),
        dtype=np.int64,
    )

    full_positives, full_prevalence = (
        compute_class_statistics(
            labels,
            full_indices,
        )
    )

    manifest_rows = []
    class_rows = []

    for seed in seeds:
        subsets = (
            build_nested_multilabel_subsets(
                labels=labels,
                fractions=fractions,
                seed=seed,
            )
        )

        validate_nested_subsets(
            subsets
        )

        archive_path = (
            output_dir
            / f"seed_{seed}.npz"
        )

        archive_arrays = {
            fraction_key(
                fraction
            ): indices
            for fraction, indices in subsets.items()
        }

        np.savez_compressed(
            archive_path,
            **archive_arrays,
        )

        for fraction, indices in subsets.items():
            positives, prevalence = (
                compute_class_statistics(
                    labels,
                    indices,
                )
            )

            manifest_rows.append(
                {
                    "seed": seed,
                    "fraction": fraction,
                    "samples": len(indices),
                    "index_hash_sha256": hash_indices(
                        indices
                    ),
                    "archive": (
                        archive_path
                        .relative_to(
                            PROJECT_ROOT
                        )
                        .as_posix()
                    ),
                    "archive_key": fraction_key(
                        fraction
                    ),
                }
            )

            for class_index, class_name in enumerate(
                CLASS_NAMES
            ):
                class_rows.append(
                    {
                        "seed": seed,
                        "fraction": fraction,
                        "class_index": class_index,
                        "class_name": class_name,
                        "positive_count": int(
                            positives[
                                class_index
                            ]
                        ),
                        "prevalence": float(
                            prevalence[
                                class_index
                            ]
                        ),
                        "full_positive_count": int(
                            full_positives[
                                class_index
                            ]
                        ),
                        "full_prevalence": float(
                            full_prevalence[
                                class_index
                            ]
                        ),
                        "prevalence_delta": float(
                            prevalence[
                                class_index
                            ]
                            - full_prevalence[
                                class_index
                            ]
                        ),
                    }
                )

    save_csv(
        results_dir
        / "subset_manifest.csv",
        manifest_rows,
        [
            "seed",
            "fraction",
            "samples",
            "index_hash_sha256",
            "archive",
            "archive_key",
        ],
    )

    save_csv(
        results_dir
        / "class_distribution.csv",
        class_rows,
        [
            "seed",
            "fraction",
            "class_index",
            "class_name",
            "positive_count",
            "prevalence",
            "full_positive_count",
            "full_prevalence",
            "prevalence_delta",
        ],
    )

    protocol = {
        "dataset": "ChestMNIST",
        "split": "train",
        "total_samples": int(
            len(labels)
        ),
        "num_classes": int(
            labels.shape[1]
        ),
        "fractions": list(
            fractions
        ),
        "seeds": list(
            seeds
        ),
        "nested": True,
        "stratification": (
            "MultilabelStratifiedShuffleSplit"
        ),
        "iterative_stratification_version": (
            importlib.metadata.version(
                "iterative-stratification"
            )
        ),
        "labels_sha256": hash_file(
            labels_path
        ),
    }

    with (
        results_dir
        / "subset_protocol.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            protocol,
            file,
            indent=2,
        )

    print(
        f"Labels: {labels.shape}"
    )

    print(
        f"Seeds: {seeds}"
    )

    print(
        f"Fractions: {fractions}"
    )

    print(
        f"Local subset archives: {output_dir}"
    )

    print(
        f"Tracked audit results: {results_dir}"
    )


if __name__ == "__main__":
    main()