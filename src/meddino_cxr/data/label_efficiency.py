from __future__ import annotations

import hashlib

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


DEFAULT_FRACTIONS = (
    0.05,
    0.10,
    0.25,
    0.50,
    1.00,
)


def validate_labels(
    labels: np.ndarray,
) -> None:
    if labels.ndim != 2:
        raise ValueError(
            "labels must have shape (N, C)."
        )

    if len(labels) == 0:
        raise ValueError(
            "labels must not be empty."
        )

    unique_values = np.unique(labels)

    if not np.all(
        np.isin(
            unique_values,
            [0, 1],
        )
    ):
        raise ValueError(
            "labels must contain only binary values."
        )


def validate_fractions(
    fractions: tuple[float, ...],
) -> tuple[float, ...]:
    if not fractions:
        raise ValueError(
            "At least one fraction is required."
        )

    normalized = tuple(
        sorted(
            {
                float(value)
                for value in fractions
            }
        )
    )

    for value in normalized:
        if not 0.0 < value <= 1.0:
            raise ValueError(
                "Fractions must be in the interval (0, 1]."
            )

    if 1.0 not in normalized:
        raise ValueError(
            "Fractions must include 1.0."
        )

    return normalized


def compute_target_size(
    total_samples: int,
    fraction: float,
) -> int:
    target_size = int(
        round(
            total_samples
            * fraction
        )
    )

    return max(
        1,
        min(
            total_samples,
            target_size,
        ),
    )


def select_stratified_subset(
    candidate_indices: np.ndarray,
    labels: np.ndarray,
    target_size: int,
    random_state: int,
) -> np.ndarray:
    if target_size > len(
        candidate_indices
    ):
        raise ValueError(
            "target_size cannot exceed the candidate pool."
        )

    if target_size == len(
        candidate_indices
    ):
        return np.sort(
            candidate_indices.copy()
        )

    candidate_labels = labels[
        candidate_indices
    ]

    remaining_size = (
        len(candidate_indices)
        - target_size
    )

    splitter = (
        MultilabelStratifiedShuffleSplit(
            n_splits=1,
            train_size=target_size,
            test_size=remaining_size,
            random_state=random_state,
        )
    )

    local_indices = np.arange(
        len(candidate_indices)
    ).reshape(-1, 1)

    selected_local, _ = next(
        splitter.split(
            local_indices,
            candidate_labels,
        )
    )

    selected = candidate_indices[
        selected_local
    ]

    return np.sort(
        selected.astype(
            np.int64,
            copy=False,
        )
    )


def build_nested_multilabel_subsets(
    labels: np.ndarray,
    fractions: tuple[float, ...] = DEFAULT_FRACTIONS,
    seed: int = 42,
) -> dict[float, np.ndarray]:
    labels = np.asarray(labels)

    validate_labels(
        labels
    )

    fractions = validate_fractions(
        fractions
    )

    total_samples = len(labels)

    current_indices = np.arange(
        total_samples,
        dtype=np.int64,
    )

    subsets: dict[
        float,
        np.ndarray,
    ] = {
        1.0: current_indices.copy()
    }

    descending_fractions = sorted(
        (
            value
            for value in fractions
            if value < 1.0
        ),
        reverse=True,
    )

    for stage_index, fraction in enumerate(
        descending_fractions,
        start=1,
    ):
        target_size = compute_target_size(
            total_samples,
            fraction,
        )

        current_indices = select_stratified_subset(
            candidate_indices=current_indices,
            labels=labels,
            target_size=target_size,
            random_state=seed + stage_index,
        )

        subsets[fraction] = (
            current_indices.copy()
        )

    return {
        fraction: subsets[
            fraction
        ]
        for fraction in fractions
    }


def validate_nested_subsets(
    subsets: dict[
        float,
        np.ndarray,
    ],
) -> None:
    fractions = sorted(
        subsets
    )

    for fraction in fractions:
        indices = subsets[
            fraction
        ]

        if len(
            np.unique(
                indices
            )
        ) != len(
            indices
        ):
            raise ValueError(
                f"Duplicate indices found in fraction {fraction}."
            )

    for smaller, larger in zip(
        fractions[:-1],
        fractions[1:],
        strict=True,
    ):
        smaller_indices = subsets[
            smaller
        ]

        larger_indices = subsets[
            larger
        ]

        if not np.all(
            np.isin(
                smaller_indices,
                larger_indices,
            )
        ):
            raise ValueError(
                f"Fraction {smaller} is not nested inside {larger}."
            )


def compute_class_statistics(
    labels: np.ndarray,
    indices: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    selected_labels = labels[
        indices
    ]

    positives = selected_labels.sum(
        axis=0,
        dtype=np.int64,
    )

    prevalence = selected_labels.mean(
        axis=0,
        dtype=np.float64,
    )

    return (
        positives,
        prevalence,
    )


def hash_indices(
    indices: np.ndarray,
) -> str:
    normalized = np.asarray(
        indices,
        dtype=np.int64,
    )

    return hashlib.sha256(
        normalized.tobytes()
    ).hexdigest()