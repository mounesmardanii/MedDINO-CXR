from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


def _to_numpy(
    array: np.ndarray | torch.Tensor,
) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()

    return np.asarray(array)


def compute_multilabel_metrics(
    targets: np.ndarray | torch.Tensor,
    probabilities: np.ndarray | torch.Tensor,
) -> dict:
    targets = _to_numpy(targets)
    probabilities = _to_numpy(probabilities)

    if targets.shape != probabilities.shape:
        raise ValueError(
            f"Shape mismatch: {targets.shape} vs "
            f"{probabilities.shape}"
        )

    if targets.ndim != 2:
        raise ValueError(
            "Expected arrays with shape (N, C)."
        )

    per_class_roc_auc = []
    per_class_average_precision = []

    for class_index in range(targets.shape[1]):
        class_targets = targets[:, class_index]
        class_probabilities = probabilities[:, class_index]

        if np.unique(class_targets).size < 2:
            roc_auc = np.nan
        else:
            roc_auc = roc_auc_score(
                class_targets,
                class_probabilities,
            )

        if class_targets.sum() == 0:
            average_precision = np.nan
        else:
            average_precision = average_precision_score(
                class_targets,
                class_probabilities,
            )

        per_class_roc_auc.append(
            float(roc_auc)
        )

        per_class_average_precision.append(
            float(average_precision)
        )

    roc_array = np.asarray(
        per_class_roc_auc,
        dtype=np.float64,
    )

    ap_array = np.asarray(
        per_class_average_precision,
        dtype=np.float64,
    )

    macro_roc_auc = float(
        np.nanmean(roc_array)
    )

    macro_average_precision = float(
        np.nanmean(ap_array)
    )

    micro_roc_auc = float(
        roc_auc_score(
            targets.reshape(-1),
            probabilities.reshape(-1),
        )
    )

    micro_average_precision = float(
        average_precision_score(
            targets.reshape(-1),
            probabilities.reshape(-1),
        )
    )

    return {
        "macro_roc_auc": macro_roc_auc,
        "micro_roc_auc": micro_roc_auc,
        "macro_average_precision": macro_average_precision,
        "micro_average_precision": micro_average_precision,
        "per_class_roc_auc": per_class_roc_auc,
        "per_class_average_precision": (
            per_class_average_precision
        ),
    }