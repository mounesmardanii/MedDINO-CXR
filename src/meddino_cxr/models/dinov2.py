from __future__ import annotations

import torch
import torch.nn as nn


DINOV2_REPO = "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINOV2_MODEL = "dinov2_vits14"
DINOV2_FEATURE_DIM = 384


def build_dinov2_vits14(
    freeze: bool = True,
) -> nn.Module:
    model = torch.hub.load(
        DINOV2_REPO,
        DINOV2_MODEL,
    )

    if freeze:
        for parameter in model.parameters():
            parameter.requires_grad = False

        model.eval()

    return model