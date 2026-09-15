"""Selective feature and attention distillation objectives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


FEATURE_MODES: dict[str, tuple[str, ...]] = {
    "b1": ("block1",),
    "b2": ("block2",),
    "b3": ("block3",),
    "b1_b2": ("block1", "block2"),
    "b1_b3": ("block1", "block3"),
    "b2_b3": ("block2", "block3"),
    "b1_b2_b3": ("block1", "block2", "block3"),
}

ATTENTION_MODES: dict[str, tuple[str, ...]] = {
    "base": (),
    "ta_b1": ("ta1",),
    "sa_b2": ("sa2",),
    "ta_b3": ("ta3",),
    "ta_b1_ta_b3": ("ta1", "ta3"),
    "sa_b2_ta_b3": ("sa2", "ta3"),
    "ta_b1_sa_b2": ("ta1", "sa2"),
}


class FeatureAdapters(nn.Module):
    """Align student channels (64/128/256) with teacher channels (128/256/512)."""

    def __init__(self) -> None:
        super().__init__()
        self.adapters = nn.ModuleDict(
            {
                "block1": nn.Conv2d(64, 128, kernel_size=1),
                "block2": nn.Conv2d(128, 256, kernel_size=1),
                "block3": nn.Conv2d(256, 512, kernel_size=1),
            }
        )

    def forward(self, name: str, feature: Tensor) -> Tensor:
        return self.adapters[name](feature)


def _sum_losses(losses: Sequence[Tensor], reference: Tensor) -> Tensor:
    if not losses:
        return reference.new_zeros(())
    return torch.stack(tuple(losses)).sum()


def feature_distillation_loss(
    student_features: Mapping[str, Tensor],
    teacher_features: Mapping[str, Tensor],
    adapters: FeatureAdapters,
    mode: str,
) -> Tensor:
    if mode not in FEATURE_MODES:
        raise ValueError(f"Unknown feature mode: {mode}")
    losses = [
        F.mse_loss(adapters(name, student_features[name]), teacher_features[name].detach())
        for name in FEATURE_MODES[mode]
    ]
    return _sum_losses(losses, student_features["logits"])


def attention_distillation_loss(
    student_features: Mapping[str, Tensor],
    teacher_features: Mapping[str, Tensor],
    mode: str,
) -> Tensor:
    if mode not in ATTENTION_MODES:
        raise ValueError(f"Unknown attention mode: {mode}")
    losses = [
        F.mse_loss(student_features[name], teacher_features[name].detach())
        for name in ATTENTION_MODES[mode]
    ]
    return _sum_losses(losses, student_features["logits"])
