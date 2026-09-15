"""Shared training loops."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_supervised_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    loss_sum = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True).permute(0, 3, 1, 2)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    return loss_sum / max(len(loader), 1)
