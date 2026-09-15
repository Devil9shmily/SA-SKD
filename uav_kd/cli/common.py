"""Shared command-line helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from uav_kd.data import load_splits, make_loaders


def add_training_arguments(
    parser: argparse.ArgumentParser,
    *,
    output: str,
    learning_rate: float,
) -> None:
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Directory containing train.pt, val.pt, and test.pt.",
    )
    parser.add_argument("--output", type=Path, default=Path(output))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=learning_rate)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--pos-weight", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")


def build_loaders(args, device: torch.device, train_dataset=None):
    train, validation, test = load_splits(args.data_root)
    for dataset in (train, validation, test):
        print(dataset.summary())
    if train_dataset is not None:
        train = train_dataset(train)
        print(f"Resampled training samples: {len(train)}")
    return make_loaders(
        (train, validation, test),
        batch_size=args.batch_size,
        workers=args.workers,
        pin_memory=device.type == "cuda",
    )


def binary_criterion(pos_weight: float, device: torch.device) -> nn.Module:
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
