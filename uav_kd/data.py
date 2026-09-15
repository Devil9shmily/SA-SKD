"""Dataset loading and DataLoader construction."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset


def _load_tensor_file(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0
        return torch.load(path, map_location="cpu")


class GraphDataset(Dataset):
    """Load a ``.pt`` mapping containing ``x`` and binary ``y`` tensors."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Dataset split not found: {self.path}")

        payload = _load_tensor_file(self.path)
        if not isinstance(payload, dict) or "x" not in payload or "y" not in payload:
            raise ValueError(f"{self.path} must contain a mapping with 'x' and 'y'")

        self.x = torch.as_tensor(payload["x"])
        self.y = torch.as_tensor(payload["y"]).reshape(-1)
        if self.x.ndim != 4:
            raise ValueError(
                f"Expected x with shape [samples, time, nodes, features], got {tuple(self.x.shape)}"
            )
        if self.x.size(-1) != 11:
            raise ValueError(f"Expected 11 input features, got {self.x.size(-1)}")
        if self.x.size(0) != self.y.numel():
            raise ValueError("x and y contain different numbers of samples")

    def __len__(self) -> int:
        return self.y.numel()

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.x[index].float(), self.y[index].float()

    def summary(self) -> str:
        positives = int((self.y == 1).sum())
        negatives = int((self.y == 0).sum())
        return (
            f"{self.path}: samples={len(self)}, shape={tuple(self.x.shape)}, "
            f"positive={positives}, negative={negatives}"
        )


def resampled_binary_subset(
    dataset: GraphDataset,
    positive_repeat: int = 2,
    negative_to_positive: float = 2.0,
    seed: int = 42,
) -> Subset:
    """Create the optional Stage-2 class-balanced subset used by the draft code."""

    if positive_repeat < 1:
        raise ValueError("positive_repeat must be at least 1")
    if negative_to_positive <= 0:
        raise ValueError("negative_to_positive must be positive")

    positive = torch.where(dataset.y == 1)[0]
    negative = torch.where(dataset.y == 0)[0]
    if len(positive) == 0 or len(negative) == 0:
        raise ValueError("Stage-2 resampling requires both positive and negative samples")

    repeated_positive = positive.repeat(positive_repeat)
    negative_count = min(
        len(negative), int(round(len(repeated_positive) * negative_to_positive))
    )
    generator = torch.Generator().manual_seed(seed)
    chosen_negative = negative[torch.randperm(len(negative), generator=generator)[:negative_count]]
    indices = torch.cat((repeated_positive, chosen_negative))
    indices = indices[torch.randperm(len(indices), generator=generator)]
    return Subset(dataset, indices.tolist())


def load_splits(data_root: str | Path) -> tuple[GraphDataset, GraphDataset, GraphDataset]:
    root = Path(data_root)
    return tuple(GraphDataset(root / f"{split}.pt") for split in ("train", "val", "test"))


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
    )


def make_loaders(
    datasets: Sequence[Dataset],
    batch_size: int,
    workers: int,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    if len(datasets) != 3:
        raise ValueError("datasets must contain train, validation, and test splits")
    return (
        make_loader(datasets[0], batch_size, True, workers, pin_memory),
        make_loader(datasets[1], batch_size, False, workers, pin_memory),
        make_loader(datasets[2], batch_size, False, workers, pin_memory),
    )
