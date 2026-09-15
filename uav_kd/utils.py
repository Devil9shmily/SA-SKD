"""Reproducibility, device, and checkpoint helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch
from torch import nn


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _safe_torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch < 2.0
        return torch.load(path, map_location=device)


def remove_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }


def load_model_checkpoint(
    model: nn.Module,
    path: str | Path,
    device: torch.device,
    keys: Iterable[str] = ("model", "student", "teacher", "state_dict"),
) -> dict:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    payload = _safe_torch_load(checkpoint_path, device)
    state_dict = payload
    if isinstance(payload, dict):
        for key in keys:
            if key in payload and isinstance(payload[key], dict):
                state_dict = payload[key]
                break
    if not isinstance(state_dict, dict):
        raise ValueError(f"No state dictionary found in {checkpoint_path}")
    model.load_state_dict(remove_module_prefix(state_dict), strict=True)
    return payload if isinstance(payload, dict) else {"state_dict": payload}


def save_checkpoint(payload: dict, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)


def serializable_config(args) -> dict:
    """Convert argparse values to checkpoint-safe primitive values."""

    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def print_run_header(
    name: str,
    args,
    device: torch.device,
    models: Mapping[str, nn.Module],
) -> None:
    print(f"\n{name}")
    print(f"Device: {device}")
    for model_name, model in models.items():
        print(f"{model_name} parameters: {count_parameters(model) / 1e6:.3f}M")
    print("Configuration:")
    print(json.dumps(vars(args), indent=2, default=str))
