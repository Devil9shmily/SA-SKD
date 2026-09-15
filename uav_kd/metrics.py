"""Binary accident-detection metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class BinaryMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"Acc {self.accuracy:.4f} | Prec {self.precision:.4f} | "
            f"Recall {self.recall:.4f} | F1 {self.f1:.4f} | AUC {self.auc:.4f}"
        )


def compute_metrics(labels, probabilities, threshold: float = 0.5) -> BinaryMetrics:
    labels = np.asarray(labels).reshape(-1)
    probabilities = np.asarray(probabilities).reshape(-1)
    predictions = (probabilities > threshold).astype(np.int64)
    try:
        auc = float(roc_auc_score(labels, probabilities))
    except ValueError:
        auc = float("nan")
    return BinaryMetrics(
        accuracy=float(accuracy_score(labels, predictions)),
        precision=float(precision_score(labels, predictions, zero_division=0)),
        recall=float(recall_score(labels, predictions, zero_division=0)),
        f1=float(f1_score(labels, predictions, zero_division=0)),
        auc=auc,
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> BinaryMetrics:
    model.eval()
    labels: list[float] = []
    probabilities: list[float] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True).permute(0, 3, 1, 2)
        logits = model(x)
        probabilities.extend(torch.sigmoid(logits).detach().cpu().reshape(-1).tolist())
        labels.extend(y.detach().cpu().reshape(-1).tolist())
    return compute_metrics(labels, probabilities)
