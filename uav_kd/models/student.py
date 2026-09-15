"""Lightweight student ST-GCN (approximately 3.1M parameters)."""

from __future__ import annotations

from typing import Dict, Tuple

from torch import Tensor, nn

from .graph import DynamicGraphGenerator, SpatialAttention, SpatialGraphConv, TemporalAttention
from .temporal import TemporalConvUnit


class StudentBlock1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_proj = nn.Conv2d(11, 64, kernel_size=1)
        self.ta = TemporalAttention(64)
        self.t1 = TemporalConvUnit(64, 1)
        self.t2 = TemporalConvUnit(64, 2)
        self.t3 = TemporalConvUnit(64, 4)
        self.s1 = SpatialGraphConv(64, 64)
        self.t4 = TemporalConvUnit(64, 2)
        self.t5 = TemporalConvUnit(64, 1)

    def forward(self, x: Tensor, adjacency: Tensor, capture: bool = False):
        x = self.input_proj(x)
        if capture:
            x, temporal_attention = self.ta(x, return_weights=True)
        else:
            x = self.ta(x)
            temporal_attention = None
        x = self.t3(self.t2(self.t1(x)))
        x = self.s1(x, adjacency)
        x = self.t5(self.t4(x))
        return (x, temporal_attention) if capture else x


class StudentBlock2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(64, 128, kernel_size=1)
        self.t1 = TemporalConvUnit(128, 1)
        self.t2 = TemporalConvUnit(128, 2)
        self.t3 = TemporalConvUnit(128, 4)
        self.s1 = SpatialGraphConv(128, 128)
        self.s2 = SpatialGraphConv(128, 128)
        self.sa = SpatialAttention(128)
        self.t4 = TemporalConvUnit(128, 2)
        self.t5 = TemporalConvUnit(128, 1)

    def forward(self, x: Tensor, adjacency: Tensor, capture: bool = False):
        x = self.t3(self.t2(self.t1(self.proj(x))))
        x = self.s2(self.s1(x, adjacency), adjacency)
        if capture:
            x, spatial_attention = self.sa(x, return_weights=True)
        else:
            x = self.sa(x)
            spatial_attention = None
        x = self.t5(self.t4(x))
        return (x, spatial_attention) if capture else x


class StudentBlock3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(128, 256, kernel_size=1)
        self.ta = TemporalAttention(256)
        self.t1 = TemporalConvUnit(256, 1)
        self.t2 = TemporalConvUnit(256, 2)
        self.t3 = TemporalConvUnit(256, 4)
        self.s1 = SpatialGraphConv(256, 256)
        self.s2 = SpatialGraphConv(256, 256)
        self.sa = SpatialAttention(256)
        self.t4 = TemporalConvUnit(256, 2)
        self.t5 = TemporalConvUnit(256, 1)
        self.ffn = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(512, 256, kernel_size=1),
        )

    def forward(self, x: Tensor, adjacency: Tensor, capture: bool = False):
        x = self.proj(x)
        if capture:
            x, temporal_attention = self.ta(x, return_weights=True)
        else:
            x = self.ta(x)
            temporal_attention = None
        x = self.t3(self.t2(self.t1(x)))
        x = self.s2(self.s1(x, adjacency), adjacency)
        if capture:
            x, spatial_attention = self.sa(x, return_weights=True)
        else:
            x = self.sa(x)
            spatial_attention = None
        x = self.ffn(self.t5(self.t4(x)))
        if capture:
            return x, temporal_attention, spatial_attention
        return x


class StudentSTGCN(nn.Module):
    """Student network used in both distillation stages."""

    def __init__(self) -> None:
        super().__init__()
        self.graph = DynamicGraphGenerator()
        self.block1 = StudentBlock1()
        self.block2 = StudentBlock2()
        self.block3 = StudentBlock3()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.cls = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(
        self, x: Tensor, return_features: bool = False
    ) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
        adjacency = self.graph(x)
        if return_features:
            features: Dict[str, Tensor] = {"graph": adjacency}
            x, features["ta1"] = self.block1(x, adjacency, capture=True)
            features["block1"] = x
            x, features["sa2"] = self.block2(x, adjacency, capture=True)
            features["block2"] = x
            x, features["ta3"], features["sa3"] = self.block3(
                x, adjacency, capture=True
            )
            features["block3"] = x
        else:
            x = self.block1(x, adjacency)
            x = self.block2(x, adjacency)
            x = self.block3(x, adjacency)

        pooled = self.pool(x).flatten(1)
        logits = self.cls(pooled).squeeze(-1)
        if return_features:
            features["feature"] = pooled
            features["logits"] = logits
            return logits, features
        return logits
