"""Dynamic graph, graph convolution, and attention modules."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DynamicGraphGenerator(nn.Module):
    """Build a frame-wise KNN graph from positions and velocities.

    Input tensors use the model layout ``[batch, channels, time, nodes]``.
    Position is read from channels 0-1 and velocity from channels 6-7.
    """

    def __init__(
        self,
        k: int = 8,
        alpha: float = 0.8,
        beta: float = 0.2,
        sigma_d: float = 50.0,
        sigma_v: float = 10.0,
    ) -> None:
        super().__init__()
        if k < 1:
            raise ValueError("k must be at least 1")
        if sigma_d <= 0 or sigma_v <= 0:
            raise ValueError("sigma_d and sigma_v must be positive")
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.sigma_d = sigma_d
        self.sigma_v = sigma_v

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B, C, T, N], got shape {tuple(x.shape)}")
        if x.size(1) < 8:
            raise ValueError("At least 8 input channels are required")

        _, _, time_steps, node_count = x.shape
        positions = x[:, 0:2].permute(0, 2, 3, 1)
        velocities = x[:, 6:8].permute(0, 2, 3, 1)
        identity = torch.eye(node_count, device=x.device, dtype=x.dtype).unsqueeze(0)
        adjacency = []

        for time_index in range(time_steps):
            distance = torch.cdist(positions[:, time_index], positions[:, time_index])
            velocity_difference = torch.cdist(
                velocities[:, time_index], velocities[:, time_index]
            )
            spatial = torch.exp(-distance / self.sigma_d)
            motion = torch.exp(-velocity_difference / self.sigma_v)
            frame_adjacency = (self.alpha * spatial + self.beta * motion) * (1 - identity)

            if self.k < node_count:
                indices = torch.topk(frame_adjacency, self.k, dim=-1).indices
                mask = torch.zeros_like(frame_adjacency)
                mask.scatter_(-1, indices, 1)
                frame_adjacency = frame_adjacency * mask

            frame_adjacency = 0.5 * (
                frame_adjacency + frame_adjacency.transpose(-1, -2)
            )
            degree = frame_adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            adjacency.append(frame_adjacency / degree)

        return torch.stack(adjacency, dim=1)


class SpatialGraphConv(nn.Module):
    """Apply a 1x1 projection followed by frame-wise graph aggregation."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.theta = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor, adjacency: Tensor) -> Tensor:
        x = self.theta(x)
        x = torch.einsum("btij,bctj->bcti", adjacency, x)
        return self.relu(self.bn(x))


class TemporalAttention(nn.Module):
    """Normalize attention over the temporal dimension for each node."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.attn = nn.Sequential(nn.Conv2d(channels, 1, kernel_size=1), nn.Softmax(dim=2))

    def forward(self, x: Tensor, return_weights: bool = False):
        weights = self.attn(x)
        output = x * weights
        return (output, weights) if return_weights else output


class SpatialAttention(nn.Module):
    """Normalize attention over the node dimension for each frame."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.attn = nn.Sequential(nn.Conv2d(channels, 1, kernel_size=1), nn.Softmax(dim=3))

    def forward(self, x: Tensor, return_weights: bool = False):
        weights = self.attn(x)
        output = x * weights
        return (output, weights) if return_weights else output
