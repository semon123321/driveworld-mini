"""Rollout loss, horizon-wise metrics, and a geometry-only reference baseline."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def occupancy_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if logits.shape != target.shape:
        raise ValueError("Prediction and target shapes differ")
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=target.new_tensor(8.0))
    probability = logits.sigmoid()
    dims = (0, 1, 3, 4)
    overlap = 2 * (probability * target).sum(dims) + 1
    total = probability.sum(dims) + target.sum(dims) + 1
    return bce + (1 - overlap / total).mean()


def intersection_union(prediction: torch.Tensor, target: torch.Tensor):
    """Return per-horizon/channel counts. Empty unions are excluded from averages."""
    binary = prediction >= 0.5
    truth = target >= 0.5
    intersection = (binary & truth).sum(dim=(0, 3, 4)).to(torch.float64)
    union = (binary | truth).sum(dim=(0, 3, 4)).to(torch.float64)
    return intersection, union


def static_world_baseline(last: torch.Tensor, actions: torch.Tensor, resolution: float) -> torch.Tensor:
    """Warp the last ego-frame occupancy under proposed ego motion; agents stay fixed in world."""
    batch, channels, height, width = last.shape
    if actions.shape[0] != batch or actions.shape[-1] != 3:
        raise ValueError("Invalid action shape")
    device, dtype = last.device, last.dtype
    rows, cols = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype), indexing="ij",
    )
    x_future = (height / 2 - rows) * resolution
    y_future = (width / 2 - cols) * resolution
    origin_x = last.new_zeros(batch)
    origin_y = last.new_zeros(batch)
    heading = last.new_zeros(batch)
    outputs = []
    for t in range(actions.shape[1]):
        forward, lateral, turn = actions[:, t].unbind(-1)
        c, s = heading.cos(), heading.sin()
        origin_x, origin_y = origin_x + c * forward - s * lateral, origin_y + s * forward + c * lateral
        heading = heading + turn
        c, s = heading.cos()[:, None, None], heading.sin()[:, None, None]
        x_current = origin_x[:, None, None] + c * x_future - s * y_future
        y_current = origin_y[:, None, None] + s * x_future + c * y_future
        row_current = height / 2 - x_current / resolution
        col_current = width / 2 - y_current / resolution
        sample_grid = torch.stack([
            2 * col_current / (width - 1) - 1,
            2 * row_current / (height - 1) - 1,
        ], dim=-1)
        outputs.append(F.grid_sample(last, sample_grid, mode="nearest", align_corners=True))
    return torch.stack(outputs, dim=1)
