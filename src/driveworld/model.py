"""Autoregressive action-conditioned occupancy predictor."""

from __future__ import annotations

import torch
from torch import nn

from .metrics import static_world_baseline


class ConvGRUCell(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gates = nn.Conv2d(channels * 2, channels * 2, 3, padding=1)
        self.candidate = nn.Conv2d(channels * 2, channels, 3, padding=1)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        reset, update = self.gates(torch.cat([x, state], dim=1)).sigmoid().chunk(2, dim=1)
        candidate = torch.tanh(self.candidate(torch.cat([x, reset * state], dim=1)))
        return (1 - update) * state + update * candidate


class ActionRollout(nn.Module):
    def __init__(self, channels: int = 32, resolution: float = 0.5):
        super().__init__()
        self.channels = channels
        self.resolution = resolution
        self.encoder = nn.Sequential(
            nn.Conv2d(2, channels // 2, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(channels // 2, channels, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.action = nn.Sequential(nn.Linear(3, channels), nn.Tanh())
        self.recurrent = ConvGRUCell(channels)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channels, channels // 2, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(channels // 2, 2, 4, stride=2, padding=1),
        )
        self.register_buffer("action_scale", torch.tensor([10.0, 10.0, 0.5]))

    def forward(self, history: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if history.ndim != 5 or actions.ndim != 3:
            raise ValueError("Expected history [B,T,2,H,W], actions [B,F,3]")
        if history.shape[2] != 2 or actions.shape[-1] != 3:
            raise ValueError("Expected two occupancy channels and three action values")
        batch, steps, _, height, width = history.shape
        if height % 4 or width % 4:
            raise ValueError("BEV height and width must be divisible by four")
        state = history.new_zeros(batch, self.channels, height // 4, width // 4)
        for t in range(steps):
            state = self.recurrent(self.encoder(history[:, t]), state)
        last = history[:, -1]
        geometry_prior = static_world_baseline(last, actions, self.resolution)
        predictions = []
        for t in range(actions.shape[1]):
            condition = self.action(actions[:, t] / self.action_scale).view(batch, self.channels, 1, 1)
            state = self.recurrent(self.encoder(last) + condition, state)
            # A fixed ego-motion warp handles coordinate changes; the network learns residual dynamics.
            prior_logits = torch.logit(geometry_prior[:, t] * 0.9 + 0.05)
            logits = prior_logits + self.decoder(state)
            predictions.append(logits)
            last = logits.sigmoid()
        return torch.stack(predictions, dim=1)
