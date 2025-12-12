# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from Bagel repository to avoid importing external Bagel package.

from __future__ import annotations

import math

import torch
from torch import nn


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    @staticmethod
    def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        # Standard sinusoidal timestep embedding.
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=timesteps.device) / half)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        return embedding

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        emb = self.timestep_embedding(timesteps, self.hidden_size).to(torch.bfloat16)
        return self.mlp(emb)


class PositionEmbedding(nn.Module):
    def __init__(self, max_size: int, hidden_size: int):
        super().__init__()
        self.max_size = max_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(max_size * max_size, hidden_size)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(position_ids)


class MLPconnector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, act: str = "gelu_pytorch_tanh"):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU() if "gelu" in act else nn.SiLU()
        self.fc2 = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))
