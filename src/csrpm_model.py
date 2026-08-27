from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class AdaptiveSpatialLayer(nn.Module):
    """Learn context-dependent graph mixing while respecting a physical adjacency prior."""

    def __init__(self, n_communities: int, hidden_dim: int):
        super().__init__()
        self.community_embed = nn.Parameter(torch.randn(n_communities, hidden_dim) * 0.05)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, h: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # h: [B, N, H]
        sim = F.normalize(self.community_embed, dim=-1) @ F.normalize(self.community_embed, dim=-1).T
        mask = (adjacency > 0).float()
        eye = torch.eye(adjacency.shape[0], device=adjacency.device)
        mask = torch.maximum(mask, eye)
        logits = sim / self.temperature.abs().clamp_min(0.1)
        logits = logits.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(logits, dim=-1)
        return torch.einsum("ij,bjh->bih", weights, h)


class CSRPMNet(nn.Module):
    """Adaptive spatial-temporal risk predictor with explicit uncertainty-ready logits.

    Input shape: [batch, time, community, feature]
    Output shape: [batch, community, class]
    """

    def __init__(
        self,
        n_features: int,
        n_communities: int,
        n_classes: int = 3,
        hidden_dim: int = 64,
        temporal_layers: int = 1,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.n_communities = n_communities
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.spatial = AdaptiveSpatialLayer(n_communities, hidden_dim)
        self.temporal = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=temporal_layers,
            batch_first=True,
            dropout=dropout if temporal_layers > 1 else 0.0,
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        B, T, N, Fdim = x.shape
        if N != self.n_communities:
            raise ValueError(f"Expected {self.n_communities} communities, got {N}.")
        h = self.encoder(x)
        spatial_states = []
        for t in range(T):
            spatial_states.append(self.spatial(h[:, t], adjacency))
        hs = torch.stack(spatial_states, dim=1)  # [B,T,N,H]

        temporal_in = hs.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        temporal_out, _ = self.temporal(temporal_in)
        ht = temporal_out[:, -1].reshape(B, N, -1)
        hs_last = hs[:, -1]
        gate = self.fusion_gate(torch.cat([ht, hs_last], dim=-1))
        fused = gate * ht + (1.0 - gate) * hs_last
        return self.head(fused)
