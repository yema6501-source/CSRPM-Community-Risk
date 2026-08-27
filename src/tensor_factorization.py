from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def normalized_laplacian(adjacency: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a = adjacency.float()
    a = torch.maximum(a, a.T)
    d = a.sum(dim=1)
    inv_sqrt = torch.rsqrt(d + eps)
    I = torch.eye(a.shape[0], device=a.device)
    return I - inv_sqrt[:, None] * a * inv_sqrt[None, :]


class SupervisedTucker(nn.Module):
    """Masked Tucker factorization with spatial regularization and supervised head.

    X has shape [community, time, feature]. The model learns:
      G: core tensor
      Uc: community factors
      Ut: temporal factors
      Uf: feature factors
    A prediction head maps reconstructed/factorized latent state to class logits.
    """

    def __init__(
        self,
        n_communities: int,
        n_times: int,
        n_features: int,
        ranks: Tuple[int, int, int] = (12, 16, 10),
        n_classes: int = 3,
        hidden_dim: int = 32,
    ):
        super().__init__()
        rc, rt, rf = ranks
        self.ranks = ranks
        self.Uc = nn.Parameter(torch.randn(n_communities, rc) * 0.08)
        self.Ut = nn.Parameter(torch.randn(n_times, rt) * 0.08)
        self.Uf = nn.Parameter(torch.randn(n_features, rf) * 0.08)
        self.core = nn.Parameter(torch.randn(rc, rt, rf) * 0.05)

        self.latent_projection = nn.Sequential(
            nn.Linear(rc + rt + rf, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def reconstruct(self) -> torch.Tensor:
        return torch.einsum("ia,jb,kc,abc->ijk", self.Uc, self.Ut, self.Uf, self.core)

    def latent(self, community_idx: torch.Tensor, time_idx: torch.Tensor, x_feature_summary: torch.Tensor) -> torch.Tensor:
        uc = self.Uc[community_idx]
        ut = self.Ut[time_idx]
        # Map observed feature summary to factor space; summary is [B, n_features].
        uf_summary = x_feature_summary @ self.Uf
        z = torch.cat([uc, ut, uf_summary], dim=-1)
        return self.latent_projection(z)

    def forward(self, community_idx: torch.Tensor, time_idx: torch.Tensor, x_feature_summary: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.latent(community_idx, time_idx, x_feature_summary))

    def objective(
        self,
        X: torch.Tensor,
        observed_mask: torch.Tensor,
        community_idx: torch.Tensor,
        time_idx: torch.Tensor,
        x_feature_summary: torch.Tensor,
        y: torch.Tensor,
        adjacency: Optional[torch.Tensor] = None,
        lambda_recon: float = 0.15,
        lambda_spatial: float = 0.02,
        lambda_l2: float = 1e-5,
    ):
        recon = self.reconstruct()
        obs = observed_mask.float()
        recon_loss = ((recon - X).pow(2) * obs).sum() / obs.sum().clamp_min(1.0)
        logits = self.forward(community_idx, time_idx, x_feature_summary)
        pred_loss = F.cross_entropy(logits, y)

        spatial_loss = torch.zeros((), device=X.device)
        if adjacency is not None:
            L = normalized_laplacian(adjacency)
            spatial_loss = torch.trace(self.Uc.T @ L @ self.Uc) / max(1, self.Uc.shape[0])

        l2 = sum(p.pow(2).sum() for p in self.parameters()) / max(1, sum(p.numel() for p in self.parameters()))
        total = pred_loss + lambda_recon * recon_loss + lambda_spatial * spatial_loss + lambda_l2 * l2
        parts = {
            "total": total.detach(),
            "prediction": pred_loss.detach(),
            "reconstruction": recon_loss.detach(),
            "spatial": spatial_loss.detach(),
            "l2": l2.detach(),
        }
        return total, logits, parts
