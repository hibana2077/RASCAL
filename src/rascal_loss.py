"""RASCAL loss: rank-agreement supervised contrastive alignment.

Design goals:
- Plug-in replacement for SupConLoss in the pretraining loop.
- Single loss only (no extra lambda terms).
- Uses a detached per-sample feature cache (sample_id -> last embedding).
- Keeps implementation readable (row loop for per-anchor positive ranking).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RASCALLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
        num_samples: int | None = None,
        feat_dim: int | None = None,
        persistent_cache: bool = False,
        contrast_mode: str = "all",
    ):
        super().__init__()
        if contrast_mode != "all":
            # keep it simple for now; SupContrast uses 'all' by default
            raise ValueError("RASCALLoss currently supports contrast_mode='all' only")

        if num_samples is None or feat_dim is None:
            raise ValueError("num_samples and feat_dim are required for the cache")

        self.temperature = float(temperature)
        self.base_temperature = float(base_temperature)
        self.contrast_mode = contrast_mode

        self.register_buffer(
            "cache_feat",
            torch.zeros(num_samples, feat_dim, dtype=torch.float32),
            persistent=persistent_cache,
        )
        self.register_buffer(
            "cache_valid",
            torch.zeros(num_samples, dtype=torch.bool),
            persistent=persistent_cache,
        )

    @torch.no_grad()
    def update_cache(self, sample_idx: torch.Tensor, sample_feat: torch.Tensor) -> None:
        self.cache_feat[sample_idx] = sample_feat
        self.cache_valid[sample_idx] = True

    def _build_pos_mask(self, labels: torch.Tensor, n_views: int) -> torch.Tensor:
        # labels: [bsz]
        bsz = labels.shape[0]
        labels_ = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels_, labels_.T).float()  # [bsz, bsz]
        mask = mask.repeat(n_views, n_views)  # [M, M]
        return mask

    def _build_rank_weights(
        self,
        contrast_feat_det: torch.Tensor,  # [M, d]
        sample_idx_rep: torch.Tensor,  # [M]
        pos_mask: torch.Tensor,  # [M, M] positives only (self masked out)
    ) -> torch.Tensor:
        device = contrast_feat_det.device
        dtype = contrast_feat_det.dtype
        M = contrast_feat_det.size(0)

        cache_valid_rep = self.cache_valid[sample_idx_rep]  # [M]

        W = torch.zeros((M, M), device=device, dtype=dtype)

        # Per-anchor row loop keeps code easy to validate.
        for r in range(M):
            pos_idx = torch.where(pos_mask[r] > 0)[0]
            m = int(pos_idx.numel())
            if m == 0:
                continue

            # If cache is missing for anchor or any positive, fall back to uniform.
            if (not bool(cache_valid_rep[r])) or (not bool(torch.all(cache_valid_rep[pos_idx]))):
                W[r, pos_idx] = 1.0 / m
                continue

            anchor_view = contrast_feat_det[r]  # [d]
            pos_views = contrast_feat_det[pos_idx]  # [m, d]

            anchor_sid = sample_idx_rep[r]
            pos_sid = sample_idx_rep[pos_idx]
            anchor_cache = self.cache_feat[anchor_sid]  # [d]
            pos_cache = self.cache_feat[pos_sid]  # [m, d]

            cur_scores = torch.matmul(pos_views, anchor_view)  # [m]
            cache_scores = torch.matmul(pos_cache, anchor_cache)  # [m]

            # rank: 0 = most similar
            rank_cur = torch.argsort(torch.argsort(-cur_scores))
            rank_cache = torch.argsort(torch.argsort(-cache_scores))

            if m == 1:
                drift = torch.zeros_like(cur_scores)
            else:
                drift = (rank_cur - rank_cache).abs().float() / (m - 1)

            w = (1.0 - drift).clamp_min(0.0)
            s = w.sum()
            if float(s) <= 1e-12:
                W[r, pos_idx] = 1.0 / m
            else:
                W[r, pos_idx] = w / s

        return W

    def forward(self, features: torch.Tensor, labels: torch.Tensor, sample_idx: torch.Tensor) -> torch.Tensor:
        """Compute RASCAL loss.

        Args:
            features:   [bsz, n_views, d]
            labels:     [bsz]
            sample_idx: [bsz] (must be stable indices within the pretraining dataset)
        """
        device = features.device
        if features.ndim < 3:
            raise ValueError("features must be [bsz, n_views, ...]")
        if features.ndim > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        bsz, n_views, d = features.shape
        features = F.normalize(features, dim=-1)

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # [M, d]
        M = contrast_feature.size(0)

        pos_mask = self._build_pos_mask(labels, n_views).to(device)
        logits_mask = torch.ones((M, M), device=device, dtype=pos_mask.dtype)
        logits_mask.fill_diagonal_(0.0)
        pos_mask = pos_mask * logits_mask

        logits = torch.div(torch.matmul(contrast_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # repeat sample_idx to view-level
        sample_idx_rep = sample_idx.view(-1).repeat(n_views).to(device)

        W = self._build_rank_weights(
            contrast_feat_det=contrast_feature.detach(),
            sample_idx_rep=sample_idx_rep,
            pos_mask=pos_mask,
        )

        weighted_log_prob_pos = (W * log_prob).sum(1)  # [M]
        loss = -(self.temperature / self.base_temperature) * weighted_log_prob_pos
        loss = loss.mean()

        # Update cache with per-sample feature (mean over views)
        with torch.no_grad():
            sample_feat = features.mean(dim=1)
            sample_feat = F.normalize(sample_feat, dim=-1)
            self.update_cache(sample_idx.to(device), sample_feat.detach())

        return loss
