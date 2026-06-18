"""InfoNCE / NT-Xent contrastive loss with optional hard-negative mining."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """Symmetric InfoNCE (NT-Xent) contrastive loss for cross-modal retrieval.

    Given a batch of L2-normalized SAR and Optical embeddings, computes the
    symmetric cross-modal contrastive loss where diagonal entries are positive
    pairs and all off-diagonal entries are in-batch negatives.

    Args:
        temperature: Softmax temperature τ (default 0.07).
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        sar_emb: torch.Tensor,
        optical_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """Compute symmetric InfoNCE loss.

        Args:
            sar_emb:     L2-normalized SAR embeddings, shape (B, D).
            optical_emb: L2-normalized Optical embeddings, shape (B, D).

        Returns:
            A (loss, accuracy) tuple where:
              - loss     is a scalar Tensor.
              - accuracy is the fraction of correct top-1 retrievals in the batch
                         (average of SAR→Optical and Optical→SAR directions).
        """
        B = sar_emb.size(0)

        # Cosine similarity matrix: (B, B)
        # Both embeddings are already L2-normalized, so dot product = cosine sim.
        sim = torch.mm(sar_emb, optical_emb.t()) / self.temperature  # (B, B)

        # Ground-truth labels: diagonal indices are positive pairs
        labels = torch.arange(B, device=sar_emb.device)

        # SAR → Optical direction
        loss_s2o = F.cross_entropy(sim, labels)
        # Optical → SAR direction
        loss_o2s = F.cross_entropy(sim.t(), labels)

        loss = 0.5 * (loss_s2o + loss_o2s)

        # Top-1 accuracy (average of both directions)
        with torch.no_grad():
            acc_s2o = (sim.argmax(dim=1) == labels).float().mean().item()
            acc_o2s = (sim.t().argmax(dim=1) == labels).float().mean().item()
            accuracy = 0.5 * (acc_s2o + acc_o2s)

        return loss, accuracy


class HardNegativeInfoNCELoss(nn.Module):
    """InfoNCE with hard-negative mining.

    Weights the hardest 25% of in-batch negatives by 2× in the cross-entropy
    computation, forcing the encoder to focus on the most confusing pairs.

    Args:
        temperature:    Softmax temperature τ (default 0.07).
        hard_neg_ratio: Fraction of negatives treated as "hard" (default 0.25).
        hard_neg_weight: Multiplier applied to hard negatives (default 2.0).
    """

    def __init__(
        self,
        temperature: float = 0.07,
        hard_neg_ratio: float = 0.25,
        hard_neg_weight: float = 2.0,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.hard_neg_ratio = hard_neg_ratio
        self.hard_neg_weight = hard_neg_weight

    def _weighted_cross_entropy(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-entropy with up-weighted hard negatives.

        Args:
            logits: Similarity matrix (B, B) / temperature.
            labels: Ground-truth indices (B,).

        Returns:
            Scalar loss tensor.
        """
        B = logits.size(0)

        # Build a weight matrix: 1.0 everywhere, hard_neg_weight for hard negatives
        weight = torch.ones_like(logits)

        with torch.no_grad():
            # Mask out positive pairs when ranking negatives
            neg_logits = logits.clone()
            pos_mask = torch.zeros_like(neg_logits, dtype=torch.bool)
            pos_mask[torch.arange(B), labels] = True
            neg_logits[pos_mask] = -1e9

            # Number of hard negatives per row
            n_hard = max(1, int((B - 1) * self.hard_neg_ratio))

            # Top-n_hard hardest negatives per row
            _, hard_indices = neg_logits.topk(n_hard, dim=1)
            weight.scatter_(1, hard_indices, self.hard_neg_weight)

        # Weighted cross-entropy: scale logits by weights, then standard CE
        weighted_logits = logits * weight
        return F.cross_entropy(weighted_logits, labels)

    def forward(
        self,
        sar_emb: torch.Tensor,
        optical_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """Compute symmetric hard-negative InfoNCE loss.

        Args:
            sar_emb:     L2-normalized SAR embeddings, shape (B, D).
            optical_emb: L2-normalized Optical embeddings, shape (B, D).

        Returns:
            (loss, accuracy) — same semantics as :class:`InfoNCELoss`.
        """
        B = sar_emb.size(0)
        labels = torch.arange(B, device=sar_emb.device)

        sim = torch.mm(sar_emb, optical_emb.t()) / self.temperature  # (B, B)

        loss_s2o = self._weighted_cross_entropy(sim, labels)
        loss_o2s = self._weighted_cross_entropy(sim.t(), labels)
        loss = 0.5 * (loss_s2o + loss_o2s)

        with torch.no_grad():
            acc_s2o = (sim.argmax(dim=1) == labels).float().mean().item()
            acc_o2s = (sim.t().argmax(dim=1) == labels).float().mean().item()
            accuracy = 0.5 * (acc_s2o + acc_o2s)

        return loss, accuracy
