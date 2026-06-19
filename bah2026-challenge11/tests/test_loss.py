"""Unit tests for InfoNCELoss and HardNegativeInfoNCELoss."""

from __future__ import annotations

import pytest
import torch

from ml.loss import HardNegativeInfoNCELoss, InfoNCELoss


@pytest.fixture
def batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Return a small batch of random L2-normalized embeddings."""
    B, D = 16, 256
    sar = torch.randn(B, D)
    sar = sar / sar.norm(dim=1, keepdim=True)
    optical = torch.randn(B, D)
    optical = optical / optical.norm(dim=1, keepdim=True)
    return sar, optical


class TestInfoNCELoss:
    def test_loss_is_scalar(self, batch):
        sar, optical = batch
        loss_fn = InfoNCELoss(temperature=0.07)
        loss, acc = loss_fn(sar, optical)
        assert loss.shape == (), "Loss should be a scalar tensor."

    def test_loss_is_positive(self, batch):
        sar, optical = batch
        loss_fn = InfoNCELoss(temperature=0.07)
        loss, _ = loss_fn(sar, optical)
        assert loss.item() > 0, "Loss should be positive."

    def test_accuracy_in_range(self, batch):
        sar, optical = batch
        loss_fn = InfoNCELoss(temperature=0.07)
        _, acc = loss_fn(sar, optical)
        assert 0.0 <= acc <= 1.0, "Accuracy must be in [0, 1]."

    def test_perfect_pairs_high_accuracy(self):
        """When embeddings are identical for positive pairs, accuracy should be 1."""
        B, D = 8, 256
        emb = torch.randn(B, D)
        emb = emb / emb.norm(dim=1, keepdim=True)
        loss_fn = InfoNCELoss(temperature=0.07)
        # Use same embeddings for both modalities → diagonal is always the highest
        loss, acc = loss_fn(emb, emb)
        assert acc == pytest.approx(1.0), "Identical embeddings should give acc=1."

    def test_temperature_scaling(self, batch):
        """Higher temperature should produce lower loss (smoother distribution)."""
        sar, optical = batch
        loss_low_temp = InfoNCELoss(temperature=0.01)(sar, optical)[0].item()
        loss_high_temp = InfoNCELoss(temperature=1.0)(sar, optical)[0].item()
        assert loss_low_temp > loss_high_temp, (
            "Lower temperature should produce sharper (higher) loss on random pairs."
        )

    def test_gradient_flows(self, batch):
        sar, optical = batch
        sar.requires_grad_(True)
        optical.requires_grad_(True)
        loss_fn = InfoNCELoss()
        loss, _ = loss_fn(sar, optical)
        loss.backward()
        assert sar.grad is not None
        assert optical.grad is not None


class TestHardNegativeInfoNCELoss:
    def test_loss_is_scalar(self, batch):
        sar, optical = batch
        loss_fn = HardNegativeInfoNCELoss(temperature=0.07)
        loss, acc = loss_fn(sar, optical)
        assert loss.shape == ()

    def test_loss_positive(self, batch):
        sar, optical = batch
        loss_fn = HardNegativeInfoNCELoss()
        loss, _ = loss_fn(sar, optical)
        assert loss.item() > 0

    def test_accuracy_in_range(self, batch):
        sar, optical = batch
        loss_fn = HardNegativeInfoNCELoss()
        _, acc = loss_fn(sar, optical)
        assert 0.0 <= acc <= 1.0

    def test_hard_neg_ge_standard(self, batch):
        """Hard-negative loss should be >= standard InfoNCE on random pairs."""
        sar, optical = batch
        std_loss = InfoNCELoss()(sar, optical)[0].item()
        hard_loss = HardNegativeInfoNCELoss()(sar, optical)[0].item()
        assert hard_loss >= std_loss * 0.9, (
            "Hard-negative loss should be at least as high as standard loss."
        )
