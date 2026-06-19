"""Unit tests for DualEncoder and ProjectionHead."""

from __future__ import annotations

import pytest
import torch

from ml.model import DualEncoder, ProjectionHead


class TestProjectionHead:
    def test_output_shape(self):
        head = ProjectionHead(in_dim=2048, out_dim=256)
        x = torch.randn(8, 2048)
        out = head(x)
        assert out.shape == (8, 256), f"Expected (8, 256), got {out.shape}"

    def test_output_is_l2_normalized(self):
        head = ProjectionHead(in_dim=2048, out_dim=256)
        x = torch.randn(8, 2048)
        out = head(x)
        norms = out.norm(dim=1)
        assert torch.allclose(norms, torch.ones(8), atol=1e-5), (
            "ProjectionHead output must be L2-normalized."
        )

    def test_configurable_dims(self):
        head = ProjectionHead(in_dim=768, out_dim=256)
        x = torch.randn(4, 768)
        out = head(x)
        assert out.shape == (4, 256)

    def test_gradient_flows(self):
        head = ProjectionHead(in_dim=2048, out_dim=256)
        x = torch.randn(4, 2048, requires_grad=True)
        out = head(x)
        out.sum().backward()
        assert x.grad is not None


class TestDualEncoder:
    @pytest.fixture(scope="class")
    def model(self):
        from unittest.mock import patch
        import torchvision.models
        original_resnet50 = torchvision.models.resnet50
        def mock_resnet50(*args, **kwargs):
            if "weights" in kwargs:
                kwargs["weights"] = None
            return original_resnet50(*args, **kwargs)
        with patch("torchvision.models.resnet50", mock_resnet50):
            return DualEncoder(backbone="resnet50", emb_dim=256)

    def test_forward_output_shapes(self, model):
        sar = torch.randn(4, 1, 224, 224)
        optical = torch.randn(4, 3, 224, 224)
        sar_emb, opt_emb = model(sar, optical)
        assert sar_emb.shape == (4, 256), f"Expected (4, 256), got {sar_emb.shape}"
        assert opt_emb.shape == (4, 256), f"Expected (4, 256), got {opt_emb.shape}"

    def test_embeddings_are_normalized(self, model):
        sar = torch.randn(4, 1, 224, 224)
        optical = torch.randn(4, 3, 224, 224)
        sar_emb, opt_emb = model(sar, optical)
        assert torch.allclose(sar_emb.norm(dim=1), torch.ones(4), atol=1e-5)
        assert torch.allclose(opt_emb.norm(dim=1), torch.ones(4), atol=1e-5)

    def test_encode_sar_convenience(self, model):
        sar = torch.randn(2, 1, 224, 224)
        out = model.encode_sar(sar)
        assert out.shape == (2, 256)

    def test_encode_optical_convenience(self, model):
        optical = torch.randn(2, 3, 224, 224)
        out = model.encode_optical(optical)
        assert out.shape == (2, 256)

    def test_freeze_schedule_epoch_0(self, model):
        model.set_freeze_schedule(epoch=0)
        for encoder in (model.sar_encoder, model.optical_encoder):
            for param in encoder.layer1.parameters():
                assert not param.requires_grad, "layer1 should be frozen at epoch 0."
            for param in encoder.layer2.parameters():
                assert not param.requires_grad, "layer2 should be frozen at epoch 0."

    def test_freeze_schedule_epoch_5(self, model):
        model.set_freeze_schedule(epoch=5)
        for encoder in (model.sar_encoder, model.optical_encoder):
            for param in encoder.layer1.parameters():
                assert param.requires_grad, "layer1 should be trainable at epoch 5."
            for param in encoder.layer2.parameters():
                assert param.requires_grad, "layer2 should be trainable at epoch 5."

    def test_sar_encoder_single_channel(self, model):
        """Ensure SAR encoder accepts 1-channel input without error."""
        model.eval()
        sar = torch.randn(1, 1, 224, 224)
        out = model.encode_sar(sar)
        assert out.shape == (1, 256)

    def test_separate_encoder_weights(self, model):
        """Verify SAR and Optical encoders have independent weight tensors."""
        sar_param = next(model.sar_encoder.parameters())
        opt_param = next(model.optical_encoder.parameters())
        assert sar_param.data_ptr() != opt_param.data_ptr(), (
            "SAR and Optical encoders must not share weights."
        )
