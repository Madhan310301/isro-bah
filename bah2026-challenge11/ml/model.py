"""DualEncoder model with ResNet50 backbones and ProjectionHead MLP."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50

from loguru import logger


class ProjectionHead(nn.Module):
    """Two-layer MLP projection head that maps backbone features to unit-sphere embeddings.

    Architecture: Linear → BatchNorm1d → ReLU → Linear → L2-normalize

    Args:
        in_dim:  Dimensionality of input features (default 2048 for ResNet50).
        out_dim: Dimensionality of output embeddings (default 512).
    """

    def __init__(self, in_dim: int = 2048, out_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, in_dim).

        Returns:
            L2-normalized tensor of shape (B, out_dim).
        """
        return F.normalize(self.net(x), p=2, dim=1)


def _build_encoder(in_channels: int = 3, emb_dim: int = 512) -> nn.Module:
    """Build a ResNet50 encoder with a ProjectionHead replacing the FC layer.

    Args:
        in_channels: Number of input image channels (1 for SAR, 3 for Optical).
        emb_dim:     Output embedding dimension.

    Returns:
        A Sequential-like module: conv_stem → ... → avgpool → projection_head.
    """
    backbone = resnet50(weights=ResNet50_Weights.DEFAULT)

    # Adapt first conv for single-channel SAR input
    if in_channels != 3:
        original_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            in_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )
        # Average pretrained RGB weights across the channel dimension
        with torch.no_grad():
            backbone.conv1.weight.copy_(
                original_conv.weight.mean(dim=1, keepdim=True)
            )

    # Replace the final FC with our projection head
    in_features = backbone.fc.in_features  # 2048 for ResNet50
    backbone.fc = ProjectionHead(in_dim=in_features, out_dim=emb_dim)

    return backbone


class DualEncoder(nn.Module):
    """Dual-encoder model for cross-modal SAR ↔ Optical contrastive learning.

    Each encoder is a pretrained ResNet50 with a ProjectionHead replacing the FC.
    The SAR encoder accepts single-channel (1, H, W) inputs.
    The Optical encoder accepts three-channel (3, H, W) inputs.

    Layer-freezing schedule (call :meth:`set_freeze_schedule` each epoch):
        - Epochs 0–4  : layer1 and layer2 frozen
        - Epochs 5+   : all layers trainable

    Args:
        emb_dim: Shared embedding dimension (default 512).
    """

    def __init__(self, emb_dim: int = 512) -> None:
        super().__init__()
        self.emb_dim = emb_dim

        logger.info("Building SAR encoder (1-channel ResNet50)…")
        self.sar_encoder = _build_encoder(in_channels=1, emb_dim=emb_dim)

        logger.info("Building Optical encoder (3-channel ResNet50)…")
        self.optical_encoder = _build_encoder(in_channels=3, emb_dim=emb_dim)

        # Freeze first two blocks initially
        self._freeze_blocks(frozen=True)
        logger.info("DualEncoder ready. layer1 & layer2 frozen for first 5 epochs.")

    # ------------------------------------------------------------------
    # Freeze schedule
    # ------------------------------------------------------------------

    def _freeze_blocks(self, frozen: bool) -> None:
        """Freeze / unfreeze layer1 and layer2 in both encoders."""
        for encoder in (self.sar_encoder, self.optical_encoder):
            for name in ("layer1", "layer2"):
                for param in getattr(encoder, name).parameters():
                    param.requires_grad = not frozen

    def set_freeze_schedule(self, epoch: int) -> None:
        """Apply the freezing schedule based on the current epoch.

        Epochs 0–4 : layer1 & layer2 frozen.
        Epoch 5+   : all layers trainable.

        Args:
            epoch: Current training epoch (0-indexed).
        """
        if epoch < 5:
            self._freeze_blocks(frozen=True)
            logger.debug(f"Epoch {epoch}: layer1/layer2 frozen.")
        else:
            self._freeze_blocks(frozen=False)
            logger.debug(f"Epoch {epoch}: all layers trainable.")

    # ------------------------------------------------------------------
    # Forward / convenience encoders
    # ------------------------------------------------------------------

    def encode_sar(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of SAR images.

        Args:
            x: SAR tensor of shape (B, 1, H, W).

        Returns:
            L2-normalized embeddings of shape (B, emb_dim).
        """
        return self.sar_encoder(x)

    def encode_optical(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of Optical images.

        Args:
            x: Optical tensor of shape (B, 3, H, W).

        Returns:
            L2-normalized embeddings of shape (B, emb_dim).
        """
        return self.optical_encoder(x)

    def forward(
        self, sar: torch.Tensor, optical: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a paired batch of SAR and Optical images.

        Args:
            sar:     SAR tensor of shape (B, 1, H, W).
            optical: Optical tensor of shape (B, 3, H, W).

        Returns:
            (sar_emb, optical_emb) — both shape (B, emb_dim), L2-normalized.
        """
        return self.encode_sar(sar), self.encode_optical(optical)
