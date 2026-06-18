"""DualEncoder model: ViT-B/16 or ResNet50 backbone with ProjectionHead and MomentumEncoder."""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from loguru import logger


# ---------------------------------------------------------------------------
# Projection head
# ---------------------------------------------------------------------------


class ProjectionHead(nn.Module):
    """Two-layer MLP projection head that maps backbone features to unit-sphere embeddings.

    Architecture: Linear → BatchNorm1d → ReLU → Linear → L2-normalize.

    Args:
        in_dim:  Dimensionality of input features (default 768 for ViT-B/16).
        out_dim: Dimensionality of output embeddings (default 512).
    """

    def __init__(self, in_dim: int = 768, out_dim: int = 512) -> None:
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


# ---------------------------------------------------------------------------
# Backbone builders
# ---------------------------------------------------------------------------


def _build_vit_encoder(in_channels: int = 3, emb_dim: int = 512) -> nn.Module:
    """Build a ViT-B/16 encoder with ProjectionHead (requires timm).

    Args:
        in_channels: Number of input channels (1 for SAR, 3 for Optical).
        emb_dim:     Output embedding dimension.

    Returns:
        An nn.Module with a ``forward(x)`` that returns (B, emb_dim) embeddings.
    """
    try:
        import timm
    except ImportError as exc:
        raise ImportError("timm is required for ViT-B/16 backbone. pip install timm") from exc

    # timm ViT-B/16 with in_chans adaptation and no classifier head
    backbone = timm.create_model(
        "vit_base_patch16_224",
        pretrained=True,
        num_classes=0,          # remove classifier → outputs (B, 768)
        in_chans=in_channels,
    )
    feat_dim = backbone.num_features  # 768 for ViT-B/16

    class ViTEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.projection = ProjectionHead(in_dim=feat_dim, out_dim=emb_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            features = self.backbone(x)  # (B, 768)
            return self.projection(features)

    return ViTEncoder()


def _build_resnet_encoder(in_channels: int = 3, emb_dim: int = 512) -> nn.Module:
    """Build a ResNet50 encoder with ProjectionHead (ImageNet pretrained).

    Args:
        in_channels: Input channels (1 for SAR, 3 for Optical).
        emb_dim:     Output embedding dimension.

    Returns:
        An nn.Module with a ``forward(x)`` that returns (B, emb_dim) embeddings.
    """
    from torchvision.models import ResNet50_Weights, resnet50

    backbone = resnet50(weights=ResNet50_Weights.DEFAULT)

    if in_channels != 3:
        original = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            in_channels,
            original.out_channels,
            kernel_size=original.kernel_size,
            stride=original.stride,
            padding=original.padding,
            bias=False,
        )
        with torch.no_grad():
            backbone.conv1.weight.copy_(original.weight.mean(dim=1, keepdim=True))

    in_features = backbone.fc.in_features  # 2048
    backbone.fc = ProjectionHead(in_dim=in_features, out_dim=emb_dim)
    return backbone


# ---------------------------------------------------------------------------
# Momentum Encoder (MoCo-style)
# ---------------------------------------------------------------------------


class MomentumEncoder(nn.Module):
    """EMA (momentum) copy of an online encoder for stable contrastive training.

    The momentum encoder is never optimised directly — its weights are updated
    via exponential moving average of the online encoder at each step.

    Args:
        online_encoder: The encoder whose weights are tracked.
        momentum:       EMA momentum coefficient (default 0.995).
    """

    def __init__(self, online_encoder: nn.Module, momentum: float = 0.995) -> None:
        super().__init__()
        self.momentum = momentum
        # Deep-copy the online encoder; disable gradients
        self.encoder = copy.deepcopy(online_encoder)
        for param in self.encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, online_encoder: nn.Module) -> None:
        """Update momentum encoder weights via EMA.

        Args:
            online_encoder: Current online encoder whose weights are used to
                            update the momentum encoder.
        """
        for param_m, param_o in zip(
            self.encoder.parameters(), online_encoder.parameters()
        ):
            param_m.data.mul_(self.momentum).add_(param_o.data, alpha=1.0 - self.momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch with the momentum encoder (no grad).

        Args:
            x: Image tensor (B, C, H, W).

        Returns:
            L2-normalized embeddings (B, emb_dim).
        """
        with torch.no_grad():
            return self.encoder(x)


# ---------------------------------------------------------------------------
# DualEncoder
# ---------------------------------------------------------------------------


class DualEncoder(nn.Module):
    """Dual-encoder for cross-modal SAR ↔ Optical contrastive learning.

    Supports ViT-B/16 (default) or ResNet50 backbones.  Includes an optional
    third encoder slot for Thermal/IR modality (future-proofing).

    Momentum encoders are maintained for each active modality encoder for
    MoCo-style training stability.

    Args:
        emb_dim:  Shared embedding dimension (default 512).
        backbone: ``"vit"`` or ``"resnet50"``.
        with_thermal: If True, instantiate a third encoder for thermal/IR.
        momentum: EMA momentum for momentum encoders (default 0.995).
    """

    def __init__(
        self,
        emb_dim: int = 512,
        backbone: str = "vit",
        with_thermal: bool = False,
        momentum: float = 0.995,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.backbone_type = backbone
        self.with_thermal = with_thermal

        build = _build_vit_encoder if backbone == "vit" else _build_resnet_encoder

        logger.info(f"Building SAR encoder ({backbone}, 1-ch)…")
        self.sar_encoder = build(in_channels=1, emb_dim=emb_dim)

        logger.info(f"Building Optical encoder ({backbone}, 3-ch)…")
        self.optical_encoder = build(in_channels=3, emb_dim=emb_dim)

        # Momentum (EMA) encoders
        self.sar_momentum = MomentumEncoder(self.sar_encoder, momentum=momentum)
        self.optical_momentum = MomentumEncoder(self.optical_encoder, momentum=momentum)

        # Optional Thermal/IR encoder
        self.thermal_encoder: Optional[nn.Module] = None
        self.thermal_momentum: Optional[MomentumEncoder] = None
        if with_thermal:
            logger.info(f"Building Thermal/IR encoder ({backbone}, 1-ch)…")
            self.thermal_encoder = build(in_channels=1, emb_dim=emb_dim)
            self.thermal_momentum = MomentumEncoder(self.thermal_encoder, momentum=momentum)

        # Freeze schedule state
        self._freeze_applied = False
        if backbone == "resnet50":
            self._freeze_blocks(frozen=True)

        logger.info(f"DualEncoder ready (backbone={backbone}, emb_dim={emb_dim}).")

    # ------------------------------------------------------------------
    # Freeze schedule (ResNet50 only)
    # ------------------------------------------------------------------

    def _freeze_blocks(self, frozen: bool) -> None:
        """Freeze / unfreeze layer1 and layer2 in ResNet50 encoders."""
        if self.backbone_type != "resnet50":
            return
        for enc in self._all_online_encoders():
            for name in ("layer1", "layer2"):
                layer = getattr(enc, name, None)
                if layer is not None:
                    for param in layer.parameters():
                        param.requires_grad = not frozen

    def _all_online_encoders(self) -> list[nn.Module]:
        encs = [self.sar_encoder, self.optical_encoder]
        if self.thermal_encoder is not None:
            encs.append(self.thermal_encoder)
        return encs

    def set_freeze_schedule(self, epoch: int) -> None:
        """Apply the freezing schedule based on the current epoch.

        Epochs 0–4: layer1 & layer2 frozen (ResNet50 only).
        Epoch 5+:   all layers trainable.

        Args:
            epoch: Current training epoch (0-indexed).
        """
        if self.backbone_type != "resnet50":
            return
        if epoch < 5:
            self._freeze_blocks(frozen=True)
            logger.debug(f"Epoch {epoch}: layer1/layer2 frozen.")
        else:
            self._freeze_blocks(frozen=False)
            logger.debug(f"Epoch {epoch}: all layers trainable.")

    # ------------------------------------------------------------------
    # Momentum update
    # ------------------------------------------------------------------

    def update_momentum_encoders(self) -> None:
        """Update all momentum encoders via EMA after each optimiser step."""
        self.sar_momentum.update(self.sar_encoder)
        self.optical_momentum.update(self.optical_encoder)
        if self.thermal_encoder is not None and self.thermal_momentum is not None:
            self.thermal_momentum.update(self.thermal_encoder)

    # ------------------------------------------------------------------
    # Encode methods
    # ------------------------------------------------------------------

    def encode_sar(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of SAR images.

        Args:
            x: SAR tensor (B, 1, H, W).

        Returns:
            L2-normalized embeddings (B, emb_dim).
        """
        return self.sar_encoder(x)

    def encode_optical(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of Optical images.

        Args:
            x: Optical tensor (B, 3, H, W).

        Returns:
            L2-normalized embeddings (B, emb_dim).
        """
        return self.optical_encoder(x)

    def encode_thermal(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of Thermal/IR images (requires with_thermal=True).

        Args:
            x: Thermal tensor (B, 1, H, W).

        Returns:
            L2-normalized embeddings (B, emb_dim).

        Raises:
            RuntimeError: If the thermal encoder was not initialised.
        """
        if self.thermal_encoder is None:
            raise RuntimeError("Thermal encoder not initialised (with_thermal=False).")
        return self.thermal_encoder(x)

    def forward(
        self, sar: torch.Tensor, optical: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a paired batch of SAR and Optical images.

        Args:
            sar:     SAR tensor (B, 1, H, W).
            optical: Optical tensor (B, 3, H, W).

        Returns:
            (sar_emb, optical_emb) — both shape (B, emb_dim), L2-normalized.
        """
        return self.encode_sar(sar), self.encode_optical(optical)
