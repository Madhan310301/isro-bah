"""Model factory — supports ResNet50 baseline, RemoteCLIP, and GeoRSCLIP backbones."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from loguru import logger

from ml.model import DualEncoder, ProjectionHead

BackboneType = Literal["resnet50", "remoteclip", "georsCLIP"]


# ---------------------------------------------------------------------------
# RemoteCLIP wrapper
# ---------------------------------------------------------------------------


class RemoteCLIPEncoder(nn.Module):
    """Visual encoder built on top of a RemoteCLIP or GeoRSCLIP visual backbone.

    Downloads the pretrained model from HuggingFace via ``huggingface_hub``,
    extracts the vision tower, and appends a trainable :class:`~ml.model.ProjectionHead`.

    Args:
        model_name: HuggingFace repo ID (e.g. ``"torchgeo/RemoteCLIP"``).
        emb_dim:    Output embedding dimension (default 512).
    """

    def __init__(self, model_name: str = "torchgeo/RemoteCLIP", emb_dim: int = 512) -> None:
        super().__init__()
        self.model_name = model_name
        self.emb_dim = emb_dim

        logger.info(f"Downloading {model_name} from HuggingFace…")
        try:
            import open_clip
            from huggingface_hub import snapshot_download

            local_dir = snapshot_download(model_name)
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained=f"{local_dir}/RemoteCLIP-ViT-B-32.pt"
            )
            self.backbone = model.visual
            backbone_dim = model.visual.output_dim  # typically 512 for ViT-B-32
        except Exception as exc:
            logger.warning(
                f"Failed to load {model_name}: {exc}. "
                "Falling back to vanilla ResNet50 backbone."
            )
            from torchvision.models import ResNet50_Weights, resnet50
            base = resnet50(weights=ResNet50_Weights.DEFAULT)
            backbone_dim = base.fc.in_features
            base.fc = nn.Identity()
            self.backbone = base

        self.projection = ProjectionHead(in_dim=backbone_dim, out_dim=emb_dim)
        logger.info(f"RemoteCLIPEncoder ready (backbone_dim={backbone_dim}, emb_dim={emb_dim}).")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images and return L2-normalized embeddings.

        Args:
            x: Image tensor (B, C, H, W).

        Returns:
            L2-normalized embeddings (B, emb_dim).
        """
        features = self.backbone(x)
        if isinstance(features, (list, tuple)):
            features = features[0]
        return self.projection(features)


class RemoteCLIPDualEncoder(nn.Module):
    """Dual encoder backed by RemoteCLIP / GeoRSCLIP visual towers.

    For remote-sensing pretrained backbones both SAR and Optical encoders
    share the same architecture but have separate weights, allowing the
    model to specialise per modality while benefiting from remote-sensing
    pretraining.

    Args:
        model_name: HuggingFace repo ID.
        emb_dim:    Shared embedding dimension.
    """

    def __init__(self, model_name: str = "torchgeo/RemoteCLIP", emb_dim: int = 512) -> None:
        super().__init__()
        self.sar_encoder = RemoteCLIPEncoder(model_name, emb_dim)
        self.optical_encoder = RemoteCLIPEncoder(model_name, emb_dim)
        self.emb_dim = emb_dim

    def encode_sar(self, x: torch.Tensor) -> torch.Tensor:
        """Encode SAR images."""
        return self.sar_encoder(x)

    def encode_optical(self, x: torch.Tensor) -> torch.Tensor:
        """Encode Optical images."""
        return self.optical_encoder(x)

    def forward(
        self, sar: torch.Tensor, optical: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a paired batch."""
        return self.encode_sar(sar), self.encode_optical(optical)

    def set_freeze_schedule(self, epoch: int) -> None:
        """No-op freeze schedule (RemoteCLIP backbone is fine-tuned from epoch 0)."""
        pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_model(
    backbone: BackboneType = "resnet50",
    emb_dim: int = 512,
    checkpoint_path: str | None = None,
    device: str | None = None,
) -> nn.Module:
    """Instantiate a dual encoder model with the chosen backbone.

    Args:
        backbone:        One of ``"resnet50"``, ``"remoteclip"``, ``"georsCLIP"``.
        emb_dim:         Output embedding dimension (default 512).
        checkpoint_path: Optional path to a ``best_model.pt`` checkpoint to
                         load weights from.
        device:          Device string (default: auto-detect).

    Returns:
        A model with ``encode_sar``, ``encode_optical``, and ``forward`` methods.

    Example::

        model = create_model(backbone="remoteclip", emb_dim=512)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    backbone_lower = backbone.lower()

    if backbone_lower == "resnet50":
        logger.info("Creating DualEncoder with ResNet50 backbones.")
        model: nn.Module = DualEncoder(emb_dim=emb_dim)

    elif backbone_lower == "remoteclip":
        logger.info("Creating DualEncoder with RemoteCLIP backbones.")
        model = RemoteCLIPDualEncoder(model_name="torchgeo/RemoteCLIP", emb_dim=emb_dim)

    elif backbone_lower in ("georsclip", "georsCLIP"):
        logger.info("Creating DualEncoder with GeoRSCLIP backbones.")
        model = RemoteCLIPDualEncoder(model_name="torchgeo/GeoRSCLIP", emb_dim=emb_dim)

    else:
        raise ValueError(
            f"Unknown backbone {backbone!r}. "
            "Choose from 'resnet50', 'remoteclip', 'georsCLIP'."
        )

    if checkpoint_path is not None:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            logger.warning(f"Missing keys: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected}")
        logger.info("Checkpoint loaded.")

    model = model.to(device)
    model.eval()
    return model
