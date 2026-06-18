"""Model factory — ResNet50, ViT-B/16, RemoteCLIP, and GeoRSCLIP backbones."""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from loguru import logger

from ml.model import DualEncoder, ProjectionHead

BackboneType = Literal["resnet50", "vit", "remoteclip", "georsCLIP"]


# ---------------------------------------------------------------------------
# RemoteCLIP / GeoRSCLIP wrapper
# ---------------------------------------------------------------------------


class RemoteCLIPSingleEncoder(nn.Module):
    """Single encoder backed by a RemoteCLIP or GeoRSCLIP visual tower.

    Downloads pretrained weights from HuggingFace (``open_clip`` + ``huggingface_hub``),
    extracts the vision backbone, and appends a trainable ProjectionHead.

    Falls back to a vanilla ResNet50 if the download fails, so the rest of the
    pipeline stays functional without internet access.

    Args:
        hf_repo:     HuggingFace repo ID (e.g. ``"flywire/RemoteCLIP"``).
        model_arch:  open_clip model architecture string (default ``"ViT-B-32"``).
        emb_dim:     Output embedding dimension (default 512).
    """

    def __init__(
        self,
        hf_repo: str = "flywire/RemoteCLIP",
        model_arch: str = "ViT-B-32",
        emb_dim: int = 512,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        backbone_dim: int

        logger.info(f"Loading {hf_repo} from HuggingFace…")
        try:
            import open_clip
            from huggingface_hub import hf_hub_download

            # Download the pretrained weights file
            ckpt_filename = f"RemoteCLIP-{model_arch}.pt"
            ckpt_path = hf_hub_download(repo_id=hf_repo, filename=ckpt_filename)

            clip_model, _, _ = open_clip.create_model_and_transforms(
                model_arch, pretrained=ckpt_path
            )
            self.backbone = clip_model.visual
            backbone_dim = getattr(clip_model.visual, "output_dim", 512)
            logger.info(f"RemoteCLIP loaded (arch={model_arch}, backbone_dim={backbone_dim}).")

        except Exception as exc:
            logger.warning(
                f"Failed to load {hf_repo}: {exc}\n"
                "Falling back to ImageNet ResNet50."
            )
            from torchvision.models import ResNet50_Weights, resnet50

            base = resnet50(weights=ResNet50_Weights.DEFAULT)
            backbone_dim = base.fc.in_features  # 2048
            base.fc = nn.Identity()
            self.backbone = base

        self.projection = ProjectionHead(in_dim=backbone_dim, out_dim=emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to L2-normalized embeddings.

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

    Args:
        hf_repo:    HuggingFace repo ID.
        model_arch: open_clip architecture string.
        emb_dim:    Shared embedding dimension.
    """

    def __init__(
        self,
        hf_repo: str = "flywire/RemoteCLIP",
        model_arch: str = "ViT-B-32",
        emb_dim: int = 512,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        logger.info("Building SAR RemoteCLIP encoder…")
        self.sar_encoder = RemoteCLIPSingleEncoder(hf_repo, model_arch, emb_dim)
        logger.info("Building Optical RemoteCLIP encoder…")
        self.optical_encoder = RemoteCLIPSingleEncoder(hf_repo, model_arch, emb_dim)

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
        """No-op — RemoteCLIP backbone is fine-tuned from epoch 0."""

    def update_momentum_encoders(self) -> None:
        """No-op — RemoteCLIPDualEncoder has no momentum encoders."""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_model(
    backbone: BackboneType = "vit",
    emb_dim: int = 512,
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None,
    with_thermal: bool = False,
) -> nn.Module:
    """Instantiate a dual encoder model with the chosen backbone.

    Args:
        backbone:        One of ``"resnet50"``, ``"vit"``, ``"remoteclip"``,
                         ``"georsCLIP"``.
        emb_dim:         Output embedding dimension (default 512).
        checkpoint_path: Optional path to a ``best_model.pt`` checkpoint.
        device:          Device string (default: auto-detect).
        with_thermal:    If True, also build a thermal encoder (DualEncoder only).

    Returns:
        A model with ``encode_sar``, ``encode_optical``, and ``forward`` methods.

    Example::

        model = create_model(backbone="remoteclip", emb_dim=512)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    key = backbone.lower()

    if key in ("resnet50", "vit"):
        logger.info(f"Creating DualEncoder with backbone={backbone}.")
        model: nn.Module = DualEncoder(
            emb_dim=emb_dim,
            backbone=key,
            with_thermal=with_thermal,
        )

    elif key == "remoteclip":
        logger.info("Creating DualEncoder with RemoteCLIP backbone.")
        model = RemoteCLIPDualEncoder(
            hf_repo="flywire/RemoteCLIP",
            emb_dim=emb_dim,
        )

    elif key in ("georsclip", "georsCLIP"):
        logger.info("Creating DualEncoder with GeoRSCLIP backbone.")
        model = RemoteCLIPDualEncoder(
            hf_repo="torchgeo/GeoRSCLIP",
            emb_dim=emb_dim,
        )

    else:
        raise ValueError(
            f"Unknown backbone {backbone!r}. "
            "Choose from 'resnet50', 'vit', 'remoteclip', 'georsCLIP'."
        )

    if checkpoint_path is not None:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            logger.warning(f"Missing keys ({len(missing)}): {missing[:5]}…")
        if unexpected:
            logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}…")
        logger.info("Checkpoint loaded.")

    model = model.to(device)
    model.eval()
    return model
