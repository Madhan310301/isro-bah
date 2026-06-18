"""Model loading, FAISS index management, and inference logic."""

from __future__ import annotations

import base64
import io
import json
import os
import time
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from PIL import Image
from torchvision import transforms

IMAGE_SIZE = 224
SAR_MEAN = [0.0]
SAR_STD = [1.0]
OPTICAL_MEAN = [0.485, 0.456, 0.406]
OPTICAL_STD = [0.229, 0.224, 0.225]

_sar_preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Lambda(lambda t: torch.log1p(t)),
    transforms.Normalize(mean=SAR_MEAN, std=SAR_STD),
])

_optical_preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=OPTICAL_MEAN, std=OPTICAL_STD),
])


# ---------------------------------------------------------------------------
# Singletons loaded at startup
# ---------------------------------------------------------------------------


class InferenceEngine:
    """Holds the model, FAISS indices, and metadata loaded at app startup.

    Attributes:
        model:           Loaded DualEncoder (or compatible) model.
        sar_index:       FAISS IndexIDMap over SAR embeddings.
        optical_index:   FAISS IndexIDMap over Optical embeddings.
        metadata:        Dict mapping str(embedding_id) → metadata dict.
        device:          Torch device string.
        model_name:      Checkpoint identifier string.
    """

    def __init__(self) -> None:
        self.model: Optional[nn.Module] = None
        self.sar_index: Optional[faiss.IndexIDMap] = None
        self.optical_index: Optional[faiss.IndexIDMap] = None
        self.metadata: dict[str, dict] = {}
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name: str = "DualEncoder-v1"

    def load(
        self,
        checkpoint_path: str,
        sar_index_path: str,
        optical_index_path: str,
        metadata_path: str,
        backbone: str = "resnet50",
    ) -> None:
        """Load model, indices, and metadata from disk.

        Args:
            checkpoint_path:    Path to ``best_model.pt``.
            sar_index_path:     Path to ``sar_index.faiss``.
            optical_index_path: Path to ``optical_index.faiss``.
            metadata_path:      Path to ``metadata.json``.
            backbone:           Backbone type passed to :func:`~ml.model_factory.create_model`.
        """
        from ml.model_factory import create_model

        logger.info(f"Loading model checkpoint: {checkpoint_path}")
        self.model = create_model(
            backbone=backbone,
            checkpoint_path=checkpoint_path,
            device=self.device,
        )
        self.model.eval()

        logger.info(f"Loading SAR FAISS index: {sar_index_path}")
        if Path(sar_index_path).exists():
            self.sar_index = faiss.read_index(sar_index_path)
            logger.info(f"SAR index size: {self.sar_index.ntotal}")
        else:
            logger.warning(f"SAR index not found at {sar_index_path}")

        logger.info(f"Loading Optical FAISS index: {optical_index_path}")
        if Path(optical_index_path).exists():
            self.optical_index = faiss.read_index(optical_index_path)
            logger.info(f"Optical index size: {self.optical_index.ntotal}")
        else:
            logger.warning(f"Optical index not found at {optical_index_path}")

        logger.info(f"Loading metadata: {metadata_path}")
        if Path(metadata_path).exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
            logger.info(f"Metadata entries: {len(self.metadata)}")
        else:
            logger.warning(f"Metadata not found at {metadata_path}")

        logger.info(f"InferenceEngine ready on {self.device}.")

    # ------------------------------------------------------------------
    # Image pre-processing
    # ------------------------------------------------------------------

    def preprocess_image(
        self, image_b64: str, modality: str
    ) -> torch.Tensor:
        """Decode base64 image and return a pre-processed tensor (1, C, H, W).

        Args:
            image_b64: Base64-encoded image bytes.
            modality:  ``"sar"`` or ``"optical"``.

        Returns:
            Float tensor of shape (1, C, H, W) on self.device.
        """
        raw = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(raw))

        if modality == "sar":
            img = img.convert("L")  # single-channel
            tensor = _sar_preprocess(img)  # (1, H, W)
        else:
            img = img.convert("RGB")
            tensor = _optical_preprocess(img)  # (3, H, W)

        return tensor.unsqueeze(0).to(self.device)  # (1, C, H, W)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed(self, image_b64: str, modality: str) -> np.ndarray:
        """Encode a single image and return its L2-normalized embedding.

        Args:
            image_b64: Base64-encoded image.
            modality:  ``"sar"`` or ``"optical"``.

        Returns:
            Float32 numpy array of shape (512,).
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        tensor = self.preprocess_image(image_b64, modality)

        with torch.no_grad():
            if modality == "sar":
                emb = self.model.encode_sar(tensor)
            else:
                emb = self.model.encode_optical(tensor)

        return emb.float().cpu().numpy()[0]  # (512,)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        image_b64: str,
        modality: str,
        top_k: int = 10,
        supabase_client=None,
    ) -> tuple[list[dict], float]:
        """Retrieve the top-K cross-modal matches for a query image.

        SAR query  → searches the Optical index.
        Optical query → searches the SAR index.

        Args:
            image_b64:       Base64-encoded query image.
            modality:        Query modality (``"sar"`` or ``"optical"``).
            top_k:           Number of results to return.
            supabase_client: Optional :class:`~app.supabase_client.SupabaseImageClient`
                             for enriching results with URLs and coordinates.

        Returns:
            (results, inference_time_ms) where results is a list of dicts.
        """
        t0 = time.perf_counter()

        query_emb = self.embed(image_b64, modality)
        query_emb = query_emb.reshape(1, -1).astype("float32")

        # Choose target index (cross-modal)
        target_modality = "optical" if modality == "sar" else "sar"
        index = self.optical_index if modality == "sar" else self.sar_index

        if index is None:
            raise RuntimeError(f"{target_modality.capitalize()} FAISS index not loaded.")

        D, I = index.search(query_emb, top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        results = []
        for rank, (sim, idx) in enumerate(zip(D[0], I[0]), start=1):
            meta = self.metadata.get(str(idx), {})
            pair_id = meta.get("pair_id", f"id_{idx}")

            # Enrich with Supabase if available
            supabase_url = meta.get("supabase_url")
            if supabase_url is None and supabase_client is not None:
                sb_meta = supabase_client.get_metadata(pair_id, target_modality)
                if sb_meta:
                    supabase_url = sb_meta.get("supabase_url")
                    meta.update(sb_meta)

            results.append(
                {
                    "rank": rank,
                    "similarity": float(sim),
                    "pair_id": pair_id,
                    "supabase_url": supabase_url,
                    "lat": meta.get("lat"),
                    "lon": meta.get("lon"),
                    "season": meta.get("season"),
                }
            )

        return results, elapsed_ms

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def optical_index_size(self) -> int:
        return self.optical_index.ntotal if self.optical_index else 0

    @property
    def sar_index_size(self) -> int:
        return self.sar_index.ntotal if self.sar_index else 0


# Global singleton — populated by app startup
engine = InferenceEngine()
