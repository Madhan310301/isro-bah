"""Model loading, FAISS index management, and inference logic."""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

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
# HNSW index builder
# ---------------------------------------------------------------------------


def _build_hnsw_index(embs: np.ndarray, ids: np.ndarray, M: int = 32) -> faiss.IndexIDMap:
    """Build a FAISS IndexHNSWFlat wrapped in IndexIDMap.

    IndexHNSWFlat gives O(log N) approximate nearest-neighbour search,
    significantly faster than IndexFlatIP for large corpora.

    Args:
        embs: Float32 embeddings (N, D).
        ids:  Int64 vector IDs (N,).
        M:    HNSW M parameter — neighbours per layer (default 32).

    Returns:
        Populated faiss.IndexIDMap.
    """
    d = embs.shape[1]
    hnsw = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 200
    hnsw.hnsw.efSearch = 64
    index = faiss.IndexIDMap(hnsw)
    index.add_with_ids(embs, ids)
    return index


# ---------------------------------------------------------------------------
# InferenceEngine
# ---------------------------------------------------------------------------


class InferenceEngine:
    """Holds the model, FAISS indices, and metadata loaded at app startup.

    Attributes:
        model:           Loaded dual encoder model.
        sar_index:       FAISS index over SAR embeddings.
        optical_index:   FAISS index over Optical embeddings.
        metadata:        Dict mapping str(embedding_id) → metadata dict.
        device:          Torch device string.
        model_name:      Checkpoint identifier.
        index_version:   Timestamp string of when the indices were built.
        index_type:      FAISS index type string (e.g. ``"IndexHNSWFlat"``).
        emb_dim:         Embedding dimension.
    """

    def __init__(self) -> None:
        self.model: Optional[nn.Module] = None
        self.sar_index: Optional[faiss.IndexIDMap] = None
        self.optical_index: Optional[faiss.IndexIDMap] = None
        self.metadata: dict[str, dict] = {}
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name: str = "DualEncoder-v1"
        self.index_version: str = "unknown"
        self.index_type: str = "IndexHNSWFlat"
        self.emb_dim: int = 256

    def load(
        self,
        checkpoint_path: str,
        sar_index_path: str,
        optical_index_path: str,
        metadata_path: str,
        backbone: str = "vit",
    ) -> None:
        """Load model, FAISS indices, and metadata from disk.

        Args:
            checkpoint_path:    Path to ``best_model.pt``.
            sar_index_path:     Path to SAR FAISS index file.
            optical_index_path: Path to Optical FAISS index file.
            metadata_path:      Path to ``metadata.json``.
            backbone:           Backbone type for :func:`~ml.model_factory.create_model`.
        """
        from ml.model_factory import create_model

        logger.info(f"Loading model checkpoint: {checkpoint_path}")
        self.model = create_model(
            backbone=backbone,
            checkpoint_path=checkpoint_path,
            device=self.device,
        )
        self.model.eval()

        for attr, path, name in [
            ("sar_index", sar_index_path, "SAR"),
            ("optical_index", optical_index_path, "Optical"),
        ]:
            if Path(path).exists():
                idx = faiss.read_index(path)
                setattr(self, attr, idx)
                self.index_type = type(idx).__name__
                logger.info(f"{name} index loaded: {idx.ntotal} vectors ({self.index_type})")
            else:
                logger.warning(f"{name} index not found at {path}")

        if Path(metadata_path).exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
            logger.info(f"Metadata: {len(self.metadata)} entries.")
        else:
            logger.warning(f"Metadata not found at {metadata_path}")

        # Detect version from filename timestamp if present
        # e.g. optical_v2_20260806.faiss
        stem = Path(optical_index_path).stem
        parts = stem.split("_")
        self.index_version = parts[-1] if len(parts) > 1 and parts[-1].isdigit() else "latest"

        logger.info(f"InferenceEngine ready on {self.device} (version={self.index_version}).")

    # ------------------------------------------------------------------
    # Incremental index update
    # ------------------------------------------------------------------

    def add_embeddings(
        self,
        sar_embs: np.ndarray,
        optical_embs: np.ndarray,
        new_metadata: dict[str, dict],
    ) -> None:
        """Incrementally add new embeddings to both indices without full rebuild.

        Args:
            sar_embs:     Float32 SAR embeddings (N, D).
            optical_embs: Float32 Optical embeddings (N, D).
            new_metadata: Dict mapping new embedding id → metadata dict.
        """
        if self.sar_index is None or self.optical_index is None:
            raise RuntimeError("Indices not loaded. Call load() first.")

        new_ids = np.array(
            [int(k) for k in new_metadata.keys()], dtype="int64"
        )
        self.sar_index.add_with_ids(sar_embs.astype("float32"), new_ids)
        self.optical_index.add_with_ids(optical_embs.astype("float32"), new_ids)
        self.metadata.update(new_metadata)
        logger.info(f"Added {len(new_ids)} new embeddings. Total: {self.optical_index.ntotal}")

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess_image(self, image_b64: str, modality: str) -> torch.Tensor:
        """Decode base64 image and return a preprocessed tensor.

        Args:
            image_b64: Base64-encoded image bytes.
            modality:  ``"sar"`` or ``"optical"``.

        Returns:
            Float tensor (1, C, H, W) on the engine device.
        """
        raw = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(raw))

        if modality == "sar":
            img = img.convert("L")
            tensor = _sar_preprocess(img)
        else:
            img = img.convert("RGB")
            tensor = _optical_preprocess(img)

        return tensor.unsqueeze(0).to(self.device)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed(self, image_b64: str, modality: str) -> np.ndarray:
        """Encode a single image and return its L2-normalized embedding.

        Args:
            image_b64: Base64-encoded image.
            modality:  ``"sar"`` or ``"optical"``.

        Returns:
            Float32 numpy array of shape (emb_dim,).
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        tensor = self.preprocess_image(image_b64, modality)

        with torch.no_grad():
            if modality == "sar":
                emb = self.model.encode_sar(tensor)
            else:
                emb = self.model.encode_optical(tensor)

        return emb.float().cpu().numpy()[0]

    # ------------------------------------------------------------------
    # Search helpers
    # ------------------------------------------------------------------

    def _results_from_faiss(
        self,
        D: np.ndarray,
        I: np.ndarray,
        target_modality: str,
        supabase_client: Any = None,
    ) -> list[dict]:
        """Convert raw FAISS distances/indices to result dicts.

        Args:
            D:                 Similarity scores (1, K).
            I:                 Index IDs (1, K).
            target_modality:   Modality of results (for Supabase lookup).
            supabase_client:   Optional Supabase client for URL enrichment.

        Returns:
            List of result dicts.
        """
        results = []
        for rank, (sim, idx) in enumerate(zip(D[0], I[0]), start=1):
            meta = self.metadata.get(str(idx), {})
            pair_id = meta.get("pair_id", f"id_{idx}")

            supabase_url = meta.get("supabase_url")
            if supabase_url is None and supabase_client is not None:
                sb = supabase_client.get_metadata(pair_id, target_modality)
                if sb:
                    supabase_url = sb.get("supabase_url")
                    meta.update(sb)

            bbox = meta.get("bbox")

            results.append({
                "rank": rank,
                "similarity": float(sim),
                "pair_id": pair_id,
                "supabase_url": supabase_url,
                "lat": meta.get("lat"),
                "lon": meta.get("lon"),
                "season": meta.get("season"),
                "bbox": bbox,
            })
        return results

    def search(
        self,
        image_b64: str,
        modality: str,
        top_k: int = 10,
        supabase_client: Any = None,
    ) -> tuple[list[dict], float]:
        """Cross-modal retrieval: SAR query → Optical results (or vice versa).

        Args:
            image_b64:       Base64-encoded query image.
            modality:        Query modality (``"sar"`` or ``"optical"``).
            top_k:           Number of results.
            supabase_client: Optional Supabase client.

        Returns:
            (results, inference_time_ms).
        """
        t0 = time.perf_counter()

        query_emb = self.embed(image_b64, modality).reshape(1, -1).astype("float32")
        target_modality = "optical" if modality == "sar" else "sar"
        index = self.optical_index if modality == "sar" else self.sar_index

        if index is None:
            raise RuntimeError(f"{target_modality.capitalize()} FAISS index not loaded.")

        D, I = index.search(query_emb, top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            f"search | modality={modality} top_k={top_k} "
            f"latency={elapsed_ms:.1f}ms top_score={D[0][0]:.4f}"
        )

        results = self._results_from_faiss(D, I, target_modality, supabase_client)
        return results, elapsed_ms

    def search_similar(
        self,
        image_b64: str,
        modality: str,
        top_k: int = 10,
        supabase_client: Any = None,
    ) -> tuple[list[dict], float]:
        """Same-modality similarity search.

        Args:
            image_b64: Base64-encoded query image.
            modality:  Modality — search within this index.
            top_k:     Number of results.
            supabase_client: Optional Supabase client.

        Returns:
            (results, inference_time_ms).
        """
        t0 = time.perf_counter()

        query_emb = self.embed(image_b64, modality).reshape(1, -1).astype("float32")
        index = self.sar_index if modality == "sar" else self.optical_index

        if index is None:
            raise RuntimeError(f"{modality.capitalize()} FAISS index not loaded.")

        D, I = index.search(query_emb, top_k + 1)  # +1 to skip the query itself
        elapsed_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            f"similar | modality={modality} top_k={top_k} latency={elapsed_ms:.1f}ms"
        )

        results = self._results_from_faiss(D[:, 1:], I[:, 1:], modality, supabase_client)
        return results[:top_k], elapsed_ms

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def optical_index_size(self) -> int:
        return self.optical_index.ntotal if self.optical_index else 0

    @property
    def sar_index_size(self) -> int:
        return self.sar_index.ntotal if self.sar_index else 0

    @property
    def memory_usage_mb(self) -> float:
        """Approximate combined memory usage of both FAISS indices in MB."""
        try:
            return (
                faiss.vector_float_size(
                    self.optical_index_size * self.emb_dim
                ) * 2  # SAR + Optical
            ) / (1024 ** 2)
        except Exception:
            return float(self.optical_index_size * self.emb_dim * 4 * 2) / (1024 ** 2)


# Global singleton
engine = InferenceEngine()
