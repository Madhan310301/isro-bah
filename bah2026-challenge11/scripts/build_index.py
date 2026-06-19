"""Build FAISS indices from a trained DualEncoder checkpoint.

Usage::

    python -m scripts.build_index \
        --checkpoint ./checkpoints/best_model.pt \
        --data_dir   ./data/SEN12 \
        --output_dir ./faiss_index \
        --max_pairs  10000
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import faiss
import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from ml.dataset import SAROpticalPairDataset
from ml.model_factory import create_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build FAISS indices from a trained checkpoint.")
    p.add_argument("--checkpoint", default="./checkpoints/best_model.pt")
    p.add_argument("--data_dir", default="./data/SEN12")
    p.add_argument("--output_dir", default="./faiss_index")
    p.add_argument("--max_pairs", type=int, default=None,
                   help="Limit to first N pairs (default: all).")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--backbone", default="resnet50",
                   choices=["resnet50", "remoteclip", "georsCLIP"])
    p.add_argument("--split", default="test",
                   choices=["train", "val", "test"],
                   help="Dataset split to index.")
    return p.parse_args()


def build_index(args: argparse.Namespace) -> None:
    """Run the full index-build pipeline."""
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    logger.info(f"Loading model from {args.checkpoint} on {device_str}…")
    model = create_model(
        backbone=args.backbone,
        checkpoint_path=args.checkpoint,
        device=device_str,
    )
    model.eval()

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset = SAROpticalPairDataset(args.data_dir, split=args.split)
    if args.max_pairs is not None:
        from torch.utils.data import Subset
        dataset = Subset(dataset, list(range(min(args.max_pairs, len(dataset)))))
        logger.info(f"Using subset of {len(dataset)} pairs.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # Embed all images
    # ------------------------------------------------------------------
    all_sar_embs: list[np.ndarray] = []
    all_opt_embs: list[np.ndarray] = []
    all_pair_ids: list[str] = []

    t0 = time.perf_counter()

    with torch.no_grad():
        for i, batch in enumerate(loader):
            sar = batch["sar"].to(device)
            optical = batch["optical"].to(device)
            pair_ids = batch["pair_id"]

            with autocast(enabled=device.type == "cuda"):
                sar_emb, opt_emb = model(sar, optical)

            all_sar_embs.append(sar_emb.float().cpu().numpy())
            all_opt_embs.append(opt_emb.float().cpu().numpy())
            all_pair_ids.extend(pair_ids)

            if (i + 1) % 20 == 0:
                logger.info(f"  Embedded {(i + 1) * args.batch_size} pairs…")

    sar_embs = np.vstack(all_sar_embs).astype("float32")
    opt_embs = np.vstack(all_opt_embs).astype("float32")
    elapsed = time.perf_counter() - t0
    logger.info(
        f"Embedded {len(all_pair_ids)} pairs in {elapsed:.1f}s "
        f"({len(all_pair_ids) / elapsed:.0f} pairs/s)."
    )

    # ------------------------------------------------------------------
    # Build FAISS indices (IndexFlatIP wrapped in IndexIDMap)
    # ------------------------------------------------------------------
    dim = sar_embs.shape[1]
    ids = np.arange(len(all_pair_ids), dtype="int64")

    def _build(embs: np.ndarray, name: str) -> faiss.IndexIDMap:
        flat = faiss.IndexFlatIP(dim)
        index = faiss.IndexIDMap(flat)
        t_build = time.perf_counter()
        index.add_with_ids(embs, ids)
        elapsed_build = time.perf_counter() - t_build
        logger.info(
            f"Built {name} index: {index.ntotal} vectors in {elapsed_build:.2f}s."
        )
        return index

    t_idx = time.perf_counter()
    sar_index = _build(sar_embs, "SAR")
    opt_index = _build(opt_embs, "Optical")
    logger.info(f"Total index build time: {time.perf_counter() - t_idx:.2f}s")

    # ------------------------------------------------------------------
    # Save indices
    # ------------------------------------------------------------------
    sar_path = str(output_dir / "sar_index.faiss")
    opt_path = str(output_dir / "optical_index.faiss")

    faiss.write_index(sar_index, sar_path)
    faiss.write_index(opt_index, opt_path)
    logger.info(f"Saved SAR index → {sar_path}")
    logger.info(f"Saved Optical index → {opt_path}")

    # ------------------------------------------------------------------
    # metadata.json: id → {pair_id, modality, ...}
    # ------------------------------------------------------------------
    metadata: dict[str, dict] = {}
    for idx, pair_id in enumerate(all_pair_ids):
        # Parse season from pair_id if it follows SEN1-2 naming convention
        # e.g. "ROIs1158_spring_s1_1" → season="spring"
        parts = pair_id.split("_")
        season = parts[1] if len(parts) > 1 else "unknown"

        metadata[str(idx)] = {
            "pair_id": pair_id,
            "modality": "optical",   # target modality stored
            "season": season,
            "supabase_url": None,
            "lat": None,
            "lon": None,
        }

    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata → {meta_path}")
    logger.info(
        f"Done. Indices contain {sar_index.ntotal} SAR and "
        f"{opt_index.ntotal} Optical vectors."
    )


if __name__ == "__main__":
    build_index(parse_args())
