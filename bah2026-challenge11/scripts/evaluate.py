"""Evaluation script — computes Recall@K, mAP, MRR, latency, and throughput.

Usage::

    python -m scripts.evaluate \
        --checkpoint ./checkpoints/best_model.pt \
        --data_dir   ./data/SEN12 \
        --output     ./metrics.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np
import torch
from loguru import logger
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from ml.dataset import SAROpticalPairDataset
from ml.model_factory import create_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate retrieval metrics on the test split.")
    p.add_argument("--checkpoint", default="./checkpoints/best_model.pt")
    p.add_argument("--data_dir", default="./data/SEN12")
    p.add_argument("--output", default="./metrics.json")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--backbone", default="resnet50",
                   choices=["resnet50", "remoteclip", "georsCLIP"])
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    return p.parse_args()


def compute_metrics(
    sar_embs: np.ndarray,
    opt_embs: np.ndarray,
) -> dict[str, float]:
    """Compute Recall@K, mAP, and MRR between SAR query and Optical gallery.

    Args:
        sar_embs: Normalized SAR embeddings (N, D).
        opt_embs: Normalized Optical embeddings (N, D).

    Returns:
        Dict of metric name → float value.
    """
    n, d = sar_embs.shape
    index = faiss.IndexFlatIP(d)
    index.add(opt_embs)

    K_values = [1, 5, 10]
    D, I = index.search(sar_embs, max(K_values))  # (N, 10)

    gt = np.arange(n)
    metrics: dict[str, float] = {}

    for k in K_values:
        hits = np.any(I[:, :k] == gt[:, None], axis=1)
        metrics[f"recall_at_{k}"] = float(hits.mean())

    # mAP and MRR
    ap_list, rr_list = [], []
    for i in range(n):
        rank_positions = np.where(I[i] == gt[i])[0]
        if len(rank_positions) > 0:
            rank = rank_positions[0] + 1
            ap_list.append(1.0 / rank)
            rr_list.append(1.0 / rank)
        else:
            ap_list.append(0.0)
            rr_list.append(0.0)

    metrics["map"] = float(np.mean(ap_list))
    metrics["mrr"] = float(np.mean(rr_list))
    return metrics


def measure_latency(
    model,
    sar_embs: np.ndarray,
    index: faiss.IndexFlatIP,
    n_queries: int = 200,
) -> dict[str, float]:
    """Measure per-query latency (FAISS search only, embeddings pre-computed).

    Args:
        model:     (unused — embeddings are pre-computed for latency test)
        sar_embs:  Pre-computed SAR embeddings (N, D).
        index:     Optical FAISS index.
        n_queries: Number of random queries to sample.

    Returns:
        Dict with p50, p95, p99 latency in ms and throughput (qps).
    """
    indices = np.random.choice(len(sar_embs), size=min(n_queries, len(sar_embs)), replace=False)
    latencies: list[float] = []

    for i in indices:
        q = sar_embs[i : i + 1]
        t0 = time.perf_counter()
        index.search(q, 10)
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies_arr = np.array(latencies)
    return {
        "latency_p50_ms": float(np.percentile(latencies_arr, 50)),
        "latency_p95_ms": float(np.percentile(latencies_arr, 95)),
        "latency_p99_ms": float(np.percentile(latencies_arr, 99)),
        "throughput_qps": float(1000.0 / np.mean(latencies_arr)),
    }


def _pretty_table(metrics: dict[str, float]) -> str:
    """Format a metrics dict as a readable table string."""
    lines = [
        "╔══════════════════════════╦══════════════╗",
        "║ Metric                   ║ Value        ║",
        "╠══════════════════════════╬══════════════╣",
    ]
    for k, v in metrics.items():
        name = k.replace("_", " ").title()
        if "ms" in k.lower():
            val_str = f"{v:>10.2f} ms"
        elif "qps" in k.lower():
            val_str = f"{v:>9.1f} q/s"
        else:
            val_str = f"{v:>12.4f}"
        lines.append(f"║ {name:<24} ║ {val_str} ║")
    lines.append("╚══════════════════════════╩══════════════╝")
    return "\n".join(lines)


def evaluate(args: argparse.Namespace) -> None:
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    logger.info(f"Evaluating on {device_str}…")

    model = create_model(
        backbone=args.backbone,
        checkpoint_path=args.checkpoint,
        device=device_str,
    )
    model.eval()

    dataset = SAROpticalPairDataset(args.data_dir, split=args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    logger.info(f"Evaluating on {len(dataset)} pairs ({args.split} split).")

    sar_list: list[np.ndarray] = []
    opt_list: list[np.ndarray] = []

    t_embed_start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            sar = batch["sar"].to(device)
            opt = batch["optical"].to(device)
            with autocast(enabled=device.type == "cuda"):
                sar_emb, opt_emb = model(sar, opt)
            sar_list.append(sar_emb.float().cpu().numpy())
            opt_list.append(opt_emb.float().cpu().numpy())

    sar_embs = np.vstack(sar_list).astype("float32")
    opt_embs = np.vstack(opt_list).astype("float32")
    embed_time = time.perf_counter() - t_embed_start
    logger.info(f"Embedded {len(sar_embs)} pairs in {embed_time:.1f}s.")

    # Retrieval metrics
    retrieval_metrics = compute_metrics(sar_embs, opt_embs)

    # Latency
    flat_index = faiss.IndexFlatIP(sar_embs.shape[1])
    flat_index.add(opt_embs)
    latency_metrics = measure_latency(model, sar_embs, flat_index, n_queries=200)

    all_metrics = {**retrieval_metrics, **latency_metrics}

    table = _pretty_table(all_metrics)
    logger.info("\n" + table)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Metrics saved to {out_path}.")


if __name__ == "__main__":
    evaluate(parse_args())
