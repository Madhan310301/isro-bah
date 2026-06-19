"""Evaluation script — Recall@K, mAP, MRR, latency, t-SNE/UMAP, PDF report.

Usage::

    # Full evaluation
    python -m ml.evaluate --checkpoint ./checkpoints/best_model.pt --data_dir ./data/SEN12

    # Quick mode (500 random pairs)
    python -m ml.evaluate --checkpoint ./checkpoints/best_model.pt --quick
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import torch
from loguru import logger
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Subset

from ml.dataset import SAROpticalPairDataset
from ml.model_factory import create_model


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Evaluate cross-modal retrieval metrics.")
    p.add_argument("--checkpoint", default="./checkpoints/best_model.pt")
    p.add_argument("--data_dir", default="./data/SEN12")
    p.add_argument("--output", default="./metrics.json")
    p.add_argument("--report", default="./metrics_report.pdf",
                   help="Output PDF report path.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--backbone", default="vit",
                   choices=["resnet50", "vit", "remoteclip", "georsCLIP"])
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--quick", action="store_true",
                   help="Evaluate on 500 random pairs only.")
    p.add_argument("--no_viz", action="store_true",
                   help="Skip t-SNE/UMAP and heatmap visualizations.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------


def compute_retrieval_metrics(
    sar_embs: np.ndarray,
    opt_embs: np.ndarray,
) -> dict[str, float]:
    """Compute Recall@K, mAP, and MRR.

    Args:
        sar_embs: L2-normalized SAR embeddings (N, D).
        opt_embs: L2-normalized Optical embeddings (N, D).

    Returns:
        Dict with recall_at_1, recall_at_5, recall_at_10, map, mrr.
    """
    n, d = sar_embs.shape
    index = faiss.IndexFlatIP(d)
    index.add(opt_embs)

    D, I = index.search(sar_embs, 10)
    gt = np.arange(n)

    metrics: dict[str, float] = {}
    for k in (1, 5, 10):
        hits = np.any(I[:, :k] == gt[:, None], axis=1)
        metrics[f"recall_at_{k}"] = float(hits.mean())

    ap_list, rr_list = [], []
    for i in range(n):
        pos = np.where(I[i] == gt[i])[0]
        rank = (pos[0] + 1) if len(pos) > 0 else None
        ap_list.append(1.0 / rank if rank else 0.0)
        rr_list.append(1.0 / rank if rank else 0.0)

    metrics["map"] = float(np.mean(ap_list))
    metrics["mrr"] = float(np.mean(rr_list))
    return metrics


def compute_latency_metrics(
    sar_embs: np.ndarray,
    flat_index: faiss.IndexFlatIP,
    n_queries: int = 200,
) -> dict[str, float]:
    """Measure FAISS search latency percentiles and throughput.

    Args:
        sar_embs:    Pre-computed SAR embeddings (N, D).
        flat_index:  Optical FAISS IndexFlatIP.
        n_queries:   Number of random queries to sample.

    Returns:
        Dict with latency_p50_ms, latency_p95_ms, latency_p99_ms,
        throughput_qps.
    """
    indices = np.random.choice(len(sar_embs), size=min(n_queries, len(sar_embs)), replace=False)
    latencies: list[float] = []

    for i in indices:
        q = sar_embs[i : i + 1]
        t0 = time.perf_counter()
        flat_index.search(q, 10)
        latencies.append((time.perf_counter() - t0) * 1000)

    arr = np.array(latencies)
    return {
        "latency_p50_ms": float(np.percentile(arr, 50)),
        "latency_p95_ms": float(np.percentile(arr, 95)),
        "latency_p99_ms": float(np.percentile(arr, 99)),
        "throughput_qps": float(1000.0 / arr.mean()),
    }


def compute_per_season_metrics(
    sar_embs: np.ndarray,
    opt_embs: np.ndarray,
    seasons: list[str],
) -> dict[str, dict[str, float]]:
    """Compute Recall@1/10 per season.

    Args:
        sar_embs: SAR embeddings (N, D).
        opt_embs: Optical embeddings (N, D).
        seasons:  Season label per pair (length N).

    Returns:
        Dict mapping season → metrics dict.
    """
    unique_seasons = sorted(set(seasons))
    season_arr = np.array(seasons)
    per_season: dict[str, dict[str, float]] = {}

    for season in unique_seasons:
        mask = season_arr == season
        idx = np.where(mask)[0]
        if len(idx) < 2:
            continue
        s_metrics = compute_retrieval_metrics(sar_embs[idx], opt_embs[idx])
        per_season[season] = s_metrics
        logger.info(
            f"  Season {season}: R@1={s_metrics['recall_at_1']:.3f}  "
            f"R@10={s_metrics['recall_at_10']:.3f}  n={len(idx)}"
        )

    return per_season


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------


def _save_tsne(
    sar_embs: np.ndarray,
    opt_embs: np.ndarray,
    output_path: str,
    n_samples: int = 1000,
) -> None:
    """Save a t-SNE scatter plot of SAR and Optical embeddings.

    Args:
        sar_embs:    SAR embeddings.
        opt_embs:    Optical embeddings.
        output_path: Output PNG path.
        n_samples:   Max samples to visualise.
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        n = min(n_samples, len(sar_embs))
        idx = np.random.choice(len(sar_embs), n, replace=False)
        combined = np.vstack([sar_embs[idx], opt_embs[idx]])

        if combined.shape[0] < 2:
            logger.warning("t-SNE skipped: Not enough samples to fit t-SNE.")
            return

        perp = min(30, max(1, combined.shape[0] - 1))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perp, max_iter=1000)
        coords = tsne.fit_transform(combined)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(coords[:n, 0], coords[:n, 1], c="steelblue", alpha=0.5,
                   s=8, label="SAR")
        ax.scatter(coords[n:, 0], coords[n:, 1], c="coral", alpha=0.5,
                   s=8, label="Optical")
        # Draw lines between paired points
        for i in range(min(100, n)):
            ax.plot(
                [coords[i, 0], coords[n + i, 0]],
                [coords[i, 1], coords[n + i, 1]],
                "gray", alpha=0.1, linewidth=0.5,
            )
        ax.legend(markerscale=3)
        ax.set_title("t-SNE: SAR (blue) vs Optical (red) Embedding Space")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"t-SNE saved → {output_path}")
    except Exception as exc:
        logger.warning(f"t-SNE skipped: {exc}")


def _save_umap(
    sar_embs: np.ndarray,
    opt_embs: np.ndarray,
    output_path: str,
    n_samples: int = 2000,
) -> None:
    """Save a UMAP projection of SAR and Optical embeddings.

    Args:
        sar_embs:    SAR embeddings.
        opt_embs:    Optical embeddings.
        output_path: Output PNG path.
        n_samples:   Max samples to visualise.
    """
    try:
        import matplotlib.pyplot as plt
        import umap

        n = min(n_samples, len(sar_embs))
        idx = np.random.choice(len(sar_embs), n, replace=False)
        combined = np.vstack([sar_embs[idx], opt_embs[idx]])

        if combined.shape[0] < 2:
            logger.warning("UMAP skipped: Not enough samples to fit UMAP.")
            return

        n_neighbors = min(15, max(2, combined.shape[0] - 1))
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42, metric="cosine")
        coords = reducer.fit_transform(combined)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(coords[:n, 0], coords[:n, 1], c="steelblue", alpha=0.5,
                   s=6, label="SAR")
        ax.scatter(coords[n:, 0], coords[n:, 1], c="coral", alpha=0.5,
                   s=6, label="Optical")
        ax.legend(markerscale=3)
        ax.set_title("UMAP: SAR (blue) vs Optical (red) Embedding Space")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"UMAP saved → {output_path}")
    except Exception as exc:
        logger.warning(f"UMAP skipped: {exc}")


def _save_similarity_heatmap(
    sar_embs: np.ndarray,
    opt_embs: np.ndarray,
    output_path: str,
    n_samples: int = 200,
) -> None:
    """Save a cross-modal similarity score heatmap.

    Plots the full (n×n) cosine similarity matrix for a random subset,
    with positive pairs on the diagonal.

    Args:
        sar_embs:    SAR embeddings.
        opt_embs:    Optical embeddings.
        output_path: Output PNG path.
        n_samples:   Number of pairs to include in the heatmap.
    """
    try:
        import matplotlib.pyplot as plt

        n = min(n_samples, len(sar_embs))
        idx = np.random.choice(len(sar_embs), n, replace=False)
        s = sar_embs[idx]
        o = opt_embs[idx]
        sim = s @ o.T  # (n, n)

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(sim, cmap="viridis", aspect="auto", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, label="Cosine Similarity")
        ax.set_title(f"Cross-Modal Similarity Heatmap (n={n})")
        ax.set_xlabel("Optical Index")
        ax.set_ylabel("SAR Index")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Similarity heatmap saved → {output_path}")
    except Exception as exc:
        logger.warning(f"Heatmap skipped: {exc}")


# ---------------------------------------------------------------------------
# PDF report
# ---------------------------------------------------------------------------


def _save_pdf_report(
    metrics: dict,
    per_season: dict,
    viz_paths: dict[str, str],
    output_path: str,
) -> None:
    """Export a full metrics report as a PDF using reportlab.

    Args:
        metrics:     Combined retrieval + latency metrics dict.
        per_season:  Per-season metrics dict.
        viz_paths:   Dict mapping label → PNG path.
        output_path: Output PDF path.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            Image as RLImage,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
        from reportlab.lib.styles import getSampleStyleSheet

        doc = SimpleDocTemplate(output_path, pagesize=A4,
                                leftMargin=2*cm, rightMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        story = []

        # Title
        story.append(Paragraph(
            "BAH2026 Challenge 11 — Cross-Modal Satellite Image Retrieval",
            styles["Title"]
        ))
        story.append(Paragraph(
            "Evaluation Metrics Report", styles["Heading2"]
        ))
        story.append(Spacer(1, 0.5*cm))

        # Overall metrics table
        story.append(Paragraph("Overall Retrieval Metrics", styles["Heading3"]))
        data = [["Metric", "Value"]] + [
            [k.replace("_", " ").title(), f"{v:.4f}"]
            for k, v in metrics.items()
        ]
        t = Table(data, colWidths=[9*cm, 5*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F3F4")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.5*cm))

        # Per-season table
        if per_season:
            story.append(Paragraph("Per-Season Breakdown", styles["Heading3"]))
            ps_data = [["Season", "Recall@1", "Recall@10", "mAP"]]
            for season, sm in per_season.items():
                ps_data.append([
                    season.title(),
                    f"{sm.get('recall_at_1', 0):.3f}",
                    f"{sm.get('recall_at_10', 0):.3f}",
                    f"{sm.get('map', 0):.3f}",
                ])
            t2 = Table(ps_data, colWidths=[4*cm, 4*cm, 4*cm, 4*cm])
            t2.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A5276")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EAF2FF")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ]))
            story.append(t2)
            story.append(Spacer(1, 0.5*cm))

        # Visualizations
        for label, path in viz_paths.items():
            if path and Path(path).exists():
                story.append(Paragraph(label, styles["Heading3"]))
                story.append(RLImage(path, width=14*cm, height=10*cm))
                story.append(Spacer(1, 0.3*cm))

        doc.build(story)
        logger.info(f"PDF report saved → {output_path}")
    except Exception as exc:
        logger.warning(f"PDF report skipped: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _pretty_table(metrics: dict[str, float]) -> str:
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
    """Run the full evaluation pipeline.

    Args:
        args: Parsed CLI arguments.
    """
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    logger.info(f"Evaluating on {device_str}…")

    model = create_model(backbone=args.backbone, checkpoint_path=args.checkpoint,
                         device=device_str)
    model.eval()

    dataset: SAROpticalPairDataset | Subset = SAROpticalPairDataset(
        args.data_dir, split=args.split
    )

    if args.quick:
        n = min(500, len(dataset))
        idxs = random.sample(range(len(dataset)), n)
        dataset = Subset(dataset, idxs)
        logger.info(f"Quick mode: evaluating on {n} random pairs.")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    logger.info(f"Evaluating on {len(dataset)} pairs ({args.split} split).")

    sar_list, opt_list, seasons = [], [], []

    t_start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            sar = batch["sar"].to(device)
            opt = batch["optical"].to(device)
            with autocast(enabled=device.type == "cuda"):
                sar_emb, opt_emb = model(sar, opt)
            sar_list.append(sar_emb.float().cpu().numpy())
            opt_list.append(opt_emb.float().cpu().numpy())
            for pid in batch["pair_id"]:
                parts = str(pid).split("_")
                seasons.append(parts[1] if len(parts) > 1 else "unknown")

    sar_embs = np.vstack(sar_list).astype("float32")
    opt_embs = np.vstack(opt_list).astype("float32")
    logger.info(f"Embedded {len(sar_embs)} pairs in {time.perf_counter() - t_start:.1f}s.")

    # Core metrics
    retrieval = compute_retrieval_metrics(sar_embs, opt_embs)
    flat_index = faiss.IndexFlatIP(sar_embs.shape[1])
    flat_index.add(opt_embs)
    latency = compute_latency_metrics(sar_embs, flat_index, n_queries=200)
    all_metrics = {**retrieval, **latency}

    # Per-season
    per_season = compute_per_season_metrics(sar_embs, opt_embs, seasons)

    logger.info("\n" + _pretty_table(all_metrics))

    # Save JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"overall": all_metrics, "per_season": per_season}, f, indent=2)
    logger.info(f"Metrics saved → {out_path}")

    # Visualizations
    viz_dir = out_path.parent / "viz"
    viz_dir.mkdir(exist_ok=True)
    viz_paths: dict[str, str] = {}

    if not args.no_viz:
        tsne_path = str(viz_dir / "tsne.png")
        umap_path = str(viz_dir / "umap.png")
        heat_path = str(viz_dir / "similarity_heatmap.png")

        _save_tsne(sar_embs, opt_embs, tsne_path)
        _save_umap(sar_embs, opt_embs, umap_path)
        _save_similarity_heatmap(sar_embs, opt_embs, heat_path)

        viz_paths = {
            "t-SNE Embedding Space": tsne_path,
            "UMAP Embedding Space": umap_path,
            "Cross-Modal Similarity Heatmap": heat_path,
        }

    # PDF report
    _save_pdf_report(all_metrics, per_season, viz_paths, args.report)


if __name__ == "__main__":
    evaluate(parse_args())
