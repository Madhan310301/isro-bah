"""Training script for the DualEncoder cross-modal SAR/Optical retrieval model."""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from loguru import logger

from ml.dataset import create_dataloaders
from ml.loss import InfoNCELoss
from ml.model import DualEncoder


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-4
    temperature: float = 0.07
    data_dir: str = "./data/SEN12"
    checkpoint_dir: str = "./checkpoints"
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    num_workers: int = 4
    use_wandb: bool = False
    emb_dim: int = 512
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train DualEncoder for SAR/Optical retrieval")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--data_dir", type=str, default="./data/SEN12")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--emb_dim", type=int, default=512)
    a = parser.parse_args()
    return TrainConfig(**vars(a))


# ---------------------------------------------------------------------------
# LR scheduler with linear warmup + cosine annealing
# ---------------------------------------------------------------------------


def _get_cosine_schedule_with_warmup(
    optimizer: AdamW,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """Cosine decay scheduler with linear warmup."""

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    model: DualEncoder,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Encode the entire validation set and compute retrieval metrics.

    Builds an in-memory FAISS IndexFlatIP from optical embeddings, then
    queries with SAR embeddings and computes Recall@K, mAP, and MRR.

    Args:
        model:      Trained DualEncoder.
        dataloader: Validation DataLoader.
        device:     Torch device.

    Returns:
        Dict with keys: recall_at_1, recall_at_5, recall_at_10, map, mrr.
    """
    model.eval()
    all_sar_embs: list[np.ndarray] = []
    all_opt_embs: list[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            sar = batch["sar"].to(device)
            optical = batch["optical"].to(device)
            with autocast(enabled=device.type == "cuda"):
                sar_emb, opt_emb = model(sar, optical)
            all_sar_embs.append(sar_emb.float().cpu().numpy())
            all_opt_embs.append(opt_emb.float().cpu().numpy())

    sar_embs = np.vstack(all_sar_embs).astype("float32")
    opt_embs = np.vstack(all_opt_embs).astype("float32")
    n = len(sar_embs)

    # Build FAISS index from optical embeddings
    index = faiss.IndexFlatIP(sar_embs.shape[1])
    index.add(opt_embs)

    # Query with SAR embeddings — retrieve top-10
    K = 10
    D, I = index.search(sar_embs, K)  # (n, K)

    # Ground truth: pair i → optical index i
    gt = np.arange(n)

    recall_at: dict[int, float] = {}
    for k in (1, 5, 10):
        hits = np.any(I[:, :k] == gt[:, None], axis=1)
        recall_at[k] = float(hits.mean())

    # mAP and MRR
    ap_list: list[float] = []
    rr_list: list[float] = []
    for i in range(n):
        ranks = np.where(I[i] == gt[i])[0]
        if len(ranks) > 0:
            rank = ranks[0] + 1  # 1-indexed
            ap_list.append(1.0 / rank)
            rr_list.append(1.0 / rank)
        else:
            ap_list.append(0.0)
            rr_list.append(0.0)

    metrics = {
        "recall_at_1": recall_at[1],
        "recall_at_5": recall_at[5],
        "recall_at_10": recall_at[10],
        "map": float(np.mean(ap_list)),
        "mrr": float(np.mean(rr_list)),
    }
    return metrics


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(cfg: TrainConfig) -> None:
    """Run the full training loop.

    Args:
        cfg: Training configuration dataclass.
    """
    device = torch.device(cfg.device)
    logger.info(f"Using device: {device}")

    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Data
    train_loader, val_loader, _ = create_dataloaders(
        cfg.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    model = DualEncoder(emb_dim=cfg.emb_dim).to(device)

    # Loss
    criterion = InfoNCELoss(temperature=cfg.temperature)

    # Optimizer
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # Scheduler
    total_steps = cfg.epochs * len(train_loader)
    warmup_steps = cfg.warmup_epochs * len(train_loader)
    scheduler = _get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # AMP scaler
    scaler = GradScaler(enabled=device.type == "cuda")

    # Optional wandb
    if cfg.use_wandb:
        try:
            import wandb
            wandb.init(project="bah2026-challenge11", config=vars(cfg))
        except ImportError:
            logger.warning("wandb not installed — skipping.")
            cfg.use_wandb = False

    best_recall_at_1 = 0.0

    for epoch in range(cfg.epochs):
        model.train()
        model.set_freeze_schedule(epoch)

        epoch_loss = 0.0
        epoch_acc = 0.0
        n_batches = 0

        for batch in train_loader:
            sar = batch["sar"].to(device, non_blocking=True)
            optical = batch["optical"].to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast(enabled=device.type == "cuda"):
                sar_emb, opt_emb = model(sar, optical)
                loss, acc = criterion(sar_emb, opt_emb)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_acc += acc
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        avg_acc = epoch_acc / n_batches
        current_lr = scheduler.get_last_lr()[0]

        logger.info(
            f"Epoch [{epoch + 1}/{cfg.epochs}] "
            f"loss={avg_loss:.4f}  acc={avg_acc:.3f}  lr={current_lr:.2e}"
        )

        # Validation
        val_metrics = evaluate(model, val_loader, device)
        logger.info(
            f"  Val → R@1={val_metrics['recall_at_1']:.3f}  "
            f"R@5={val_metrics['recall_at_5']:.3f}  "
            f"R@10={val_metrics['recall_at_10']:.3f}  "
            f"mAP={val_metrics['map']:.3f}  "
            f"MRR={val_metrics['mrr']:.3f}"
        )

        if cfg.use_wandb:
            import wandb
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": avg_loss,
                    "train/accuracy": avg_acc,
                    "train/lr": current_lr,
                    **{f"val/{k}": v for k, v in val_metrics.items()},
                }
            )

        # Save per-epoch checkpoint
        ckpt_path = Path(cfg.checkpoint_dir) / f"epoch_{epoch + 1:03d}.pt"
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
            },
            ckpt_path,
        )

        # Save best checkpoint
        if val_metrics["recall_at_1"] > best_recall_at_1:
            best_recall_at_1 = val_metrics["recall_at_1"]
            best_path = Path(cfg.checkpoint_dir) / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "val_metrics": val_metrics,
                    "emb_dim": cfg.emb_dim,
                },
                best_path,
            )
            logger.info(f"  ✓ New best R@1={best_recall_at_1:.3f} — saved to {best_path}")

    logger.info(f"Training complete. Best Recall@1 = {best_recall_at_1:.3f}")
    if cfg.use_wandb:
        import wandb
        wandb.finish()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
