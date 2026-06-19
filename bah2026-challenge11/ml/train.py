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
    """Training configuration dataclass.

    Attributes:
        epochs:          Total number of training epochs.
        batch_size:      Mini-batch size per device.
        lr:              Initial AdamW learning rate.
        weight_decay:    AdamW weight decay.
        temperature:     InfoNCE temperature τ.
        data_dir:        Path to SEN1-2 dataset root.
        checkpoint_dir:  Directory to save checkpoints.
        warmup_epochs:   Number of linear warmup epochs.
        grad_clip:       Gradient clipping max-norm.
        num_workers:     DataLoader parallel workers.
        use_wandb:       Enable Weights & Biases logging.
        emb_dim:         Embedding dimension.
        backbone:        Encoder backbone: ``"vit"`` or ``"resnet50"``.
        mixup_alpha:     Mixup alpha for optical images (0 = disabled).
        resume:          Path to checkpoint to resume training from.
        patience:        Early-stopping patience on Recall@10 (0 = disabled).
        device:          Torch device string (auto-detected).
    """

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
    emb_dim: int = 256
    backbone: str = "vit"
    mixup_alpha: float = 0.2
    resume: str = ""
    patience: int = 5
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> TrainConfig:
    """Parse CLI arguments into a TrainConfig.

    Returns:
        Populated TrainConfig dataclass.
    """
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
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--backbone", type=str, default="vit", choices=["vit", "resnet50"])
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument("--resume", type=str, default="",
                        help="Path to checkpoint to resume from.")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early-stopping patience on Recall@10 (0 = off).")
    a = parser.parse_args()
    return TrainConfig(**vars(a))


# ---------------------------------------------------------------------------
# LR scheduler
# ---------------------------------------------------------------------------


def _get_cosine_schedule_with_warmup(
    optimizer: AdamW,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """Cosine decay scheduler with linear warmup.

    Args:
        optimizer:     AdamW optimizer.
        warmup_steps:  Number of linear warmup steps.
        total_steps:   Total training steps.

    Returns:
        LambdaLR scheduler.
    """

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

    Builds an in-memory FAISS index from optical embeddings, then queries
    with SAR embeddings to compute Recall@K, mAP, and MRR.

    Args:
        model:      Trained DualEncoder.
        dataloader: Validation DataLoader.
        device:     Torch device.

    Returns:
        Dict with keys: ``recall_at_1``, ``recall_at_5``, ``recall_at_10``,
        ``map``, ``mrr``.
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

    index = faiss.IndexFlatIP(sar_embs.shape[1])
    index.add(opt_embs)

    K = 10
    _, I = index.search(sar_embs, K)  # (n, K)
    gt = np.arange(n)

    recall_at: dict[int, float] = {}
    for k in (1, 5, 10):
        hits = np.any(I[:, :k] == gt[:, None], axis=1)
        recall_at[k] = float(hits.mean())

    ap_list: list[float] = []
    rr_list: list[float] = []
    for i in range(n):
        ranks = np.where(I[i] == gt[i])[0]
        if len(ranks) > 0:
            rank = ranks[0] + 1
            ap_list.append(1.0 / rank)
            rr_list.append(1.0 / rank)
        else:
            ap_list.append(0.0)
            rr_list.append(0.0)

    return {
        "recall_at_1": recall_at[1],
        "recall_at_5": recall_at[5],
        "recall_at_10": recall_at[10],
        "map": float(np.mean(ap_list)),
        "mrr": float(np.mean(rr_list)),
    }


# ---------------------------------------------------------------------------
# Training curve plotting
# ---------------------------------------------------------------------------


def _save_training_curves(
    history: dict[str, list[float]],
    output_path: str,
) -> None:
    """Save loss and Recall@K curves to a PNG file.

    Args:
        history:     Dict of metric name → list of per-epoch values.
        output_path: Output PNG filepath.
    """
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Loss
        ax = axes[0]
        if "train_loss" in history:
            ax.plot(history["train_loss"], label="Train Loss")
        if "val_loss" in history:
            ax.plot(history["val_loss"], label="Val Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Recall metrics
        ax = axes[1]
        for key, label in [
            ("recall_at_1", "Recall@1"),
            ("recall_at_10", "Recall@10"),
        ]:
            if key in history:
                ax.plot(history[key], label=label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Recall")
        ax.set_title("Validation Recall@K")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Training curves saved to {output_path}")
    except ImportError:
        logger.warning("matplotlib not installed — training curves not saved.")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(cfg: TrainConfig) -> None:
    """Run the full training loop with early stopping and multi-GPU support.

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
        mixup_alpha=cfg.mixup_alpha,
    )
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    model = DualEncoder(emb_dim=cfg.emb_dim, backbone=cfg.backbone).to(device)

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model = nn.DataParallel(model)

    # Loss
    criterion = InfoNCELoss(temperature=cfg.temperature)

    # Optimizer
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
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

    # Resume from checkpoint
    start_epoch = 0
    if cfg.resume and Path(cfg.resume).exists():
        logger.info(f"Resuming from {cfg.resume}")
        ckpt = torch.load(cfg.resume, map_location=device)
        raw_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt.get("optimizer_state_dict", {}))
        start_epoch = ckpt.get("epoch", 0)
        logger.info(f"Resumed at epoch {start_epoch}.")

    # Optional wandb
    if cfg.use_wandb:
        try:
            import wandb
            wandb.init(project="bah2026-challenge11", config=vars(cfg))
        except ImportError:
            logger.warning("wandb not installed — skipping.")
            cfg.use_wandb = False

    best_recall_at_10 = 0.0
    best_recall_at_1 = 0.0
    patience_counter = 0

    # Training history for curves
    history: dict[str, list[float]] = {
        "train_loss": [],
        "recall_at_1": [],
        "recall_at_10": [],
    }

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        raw_model.set_freeze_schedule(epoch)

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

            # Update momentum encoders
            raw_model.update_momentum_encoders()

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
        val_metrics = evaluate(raw_model, val_loader, device)
        logger.info(
            f"  Val → R@1={val_metrics['recall_at_1']:.3f}  "
            f"R@5={val_metrics['recall_at_5']:.3f}  "
            f"R@10={val_metrics['recall_at_10']:.3f}  "
            f"mAP={val_metrics['map']:.3f}  "
            f"MRR={val_metrics['mrr']:.3f}"
        )

        # History
        history["train_loss"].append(avg_loss)
        history["recall_at_1"].append(val_metrics["recall_at_1"])
        history["recall_at_10"].append(val_metrics["recall_at_10"])

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
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "emb_dim": cfg.emb_dim,
                "backbone": cfg.backbone,
            },
            ckpt_path,
        )

        # Best checkpoint by Recall@1
        if val_metrics["recall_at_1"] > best_recall_at_1:
            best_recall_at_1 = val_metrics["recall_at_1"]
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": raw_model.state_dict(),
                    "val_metrics": val_metrics,
                    "emb_dim": cfg.emb_dim,
                    "backbone": cfg.backbone,
                },
                Path(cfg.checkpoint_dir) / "best_model.pt",
            )
            logger.info(f"  ✓ New best R@1={best_recall_at_1:.3f} — saved best_model.pt")

        # Early stopping on Recall@10
        if val_metrics["recall_at_10"] > best_recall_at_10:
            best_recall_at_10 = val_metrics["recall_at_10"]
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(f"  Early-stop patience: {patience_counter}/{cfg.patience}")

        if cfg.patience > 0 and patience_counter >= cfg.patience:
            logger.info(f"Early stopping triggered at epoch {epoch + 1}.")
            break

    # Save training curves
    curves_path = str(Path(cfg.checkpoint_dir) / "training_curves.png")
    _save_training_curves(history, curves_path)

    logger.info(f"Training complete. Best Recall@1={best_recall_at_1:.3f}  R@10={best_recall_at_10:.3f}")
    if cfg.use_wandb:
        import wandb
        wandb.finish()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
