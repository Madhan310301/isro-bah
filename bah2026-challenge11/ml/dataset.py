"""SAROpticalPairDataset with SAR-specific augmentations and mixup support."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from loguru import logger

try:
    import rasterio

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

# ImageNet mean/std for Optical (Sentinel-2 RGB)
OPTICAL_MEAN = [0.485, 0.456, 0.406]
OPTICAL_STD = [0.229, 0.224, 0.225]

# SAR normalization (post log1p transform)
SAR_MEAN = [0.0]
SAR_STD = [1.0]

IMAGE_SIZE = 224


# ---------------------------------------------------------------------------
# SAR-specific augmentations
# ---------------------------------------------------------------------------


class SpeckleNoise:
    """Multiplicative speckle noise for SAR images.

    Adds Rayleigh/Gamma-distributed multiplicative noise that mimics SAR
    speckle patterns.

    Args:
        sigma: Standard deviation of the noise (default 0.1).
        p:     Probability of applying this transform (default 0.5).
    """

    def __init__(self, sigma: float = 0.1, p: float = 0.5) -> None:
        self.sigma = sigma
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply speckle noise to a SAR tensor.

        Args:
            x: Float tensor of any shape.

        Returns:
            Noise-augmented tensor of the same shape.
        """
        if random.random() > self.p:
            return x
        noise = torch.randn_like(x) * self.sigma
        return x * (1.0 + noise)


class SARNormalize(transforms.Normalize):
    """Convenience wrapper — normalise after log1p is already applied."""

    pass


def _build_sar_transforms(augment: bool = True) -> transforms.Compose:
    """Build SAR-specific transform pipeline including speckle noise.

    Args:
        augment: Whether to apply augmentations (only during training).

    Returns:
        A composed transform pipeline.
    """
    ops: list = []
    if augment:
        ops += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            SpeckleNoise(sigma=0.1, p=0.5),
        ]
    ops += [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Normalize(mean=SAR_MEAN, std=SAR_STD),
    ]
    return transforms.Compose(ops)


def _build_optical_transforms(augment: bool = True) -> transforms.Compose:
    """Build Optical-specific transform pipeline.

    Args:
        augment: Whether to apply augmentations.

    Returns:
        A composed transform pipeline.
    """
    ops: list = []
    if augment:
        ops += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        ]
    ops += [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Normalize(mean=OPTICAL_MEAN, std=OPTICAL_STD),
    ]
    return transforms.Compose(ops)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------


def _load_sar_image(path: str) -> torch.Tensor:
    """Load a single-channel GeoTIFF SAR image and apply log1p transform.

    Args:
        path: Filesystem path to the SAR GeoTIFF.

    Returns:
        Float32 tensor of shape (1, H, W).
    """
    if HAS_RASTERIO:
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
    else:
        from PIL import Image

        arr = np.array(Image.open(path).convert("L"), dtype=np.float32)

    arr = np.log1p(np.clip(arr, 0, None))
    return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)


def _load_optical_image(path: str) -> torch.Tensor:
    """Load a 3-channel GeoTIFF Optical image.

    Args:
        path: Filesystem path to the Optical GeoTIFF.

    Returns:
        Float32 tensor of shape (3, H, W) in [0, 1] range.
    """
    if HAS_RASTERIO:
        with rasterio.open(path) as src:
            arr = src.read([1, 2, 3]).astype(np.float32)
            arr = np.clip(arr / 10000.0, 0, 1)
    else:
        from PIL import Image

        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)

    return torch.from_numpy(arr)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SAROpticalPairDataset(Dataset):
    """PyTorch Dataset for paired SAR (Sentinel-1) and Optical (Sentinel-2) images.

    Expected directory layout::

        root_dir/
          s1/   <- SAR .tif files, named <pair_id>_s1.tif
          s2/   <- Optical .tif files, named <pair_id>_s2.tif

    Args:
        root_dir:  Path to the dataset root.
        split:     One of ``"train"``, ``"val"``, or ``"test"``.
        transform: Optional callable applied to the full sample dict after
                   per-modality augmentations.
    """

    SPLITS = ("train", "val", "test")
    SPLIT_RATIOS = (0.8, 0.1, 0.1)

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        if split not in self.SPLITS:
            raise ValueError(f"split must be one of {self.SPLITS}, got {split!r}")

        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        augment = split == "train"
        self.sar_transform = _build_sar_transforms(augment=augment)
        self.optical_transform = _build_optical_transforms(augment=augment)

        self.pair_ids = self._discover_pairs()
        logger.info(
            f"SAROpticalPairDataset [{split}]: {len(self.pair_ids)} pairs "
            f"from {self.root_dir}"
        )

    def _discover_pairs(self) -> list[str]:
        """Discover and split pair IDs deterministically."""
        s1_dir = self.root_dir / "s1"
        if not s1_dir.exists():
            raise FileNotFoundError(f"SAR directory not found: {s1_dir}")

        all_ids = sorted(
            p.stem.replace("_s1", "") for p in s1_dir.glob("*_s1.tif")
        )
        if not all_ids:
            all_ids = sorted(p.stem for p in s1_dir.glob("*.tif"))

        n = len(all_ids)
        n_train = int(n * self.SPLIT_RATIOS[0])
        n_val = int(n * self.SPLIT_RATIOS[1])

        if self.split == "train":
            return all_ids[:n_train]
        elif self.split == "val":
            return all_ids[n_train : n_train + n_val]
        else:
            return all_ids[n_train + n_val :]

    def _sar_path(self, pair_id: str) -> Path:
        p = self.root_dir / "s1" / f"{pair_id}_s1.tif"
        if not p.exists():
            p = self.root_dir / "s1" / f"{pair_id}.tif"
        return p

    def _optical_path(self, pair_id: str) -> Path:
        p = self.root_dir / "s2" / f"{pair_id}_s2.tif"
        if not p.exists():
            p = self.root_dir / "s2" / f"{pair_id}.tif"
        return p

    def __len__(self) -> int:
        return len(self.pair_ids)

    def __getitem__(self, idx: int) -> dict:
        """Return a sample dict with SAR tensor, Optical tensor, and pair_id.

        Args:
            idx: Sample index.

        Returns:
            Dict with keys ``"sar"`` (1, H, W), ``"optical"`` (3, H, W),
            ``"pair_id"`` (str).
        """
        pair_id = self.pair_ids[idx]

        sar_tensor = _load_sar_image(str(self._sar_path(pair_id)))
        optical_tensor = _load_optical_image(str(self._optical_path(pair_id)))

        sar_tensor = self.sar_transform(sar_tensor)
        optical_tensor = self.optical_transform(optical_tensor)

        sample = {
            "sar": sar_tensor,
            "optical": optical_tensor,
            "pair_id": pair_id,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


# ---------------------------------------------------------------------------
# Mixup collator
# ---------------------------------------------------------------------------


class MixupCollator:
    """Collate function that applies mixup augmentation to optical images.

    Mixup blends each optical image with a random other image in the batch,
    using a Beta(alpha, alpha) mixing coefficient.  SAR images and pair_ids
    are left unchanged.

    Args:
        alpha: Beta distribution parameter (default 0.2).  Set to 0 to
               disable mixup.
    """

    def __init__(self, alpha: float = 0.2) -> None:
        self.alpha = alpha

    def __call__(self, batch: list[dict]) -> dict:
        """Collate and apply mixup to optical images.

        Args:
            batch: List of sample dicts from the Dataset.

        Returns:
            Batched dict with mixed optical images.
        """
        collated = torch.utils.data.dataloader.default_collate(batch)

        if self.alpha <= 0:
            return collated

        optical = collated["optical"]  # (B, 3, H, W)
        B = optical.size(0)
        lam = float(np.random.beta(self.alpha, self.alpha))

        perm = torch.randperm(B)
        collated["optical"] = lam * optical + (1.0 - lam) * optical[perm]

        return collated


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------


def create_dataloaders(
    root_dir: str | Path,
    batch_size: int = 64,
    num_workers: int = 4,
    mixup_alpha: float = 0.2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, and test DataLoaders for SEN1-2.

    Args:
        root_dir:     Path to dataset root (must contain s1/ and s2/).
        batch_size:   Mini-batch size.
        num_workers:  Parallel data-loading workers.
        mixup_alpha:  Mixup alpha for optical images during training (0 = off).

    Returns:
        A (train_loader, val_loader, test_loader) tuple.
    """
    root_dir = Path(root_dir)

    train_ds = SAROpticalPairDataset(root_dir, split="train")
    val_ds = SAROpticalPairDataset(root_dir, split="val")
    test_ds = SAROpticalPairDataset(root_dir, split="test")

    train_collate = MixupCollator(alpha=mixup_alpha) if mixup_alpha > 0 else None

    def _loader(
        ds: Dataset,
        shuffle: bool,
        collate_fn: Optional[Callable] = None,
    ) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=shuffle,
            persistent_workers=num_workers > 0,
            collate_fn=collate_fn,
        )

    return (
        _loader(train_ds, shuffle=True, collate_fn=train_collate),
        _loader(val_ds, shuffle=False),
        _loader(test_ds, shuffle=False),
    )
