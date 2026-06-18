"""SAROpticalPairDataset — PyTorch Dataset for SEN1-2 paired SAR/Optical images."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

from loguru import logger


# ImageNet mean/std for Optical (Sentinel-2 RGB)
OPTICAL_MEAN = [0.485, 0.456, 0.406]
OPTICAL_STD = [0.229, 0.224, 0.225]

# SAR normalization (post log1p transform)
SAR_MEAN = [0.0]
SAR_STD = [1.0]

IMAGE_SIZE = 224


def _build_sar_transforms(augment: bool = True) -> transforms.Compose:
    """Build SAR-specific transform pipeline."""
    ops: list = []
    if augment:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
        ]
    ops += [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Normalize(mean=SAR_MEAN, std=SAR_STD),
    ]
    return transforms.Compose(ops)


def _build_optical_transforms(augment: bool = True) -> transforms.Compose:
    """Build Optical-specific transform pipeline."""
    ops: list = []
    if augment:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    ops += [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Normalize(mean=OPTICAL_MEAN, std=OPTICAL_STD),
    ]
    return transforms.Compose(ops)


def _load_sar_image(path: str) -> torch.Tensor:
    """Load a single-channel GeoTIFF SAR image and apply log1p transform.

    Returns a float32 tensor of shape (1, H, W).
    """
    if HAS_RASTERIO:
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)  # (H, W)
    else:
        from PIL import Image
        arr = np.array(Image.open(path).convert("L"), dtype=np.float32)

    arr = np.log1p(np.clip(arr, 0, None))  # log1p for SAR backscatter
    tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)
    return tensor


def _load_optical_image(path: str) -> torch.Tensor:
    """Load a 3-channel GeoTIFF Optical image.

    Returns a float32 tensor of shape (3, H, W) in [0, 1] range.
    """
    if HAS_RASTERIO:
        with rasterio.open(path) as src:
            arr = src.read([1, 2, 3]).astype(np.float32)  # (3, H, W)
            # Normalise from [0, 10000] Sentinel-2 range to [0, 1]
            arr = np.clip(arr / 10000.0, 0, 1)
    else:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
        arr = arr.transpose(2, 0, 1)  # (3, H, W)

    return torch.from_numpy(arr)


class SAROpticalPairDataset(Dataset):
    """PyTorch Dataset for paired SAR (Sentinel-1) and Optical (Sentinel-2) images.

    Expected directory layout::

        root_dir/
          s1/       <- SAR .tif files, named <pair_id>_s1.tif
          s2/       <- Optical .tif files, named <pair_id>_s2.tif

    The dataset discovers all pair IDs from the s1 subdirectory and filters
    to the requested split using a deterministic 80/10/10 partition.

    Args:
        root_dir:  Path to the dataset root.
        split:     One of "train", "val", or "test".
        transform: Optional callable applied after the built-in per-modality
                   augmentations.  Receives the full sample dict.
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

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _discover_pairs(self) -> list[str]:
        """Discover and split pair IDs deterministically."""
        s1_dir = self.root_dir / "s1"
        if not s1_dir.exists():
            raise FileNotFoundError(f"SAR directory not found: {s1_dir}")

        all_ids = sorted(
            p.stem.replace("_s1", "") for p in s1_dir.glob("*_s1.tif")
        )
        if not all_ids:
            # Fallback: any .tif file in s1/
            all_ids = sorted(p.stem for p in s1_dir.glob("*.tif"))

        n = len(all_ids)
        n_train = int(n * self.SPLIT_RATIOS[0])
        n_val = int(n * self.SPLIT_RATIOS[1])

        if self.split == "train":
            return all_ids[:n_train]
        elif self.split == "val":
            return all_ids[n_train : n_train + n_val]
        else:  # test
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

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.pair_ids)

    def __getitem__(self, idx: int) -> dict:
        """Return a sample dict with SAR tensor, Optical tensor, and pair_id."""
        pair_id = self.pair_ids[idx]

        sar_tensor = _load_sar_image(str(self._sar_path(pair_id)))
        optical_tensor = _load_optical_image(str(self._optical_path(pair_id)))

        # Per-modality augmentations / normalization
        sar_tensor = self.sar_transform(sar_tensor)
        optical_tensor = self.optical_transform(optical_tensor)

        sample = {
            "sar": sar_tensor,       # (1, 224, 224)
            "optical": optical_tensor,  # (3, 224, 224)
            "pair_id": pair_id,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------


def create_dataloaders(
    root_dir: str | Path,
    batch_size: int = 64,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, and test DataLoaders for SEN1-2.

    Args:
        root_dir:    Path to dataset root (must contain s1/ and s2/ subdirs).
        batch_size:  Mini-batch size.
        num_workers: Parallel data-loading workers.

    Returns:
        A (train_loader, val_loader, test_loader) tuple.
    """
    root_dir = Path(root_dir)

    train_ds = SAROpticalPairDataset(root_dir, split="train")
    val_ds = SAROpticalPairDataset(root_dir, split="val")
    test_ds = SAROpticalPairDataset(root_dir, split="test")

    def _loader(ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=shuffle,
            persistent_workers=num_workers > 0,
        )

    return (
        _loader(train_ds, shuffle=True),
        _loader(val_ds, shuffle=False),
        _loader(test_ds, shuffle=False),
    )
