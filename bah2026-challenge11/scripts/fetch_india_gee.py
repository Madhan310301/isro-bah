"""Download Sentinel-1/2 paired patches for Indian cities using Google Earth Engine.

Requires:
    - ``earthengine-api`` installed (``pip install earthengine-api``)
    - GEE authentication: ``earthengine authenticate``
    - Optional: ``GOOGLE_APPLICATION_CREDENTIALS`` env var for service-account auth

Usage::

    python -m scripts.fetch_india_gee --output_dir ./data/india --patch_size 256 --year 2023
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

# ---------------------------------------------------------------------------
# City AOI definitions (WGS-84 bounding boxes)
# ---------------------------------------------------------------------------

INDIA_CITIES: dict[str, dict] = {
    "delhi": {
        "label": "Delhi",
        "lon": 77.2090, "lat": 28.6139,
        "bbox": [76.84, 28.40, 77.35, 28.88],
    },
    "mumbai": {
        "label": "Mumbai",
        "lon": 72.8777, "lat": 19.0760,
        "bbox": [72.77, 18.89, 73.05, 19.27],
    },
    "chennai": {
        "label": "Chennai",
        "lon": 80.2707, "lat": 13.0827,
        "bbox": [80.15, 12.90, 80.36, 13.23],
    },
    "bangalore": {
        "label": "Bangalore",
        "lon": 77.5946, "lat": 12.9716,
        "bbox": [77.45, 12.84, 77.75, 13.11],
    },
    "kerala_flood": {
        "label": "Kerala (Flood Region)",
        "lon": 76.5222, "lat": 10.8505,
        "bbox": [75.80, 9.50, 77.40, 12.00],
    },
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description="Download Sentinel-1/2 paired patches for Indian cities via GEE."
    )
    p.add_argument("--output_dir", default="./data/india",
                   help="Root directory for downloaded patches.")
    p.add_argument("--patch_size", type=int, default=256,
                   help="Patch size in pixels (default 256).")
    p.add_argument("--year", type=int, default=2023,
                   help="Year to fetch imagery for (default 2023).")
    p.add_argument("--cities", nargs="+", default=list(INDIA_CITIES.keys()),
                   help="City keys to download (default: all).")
    p.add_argument("--scale", type=int, default=10,
                   help="Sentinel pixel scale in metres (default 10).")
    p.add_argument("--max_cloud_pct", type=float, default=20.0,
                   help="Maximum Sentinel-2 cloud percentage (default 20).")
    p.add_argument("--stub", action="store_true",
                   help="Dry-run: print what would be downloaded without calling GEE.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# GEE helpers
# ---------------------------------------------------------------------------


def _init_gee() -> bool:
    """Initialise the GEE Python client.

    Returns:
        True if initialisation succeeded, False otherwise.
    """
    try:
        import ee

        try:
            ee.Initialize()
        except Exception:
            ee.Authenticate()
            ee.Initialize()
        return True
    except ImportError:
        logger.error(
            "earthengine-api not installed. Run: pip install earthengine-api"
        )
        return False
    except Exception as exc:
        logger.error(f"GEE initialization failed: {exc}")
        return False


def _apply_cloud_mask_s2(image):
    """Apply Sentinel-2 SCL-band cloud mask.

    Keeps only pixels with SCL class 4 (vegetation), 5 (bare soil),
    6 (water), and 11 (snow), masking clouds, cloud shadows, and saturated pixels.

    Args:
        image: ee.Image Sentinel-2 SR image.

    Returns:
        Cloud-masked ee.Image.
    """
    import ee

    scl = image.select("SCL")
    # Keep: vegetation(4), bare(5), water(6), snow(11)
    cloud_free_mask = (
        scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(11))
    )
    return image.updateMask(cloud_free_mask)


def _get_s1_collection(aoi, start_date: str, end_date: str):
    """Get filtered Sentinel-1 GRD collection for an AOI.

    Args:
        aoi:        ee.Geometry bounding box.
        start_date: ISO date string.
        end_date:   ISO date string.

    Returns:
        Filtered ee.ImageCollection.
    """
    import ee

    return (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .select(["VV", "VH"])
    )


def _get_s2_collection(aoi, start_date: str, end_date: str, max_cloud_pct: float):
    """Get filtered cloud-masked Sentinel-2 SR collection for an AOI.

    Args:
        aoi:           ee.Geometry bounding box.
        start_date:    ISO date string.
        end_date:      ISO date string.
        max_cloud_pct: Maximum cloud cover percentage.

    Returns:
        Filtered ee.ImageCollection with cloud masking applied.
    """
    import ee

    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
        .select(["B4", "B3", "B2", "SCL"])  # RGB + SCL for cloud masking
        .map(_apply_cloud_mask_s2)
        .select(["B4", "B3", "B2"])  # Keep only RGB bands
    )


def _export_patch(
    image,
    description: str,
    output_path: str,
    aoi,
    scale: int,
    patch_size: int,
) -> None:
    """Export a GEE image as a GeoTIFF to Google Drive (then download).

    In a real pipeline you would export to GCS or Drive and then
    download.  Here we use ``ee.batch.Export.image.toDrive`` and log
    the task ID for monitoring.

    Args:
        image:       ee.Image to export.
        description: Unique export task description.
        output_path: Local path where the file should eventually be saved.
        aoi:         ee.Geometry region.
        scale:       Pixel scale in metres.
        patch_size:  Dimensions in pixels.
    """
    import ee

    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder="BAH2026_India_Patches",
        fileNamePrefix=description,
        region=aoi,
        scale=scale,
        crs="EPSG:4326",
        fileFormat="GeoTIFF",
        maxPixels=1e13,
    )
    task.start()
    logger.info(f"GEE export task started: {description} (task_id={task.id})")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def fetch_india_patches(args: argparse.Namespace) -> None:
    """Download Sentinel-1/2 paired patches for configured Indian cities.

    Args:
        args: Parsed CLI arguments.
    """
    if args.stub:
        logger.info("DRY-RUN mode — no GEE calls will be made.")
        for city_key in args.cities:
            city = INDIA_CITIES.get(city_key, {})
            logger.info(f"  Would fetch: {city.get('label', city_key)}")
        return

    if not _init_gee():
        logger.error("GEE not available — exiting.")
        sys.exit(1)

    import ee

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_date = f"{args.year}-01-01"
    end_date = f"{args.year}-12-31"

    seasons = {
        "spring": (f"{args.year}-03-01", f"{args.year}-05-31"),
        "summer": (f"{args.year}-06-01", f"{args.year}-08-31"),
        "autumn": (f"{args.year}-09-01", f"{args.year}-11-30"),
        "winter": (f"{args.year}-12-01", f"{args.year + 1}-02-28"),
    }

    for city_key in args.cities:
        if city_key not in INDIA_CITIES:
            logger.warning(f"Unknown city key: {city_key} — skipping.")
            continue

        city = INDIA_CITIES[city_key]
        bbox = city["bbox"]
        aoi = ee.Geometry.Rectangle(bbox)
        logger.info(f"Processing {city['label']} | bbox={bbox}")

        city_dir = output_dir / city_key
        (city_dir / "s1").mkdir(parents=True, exist_ok=True)
        (city_dir / "s2").mkdir(parents=True, exist_ok=True)

        for season_name, (s_start, s_end) in seasons.items():
            logger.info(f"  Season: {season_name} ({s_start} → {s_end})")

            # Sentinel-1 median composite
            s1_col = _get_s1_collection(aoi, s_start, s_end)
            s1_count = s1_col.size().getInfo()
            if s1_count == 0:
                logger.warning(f"  No S1 imagery for {city_key}/{season_name} — skipping.")
                continue

            s1_image = s1_col.median().select(["VV"]).rename("VV")  # single-channel

            # Sentinel-2 median composite
            s2_col = _get_s2_collection(aoi, s_start, s_end, args.max_cloud_pct)
            s2_count = s2_col.size().getInfo()
            if s2_count == 0:
                logger.warning(f"  No S2 imagery for {city_key}/{season_name} — skipping.")
                continue

            s2_image = s2_col.median()

            # Export both
            pair_id = f"{city_key}_{season_name}"
            _export_patch(
                s1_image, f"{pair_id}_s1",
                str(city_dir / "s1" / f"{pair_id}_s1.tif"),
                aoi, args.scale, args.patch_size,
            )
            _export_patch(
                s2_image, f"{pair_id}_s2",
                str(city_dir / "s2" / f"{pair_id}_s2.tif"),
                aoi, args.scale, args.patch_size,
            )

    logger.info(
        "All GEE export tasks started. Monitor at: "
        "https://code.earthengine.google.com/tasks"
    )
    logger.info(
        "After tasks complete, download from Google Drive and place under "
        f"{output_dir}/<city>/s1/ and <city>/s2/."
    )


if __name__ == "__main__":
    fetch_india_patches(parse_args())
