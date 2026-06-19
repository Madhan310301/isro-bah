"""Upload a folder of satellite images to Supabase Storage and insert metadata rows.

Usage::

    python -m scripts.upload_to_supabase \
        --folder    ./data/SEN12/s2 \
        --modality  optical \
        --season    spring \
        --mapping   ./faiss_index/metadata.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from app.supabase_client import SupabaseImageClient

load_dotenv()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload images to Supabase Storage + insert metadata.")
    p.add_argument("--folder", required=True, help="Path to folder containing image files.")
    p.add_argument("--modality", required=True, choices=["sar", "optical"])
    p.add_argument("--season", default="unknown", help="Season label (spring/summer/fall/winter).")
    p.add_argument("--mapping", default=None,
                   help="Path to metadata.json to cross-reference embedding IDs.")
    p.add_argument("--lat", type=float, default=None, help="Latitude (optional).")
    p.add_argument("--lon", type=float, default=None, help="Longitude (optional).")
    p.add_argument("--extensions", default=".tif,.tiff,.png,.jpg",
                   help="Comma-separated list of file extensions to upload.")
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    client = SupabaseImageClient()

    folder = Path(args.folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    extensions = {e.strip().lower() for e in args.extensions.split(",")}

    # Load existing metadata for embedding_id cross-reference
    embedding_id_map: dict[str, int] = {}
    if args.mapping and Path(args.mapping).exists():
        with open(args.mapping) as f:
            raw_meta = json.load(f)
        for emb_id, meta in raw_meta.items():
            embedding_id_map[meta.get("pair_id", "")] = int(emb_id)
        logger.info(f"Loaded {len(embedding_id_map)} embedding_id mappings.")

    image_files = [
        p for p in sorted(folder.iterdir())
        if p.suffix.lower() in extensions
    ]
    logger.info(f"Found {len(image_files)} files to upload from {folder}.")

    # Map: storage_path → embedding_id for FAISS sync
    storage_map: dict[str, int | None] = {}

    success = fail = 0
    for img_path in image_files:
        # Derive pair_id from filename (strip _s1/_s2 suffix if present)
        stem = img_path.stem
        pair_id = stem.replace("_s2", "").replace("_s1", "")

        storage_path = f"{args.modality}/{args.season}/{img_path.name}"
        embedding_id = embedding_id_map.get(pair_id)

        # Upload file
        uploaded = client.upload_image(str(img_path), storage_path)
        if not uploaded:
            fail += 1
            continue

        # Insert metadata row
        inserted = client.insert_metadata(
            pair_id=pair_id,
            modality=args.modality,
            season=args.season,
            storage_path=storage_path,
            lat=args.lat,
            lon=args.lon,
            embedding_id=embedding_id,
        )

        if inserted:
            success += 1
            storage_map[storage_path] = embedding_id
            if success % 100 == 0:
                logger.info(f"  Uploaded {success} files…")
        else:
            fail += 1

    logger.info(f"Done. Uploaded: {success}, Failed: {fail}.")

    # Save storage→embedding_id mapping for FAISS metadata sync
    out_path = Path(args.folder).parent / "storage_embedding_map.json"
    with open(out_path, "w") as f:
        json.dump(storage_map, f, indent=2)
    logger.info(f"Storage→embedding_id map saved to {out_path}.")


if __name__ == "__main__":
    main(parse_args())
