"""Supabase client wrapper with in-memory URL cache."""

from __future__ import annotations

import os
from typing import Optional

from loguru import logger


class SupabaseImageClient:
    """Thin wrapper around supabase-py with caching for image metadata lookups.

    Reads ``SUPABASE_URL`` and ``SUPABASE_KEY`` from environment variables.
    If they are not set the client operates in stub mode and returns ``None``
    for all lookups (useful for local CPU-only demos without Supabase).

    Table schema expected::

        satellite_images (
          id, pair_id, modality, season,
          storage_path, lat, lon, embedding_id, created_at
        )
    """

    TABLE = "satellite_images"
    BUCKET = "satellite-images"

    def __init__(self) -> None:
        self._cache: dict[str, dict] = {}
        self._client = None

        url = os.getenv("SUPABASE_URL", "")
        key = os.getenv("SUPABASE_KEY", "")

        if url and key:
            try:
                from supabase import create_client
                self._client = create_client(url, key)
                logger.info("Supabase client initialised.")
            except Exception as exc:
                logger.warning(f"Supabase init failed: {exc}. Running in stub mode.")
        else:
            logger.warning("SUPABASE_URL / SUPABASE_KEY not set — stub mode.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_metadata(self, pair_id: str, modality: str = "optical") -> Optional[dict]:
        """Fetch metadata for a pair_id from Supabase (cached).

        Args:
            pair_id:  The pair identifier (e.g. ``"ROIs1158_spring_s1_1"``).
            modality: ``"sar"`` or ``"optical"`` (default ``"optical"``).

        Returns:
            A dict with keys ``supabase_url``, ``lat``, ``lon``, ``season``,
            or ``None`` if unavailable.
        """
        cache_key = f"{pair_id}:{modality}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self._client is None:
            return None

        try:
            res = (
                self._client.table(self.TABLE)
                .select("storage_path, lat, lon, season")
                .eq("pair_id", pair_id)
                .eq("modality", modality)
                .limit(1)
                .execute()
            )
            if res.data:
                row = res.data[0]
                public_url = self._client.storage.from_(self.BUCKET).get_public_url(
                    row["storage_path"]
                )
                meta = {
                    "supabase_url": public_url,
                    "lat": row.get("lat"),
                    "lon": row.get("lon"),
                    "season": row.get("season"),
                }
                self._cache[cache_key] = meta
                return meta
        except Exception as exc:
            logger.warning(f"Supabase lookup failed for {pair_id}: {exc}")

        return None

    def insert_metadata(
        self,
        pair_id: str,
        modality: str,
        season: str,
        storage_path: str,
        lat: Optional[float],
        lon: Optional[float],
        embedding_id: Optional[int] = None,
    ) -> bool:
        """Insert a metadata row into the satellite_images table.

        Returns:
            True on success, False on failure or stub mode.
        """
        if self._client is None:
            logger.warning("Supabase stub mode — insert skipped.")
            return False
        try:
            self._client.table(self.TABLE).insert(
                {
                    "pair_id": pair_id,
                    "modality": modality,
                    "season": season,
                    "storage_path": storage_path,
                    "lat": lat,
                    "lon": lon,
                    "embedding_id": embedding_id,
                }
            ).execute()
            return True
        except Exception as exc:
            logger.error(f"Supabase insert failed: {exc}")
            return False

    def upload_image(self, local_path: str, storage_path: str) -> bool:
        """Upload a file to the ``satellite-images`` storage bucket.

        Args:
            local_path:   Local filesystem path to the image file.
            storage_path: Destination path inside the bucket.

        Returns:
            True on success.
        """
        if self._client is None:
            logger.warning("Supabase stub mode — upload skipped.")
            return False
        try:
            with open(local_path, "rb") as f:
                self._client.storage.from_(self.BUCKET).upload(
                    storage_path, f, {"content-type": "image/tiff"}
                )
            return True
        except Exception as exc:
            logger.error(f"Supabase upload failed for {local_path}: {exc}")
            return False
