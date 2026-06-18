"""Response caching: Redis with automatic in-memory fallback."""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Optional

from loguru import logger


class CacheBackend:
    """Abstract cache backend interface."""

    def get(self, key: str) -> Optional[Any]:
        raise NotImplementedError

    def set(self, key: str, value: Any, ttl: int = 300) -> None:
        raise NotImplementedError

    def delete(self, key: str) -> None:
        raise NotImplementedError


class InMemoryCache(CacheBackend):
    """Simple in-process dict-based cache with TTL expiry.

    Args:
        max_size: Maximum number of entries before LRU eviction (default 1000).
    """

    def __init__(self, max_size: int = 1000) -> None:
        self._store: dict[str, tuple[Any, float]] = {}
        self.max_size = max_size

    def get(self, key: str) -> Optional[Any]:
        """Return cached value or None if missing/expired.

        Args:
            key: Cache key string.

        Returns:
            Cached value or None.
        """
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if time.monotonic() > expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any, ttl: int = 300) -> None:
        """Store a value with a TTL.

        Args:
            key:   Cache key.
            value: Serialisable value.
            ttl:   Time-to-live in seconds.
        """
        if len(self._store) >= self.max_size:
            # Evict oldest entry
            oldest = min(self._store, key=lambda k: self._store[k][1])
            del self._store[oldest]
        self._store[key] = (value, time.monotonic() + ttl)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)


class RedisCache(CacheBackend):
    """Redis-backed cache.

    Args:
        redis_url: Redis connection URL (default ``redis://localhost:6379/0``).
        ttl:       Default TTL in seconds.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0") -> None:
        import redis

        self._client = redis.from_url(redis_url, decode_responses=True)
        self._client.ping()  # raises ConnectionError if unavailable
        logger.info(f"Redis cache connected: {redis_url}")

    def get(self, key: str) -> Optional[Any]:
        raw = self._client.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    def set(self, key: str, value: Any, ttl: int = 300) -> None:
        self._client.setex(key, ttl, json.dumps(value))

    def delete(self, key: str) -> None:
        self._client.delete(key)


def build_cache(ttl: int = 300) -> CacheBackend:
    """Build the best available cache backend.

    Tries Redis first (from ``REDIS_URL`` env var), falls back to in-memory.

    Args:
        ttl: Default TTL in seconds (used for documentation; callers pass TTL
             explicitly to :meth:`CacheBackend.set`).

    Returns:
        A configured :class:`CacheBackend` instance.
    """
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    try:
        cache = RedisCache(redis_url)
        return cache
    except Exception as exc:
        logger.warning(f"Redis unavailable ({exc}) — using in-memory cache.")
        return InMemoryCache()


def make_cache_key(prefix: str, **kwargs: Any) -> str:
    """Create a deterministic cache key from a prefix and keyword arguments.

    Args:
        prefix: String prefix (e.g. ``"search"``).
        **kwargs: Hashable values to include in the key.

    Returns:
        A hex-digest cache key string.
    """
    payload = json.dumps(kwargs, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return f"{prefix}:{digest}"
