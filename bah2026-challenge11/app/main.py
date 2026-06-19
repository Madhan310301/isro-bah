"""FastAPI application — Cross-Modal Satellite Image Retrieval backend."""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.auth import require_auth
from app.cache import build_cache, make_cache_key
from app.inference import engine
from app.models import (
    BatchSearchRequest,
    BatchSearchResponse,
    EmbedRequest,
    EmbedResponse,
    ErrorResponse,
    HealthResponse,
    IndexStatsResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SimilarRequest,
    SimilarResponse,
)
from app.supabase_client import SupabaseImageClient

load_dotenv()

CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model, FAISS indices, Supabase client, and cache on startup."""
    checkpoint = os.getenv("MODEL_CHECKPOINT_PATH", "./checkpoints/best_model.pt")
    faiss_dir = os.getenv("FAISS_INDEX_PATH", "./faiss_index")
    backbone = os.getenv("MODEL_BACKBONE", "vit")

    try:
        engine.load(
            checkpoint_path=checkpoint,
            sar_index_path=f"{faiss_dir}/sar_index.faiss",
            optical_index_path=f"{faiss_dir}/optical_index.faiss",
            metadata_path=f"{faiss_dir}/metadata.json",
            backbone=backbone,
        )
    except Exception as exc:
        logger.error(f"Model/index load failed: {exc}")
        logger.warning("Running in degraded mode — /api/search will return 503.")

    app.state.supabase = SupabaseImageClient()
    app.state.cache = build_cache(ttl=CACHE_TTL)
    logger.info("Application startup complete.")
    yield
    logger.info("Application shutdown.")


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address, default_limits=["30/minute"])

# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="BAH2026 Cross-Modal Satellite Image Retrieval API",
    version="2.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


@app.middleware("http")
async def add_request_timing(request: Request, call_next):
    """Log and attach request timing to every response."""
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Request-Time-Ms"] = f"{elapsed_ms:.2f}"
    logger.debug(
        f"{request.method} {request.url.path} → {response.status_code} ({elapsed_ms:.1f}ms)"
    )
    return response


# ---------------------------------------------------------------------------
# Exception handler
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(error="internal_server_error", detail=str(exc)).model_dump(),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_model() -> None:
    if engine.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/health", response_model=HealthResponse, tags=["system"])
async def health():
    """Return service health and current index statistics."""
    return HealthResponse(
        status="ok",
        model=engine.model_name,
        index_size=engine.optical_index_size,
        device=engine.device,
    )


@app.get("/api/index/stats", response_model=IndexStatsResponse, tags=["system"])
async def index_stats():
    """Return detailed FAISS index statistics including memory usage."""
    return IndexStatsResponse(
        sar_index_size=engine.sar_index_size,
        optical_index_size=engine.optical_index_size,
        index_type=engine.index_type,
        index_version=engine.index_version,
        emb_dim=engine.emb_dim,
        memory_usage_mb=round(engine.memory_usage_mb, 2),
    )


@app.post("/api/search", response_model=SearchResponse, tags=["retrieval"])
@limiter.limit("30/minute")
async def search(
    request: Request,
    body: SearchRequest,
    _auth: Optional[dict] = Depends(require_auth),
):
    """Cross-modal retrieval: SAR query → Optical results (or vice versa).

    Results are cached for ``CACHE_TTL`` seconds (default 300s).
    """
    _check_model()

    cache = request.app.state.cache
    cache_key = make_cache_key(
        "search", modality=body.modality, top_k=body.top_k,
        img_prefix=body.image_b64[:64],
    )
    cached = cache.get(cache_key)
    if cached is not None:
        return SearchResponse(**cached)

    try:
        results, inference_ms = engine.search(
            image_b64=body.image_b64,
            modality=body.modality,
            top_k=body.top_k,
            supabase_client=request.app.state.supabase,
        )
    except Exception as exc:
        logger.error(f"/api/search error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    resp = SearchResponse(
        query_modality=body.modality,
        results=[SearchResult(**r) for r in results],
        inference_time_ms=round(inference_ms, 2),
    )
    cache.set(cache_key, resp.model_dump(), ttl=CACHE_TTL)
    return resp


@app.post("/api/search/batch", response_model=BatchSearchResponse, tags=["retrieval"])
@limiter.limit("10/minute")
async def search_batch(
    request: Request,
    body: BatchSearchRequest,
    _auth: Optional[dict] = Depends(require_auth),
):
    """Retrieve top-K cross-modal matches for multiple query images in one call.

    Accepts up to 20 images. Each is processed independently and results are
    returned in the same order as the input list.
    """
    _check_model()

    all_results: list[list[SearchResult]] = []
    per_image_times: list[float] = []
    t_total = time.perf_counter()

    for image_b64 in body.images:
        try:
            results, ms = engine.search(
                image_b64=image_b64,
                modality=body.modality,
                top_k=body.top_k,
                supabase_client=request.app.state.supabase,
            )
            all_results.append([SearchResult(**r) for r in results])
            per_image_times.append(round(ms, 2))
        except Exception as exc:
            logger.error(f"Batch search error for one image: {exc}")
            all_results.append([])
            per_image_times.append(0.0)

    total_ms = (time.perf_counter() - t_total) * 1000
    return BatchSearchResponse(
        query_modality=body.modality,
        results=all_results,
        total_inference_time_ms=round(total_ms, 2),
        per_image_time_ms=per_image_times,
    )


@app.post("/api/similar", response_model=SimilarResponse, tags=["retrieval"])
@limiter.limit("30/minute")
async def find_similar(
    request: Request,
    body: SimilarRequest,
    _auth: Optional[dict] = Depends(require_auth),
):
    """Find visually similar images within the *same* modality.

    Useful for browsing the dataset or finding near-duplicates.
    """
    _check_model()

    try:
        results, inference_ms = engine.search_similar(
            image_b64=body.image_b64,
            modality=body.modality,
            top_k=body.top_k,
            supabase_client=request.app.state.supabase,
        )
    except Exception as exc:
        logger.error(f"/api/similar error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    return SimilarResponse(
        query_modality=body.modality,
        results=[SearchResult(**r) for r in results],
        inference_time_ms=round(inference_ms, 2),
    )


@app.post("/api/embed", response_model=EmbedResponse, tags=["retrieval"])
@limiter.limit("30/minute")
async def embed(
    request: Request,
    body: EmbedRequest,
    _auth: Optional[dict] = Depends(require_auth),
):
    """Return the raw 256-dimensional embedding for a query image (debug/viz)."""
    _check_model()

    try:
        emb = engine.embed(body.image_b64, body.modality)
    except Exception as exc:
        logger.error(f"/api/embed error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    return EmbedResponse(
        modality=body.modality,
        embedding=emb.tolist(),
        dimension=len(emb),
    )
