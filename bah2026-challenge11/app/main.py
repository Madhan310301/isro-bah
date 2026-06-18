"""FastAPI application — Cross-Modal Satellite Image Retrieval backend."""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.inference import engine
from app.models import (
    EmbedRequest,
    EmbedResponse,
    ErrorResponse,
    HealthResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from app.supabase_client import SupabaseImageClient

load_dotenv()

# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model, FAISS indices, and Supabase client on startup."""
    checkpoint = os.getenv("MODEL_CHECKPOINT_PATH", "./checkpoints/best_model.pt")
    faiss_dir = os.getenv("FAISS_INDEX_PATH", "./faiss_index")
    backbone = os.getenv("MODEL_BACKBONE", "resnet50")

    try:
        engine.load(
            checkpoint_path=checkpoint,
            sar_index_path=f"{faiss_dir}/sar_index.faiss",
            optical_index_path=f"{faiss_dir}/optical_index.faiss",
            metadata_path=f"{faiss_dir}/metadata.json",
            backbone=backbone,
        )
    except Exception as exc:
        logger.error(f"Failed to load model/index at startup: {exc}")
        logger.warning("Continuing in degraded mode — /api/search will error.")

    app.state.supabase = SupabaseImageClient()
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
    version="1.0.0",
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
# Request timing middleware
# ---------------------------------------------------------------------------


@app.middleware("http")
async def add_request_timing(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Request-Time-Ms"] = f"{elapsed_ms:.2f}"
    logger.debug(f"{request.method} {request.url.path} → {response.status_code} ({elapsed_ms:.1f}ms)")
    return response


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_server_error",
            detail=str(exc),
        ).model_dump(),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/health", response_model=HealthResponse, tags=["system"])
async def health():
    """Return service health and index statistics."""
    return HealthResponse(
        status="ok",
        model=engine.model_name,
        index_size=engine.optical_index_size,
        device=engine.device,
    )


@app.post("/api/search", response_model=SearchResponse, tags=["retrieval"])
@limiter.limit("30/minute")
async def search(request: Request, body: SearchRequest):
    """Retrieve the top-K cross-modal matches for a query image.

    - SAR query  → returns matching Optical images.
    - Optical query → returns matching SAR images.
    """
    if engine.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

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

    return SearchResponse(
        query_modality=body.modality,
        results=[SearchResult(**r) for r in results],
        inference_time_ms=round(inference_ms, 2),
    )


@app.post("/api/embed", response_model=EmbedResponse, tags=["retrieval"])
@limiter.limit("30/minute")
async def embed(request: Request, body: EmbedRequest):
    """Return the raw 512-dimensional embedding for a query image (debug/viz)."""
    if engine.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

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
