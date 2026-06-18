"""Pydantic request/response schemas for the FastAPI backend."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    """Request body for POST /api/search."""

    image_b64: str = Field(..., description="Base64-encoded image bytes (PNG or TIFF).")
    modality: Literal["sar", "optical"] = Field(
        ..., description="Modality of the query image."
    )
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to return.")

    @field_validator("image_b64")
    @classmethod
    def validate_b64_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("image_b64 must not be empty.")
        return v


class EmbedRequest(BaseModel):
    """Request body for POST /api/embed."""

    image_b64: str = Field(..., description="Base64-encoded image bytes.")
    modality: Literal["sar", "optical"] = Field(
        ..., description="Modality of the image."
    )


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class SearchResult(BaseModel):
    """A single retrieval result."""

    rank: int
    similarity: float
    pair_id: str
    supabase_url: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    season: Optional[str] = None


class SearchResponse(BaseModel):
    """Response body for POST /api/search."""

    query_modality: str
    results: list[SearchResult]
    inference_time_ms: float


class EmbedResponse(BaseModel):
    """Response body for POST /api/embed."""

    modality: str
    embedding: list[float]
    dimension: int


class HealthResponse(BaseModel):
    """Response body for GET /api/health."""

    status: str
    model: str
    index_size: int
    device: str


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    error: str
    detail: Optional[str] = None
