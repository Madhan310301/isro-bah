"""Pydantic v2 request/response schemas for the FastAPI backend."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    """Request body for POST /api/search.

    Attributes:
        image_b64: Base64-encoded image bytes (PNG or TIFF).
        modality:  Query modality — ``"sar"`` or ``"optical"``.
        top_k:     Number of results to return (1–100, default 10).
    """

    image_b64: str = Field(..., description="Base64-encoded image bytes.")
    modality: Literal["sar", "optical"] = Field(..., description="Query modality.")
    top_k: int = Field(default=10, ge=1, le=100)

    @field_validator("image_b64")
    @classmethod
    def validate_b64_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("image_b64 must not be empty.")
        return v


class BatchSearchRequest(BaseModel):
    """Request body for POST /api/search/batch.

    Attributes:
        images:   List of base64-encoded images (max 20).
        modality: Query modality for all images.
        top_k:    Number of results per image.
    """

    images: List[str] = Field(..., min_length=1, max_length=20,
                              description="List of base64-encoded images.")
    modality: Literal["sar", "optical"] = Field(...)
    top_k: int = Field(default=10, ge=1, le=50)


class SimilarRequest(BaseModel):
    """Request body for POST /api/similar.

    Attributes:
        image_b64: Base64-encoded query image.
        modality:  Modality — search within this same modality's index.
        top_k:     Number of results.
    """

    image_b64: str = Field(..., description="Base64-encoded image bytes.")
    modality: Literal["sar", "optical"] = Field(...)
    top_k: int = Field(default=10, ge=1, le=100)


class EmbedRequest(BaseModel):
    """Request body for POST /api/embed.

    Attributes:
        image_b64: Base64-encoded image bytes.
        modality:  Modality of the image.
    """

    image_b64: str = Field(...)
    modality: Literal["sar", "optical"] = Field(...)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class BoundingBox(BaseModel):
    """Geographic bounding box in WGS-84 decimal degrees.

    Attributes:
        min_lon: West boundary.
        min_lat: South boundary.
        max_lon: East boundary.
        max_lat: North boundary.
    """

    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float


class SearchResult(BaseModel):
    """A single retrieval result.

    Attributes:
        rank:         1-indexed rank in the result list.
        similarity:   Cosine similarity score (higher = more similar).
        pair_id:      Dataset pair identifier string.
        supabase_url: Public URL to the result image in Supabase Storage.
        lat:          Centre-point latitude (WGS-84).
        lon:          Centre-point longitude (WGS-84).
        season:       Acquisition season label.
        bbox:         Geographic bounding box if available.
    """

    rank: int
    similarity: float
    pair_id: str
    supabase_url: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    season: Optional[str] = None
    bbox: Optional[BoundingBox] = None


class SearchResponse(BaseModel):
    """Response body for POST /api/search."""

    query_modality: str
    results: List[SearchResult]
    inference_time_ms: float


class BatchSearchResponse(BaseModel):
    """Response body for POST /api/search/batch."""

    query_modality: str
    results: List[List[SearchResult]]
    total_inference_time_ms: float
    per_image_time_ms: List[float]


class SimilarResponse(BaseModel):
    """Response body for POST /api/similar."""

    query_modality: str
    results: List[SearchResult]
    inference_time_ms: float


class EmbedResponse(BaseModel):
    """Response body for POST /api/embed."""

    modality: str
    embedding: List[float]
    dimension: int


class HealthResponse(BaseModel):
    """Response body for GET /api/health."""

    status: str
    model: str
    index_size: int
    device: str


class IndexStatsResponse(BaseModel):
    """Response body for GET /api/index/stats."""

    sar_index_size: int
    optical_index_size: int
    index_type: str
    index_version: str
    emb_dim: int
    memory_usage_mb: float


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    error: str
    detail: Optional[str] = None
