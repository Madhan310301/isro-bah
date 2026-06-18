"""Unit tests for the FastAPI endpoints."""

from __future__ import annotations

import base64
import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_b64_image(mode: str = "RGB", size: tuple = (224, 224)) -> str:
    img = Image.new(mode, size, color=(100, 150, 200) if mode == "RGB" else 128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Fixture: mock engine so tests run without a real checkpoint/FAISS index
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_engine():
    """Patch the inference engine singleton with a minimal mock."""
    dummy_emb = np.random.rand(512).astype("float32")
    dummy_emb /= np.linalg.norm(dummy_emb)

    mock = MagicMock()
    mock.model = MagicMock()  # truthy → endpoints won't return 503
    mock.model_name = "DualEncoder-v1"
    mock.device = "cpu"
    mock.optical_index_size = 10000
    mock.embed.return_value = dummy_emb
    mock.search.return_value = (
        [
            {
                "rank": 1,
                "similarity": 0.94,
                "pair_id": "ROIs1158_spring_s1_1",
                "supabase_url": "https://example.com/img.tif",
                "lat": 28.6,
                "lon": 77.2,
                "season": "spring",
            }
        ],
        42.0,  # inference_time_ms
    )

    with patch("app.main.engine", mock), patch("app.inference.engine", mock):
        yield mock


@pytest.fixture
def client(mock_engine):
    from app.main import app

    with patch.object(app.state, "supabase", MagicMock()):
        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_status_ok(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200

    def test_response_schema(self, client):
        data = client.get("/api/health").json()
        assert "status" in data
        assert "model" in data
        assert "index_size" in data
        assert "device" in data

    def test_status_value(self, client):
        data = client.get("/api/health").json()
        assert data["status"] == "ok"

    def test_model_name(self, client):
        data = client.get("/api/health").json()
        assert data["model"] == "DualEncoder-v1"

    def test_index_size_is_int(self, client):
        data = client.get("/api/health").json()
        assert isinstance(data["index_size"], int)


class TestSearchEndpoint:
    def test_sar_query_returns_200(self, client):
        payload = {
            "image_b64": _make_b64_image("L"),
            "modality": "sar",
            "top_k": 1,
        }
        resp = client.post("/api/search", json=payload)
        assert resp.status_code == 200

    def test_optical_query_returns_200(self, client):
        payload = {
            "image_b64": _make_b64_image("RGB"),
            "modality": "optical",
            "top_k": 5,
        }
        resp = client.post("/api/search", json=payload)
        assert resp.status_code == 200

    def test_response_fields(self, client):
        payload = {
            "image_b64": _make_b64_image("RGB"),
            "modality": "optical",
            "top_k": 1,
        }
        data = client.post("/api/search", json=payload).json()
        assert "query_modality" in data
        assert "results" in data
        assert "inference_time_ms" in data
        assert isinstance(data["results"], list)

    def test_result_fields(self, client):
        payload = {
            "image_b64": _make_b64_image("RGB"),
            "modality": "optical",
            "top_k": 1,
        }
        result = client.post("/api/search", json=payload).json()["results"][0]
        for field in ("rank", "similarity", "pair_id"):
            assert field in result, f"Missing field: {field}"

    def test_invalid_modality(self, client):
        payload = {
            "image_b64": _make_b64_image("RGB"),
            "modality": "lidar",
            "top_k": 5,
        }
        resp = client.post("/api/search", json=payload)
        assert resp.status_code == 422

    def test_missing_image_b64(self, client):
        payload = {"modality": "optical", "top_k": 5}
        resp = client.post("/api/search", json=payload)
        assert resp.status_code == 422


class TestEmbedEndpoint:
    def test_embed_returns_200(self, client):
        payload = {
            "image_b64": _make_b64_image("RGB"),
            "modality": "optical",
        }
        resp = client.post("/api/embed", json=payload)
        assert resp.status_code == 200

    def test_embedding_dimension(self, client):
        payload = {
            "image_b64": _make_b64_image("RGB"),
            "modality": "optical",
        }
        data = client.post("/api/embed", json=payload).json()
        assert data["dimension"] == 512
        assert len(data["embedding"]) == 512

    def test_embed_sar(self, client):
        payload = {
            "image_b64": _make_b64_image("L"),
            "modality": "sar",
        }
        resp = client.post("/api/embed", json=payload)
        assert resp.status_code == 200
