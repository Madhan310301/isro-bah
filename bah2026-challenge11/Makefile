.PHONY: help install train index serve test eval docker-build docker-up docker-down clean lint format typecheck

PYTHON       := python
DATA_DIR     := ./data/SEN12
CHECKPOINT   := ./checkpoints/best_model.pt
FAISS_DIR    := ./faiss_index
REPORT       := ./metrics.json
BACKBONE     := vit

help:
	@echo ""
	@echo "  BAH2026 Challenge 11 — Cross-Modal Satellite Image Retrieval"
	@echo "  ─────────────────────────────────────────────────────────────"
	@echo "  make install      Install Python dependencies"
	@echo "  make train        Train the DualEncoder model"
	@echo "  make train-resume Resume training from last checkpoint"
	@echo "  make index        Build FAISS index from best checkpoint"
	@echo "  make serve        Start FastAPI server (uvicorn)"
	@echo "  make test         Run pytest unit tests"
	@echo "  make eval         Full evaluation (Recall@K, mAP, PDF report)"
	@echo "  make eval-quick   Quick evaluation on 500 random pairs"
	@echo "  make fetch-india  Download India GEE patches (requires GEE auth)"
	@echo "  make docker-build Build Docker image"
	@echo "  make docker-up    Start all services (api + redis + nginx)"
	@echo "  make docker-down  Stop all services"
	@echo "  make lint         Run ruff linter"
	@echo "  make format       Run black + isort"
	@echo "  make typecheck    Run mypy"
	@echo "  make clean        Remove checkpoints, index, and cache files"
	@echo ""

install:
	pip install -r requirements.txt

# ── Training ──────────────────────────────────────────────────────────────

train:
	$(PYTHON) -m ml.train \
		--data_dir $(DATA_DIR) \
		--checkpoint_dir ./checkpoints \
		--backbone $(BACKBONE) \
		--epochs 30 \
		--batch_size 64 \
		--mixup_alpha 0.2 \
		--patience 5

train-resume:
	@LATEST=$$(ls -t ./checkpoints/epoch_*.pt 2>/dev/null | head -1); \
	if [ -z "$$LATEST" ]; then echo "No checkpoint found — run 'make train' first."; exit 1; fi; \
	$(PYTHON) -m ml.train \
		--data_dir $(DATA_DIR) \
		--checkpoint_dir ./checkpoints \
		--backbone $(BACKBONE) \
		--epochs 30 \
		--resume $$LATEST

# ── FAISS Index ───────────────────────────────────────────────────────────

index:
	$(PYTHON) -m scripts.build_index \
		--checkpoint $(CHECKPOINT) \
		--data_dir $(DATA_DIR) \
		--output_dir $(FAISS_DIR) \
		--backbone $(BACKBONE)

index-demo:
	$(PYTHON) -m scripts.build_index \
		--checkpoint $(CHECKPOINT) \
		--data_dir $(DATA_DIR) \
		--output_dir $(FAISS_DIR) \
		--backbone $(BACKBONE) \
		--max_pairs 10000

# ── Serving ───────────────────────────────────────────────────────────────

serve:
	MODEL_CHECKPOINT_PATH=$(CHECKPOINT) \
	FAISS_INDEX_PATH=$(FAISS_DIR) \
	MODEL_BACKBONE=$(BACKBONE) \
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# ── Testing ───────────────────────────────────────────────────────────────

test:
	pytest tests/ -v --tb=short

test-coverage:
	pytest tests/ -v --tb=short --cov=app --cov=ml --cov-report=term-missing

# ── Evaluation ────────────────────────────────────────────────────────────

eval:
	$(PYTHON) -m scripts.evaluate \
		--checkpoint $(CHECKPOINT) \
		--data_dir $(DATA_DIR) \
		--output $(REPORT) \
		--report ./metrics_report.pdf \
		--backbone $(BACKBONE)

eval-quick:
	$(PYTHON) -m scripts.evaluate \
		--checkpoint $(CHECKPOINT) \
		--data_dir $(DATA_DIR) \
		--output ./metrics_quick.json \
		--backbone $(BACKBONE) \
		--quick

# ── India GEE data ────────────────────────────────────────────────────────

fetch-india:
	$(PYTHON) -m scripts.fetch_india_gee \
		--output_dir ./data/india \
		--year 2023

fetch-india-stub:
	$(PYTHON) -m scripts.fetch_india_gee --stub

# ── Docker ────────────────────────────────────────────────────────────────

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f api

# ── Code quality ──────────────────────────────────────────────────────────

lint:
	ruff check app/ ml/ scripts/ tests/

format:
	black app/ ml/ scripts/ tests/
	isort app/ ml/ scripts/ tests/

typecheck:
	mypy app/ ml/ scripts/

# ── Cleanup ───────────────────────────────────────────────────────────────

clean:
	rm -rf ./checkpoints/epoch_*.pt
	rm -rf $(FAISS_DIR)/*.faiss
	rm -rf .pytest_cache __pycache__ .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
