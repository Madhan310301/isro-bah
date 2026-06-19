# Cross-Modal Satellite Image Retrieval (BAH2026 Challenge 11)

This subproject provides the Python machine learning training, indexing, and FastAPI inference server for cross-modal SAR-to-Optical satellite image retrieval.

## Stack
- **Python**: 3.11+
- **Deep Learning**: PyTorch, torchvision, timm
- **Vector Search**: FAISS
- **API Server**: FastAPI, Uvicorn, SlowAPI
- **Storage & Metadata**: Supabase (via `supabase` SDK)
- **Utilities**: Loguru, pytest, python-dotenv, Weights & Biases (optional)

## Setup

1. **Activate Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   Create a `.env` file based on `.env.example` and set:
   - `SUPABASE_URL`
   - `SUPABASE_KEY`

## How to Run

### 1. Training
To train the dual-encoder model using the SEN1-2 dataset:
```bash
python ml/train.py --data_dir ./data/SEN12 --epochs 30
```

### 2. Building the Vector Index
To encode the dataset images and build the FAISS index:
```bash
python scripts/build_index.py --checkpoint ./checkpoints/best_model.pt --data_dir ./data/SEN12 --output_dir ./faiss_index
```

### 3. API Server
To start the inference API server on port 8000:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 4. Running Tests
To run the test suite:
```bash
pytest tests/ -v
```
