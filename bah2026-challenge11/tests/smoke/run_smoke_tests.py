"""End-to-end Smoke Tests for bah2026-challenge11"""

import os
import sys
import time
import shutil
import base64
import subprocess
from pathlib import Path
from unittest.mock import patch

import torch
import numpy as np
from PIL import Image

# Add project root directory to sys.path
root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Mock resnet50 globally in this script for fast training/indexing
import torchvision.models
original_resnet50 = torchvision.models.resnet50
def mock_resnet50(*args, **kwargs):
    if "weights" in kwargs:
        kwargs["weights"] = None
    return original_resnet50(*args, **kwargs)
torchvision.models.resnet50 = mock_resnet50

from ml.dataset import SAROpticalPairDataset
from torch.utils.data import DataLoader

def create_synthetic_data(data_dir: Path, num_pairs: int = 20):
    """Generate mock SAR/Optical image dataset on disk"""
    s1_dir = data_dir / "s1"
    s2_dir = data_dir / "s2"
    s1_dir.mkdir(parents=True, exist_ok=True)
    s2_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_pairs} synthetic image pairs in {data_dir}...")
    for i in range(num_pairs):
        pair_id = f"ROI_smoke_{i}"
        
        # Grayscale SAR image (single-channel)
        sar_arr = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        Image.fromarray(sar_arr).save(s1_dir / f"{pair_id}_s1.tif")
        
        # RGB Optical image (3-channel)
        opt_arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        Image.fromarray(opt_arr).save(s2_dir / f"{pair_id}_s2.tif")
    
    print("Synthetic dataset generated successfully.")

def run_step_2(data_dir: Path) -> dict:
    print("\n=== STEP 2: Dataset Class Smoke Test ===")
    
    # 2. Instantiate SAROpticalPairDataset
    print("Instantiating SAROpticalPairDataset split='train'...")
    train_dataset = SAROpticalPairDataset(data_dir, split="train")
    print("Instantiating SAROpticalPairDataset split='val'...")
    val_dataset = SAROpticalPairDataset(data_dir, split="val")
    print("Instantiating SAROpticalPairDataset split='test'...")
    test_dataset = SAROpticalPairDataset(data_dir, split="test")
    
    # 3. Confirm dataset lengths (deterministic 80/10/10 split on 20 pairs)
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    assert len(train_dataset) == 16, f"Expected train size 16, got {len(train_dataset)}"
    assert len(val_dataset) == 2, f"Expected val size 2, got {len(val_dataset)}"
    assert len(test_dataset) == 2, f"Expected test size 2, got {len(test_dataset)}"
    
    # 4. Pull one sample
    sample = train_dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    assert "sar" in sample and "optical" in sample and "pair_id" in sample
    
    sar_shape = list(sample["sar"].shape)
    opt_shape = list(sample["optical"].shape)
    print(f"SAR tensor shape: {sar_shape}, dtype: {sample['sar'].dtype}")
    print(f"Optical tensor shape: {opt_shape}, dtype: {sample['optical'].dtype}")
    assert sar_shape == [1, 224, 224], f"Expected SAR shape [1, 224, 224], got {sar_shape}"
    assert opt_shape == [3, 224, 224], f"Expected Optical shape [3, 224, 224], got {opt_shape}"
    
    # 5. Wrap in DataLoader
    loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    batch = next(iter(loader))
    batched_sar_shape = list(batch["sar"].shape)
    batched_opt_shape = list(batch["optical"].shape)
    print(f"Batched SAR shape: {batched_sar_shape}")
    print(f"Batched Optical shape: {batched_opt_shape}")
    assert batched_sar_shape == [2, 1, 224, 224], f"Expected batched SAR shape [2, 1, 224, 224], got {batched_sar_shape}"
    assert batched_opt_shape == [2, 3, 224, 224], f"Expected batched Optical shape [2, 3, 224, 224], got {batched_opt_shape}"
    
    return {
        "train_len": len(train_dataset),
        "sar_shape": sar_shape,
        "opt_shape": opt_shape,
        "batched_sar_shape": batched_sar_shape
    }

def run_step_3(data_dir: Path, checkpoint_dir: Path) -> dict:
    print("\n=== STEP 3: Full Training Loop Smoke Test ===")
    
    # Execute training script via subprocess for isolated run
    cmd = [
        "py", "ml/train.py",
        "--data_dir", str(data_dir),
        "--checkpoint_dir", str(checkpoint_dir),
        "--epochs", "1",
        "--batch_size", "4",
        "--emb_dim", "256",
        "--backbone", "resnet50"
    ]
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root_dir)
    
    print(f"Executing: {' '.join(cmd)}")
    res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Print output logs
    print("--- Training Log Output ---")
    print(res.stdout)
    if res.stderr:
        print("--- Training Error Output ---")
        print(res.stderr)
        
    assert res.returncode == 0, f"Training crashed with code {res.returncode}"
    
    # Verify checkpoint
    ckpt_path = checkpoint_dir / "best_model.pt"
    assert ckpt_path.exists(), "Checkpoint file was not written to disk!"
    
    size_kb = ckpt_path.stat().st_size / 1024
    mtime = time.ctime(ckpt_path.stat().st_mtime)
    print(f"[OK] Checkpoint written: {ckpt_path}")
    print(f"  Size: {size_kb:.2f} KB")
    print(f"  Modified time: {mtime}")
    
    return {
        "checkpoint_path": str(ckpt_path),
        "size_kb": size_kb,
        "mtime": mtime
    }

def run_step_4(data_dir: Path, checkpoint_dir: Path) -> dict:
    print("\n=== STEP 4: Evaluation Script Smoke Test ===")
    
    ckpt_path = checkpoint_dir / "best_model.pt"
    eval_out = checkpoint_dir / "eval_output.json"
    
    cmd = [
        "py", "-m", "ml.evaluate",
        "--checkpoint", str(ckpt_path),
        "--data_dir", str(data_dir),
        "--output", str(eval_out),
        "--backbone", "resnet50",
        "--quick"
    ]
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root_dir)
    
    print(f"Executing: {' '.join(cmd)}")
    res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    print("--- Evaluation Log Output ---")
    print(res.stdout)
    if res.stderr:
        print("--- Evaluation Error Output ---")
        print(res.stderr)
        
    assert res.returncode == 0, f"Evaluation crashed with code {res.returncode}"
    assert eval_out.exists(), "Evaluation JSON report not written!"
    
    import json
    with open(eval_out) as f:
        metrics = json.load(f)
    print(f"Computed metrics: {metrics}")
    
    return metrics

def run_step_5(data_dir: Path, checkpoint_dir: Path, faiss_dir: Path) -> dict:
    print("\n=== STEP 5: FAISS Index Build Smoke Test ===")
    
    ckpt_path = checkpoint_dir / "best_model.pt"
    
    cmd = [
        "py", "scripts/build_index.py",
        "--checkpoint", str(ckpt_path),
        "--data_dir", str(data_dir),
        "--output_dir", str(faiss_dir),
        "--backbone", "resnet50",
        "--split", "test"
    ]
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root_dir)
    
    print(f"Executing: {' '.join(cmd)}")
    res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    print("--- Index Build Log Output ---")
    print(res.stdout)
    if res.stderr:
        print("--- Index Build Error Output ---")
        print(res.stderr)
        
    assert res.returncode == 0, f"Index build crashed with code {res.returncode}"
    
    # Confirm files exist
    sar_idx = faiss_dir / "sar_index.faiss"
    opt_idx = faiss_dir / "optical_index.faiss"
    meta_json = faiss_dir / "metadata.json"
    
    assert sar_idx.exists(), "SAR FAISS index not written!"
    assert opt_idx.exists(), "Optical FAISS index not written!"
    assert meta_json.exists(), "Metadata JSON not written!"
    
    print("[OK] FAISS indices and metadata built successfully.")
    return {
        "sar_index": str(sar_idx),
        "optical_index": str(opt_idx),
        "metadata": str(meta_json)
    }

def run_step_6(checkpoint_dir: Path, faiss_dir: Path, data_dir: Path) -> dict:
    print("\n=== STEP 6: API Smoke Test ===")
    
    import httpx
    
    ckpt_path = checkpoint_dir / "best_model.pt"
    
    # Prepare environment variables for the Uvicorn subprocess
    env = os.environ.copy()
    env["MODEL_CHECKPOINT_PATH"] = str(ckpt_path)
    env["FAISS_INDEX_PATH"] = str(faiss_dir)
    env["MODEL_BACKBONE"] = "resnet50"
    
    cmd = [
        "py", "-m", "uvicorn", "app.main:app",
        "--host", "127.0.0.1",
        "--port", "8055"  # Use a different port to avoid conflicts
    ]
    
    print(f"Starting server: {' '.join(cmd)}")
    server_proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    health_response = None
    search_response = None
    
    try:
        # Give server time to startup (poll /api/health)
        url = "http://127.0.0.1:8055/api/health"
        time.sleep(2)  # Initial wait
        
        for attempt in range(10):
            try:
                print(f"Polling {url} (attempt {attempt+1}/10)...")
                r = httpx.get(url, timeout=2.0)
                if r.status_code == 200:
                    health_response = r.json()
                    print(f"[OK] Health endpoint responded: {health_response}")
                    break
            except Exception:
                pass
            time.sleep(1)
            
        if health_response is None:
            raise RuntimeError("API Server failed to start within timeout.")
            
        # Send a search query (Step 6b)
        search_url = "http://127.0.0.1:8055/api/search"
        
        # Load a synthetic SAR image and convert to Base64
        sar_img_path = data_dir / "s1" / "ROI_smoke_0_s1.tif"
        with open(sar_img_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")
            
        payload = {
            "image_b64": b64_data,
            "modality": "sar",
            "top_k": 2
        }
        
        print(f"Sending search request to {search_url}...")
        r = httpx.post(search_url, json=payload, timeout=5.0)
        assert r.status_code == 200, f"Search endpoint returned {r.status_code}: {r.text}"
        
        search_response = r.json()
        print(f"[OK] Search endpoint responded successfully. Results count: {len(search_response.get('results', []))}")
        print(f"Search response sample: {search_response}")
        
    finally:
        print("Terminating API server...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
            print("API server terminated successfully.")
        except subprocess.TimeoutExpired:
            print("API server timed out during shutdown. Force killing...")
            server_proc.kill()
            server_proc.wait()
            
    return {
        "health": health_response,
        "search": search_response
    }

def main():
    # Set up temp paths
    temp_dir = Path("a:/ISRO - Hackathon/SatBridge/bah2026-challenge11/tests/smoke/temp")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = temp_dir / "data"
    checkpoint_dir = temp_dir / "checkpoints"
    faiss_dir = temp_dir / "faiss_index"
    
    results = {}
    
    try:
        # Create dataset
        create_synthetic_data(data_dir, num_pairs=20)
        
        # Run Step 2
        results["Step 2"] = run_step_2(data_dir)
        
        # Run Step 3
        results["Step 3"] = run_step_3(data_dir, checkpoint_dir)
        
        # Run Step 4
        results["Step 4"] = run_step_4(data_dir, checkpoint_dir)
        
        # Run Step 5
        results["Step 5"] = run_step_5(data_dir, checkpoint_dir, faiss_dir)
        
        # Run Step 6
        results["Step 6"] = run_step_6(checkpoint_dir, faiss_dir, data_dir)
        
        print("\n==========================================")
        print("ALL SMOKE TESTS COMPLETED SUCCESSFULLY!")
        print("==========================================")
        
    finally:
        # Cleanup temp directory
        print(f"Cleaning up temporary test files in {temp_dir}...")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        print("Cleanup completed.")

if __name__ == "__main__":
    main()
