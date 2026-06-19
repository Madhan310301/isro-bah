import math
import sys
from pathlib import Path

# Add project root directory to sys.path
root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import torch
import torchvision.models
from unittest.mock import patch

# Patch resnet50 so it doesn't download pretrained weights during the smoke test
original_resnet50 = torchvision.models.resnet50
def mock_resnet50(*args, **kwargs):
    if "weights" in kwargs:
        kwargs["weights"] = None
    return original_resnet50(*args, **kwargs)

with patch("torchvision.models.resnet50", mock_resnet50):
    from ml.model import DualEncoder
    from ml.loss import InfoNCELoss, HardNegativeInfoNCELoss

def run_step_1() -> dict:
    print("=== STEP 1: Model + Loss Smoke Test ===")
    
    # 2. Instantiate model
    print("Instantiating DualEncoder(backbone='resnet50', emb_dim=256)...")
    model = DualEncoder(backbone="resnet50", emb_dim=256)
    
    # 3 & 4. Create fake inputs
    print("Creating synthetic inputs (batch=8)...")
    sar_input = torch.randn(8, 1, 256, 256)
    opt_input = torch.randn(8, 3, 256, 256)
    
    # 5. Run forward pass
    sar_emb = model.encode_sar(sar_input)
    opt_emb = model.encode_optical(opt_input)
    
    print(f"SAR output shape: {sar_emb.shape}")
    print(f"Optical output shape: {opt_emb.shape}")
    assert sar_emb.shape == (8, 256), f"Expected (8, 256), got {sar_emb.shape}"
    assert opt_emb.shape == (8, 256), f"Expected (8, 256), got {opt_emb.shape}"
    
    # 6. Check L2-normalization
    sar_norms = sar_emb.norm(dim=1)
    opt_norms = opt_emb.norm(dim=1)
    print(f"SAR embedding row norms: {sar_norms.tolist()}")
    print(f"Optical embedding row norms: {opt_norms.tolist()}")
    
    for i, (sn, on) in enumerate(zip(sar_norms, opt_norms)):
        assert math.isclose(sn.item(), 1.0, abs_tol=1e-5), f"SAR row {i} norm is {sn.item()}, expected ~1.0"
        assert math.isclose(on.item(), 1.0, abs_tol=1e-5), f"Opt row {i} norm is {on.item()}, expected ~1.0"
    
    # 7. Compute InfoNCELoss
    loss_fn = InfoNCELoss()
    loss_infonce, acc_infonce = loss_fn(sar_emb, opt_emb)
    print(f"InfoNCELoss value: {loss_infonce.item():.4f} (Accuracy: {acc_infonce:.2f})")
    
    expected_random_loss = math.log(8)  # ln(8) ≈ 2.0794
    diff_infonce = abs(loss_infonce.item() - expected_random_loss)
    print(f"Difference from theoretical ln(batch_size) ({expected_random_loss:.4f}): {diff_infonce:.4f}")
    
    # 8. Compute HardNegativeInfoNCELoss
    hard_loss_fn = HardNegativeInfoNCELoss()
    loss_hard, acc_hard = hard_loss_fn(sar_emb, opt_emb)
    print(f"HardNegativeInfoNCELoss value: {loss_hard.item():.4f} (Accuracy: {acc_hard:.2f})")
    
    # 9. Backward pass and gradient flow check
    print("Testing backward pass gradient flow...")
    loss_infonce.backward()
    
    grad_params = 0
    no_grad_params = 0
    unexpected_zero_grads = []
    
    for name, param in model.named_parameters():
        # Momentum parameters are not updated via gradients, so they won't have grads
        if "momentum" in name:
            if param.grad is not None:
                print(f"Warning: Momentum param {name} has gradient.")
            continue
            
        if param.requires_grad:
            if param.grad is not None:
                grad_params += 1
                if param.grad.abs().sum().item() == 0.0:
                    unexpected_zero_grads.append(name)
            else:
                no_grad_params += 1
    
    print(f"Parameters with gradients: {grad_params}")
    print(f"Parameters without gradients (but requires_grad=True): {no_grad_params}")
    if unexpected_zero_grads:
        print(f"WARNING: Parameters with absolute zero gradients: {unexpected_zero_grads}")
    else:
        print("Gradients verified successfully for all active layers.")
        
    return {
        "sar_shape": list(sar_emb.shape),
        "opt_shape": list(opt_emb.shape),
        "sar_norms": sar_norms.tolist(),
        "opt_norms": opt_norms.tolist(),
        "infonce_loss": loss_infonce.item(),
        "hard_loss": loss_hard.item(),
        "grad_params": grad_params,
        "no_grad_params": no_grad_params
    }

if __name__ == "__main__":
    run_step_1()
