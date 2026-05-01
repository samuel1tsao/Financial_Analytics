import os
import sys
import torch
import json
import numpy as np

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from _rl_worker import PortfolioTransformerRL, _build_rl_input_fast, _softmax_normalize_top_k
    from _constants import DEFAULT_PIPELINE_CONFIG, TOP_K_ASSETS, MIN_ACTIVE_WEIGHT
    print("Imports successful")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Mock config
config = DEFAULT_PIPELINE_CONFIG.copy()
input_dim = 64 + 4 + 75 # Example dimensions (emb + user + static)
checkpoint_path = "../cache/checkpoint_rl_v2_dm64_nh4_lr001_id226.pt"

if not os.path.exists(checkpoint_path):
    # Try different relative path if needed
    checkpoint_path = "cache/checkpoint_rl_v2_dm64_nh4_lr001_id226.pt"

if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint: {checkpoint_path}")
    # We need to know the actual input_dim from the checkpoint filename or by trial/error
    # The filename says id226.
    input_dim = 226
    
    device = torch.device('cpu')
    model = PortfolioTransformerRL(input_dim, config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'agents_state' in checkpoint:
        model.load_state_dict(checkpoint['agents_state'][0])
    else:
        model.load_state_dict(checkpoint['model_state'])
    
    model.eval()
    print("Model loaded successfully")
    
    # Mock input
    # (batch, num_assets, input_dim)
    num_assets = 100
    x = torch.randn(1, num_assets, input_dim).to(device)
    
    with torch.no_grad():
        (mu_pre, _), (mu_post, _) = model(x)
        weights_pre = _softmax_normalize_top_k(mu_pre, None)
        print(f"Inference successful. Weight sum: {weights_pre.sum().item():.4f}")
else:
    print(f"Checkpoint not found at {checkpoint_path}")
