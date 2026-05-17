import sys
sys.path.append('./backend')
import numpy as np
import pandas as pd
from vector_encoder import RLRecommender
from _sim_worker import evaluate_portfolio_member_c
import logging

logging.basicConfig(level=logging.INFO)

# 1. Warm up recommender and get real returns
r = RLRecommender()
print(f"Ensemble loaded with {len(r.models)} models.")

# 2. Construct mock user profile and portfolio weights
user_profile = {
    "drawdown_sensitivity": 5.0,
    "volatility_sensitivity": 5.0,
    "goal_flexibility": 5.0,
    "concentration_pref": 5.0,
    "start_cap": 100000.0,
    "monthly_contrib": 500.0,  # $500 monthly = $6000 annually
    "goals": {
        5: 50000.0,    # Y5: $50k goal
        10: 250000.0   # Y10: $250k goal (expected to cause shortfall/bankruptcy, letting us test continuation!)
    }
}

recommendations = {
    "portfolio_weights": {
        "VOO": 0.6,
        "QQQ": 0.4
    }
}

config = {
    "sim_paths_per_episode": 2,
    "sim_horizon_mode": "loop",
    "sim_cvar_percentile": 0.95
}

# 3. Build a mini dataset dictionary
dataset = {
    "daily_returns": r.daily_returns,
    "drip_daily_returns": r.drip_daily_returns,
    "price_matrix": r.price_matrix
}

# 4. Evaluate in chronological mode starting from a real date (e.g. index 0)
print("\n--- Running Chronological Simulation with Goal Shortfalls ---")
metrics = evaluate_portfolio_member_c(
    dataset, recommendations, user_profile, config,
    debug_path=True, start_year_idx=0
)

print("\nSimulation Metrics Result:")
for k, v in metrics.items():
    if k != "sample_path_log":
        print(f"  {k}: {v}")

print("\nSample Path Log:")
print(metrics.get("sample_path_log"))
