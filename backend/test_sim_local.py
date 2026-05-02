import sys
sys.path.append('.')
from vector_encoder import simulate_multi_horizon_portfolio
import traceback

segments = [
    {
        "horizon_years": (0, 30),
        "weights": {"VOO": 0.5, "BND": 0.5}
    }
]
goals = [{"name": "Retirement", "amount": 1000000, "years": 30}]

try:
    res = simulate_multi_horizon_portfolio(segments, goals)
    print("SUCCESS")
except Exception as e:
    print("FAILED")
    traceback.print_exc()
