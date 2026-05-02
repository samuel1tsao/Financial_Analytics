"""
Vector Encoder: Transforms structured questionnaire JSON into actionable portfolio parameters.
Enhanced with RL-Transformer multi-horizon recommendation engine.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import logging
import json
import os
import sys
from typing import Any, Dict, List

# Add root to sys.path to access root modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from _constants import (
    DEFAULT_PIPELINE_CONFIG, TOP_K_ASSETS, MIN_ACTIVE_WEIGHT, SIM_TERMINAL_HORIZON,
    RISK_NORMALIZER, CAPITAL_NORMALIZER, GOAL_YEAR_NORMALIZER,
    TRADING_DAYS_PER_YEAR, CASH_BUCKET_ANNUAL_RATE, DAILY_RETURN_CLAMP, LOAN_DAILY_RATE,
    CACHE_DIR
)
from _rl_worker import PortfolioTransformerRL, _softmax_normalize_top_k, _encode_user_condition
from _data_worker import generate_dataset_member_a
from _ml_worker import load_embedding_cache
from _sim_worker import build_simulation_cache, _resolve_simulation_starts

logger = logging.getLogger(__name__)

# Singleton instance for the RL Recommender to avoid reloading data on every request
_RECOMMENDER_INSTANCE = None

# ─── Asset Universe (Legacy / Fallback) ──────────────────────────────────────
EQUITY_UNIVERSE = {
    "VOO":  {"name": "S&P 500",           "risk": 0.5, "category": "large_cap"},
    "QQQ":  {"name": "Nasdaq 100",         "risk": 0.7, "category": "tech"},
    "VTI":  {"name": "Total Market",       "risk": 0.5, "category": "broad"},
    "VXUS": {"name": "International",      "risk": 0.6, "category": "intl"},
    "VGT":  {"name": "Info Tech",          "risk": 0.8, "category": "sector_tech"},
    "ARKK": {"name": "Innovation ETF",     "risk": 0.95, "category": "speculative"},
    "VNQ":  {"name": "Real Estate",        "risk": 0.6, "category": "reit"},
    "VWO":  {"name": "Emerging Markets",   "risk": 0.75, "category": "emerging"},
}

BOND_UNIVERSE = {
    "BND":  {"name": "Total Bond Market",  "risk": 0.1, "category": "bond"},
    "SGOV": {"name": "Short Treasury",     "risk": 0.02, "category": "stbond"},
    "TLT":  {"name": "20+ Year Treasury",  "risk": 0.25, "category": "lt_bond"},
    "TIPS": {"name": "Inflation Protected", "risk": 0.1, "category": "tips"},
}

# ─── Fallback Constants (used when DB has no market data) ────────────────────
FALLBACK_EXPECTED_RETURNS = {
    "VOO": 0.10, "QQQ": 0.12, "VTI": 0.10, "VXUS": 0.07,
    "VGT": 0.13, "ARKK": 0.15, "VNQ": 0.08, "VWO": 0.09,
    "BND": 0.04, "SGOV": 0.045, "TLT": 0.035, "TIPS": 0.038,
}

FALLBACK_VOLATILITIES = {
    "VOO": 0.16, "QQQ": 0.20, "VTI": 0.16, "VXUS": 0.17,
    "VGT": 0.22, "ARKK": 0.35, "VNQ": 0.20, "VWO": 0.22,
    "BND": 0.04, "SGOV": 0.01, "TLT": 0.14, "TIPS": 0.05,
}

FALLBACK_CATEGORY_CORRELATIONS = {
    ("equity", "equity"): 0.75,
    ("equity", "bond"): -0.15,
    ("bond", "bond"): 0.60,
}

# ─── RL Recommender Service ──────────────────────────────────────────────────

class RLRecommender:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(RLRecommender, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.config = DEFAULT_PIPELINE_CONFIG.copy()
        self.device = torch.device('cpu')
        
        # 1. Load Data
        from _constants import DataSyncMode
        self.config["data_source_mode"] = DataSyncMode.OFFLINE_CSV_ONLY
        
        try:
            res = generate_dataset_member_a([], self.config)
            self.master_df = res[0]
            self.price_matrix = res[1]
            self.daily_returns = res[3]
            self.drip_daily_returns = res[4]
            
            emb_cache = load_embedding_cache(
                self.master_df, self.price_matrix, res[2], self.daily_returns, self.config, 
                drip_daily_returns=self.drip_daily_returns, folder=CACHE_DIR
            )
            if emb_cache:
                self.dynamic_embeddings = emb_cache["dynamic_embeddings"]
                self.X_mean = emb_cache["model_checkpoint"]["X_mean"]
                self.X_std = emb_cache["model_checkpoint"]["X_std"]
                logger.info(f"RL Embeddings loaded successfully from {CACHE_DIR}")
            else:
                self.dynamic_embeddings = {}
                logger.error("Failed to load Phase 1 embeddings")
        except Exception as e:
            logger.error(f"Error loading dataset for RL: {e}")
            self.dynamic_embeddings = {}

        # 2. Load Model
        # checkpoint_rl_v2_dm64_nh4_lr001_id226.pt (example from plan)
        # We'll look for the most recent RL checkpoint if this specific one isn't found
        checkpoint_name = "checkpoint_rl_v2_dm64_nh4_lr001_id226.pt"
        checkpoint_path = os.path.join(CACHE_DIR, checkpoint_name)
        
        if os.path.exists(checkpoint_path):
            input_dim = 226 # Matches the filename id226
            self.model = PortfolioTransformerRL(input_dim, self.config).to(self.device)
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if 'agents_state' in checkpoint:
                self.model.load_state_dict(checkpoint['agents_state'][0])
            else:
                self.model.load_state_dict(checkpoint['model_state'])
            self.model.eval()
            logger.info(f"RL model loaded from {checkpoint_path}")
        else:
            logger.warning(f"RL checkpoint not found at {checkpoint_path}. Multi-horizon RL disabled.")
            self.model = None

        self._initialized = True

    def get_weights(self, user_profile: dict, tickers: list) -> tuple:
        """
        Inference: Returns (pre_goal_weights, post_goal_weights) as dictionaries.
        """
        if self.model is None or not self.dynamic_embeddings:
            return {}, {}

        # 1. Build input tensor
        from _rl_worker import _get_static_feature_columns
        static_cols = _get_static_feature_columns(self.master_df)
        
        emb_rows = []
        static_rows = []
        valid_tickers = []
        for t in tickers:
            if t in self.dynamic_embeddings and t in self.master_df.index:
                emb_rows.append(self.dynamic_embeddings[t])
                static_rows.append(self.master_df.loc[t, static_cols].values)
                valid_tickers.append(t)
        
        if not valid_tickers:
            return {}, {}

        emb_matrix = np.stack(emb_rows).astype(np.float32)
        static_matrix = np.stack(static_rows).astype(np.float32)
        
        # Build user condition
        user_vec = np.array(_encode_user_condition(user_profile), dtype=np.float32)
        N = emb_matrix.shape[0]
        user_tiled = np.tile(user_vec, (N, 1))
        
        full_input = np.concatenate([emb_matrix, user_tiled, static_matrix], axis=1)
        x = torch.tensor(full_input[np.newaxis, :, :], dtype=torch.float32).to(self.device)
        
        # 2. Forward pass
        with torch.no_grad():
            (mu_pre, _), (mu_post, _) = self.model(x)
            # Softmax + Normalization (Top-K etc)
            w_pre = _softmax_normalize_top_k(mu_pre, None)[0].cpu().numpy()
            w_post = _softmax_normalize_top_k(mu_post, None)[0].cpu().numpy()
            
        # 3. Post-process (Top-K pruning and thresholding)
        def _finalize_weights(w_np):
            # Sort and take top K
            idx_sorted = np.argsort(w_np)[::-1]
            top_idx = idx_sorted[:TOP_K_ASSETS]
            
            final_w = np.zeros_like(w_np)
            final_w[top_idx] = w_np[top_idx]
            
            # Normalize
            if final_w.sum() > 0:
                final_w /= final_w.sum()
            
            # Apply min threshold
            keep = final_w > MIN_ACTIVE_WEIGHT
            if not keep.any(): keep[:] = True
            final_w[~keep] = 0
            if final_w.sum() > 0:
                final_w /= final_w.sum()
                
            return {valid_tickers[i]: float(final_w[i]) for i in range(len(valid_tickers)) if final_w[i] > 0}

        return _finalize_weights(w_pre), _finalize_weights(w_post)

# ─── Multi-Horizon Encoding ──────────────────────────────────────────────────

def encode_multi_horizon(answers: dict) -> dict:
    """
    Generate a series of weight recommendations for multiple goal horizons.
    Handles 'reserved assets' by fixing their weights and scaling the RL recommendation.
    """
    global _RECOMMENDER_INSTANCE
    if _RECOMMENDER_INSTANCE is None:
        logger.info("Initializing global RLRecommender instance...")
        _RECOMMENDER_INSTANCE = RLRecommender()
    
    recommender = _RECOMMENDER_INSTANCE
    
    risk = float(answers.get("risk_tolerance", 50))
    start_cap = float(answers.get("start_cap") or 100000)
    monthly_contrib = float(answers.get("monthly_contrib") or 500)
    goals = answers.get("goals", [])
    
    # NEW: Handle multiple hard constraints from frontend schema
    hard_constraints = answers.get("hard_constraints", [])
    reserved_weights = {}
    for c in hard_constraints:
        ticker = c.get("ticker")
        pct_val = c.get("pct")
        if ticker and pct_val is not None and str(pct_val).strip() != "":
            try:
                reserved_weights[ticker.upper()] = float(pct_val) / 100.0
            except (ValueError, TypeError):
                continue
            
    total_reserved_ratio = sum(reserved_weights.values())
    
    if total_reserved_ratio >= 1.0:
        # User reserved 100% or more (oops). Cap it at 95% to allow some recommendation or return just constraints.
        total_reserved_ratio = 1.0
        scale = 0.0
    else:
        scale = 1.0 - total_reserved_ratio
    
    if not goals:
        # Default 30 year horizon if no goals
        goals = [{"name": "Retirement", "amount": 1000000, "years": 30}]
        
    sorted_goals = sorted(goals, key=lambda g: g.get("years", 30))
    horizons = [g.get("years", 30) for g in sorted_goals]
    
    segments = []
    # Segment 1: Start to Goal 1
    # Segment 2: Goal 1 to Goal 2
    # ...
    # Segment N: Goal N-1 to Goal N
    
    prev_yr = 0
    for i, g in enumerate(sorted_goals):
        curr_yr = g.get("years", 30)
        goal_amount = g.get("amount", 0)
        
        # Condition RL agent for this specific goal horizon
        profile = {
            "risk_tolerance": risk / 10.0, # RL expects 0-10
            "start_cap": start_cap,
            "monthly_contrib": monthly_contrib,
            "goal_year": curr_yr,
            "goal_amount": goal_amount
        }
        
        # Use RL agent to get "Pre-goal" weights for this segment
        # In a multi-horizon setup, we use the pre-goal head for the active goal segment.
        w_pre, _ = recommender.get_weights(profile, list(recommender.dynamic_embeddings.keys()))
        
        # Integrate reserved assets
        # Weights = sum(reserved) + (1 - sum(reserved_ratio)) * w_pre
        combined_weights = reserved_weights.copy()
        for t, wt in w_pre.items():
            combined_weights[t] = combined_weights.get(t, 0) + wt * scale
            
        segments.append({
            "horizon_years": (prev_yr, curr_yr),
            "goal_name": g.get("name", f"Goal {i+1}"),
            "weights": combined_weights
        })
        prev_yr = curr_yr

    # NEW: Add Terminal Growth Phase (Phase N+1)
    # Uses the 'post-goal' weights from the RL model for the last goal's profile
    if prev_yr < SIM_TERMINAL_HORIZON:
        # Re-fetch weights to get mu_post from the last goal context
        _, w_post = recommender.get_weights(profile, list(recommender.dynamic_embeddings.keys()))
        
        combined_post_weights = reserved_weights.copy()
        for t, wt in w_post.items():
            combined_post_weights[t] = combined_post_weights.get(t, 0) + wt * scale
            
        segments.append({
            "horizon_years": (prev_yr, SIM_TERMINAL_HORIZON),
            "goal_name": "Growth Phase",
            "weights": combined_post_weights
        })
        
    return {
        "risk_score": risk,
        "start_cap": start_cap,
        "monthly_contrib": monthly_contrib,
        "segments": segments,
        "goals": sorted_goals,
        "hard_constraints": hard_constraints
    }


# ─── Multi-Horizon Simulation ───────────────────────────────────────────────

def simulate_multi_horizon_portfolio(
    segments: list,
    goals: list,
    initial_investment: float = 100000,
    monthly_contrib: float = 500,
    projection_years: int = 30,
) -> dict:
    """
    Run a year-by-year simulation switching weights at segment boundaries.
    Records post-cash-out balances at each goal step.
    """
    recommender = RLRecommender()
    if not recommender._initialized or recommender.daily_returns is None:
        # Fallback if no data
        return {"error": "Market data unavailable for simulation"}

    # Use projection_years as a floor for simulation length
    max_goal_yr = max(g.get("years", 0) for g in goals) if goals else 0
    total_years = max(max_goal_yr, projection_years)

    # 1. Build Simulation Cache
    base_returns = recommender.drip_daily_returns if recommender.drip_daily_returns is not None else recommender.daily_returns
    sim_cache, start_idx_to_pos, clean_returns, column_to_idx = build_simulation_cache(base_returns, max_horizon_years=total_years)
    
    # 2. Determine start dates for Monte Carlo (use monthly-shifted pool)
    sim_starts = _resolve_simulation_starts(clean_returns)
    if len(sim_starts) == 0:
        return {"error": "Insufficient historical data for simulation paths"}

    # Take a sample of paths for performance
    rng = np.random.default_rng(42)
    num_paths = min(len(sim_starts), 20)
    sim_starts = rng.choice(sim_starts, size=num_paths, replace=False)

    goal_map = {g.get("years"): g.get("amount") for g in goals}
    
    # 3. Pre-process segment weights into year-indexed arrays for speed
    num_assets = len(clean_returns.columns)
    year_weight_arrays = []
    for yr in range(1, total_years + 1):
        # Find active segment for this year
        active_w_dict = {}
        for seg in segments:
            start, end = seg["horizon_years"]
            if start < yr <= end:
                active_w_dict = seg["weights"]
                break
        if not active_w_dict:
            # Fallback to last segment if beyond
            active_w_dict = segments[-1]["weights"]
            
        wa = np.zeros(num_assets)
        for t, wt in active_w_dict.items():
            if t in column_to_idx:
                wa[column_to_idx[t]] = wt
        if wa.sum() > 0: wa /= wa.sum()
        year_weight_arrays.append(wa)

    # 4. Run Monte Carlo Paths
    n_paths = len(sim_starts)
    # Matrix: (n_paths, total_years+1) to store net balance at each year for each path
    balance_matrix = np.zeros((n_paths, total_years + 1))
    step_balances_accumulator = {yr: [] for yr in goal_map.keys()}
    cash_out_events = []  # Populated from the mean path
    
    for path_i, start_idx in enumerate(sim_starts):
        capital = initial_investment
        debt = 0.0
        balance_matrix[path_i, 0] = capital
        
        for yr in range(1, total_years + 1):
            wa = year_weight_arrays[yr-1]
            # Annual return dot product
            pos = start_idx_to_pos[start_idx]
            yr_returns = sim_cache[pos][yr-1]
            portfolio_return = np.dot(wa, yr_returns)
            
            # Growth phase
            capital *= (1 + portfolio_return)
            # Add contributions (simplified annual)
            capital += monthly_contrib * 12
            
            # Debt compounding (fallback for shortfalls)
            debt *= (1 + LOAN_DAILY_RATE * TRADING_DAYS_PER_YEAR)
            
            # Cash out at goal
            if yr in goal_map:
                needed = goal_map[yr]
                if capital >= needed:
                    capital -= needed
                else:
                    debt += (needed - capital)
                    capital = 0.0
                
                # Record "Balance-at-Step" for this path
                step_balances_accumulator[yr].append(capital - debt)
            
            balance_matrix[path_i, yr] = capital - debt

    # 5. Aggregate Results across all paths
    years = list(range(total_years + 1))
    expected_path = [round(float(np.mean(balance_matrix[:, yr])), 2) for yr in years]
    upper_bound   = [round(float(np.percentile(balance_matrix[:, yr], 90)), 2) for yr in years]
    lower_bound   = [round(float(max(0, np.percentile(balance_matrix[:, yr], 10))), 2) for yr in years]
        
    # Step balances (sorted by year, rounded to 2 decimals)
    step_balances = []
    for yr in sorted(step_balances_accumulator.keys()):
        avg_bal = float(np.mean(step_balances_accumulator[yr]))
        step_balances.append({
            "year": yr,
            "balance": round(avg_bal, 2)
        })

    # Cash-out events (from expected path)
    for yr, amount in sorted(goal_map.items()):
        yr_idx = yr if yr <= total_years else total_years
        cash_out_events.append({
            "year": yr,
            "goal_name": next((g.get("name", "Goal") for g in goals if g.get("years") == yr), "Goal"),
            "amount": amount,
            "remaining_expected": expected_path[yr_idx] if yr_idx < len(expected_path) else expected_path[-1],
        })

    # Goal annotations for the chart
    goal_annotations = []
    for g in goals:
        y = g.get("years", 10)
        if y <= total_years:
            goal_annotations.append({
                "year": y,
                "label": g.get("name", "Goal"),
                "amount": g.get("amount", 0),
                "is_short_term": y <= 5,
            })

    # 6. Calculate aggregate metrics for the dashboard cards
    # Use CAGR-equivalent considering all cash-outs
    total_withdrawn = sum(e["amount"] for e in cash_out_events)
    total_value_created = expected_path[-1] + total_withdrawn
    cagr = ((total_value_created / initial_investment) ** (1/total_years) - 1) if total_years > 0 else 0
    
    # Estimate volatility from the spread of paths at the final year
    # (Simplified: Standard deviation of final returns)
    final_returns = balance_matrix[:, -1] / initial_investment
    ann_vol = np.std(final_returns) / np.sqrt(total_years) if total_years > 0 else 0
    
    sharpe = (cagr - 0.02) / ann_vol if ann_vol > 0 else 0

    # 7. Calculate Effective Annual Returns for frontend recalculation
    # Formula: r_eff = (bal[t] + cashout - contrib) / bal[t-1] - 1
    # This allows the frontend to reproduce the path locally while changing goal amounts.
    
    expected_annual_returns = []
    upper_annual_returns = []
    lower_annual_returns = []
    
    for yr in range(1, total_years + 1):
        cashout = goal_map.get(yr, 0)
        contrib = monthly_contrib * 12
        
        # Expected
        prev_exp = expected_path[yr-1] if expected_path[yr-1] > 0 else 1.0
        r_exp = (expected_path[yr] + cashout - contrib) / prev_exp - 1
        expected_annual_returns.append(round(float(r_exp), 6))
        
        # Upper
        prev_upp = upper_bound[yr-1] if upper_bound[yr-1] > 0 else 1.0
        r_upp = (upper_bound[yr] + cashout - contrib) / prev_upp - 1
        upper_annual_returns.append(round(float(r_upp), 6))
        
        # Lower
        prev_low = lower_bound[yr-1] if lower_bound[yr-1] > 0 else 1.0
        r_low = (lower_bound[yr] + cashout - contrib) / prev_low - 1
        lower_annual_returns.append(round(float(r_low), 6))

    return {
        "years": years,
        "expected_path": expected_path,
        "upper_bound": upper_bound,
        "lower_bound": lower_bound,
        "expected_annual_returns": expected_annual_returns,
        "upper_annual_returns": upper_annual_returns,
        "lower_annual_returns": lower_annual_returns,
        "step_balances": step_balances,
        "cash_out_events": cash_out_events,
        "goal_annotations": goal_annotations,
        "portfolio_stats": {
            "initial_investment": initial_investment,
            "projected_final_expected": expected_path[-1],
            "projected_final_upper": upper_bound[-1],
            "projected_final_lower": lower_bound[-1],
            "expected_annual_return": round(float(cagr * 100), 2),
            "annual_volatility": round(float(ann_vol * 100), 2),
            "sharpe_ratio": round(float(sharpe), 2),
        },
    }

# ─── Legacy Wrappers ─────────────────────────────────────────────────────────

def classify_asset(ticker: str) -> str:
    if ticker in BOND_UNIVERSE: return "bond"
    return "equity"

def build_covariance_matrix(tickers: list[str]) -> np.ndarray:
    try:
        from market_data import get_covariance_matrix as _get_real_cov
        return _get_real_cov(tickers)
    except Exception as e:
        logger.warning(f"Could not build covariance from DB, using fallback: {e}")
    # Fallback: category-based approximation
    n = len(tickers)
    cov = np.zeros((n, n))
    for i, t1 in enumerate(tickers):
        for j, t2 in enumerate(tickers):
            vol1 = FALLBACK_VOLATILITIES.get(t1, 0.15)
            vol2 = FALLBACK_VOLATILITIES.get(t2, 0.15)
            cat1 = classify_asset(t1)
            cat2 = classify_asset(t2)
            if i == j:
                cov[i][j] = vol1 ** 2
            else:
                pair = tuple(sorted([cat1, cat2]))
                corr = FALLBACK_CATEGORY_CORRELATIONS.get(pair, 0.5)
                cov[i][j] = corr * vol1 * vol2
    return cov

def encode_questionnaire(answers: dict) -> dict:
    # Legacy heuristic logic - keep for backward compat
    # ... (omitted for brevity, or we can replace it with the RL one)
    # Actually, let's just make it return the RL one if possible.
    return encode_multi_horizon(answers)

def simulate_portfolio(weights, goals, initial_investment, years):
    # Legacy wrapper for old API
    # We'll adapt it to the multi-horizon format
    segments = [{"horizon_years": (0, years), "weights": weights}]
    res = simulate_multi_horizon_portfolio(segments, goals, initial_investment)
    return {
        "expected_path": res["expected_path"],
        "years": res["years"],
        "step_balances": res.get("step_balances", [])
    }
