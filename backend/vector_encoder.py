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

        # 2. Load Models (Ensemble)
        # checkpoint_rl_v2_dm64_nh4_lr001_id226.pt
        checkpoint_name = "checkpoint_rl_v2_dm64_nh4_lr001_id226.pt"
        checkpoint_path = os.path.join(CACHE_DIR, checkpoint_name)
        self.models = []
        
        if os.path.exists(checkpoint_path):
            input_dim = 226 # Matches the filename id226
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            if 'agents_state' in checkpoint:
                # Load all agents in the ensemble
                for state in checkpoint['agents_state']:
                    m = PortfolioTransformerRL(input_dim, self.config).to(self.device)
                    m.load_state_dict(state)
                    m.eval()
                    self.models.append(m)
                logger.info(f"Loaded ensemble of {len(self.models)} RL agents from {checkpoint_path}")
            elif 'model_state' in checkpoint:
                # Fallback for single-agent checkpoints
                m = PortfolioTransformerRL(input_dim, self.config).to(self.device)
                m.load_state_dict(checkpoint['model_state'])
                m.eval()
                self.models.append(m)
                logger.info(f"Loaded single RL agent from {checkpoint_path}")
        else:
            logger.warning(f"RL checkpoint not found at {checkpoint_path}. Multi-horizon RL disabled.")

        self._initialized = True

    def get_weights(self, user_profile: dict, tickers: list) -> tuple:
        """
        Inference: Returns (pre_goal_weights, post_goal_weights) as dictionaries.
        """
        if not self.models or not self.dynamic_embeddings:
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

        # 2. Forward pass (Ensemble Average)
        if not self.models:
            return {}, {}

        with torch.no_grad():
            mu_pre_sum = 0
            mu_post_sum = 0
            for m in self.models:
                (mu_p, _), (mu_g, _) = m(x)
                mu_pre_sum += mu_p
                mu_post_sum += mu_g
            
            # Average the logits across all agents
            mu_pre = mu_pre_sum / len(self.models)
            mu_post = mu_post_sum / len(self.models)
            
            # Use raw mu directly as done in the training notebook to ensure consistent weights
            w_pre = F.softmax(mu_pre, dim=-1)[0].cpu().numpy()
            w_post = F.softmax(mu_post, dim=-1)[0].cpu().numpy()
            
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
    projection_years: int = 30,
) -> dict:
    """
    Run a year-by-year simulation of portfolio returns, switching weights at segment boundaries.
    """
    recommender = RLRecommender()
    if not recommender._initialized or recommender.daily_returns is None:
        # Fallback if no data
        return {"error": "Market data unavailable for simulation"}

    total_years = projection_years

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

    # 4. Vectorized Path Simulation
    pos_indices = [start_idx_to_pos[s] for s in sim_starts]
    selected_returns = sim_cache[pos_indices] # (n_paths, total_years, n_assets)
    
    year_weight_arrays = np.array(year_weight_arrays)
    wa_broad = year_weight_arrays.reshape(1, total_years, num_assets)
    path_returns = np.sum(selected_returns * wa_broad, axis=2)

    # 5. Calculate Raw Portfolio Growth Rates for frontend recalculation
    expected_annual_returns = []
    upper_annual_returns = []
    lower_annual_returns = []
    
    for yr in range(total_years):
        r_mean = float(np.mean(path_returns[:, yr]))
        r_std = float(np.std(path_returns[:, yr]))
        
        expected_annual_returns.append(round(r_mean, 6))
        upper_annual_returns.append(round(r_mean + r_std * 0.8, 6)) 
        lower_annual_returns.append(round(r_mean - r_std * 0.8, 6))

    # 6. Calculate Per-Segment Stats
    segment_stats = []
    for seg in segments:
        start_yr, end_yr = seg["horizon_years"]
        seg_returns = path_returns[:, start_yr:end_yr]
        if seg_returns.size > 0:
            mean_ret = float(np.mean(seg_returns))
            std_ret = float(np.std(seg_returns))
            seg_sharpe = (mean_ret - 0.02) / std_ret if std_ret > 0 else 0
            segment_stats.append({
                "expected_return": round(mean_ret * 100, 2),
                "volatility": round(std_ret * 100, 2),
                "sharpe_ratio": round(float(seg_sharpe), 2)
            })
        else:
            segment_stats.append({"expected_return": 0, "volatility": 0, "sharpe_ratio": 0})

    return {
        "expected_annual_returns": expected_annual_returns,
        "upper_annual_returns": upper_annual_returns,
        "lower_annual_returns": lower_annual_returns,
        "segment_stats": segment_stats,
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
    res = simulate_multi_horizon_portfolio(segments, years)
    return {
        "expected_path": res.get("expected_annual_returns", []),
        "years": list(range(years)),
        "step_balances": []
    }

# Eagerly initialize the recommender to reduce first-request latency
try:
    logger.info("Eagerly warming up RLRecommender and Market Data...")
    _RECOMMENDER_INSTANCE = RLRecommender()
except Exception as e:
    logger.error(f"Failed to eager-load RLRecommender: {e}")
