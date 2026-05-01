"""
_constants.py
─────────────
Central registry of all constants, enums, file paths, and default
configuration values used across the research pipeline.

Change values HERE to update every module that depends on them.
No other module should hard-code these values.
"""


# ═══════════════════════════════════════════════════════════════════════
# FILE PATHS — CSV cache locations for the data pipeline
# ═══════════════════════════════════════════════════════════════════════

import os
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MASTER_FILE  = os.path.join(_BASE_DIR, "sp1500_master_research_dataset.csv")
PRICE_FILE   = os.path.join(_BASE_DIR, "sp1500_price_matrix.csv")
VOLUME_FILE  = os.path.join(_BASE_DIR, "sp1500_volume_matrix.csv")
DIV_CACHE    = os.path.join(_BASE_DIR, "sp1500_dividends.csv")


# ═══════════════════════════════════════════════════════════════════════
# DATA SOURCE URLS — Web scraping targets
# ═══════════════════════════════════════════════════════════════════════

SP_CONSTITUENT_URLS = [
    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
]

NASDAQ_FTP_URL = "ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqtraded.txt"


# ═══════════════════════════════════════════════════════════════════════
# ENUMS — Pipeline execution modes
# ═══════════════════════════════════════════════════════════════════════

class DataSyncMode:
    """
    Controls how generate_dataset_member_a syncs price and fundamental data.
    Set via PIPELINE_CONFIG["data_source_mode"].
    """
    # Load from existing CSV files only. No network calls.
    OFFLINE_CSV_ONLY = "OFFLINE_CSV_ONLY"

    # Fetch only missing price dates and new tickers. Fastest live mode.
    INCREMENTAL_SYNC = "INCREMENTAL_SYNC"

    # Destroy all cached data and re-download everything from scratch.
    # WARNING: Takes hours. Use sparingly.
    FULL_REBUILD = "FULL_REBUILD"

    # Re-download all fundamentals (.info) but keep existing price matrix.
    REFRESH_FUNDAMENTALS = "REFRESH_FUNDAMENTALS"


# ═══════════════════════════════════════════════════════════════════════
# SIMULATION & SCORING CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

TRADING_DAYS_PER_YEAR  = 252
SIM_YEARS              = 30
LOAN_DAILY_RATE        = (1.10) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1   # 10% APR compounded daily

# Scoring defaults
DEFAULT_ARCHETYPE_COUNT     = 30    # Top-N safe archetypes for user vector
DEFAULT_BUDGET_TOLERANCE    = 1.1   # 10% tolerance above risk budget ceiling
DEFAULT_FALLBACK_SAFE_COUNT = 20    # Fallback count when no assets within budget
DEFAULT_VOLATILITY          = 0.25  # Default annualized volatility when data is missing
DEFAULT_V_SCORE_ANCHOR      = 0.50  # Anchor vol for V_score mapping
DEFAULT_SOFTMAX_CAP         = 20    # Max exponent for softmax overflow protection

# Daily return clamp — prevents blow-ups from bad data
DAILY_RETURN_CLAMP = 0.50

# Cash bucket annual return (risk-free proxy)
CASH_BUCKET_ANNUAL_RATE = 0.03

# GFR objective threshold
GFR_OBJECTIVE_THRESHOLD = 0.90


# ═══════════════════════════════════════════════════════════════════════
# RL REWARD FUNCTION CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

# Growth: CAGR is multiplied by this to convert to reward points (10% CAGR → 10 pts)
CAGR_REWARD_SCALE         = 100.0

# Drawdown: MDD is multiplied by this before being scaled by the risk factor
MDD_PENALTY_SCALE         = 4.0

# Risk factor: penalty_multiplier = (MAX_RISK_OFFSET - user_risk_tolerance)
# Risk 1 → multiplier 10 | Risk 10 → multiplier 1
MAX_RISK_OFFSET           = 11.0

# GFR bracketed penalty/reward structure (Success Rate over N paths)
# 1.0 -> +10 | 0.8 -> 0 | 0.6 -> -5 | 0.4 -> -10 | 0.2 -> -20 | 0.0 -> -30
GFR_BRACKETS = [
    (1.0,  10.0),
    (0.8,   0.0),
    (0.6,  -5.0),
    (0.4, -10.0),
    (0.2, -20.0),
    (0.0, -30.0)
]
GFR_MISS_FLAT_PENALTY     = 0.0     # (Deprecated in favor of GFR_BRACKETS)
GFR_PERFECT_REWARD        = 10.0    # (Explicit bonus for 1.0)

# Number of top assets to select by weight — used consistently in BOTH training and evaluation.
# Replaces the old MIN_ACTIVE_WEIGHT threshold, which caused a train/eval mismatch:
# training used top-K fallback while evaluation used a hard weight cutoff.
TOP_K_ASSETS              = 10
MIN_ACTIVE_WEIGHT         = 0.001   # kept only for display/log filtering

# Random asset subset size per training iteration.
# Instead of presenting all ~6500 assets to the Transformer every step, we sample a random
# subset. This: (1) speeds up attention O(N²→n²), (2) gives stronger per-asset gradient signal,
# (3) forces the policy to learn general feature patterns rather than memorizing specific tickers.
# Set to None to disable subsetting and use the full universe.
RL_ASSET_SUBSET_SIZE      = 500

# Number of walk-forward snapshots to sample per RL episode
WF_SNAPSHOTS_PER_EPISODE  = 8

# User condition vector dimensions (single-goal: risk, capital, goal_year, goal_ratio)
USER_CONDITION_DIM        = 4
CAPITAL_NORMALIZER        = 1_000_000.0
RISK_NORMALIZER           = 10.0
GOAL_YEAR_NORMALIZER      = 30.0    # Normalize goal year to [0, 1]

# Simulation start year for monthly-shifted evaluation
SIM_MONTHLY_START_YEAR    = 2006

# All single-goal episodes simulate to this terminal horizon
# (post-goal capital keeps compounding under the growth-phase weights)
SIM_TERMINAL_HORIZON      = 30

# Debug: max years to log per path
DEBUG_LOG_MAX_YEARS       = 5


# ═══════════════════════════════════════════════════════════════════════
# DEFAULT PIPELINE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_PIPELINE_CONFIG = {
    # -----------------------
    # System Execution Limits
    # -----------------------
    "data_source_mode":  DataSyncMode.OFFLINE_CSV_ONLY,
    "data_start_date": "1962-01-01",
    "training_cutoff_date": "2015-01-01",
    "simulation_start_date": "2006-01-01",
    "simulation_end_date": "2026-12-31",
    "scrape_delay": 0.3,
    "yf_chunk_size": 300,

    # -----------------------
    # DRIP (Dividend Reinvestment)
    # -----------------------
    "drip_reinvest": False,  # True = reinvest dividends, False = hold as cash

    # -----------------------
    # Column Extraction Scope
    # -----------------------
    "ml_training_features": [
        "hist_momentum", "hist_volatility", "hist_volume",
        "sector", "industry",
        "state", "quoteType", "exchange",
    ],

    # -----------------------
    # Neural Network Geometry
    # -----------------------
    "ml_max_seq_len": 1260,          # 5 years * 252 days.
    "ml_d_model": 64,
    "ml_nhead": 4,
    "ml_num_encoder_layers": 2,
    "ml_dim_feedforward": 128,
    "ml_embedding_dim": 8,           
    "ml_target_metrics": ["return", "volatility", "volume"],
    "ml_target_horizons": [1, 3, 5, 10, 15],
    "ml_horizon_weights": {1: 1.0, 3: 0.8, 5: 0.6, 10: 0.4, 15: 0.2},
    "ml_epochs": 300,
    "ml_batch_size": 32,             # Lowered from 64 to fit sequence model into memory
    "ml_learning_rate": 0.001,
    "ml_time_decay_half_life": 10,

    # -----------------------
    # RL Transformer Geometry
    # -----------------------
    "rl_d_model": 64,
    "rl_nhead": 4,
    "rl_num_encoder_layers": 4,
    "rl_dim_feedforward": 256,
    "rl_dropout": 0.1,
    "rl_learning_rate": 0.001,
    "rl_episodes": 10000000,
    "rl_force_rebuild": False,
    "rl_initial_mu_bias": 0.0,
    "rl_checkpoint_frequency": 10,
    "sim_paths_per_episode": 50,          # (Legacy) Number of start-dates for full evaluation
    "rl_paths_per_step": 10,              # Sim paths per gradient update step (stochastic training)
    "rl_verbose_every": 10,                # Print progress every N iterations
    "rl_ensemble_size": 3,                 # Number of agents to train and average together
    "rl_convergence_window": 500,          # Rolling window size for reward smoothing
    "rl_convergence_patience": 5000,       # Iterations without improvement before convergence
    "rl_convergence_min_delta": 0.5,       # Minimum reward improvement to reset patience
    "rl_early_stopping": False,            # If True, halt training on convergence detection
    "debug_path": False,                   # Global toggle for per‑episode sample‑path logging
    "rl_parallel_cores": -1,               # Use all cores by default (set to 1 for tiny jobs)
    # -----------------------
    # Grid Search Constants
    # -----------------------
    "grid_lrs": [0.001],
    "grid_batch_sizes": [32],
    "grid_d_models": [64],
    "grid_embedding_dims": [8, 16, 32],
    # -----------------------
    # Reward Function (see RL REWARD FUNCTION CONSTANTS above)
    # -----------------------
    
    # -----------------------
    # Simulation Parameters
    # -----------------------
    "sim_horizon_mode": "loop",           # 'ignore' (truncate) or 'loop' (cyclical padding)
    "simulation_start_date": "2006-01-01",
    "simulation_end_date": "2026-12-31",
    "reward_lambda_terminal": 1.0,       # Weight for terminal wealth loss

    # -----------------------
    # Simulation & Thresholds
    # -----------------------
    "sim_glide_path_tau": 8.0,
    "sim_cvar_percentile": None,            # None = mean of ALL paths (recommended). Set to e.g. 20 to use CVaR worst-20% instead.

    # -----------------------
    # Validation & Splitting
    # -----------------------
    "ml_validation_mode": "absolute",       # 'absolute' (years) or 'proportional' (%)
    "ml_validation_horizon_years": 20,
    "ml_validation_percent": 0.20,
    "ml_cache_dir": "cache",
    "ml_force_retrain": False,
    "ml_checkpoint_frequency": 1,           # Save every N epochs
    "ml_resume_mode": "auto",               # 'auto', 'restart', or 'load_final'
    "ml_keep_backups": True,

    # -----------------------
    # Walk-Forward Analysis
    # -----------------------
    "wf_enabled": False,                    # Set to True to enable point-in-time simulation
    "wf_start_year": 2000,                  # Start point-in-time embeddings from this year
    "wf_step_months": 1,                    # How many months between generated snapshots
    "wf_force_rebuild": False,              # Set to True to overwrite existing snapshots
}


# ═══════════════════════════════════════════════════════════════════════
# TEST PROFILES — Standard user profiles for multi-profile evaluation
# ═══════════════════════════════════════════════════════════════════════

TEST_PROFILES = [
    # ── CONSERVATIVE (Risk 1-3) ──────────────────────────────────────────
    {"profile_name": "Risk-Averse Saver", "risk_tolerance": 1.5, "start_cap": 25000, "monthly_contrib": 0, "goals": {3: 30000, 10: 60000}},
    {"profile_name": "Conservative Retiree", "risk_tolerance": 2.0, "start_cap": 500000, "monthly_contrib": 0, "goals": {2: 50000, 5: 100000, 10: 200000}},
    {"profile_name": "Legacy Protector", "risk_tolerance": 1.0, "start_cap": 1500000, "monthly_contrib": 0, "goals": {20: 2000000}},
    {"profile_name": "Down Payment Safety", "risk_tolerance": 2.5, "start_cap": 80000, "monthly_contrib": 0, "goals": {4: 110000}},
    {"profile_name": "Rainy Day Buffer", "risk_tolerance": 3.0, "start_cap": 15000, "monthly_contrib": 0, "goals": {5: 25000, 15: 50000}},
    {"profile_name": "Trust Fund Anchor", "risk_tolerance": 1.8, "start_cap": 2000000, "monthly_contrib": 0, "goals": {10: 2500000, 25: 4000000}},
    {"profile_name": "Stable Income Seeker", "risk_tolerance": 2.2, "start_cap": 300000, "monthly_contrib": 0, "goals": {5: 350000, 15: 500000, 30: 800000}},
    {"profile_name": "Low-Volatility Pensioner", "risk_tolerance": 1.2, "start_cap": 1200000, "monthly_contrib": 0, "goals": {5: 1300000, 20: 1800000}},
    {"profile_name": "Frugal Conservative", "risk_tolerance": 2.8, "start_cap": 50000, "monthly_contrib": 0, "goals": {10: 100000}},
    {"profile_name": "Bond-Heavy Defensive", "risk_tolerance": 1.5, "start_cap": 100000, "monthly_contrib": 0, "goals": {3: 110000, 7: 130000, 12: 160000}},

    # ── BALANCED (Risk 4-6) ──────────────────────────────────────────────
    {"profile_name": "Balanced Growth", "risk_tolerance": 5.0, "start_cap": 150000, "monthly_contrib": 0, "goals": {5: 180000, 20: 400000}},
    {"profile_name": "Mid-Career Climber", "risk_tolerance": 4.5, "start_cap": 250000, "monthly_contrib": 0, "goals": {10: 450000, 25: 1000000}},
    {"profile_name": "House & College Fund", "risk_tolerance": 5.5, "start_cap": 100000, "monthly_contrib": 0, "goals": {7: 150000, 18: 300000}},
    {"profile_name": "Pragmatic High Earner", "risk_tolerance": 6.0, "start_cap": 400000, "monthly_contrib": 0, "goals": {5: 600000, 15: 1200000, 30: 2500000}},
    {"profile_name": "Moderate Explorer", "risk_tolerance": 4.0, "start_cap": 40000, "monthly_contrib": 0, "goals": {8: 100000}},
    {"profile_name": "Diversified Builder", "risk_tolerance": 5.8, "start_cap": 120000, "monthly_contrib": 0, "goals": {12: 300000, 28: 800000}},
    {"profile_name": "Mid-Stage Saver", "risk_tolerance": 4.2, "start_cap": 200000, "monthly_contrib": 0, "goals": {5: 250000, 10: 350000, 20: 600000}},
    {"profile_name": "Steady Path Wealth", "risk_tolerance": 4.8, "start_cap": 75000, "monthly_contrib": 0, "goals": {10: 150000, 20: 350000, 30: 700000}},
    {"profile_name": "Balanced Heritage", "risk_tolerance": 5.2, "start_cap": 800000, "monthly_contrib": 0, "goals": {15: 1500000, 30: 3000000}},
    {"profile_name": "Early Retiree Attempt", "risk_tolerance": 6.0, "start_cap": 350000, "monthly_contrib": 0, "goals": {12: 1000000}},

    # ── AGGRESSIVE (Risk 7-9) ────────────────────────────────────────────
    {"profile_name": "Aggressive Young Investor", "risk_tolerance": 9.0, "start_cap": 50000, "monthly_contrib": 0, "goals": {15: 200000, 30: 1000000}},
    {"profile_name": "Tech Bulls Catalyst", "risk_tolerance": 8.0, "start_cap": 10000, "monthly_contrib": 0, "goals": {10: 80000, 20: 400000, 30: 1500000}},
    {"profile_name": "Crypto-Minded Equity", "risk_tolerance": 8.5, "start_cap": 5000, "monthly_contrib": 0, "goals": {5: 25000, 15: 150000}},
    {"profile_name": "HNW Risk Taker", "risk_tolerance": 7.5, "start_cap": 1000000, "monthly_contrib": 0, "goals": {10: 3000000, 25: 10000000}},
    {"profile_name": "Compound Interest Maxi", "risk_tolerance": 7.0, "start_cap": 20000, "monthly_contrib": 0, "goals": {25: 500000, 30: 1000000}},
    {"profile_name": "Full Growth Engine", "risk_tolerance": 9.0, "start_cap": 100000, "monthly_contrib": 0, "goals": {5: 200000, 10: 500000, 20: 2000000, 30: 5000000}},
    {"profile_name": "Venture Style Liquid", "risk_tolerance": 8.2, "start_cap": 300000, "monthly_contrib": 0, "goals": {15: 1500000}},
    {"profile_name": "Speculative Compounder", "risk_tolerance": 7.8, "start_cap": 15000, "monthly_contrib": 0, "goals": {10: 100000, 20: 500000, 30: 2000000}},
    {"profile_name": "Unconstrained Growth", "risk_tolerance": 9.0, "start_cap": 2500, "monthly_contrib": 0, "goals": {30: 500000}},
    {"profile_name": "Aggressive Legacy", "risk_tolerance": 7.2, "start_cap": 500000, "monthly_contrib": 0, "goals": {20: 3000000, 30: 8000000}},
]


def decompose_profiles(profiles=None):
    """
    Decompose multi-goal profiles into single-goal episodes.

    Each profile with N goals becomes N single-goal entries, each with:
        goal_year:   the year of the single goal
        goal_amount: the target amount for that year
        goals:       {goal_year: goal_amount}  (kept for sim backward-compat)

    This lets the RL agent learn the optimal pre-goal/post-goal allocation
    for each specific horizon, rather than trying to satisfy all goals at once.
    """
    if profiles is None:
        profiles = TEST_PROFILES

    single_goal_profiles = []
    for p in profiles:
        for year, amount in p["goals"].items():
            single_goal_profiles.append({
                "profile_name": f"{p['profile_name']} (Y{year})",
                "risk_tolerance": p["risk_tolerance"],
                "start_cap": p["start_cap"],
                "monthly_contrib": p.get("monthly_contrib", 0),
                "goal_year": int(year),
                "goal_amount": float(amount),
                "goals": {int(year): float(amount)},
            })
    return single_goal_profiles


# ═══════════════════════════════════════════════════════════════════════
# THETA SENSITIVITY GRID — Default grid for sensitivity analysis
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_THETA_GRID = [-0.5, -0.2, 0.0, 0.1, 0.2, 0.3, 0.5]
