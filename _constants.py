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

MASTER_FILE  = "sp1500_master_research_dataset.csv"
PRICE_FILE   = "sp1500_price_matrix.csv"
VOLUME_FILE  = "sp1500_volume_matrix.csv"
DIV_CACHE    = "sp1500_dividends.csv"


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
# DEFAULT PIPELINE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_PIPELINE_CONFIG = {
    # -----------------------
    # System Execution Limits
    # -----------------------
    "data_source_mode":  DataSyncMode.OFFLINE_CSV_ONLY,
    "data_start_date": "1962-01-01",
    "training_cutoff_date": "2015-01-01",
    "simulation_start_date": "2015-01-01",
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
    "ml_epochs": 50,
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
    "rl_episodes": 1000,
    # -----------------------
    # Grid Search Constants
    # -----------------------
    "grid_lrs": [0.001],
    "grid_batch_sizes": [32],
    "grid_d_models": [64],
    "grid_embedding_dims": [8, 16, 32],
    # -----------------------
    # Reward Function (Inverse Exponential)
    # -----------------------
    "reward_terminal_target": 1000000.0, # Target for inverse exp scaling
    "reward_terminal_k": 2.0,            # Exp scaling factor
    "reward_goal_penalty_rate": 0.5,     # Penalty per percentage missed
    "reward_lambda_goal": 1.0,           # Weight for goal penalty loss
    
    # -----------------------
    # Simulation Parameters
    # -----------------------
    "sim_horizon_mode": "ignore",        # 'ignore' (truncate) or 'loop' (cyclical padding)
    "simulation_start_date": "2015-01-01",
    "simulation_end_date": "2026-12-31",
    "reward_lambda_terminal": 1.0,       # Weight for terminal wealth loss

    # -----------------------
    # Simulation & Thresholds
    # -----------------------
    "sim_glide_path_tau": 8.0,

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
}


# ═══════════════════════════════════════════════════════════════════════
# TEST PROFILES — Standard user profiles for multi-profile evaluation
# ═══════════════════════════════════════════════════════════════════════

TEST_PROFILES = [
    {
        "profile_name": "Conservative Retiree",
        "risk_tolerance": 2.0,
        "start_cap": 500000,
        "goals": {2: 50000, 5: 100000, 10: 200000}
    },
    {
        "profile_name": "Balanced Growth (House + Retirement)",
        "risk_tolerance": 5.0,
        "start_cap": 150000,
        "goals": {5: 60000, 20: 120000}
    },
    {
        "profile_name": "Aggressive Young Investor",
        "risk_tolerance": 9.0,
        "start_cap": 50000,
        "goals": {15: 80000, 30: 200000}
    },
]


# ═══════════════════════════════════════════════════════════════════════
# THETA SENSITIVITY GRID — Default grid for sensitivity analysis
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_THETA_GRID = [-0.5, -0.2, 0.0, 0.1, 0.2, 0.3, 0.5]
