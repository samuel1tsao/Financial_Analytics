"""
_rl_worker.py
─────────────
Member D: RL-Driven Transformer Portfolio Optimizer.

Architecture:
    PortfolioTransformerRL — Transformer with single Gaussian policy head:
        - Shared head for all horizons, conditioned by user profile
    train_rl_agent — Stochastic REINFORCE with per-step gradient updates

Optimizations:
    - Feature matrices pre-computed once (no per-step pandas lookups)
    - Simulation start pool pre-resolved once
    - Dense weight array built vectorized (no dict conversion)
    - Fewer sim paths per gradient update (rl_paths_per_step)
    - Single-goal episodes decomposed from multi-goal profiles
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import os
from typing import Dict, Tuple

from _constants import (
    TEST_PROFILES,
    USER_CONDITION_DIM,
    CAPITAL_NORMALIZER,
    PREFERENCE_NORMALIZER,
    CAGR_NORMALIZER,
    GOAL_YEAR_NORMALIZER,
    SIM_TERMINAL_HORIZON_CAP,
    MIN_ACTIVE_WEIGHT,
    RELATIVE_WEIGHT_THRESHOLD,
    MAX_DYNAMIC_ASSETS,
    RL_ASSET_SUBSET_SIZE,
    CAGR_REWARD_SCALE,
    MDD_PENALTY_SCALE,
    VOL_PENALTY_SCALE,
    GFR_BRACKETS,
    DIVERSITY_PENALTY_SCALE,
    decompose_profiles,
)
from _sim_worker import (
    simulate_rl_environment_step,
    build_simulation_cache,
    _resolve_simulation_starts,
    _run_single_sim_path,
    _aggregate_path_results,
    _compute_reward,
    calculate_decay_probabilities,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transformer initial bias — start by "rejecting" most assets during exploration
# Now configurable via config["rl_initial_mu_bias"]

# Categorical column prefixes used for static identity features
_CATEGORICAL_PREFIXES = ["sector", "industry", "state", "quoteType", "exchange"]


# ══════════════════════════════════════════════════════════════════════════════
# Model Architecture
# ══════════════════════════════════════════════════════════════════════════════

class PortfolioTransformerRL(nn.Module):
    """
    Transformer with single Gaussian policy head + Learnable Embedding Adapter.

    The adapter takes the frozen 8-dim embeddings from Member A and projects
    them to a richer 32-dim space via a small trainable MLP. This allows RL
    gradients to learn WHICH aspects of the embedding are useful for portfolio
    allocation, increasing the embedding's share of the input from ~3.5% to ~13%.

    Uses cross-asset attention to produce (mu, sigma) outputs:
        - Optimized for current goal/horizon specified in the user condition.
    """
    def __init__(self, input_dim, config, emb_raw_dim=8):
        super().__init__()

        self.d_model = config.get("rl_d_model", 64)
        nhead          = config.get("rl_nhead", 4)
        num_layers     = config.get("rl_num_encoder_layers", 2)
        dim_feedforward = config.get("rl_dim_feedforward", 256)
        dropout        = config.get("rl_dropout", 0.1)

        # 1. Expand frozen embeddings (8 -> 32)
        self.emb_raw_dim = emb_raw_dim
        adapter_out_dim = config.get("rl_adapter_dim", 32)
        self.emb_adapter = nn.Sequential(
            nn.Linear(emb_raw_dim, adapter_out_dim),
            nn.ReLU(),
            nn.Linear(adapter_out_dim, adapter_out_dim),
        )

        # 2. Amplify User Condition (8 -> 32)
        user_out_dim = config.get("rl_user_dim", 32)
        self.user_adapter = nn.Sequential(
            nn.Linear(8, user_out_dim),
            nn.ReLU(),
            nn.Linear(user_out_dim, user_out_dim),
        )

        # 3. Compress Static Asset Features (~214 -> 64)
        self.static_raw_dim = input_dim - emb_raw_dim - 8
        static_out_dim = config.get("rl_static_dim", 64)
        self.static_adapter = nn.Sequential(
            nn.Linear(self.static_raw_dim, static_out_dim),
            nn.ReLU(),
            nn.Linear(static_out_dim, static_out_dim),
        )

        # Effective input dim after all adapters (32 + 32 + 64 = 128)
        adapted_input_dim = adapter_out_dim + user_out_dim + static_out_dim

        # Project adapted features to transformer dimension
        self.input_proj = nn.Linear(adapted_input_dim, self.d_model)

        # Transformer encoder for cross-asset attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        initial_bias  = config.get("rl_initial_mu_bias", -0.1)
        initial_sigma = config.get("rl_initial_sigma_bias", 1.0)

        # Single policy head
        self.mu_head    = nn.Linear(self.d_model, 1)
        self.sigma_head = nn.Linear(self.d_model, 1)
        nn.init.constant_(self.mu_head.bias, initial_bias)
        nn.init.constant_(self.sigma_head.bias, initial_sigma)

    def forward(self, x, src_key_padding_mask=None):
        """
        x: (batch, num_assets, input_dim)
           Layout: [embedding(8) | user_condition(8) | static_features(~214)]
        Returns: (mu, sigma) each (batch, num_assets)
        """
        # Split out the three distinct feature groups
        emb_raw = x[:, :, :self.emb_raw_dim]               # (B, N, 8)
        user_cond_end = self.emb_raw_dim + 8
        user_c  = x[:, :, self.emb_raw_dim:user_cond_end]  # (B, N, 8)
        other   = x[:, :, user_cond_end:]                  # (B, N, ~214)
        
        # Apply specialized adapters
        emb_adapted    = self.emb_adapter(emb_raw)         # (B, N, 32)
        user_adapted   = self.user_adapter(user_c)         # (B, N, 32)
        static_adapted = self.static_adapter(other)        # (B, N, 64)

        # Concatenate re-balanced signal (Total 128 dims)
        # User condition is now 25% of the total signal (32/128) instead of 3%
        x_adapted = torch.cat([emb_adapted, user_adapted, static_adapted], dim=-1)

        h   = F.relu(self.input_proj(x_adapted))
        out = self.transformer(h, src_key_padding_mask=src_key_padding_mask)

        mu    = self.mu_head(out).squeeze(-1)
        sigma = F.softplus(self.sigma_head(out).squeeze(-1)) + 0.5

        return mu, sigma


# ══════════════════════════════════════════════════════════════════════════════
# Policy Sampling
# ══════════════════════════════════════════════════════════════════════════════

def sample_portfolio_weights(mu, sigma, mask=None):
    """
    Sample normalized portfolio weights from the Gaussian policy.
    If mask is provided (True = invalid), those assets are forced to zero weight.
    """
    dist       = torch.distributions.Normal(mu, sigma)
    raw_action = dist.sample()
    
    lp = dist.log_prob(raw_action)
    
    if mask is not None:
        # Mask out invalid assets with a large negative number so Softmax makes them 0
        raw_action = raw_action.masked_fill(mask, -1e9)
        lp = lp.masked_fill(mask, 0.0)
        
    weights    = F.softmax(raw_action, dim=-1)
    log_prob   = lp.sum(dim=-1)
    return weights, log_prob


def _softmax_normalize_top_k(samp, mask):
    """
    Apply mask and softmax to sampled logits. 
    The pruning to TOP_K happens in the weight array conversion for simulation.
    """
    if mask is not None:
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).to(samp.device)
        # Invert the mask if it's a "valid" mask (True = valid) to match masked_fill (True = masked)
        # sub_ticker_mask is True for valid assets, so we need ~mask
        if mask.dtype == torch.bool:
            samp = samp.masked_fill(~mask, -1e9)
        else:
            samp = samp.masked_fill(mask, -1e9)
    return F.softmax(samp, dim=-1)


def _run_agent_simulations_batch(w_pre, bootstrap_indices, sim_cache, clean_returns_values, profile, cvar_percentile=None):
    """
    ULTRA-FAST vectorized simulation batch runner.
    Bypasses the sequential Python loops of _run_single_sim_path.
    Calculates ETV, MDD, and GFR for all paths in a single NumPy pass.
    
    bootstrap_indices: (n_paths, goal_year) - pre-sampled indices from cache
    """
    cache_array, _ = sim_cache
    
    goal_year = profile.get('goal_year', SIM_TERMINAL_HORIZON_CAP)
    # 1. Gather annual returns for these paths using bootstrapping
    # bootstrap_indices shape: (n_paths, goal_year)
    # cache_array shape: (num_starts, num_assets)
    # batch_rets shape: (n_paths, goal_year, num_assets)
    batch_rets = cache_array[bootstrap_indices[:, :goal_year], :]
    
    # 2. Compute annual portfolio returns via vectorized dot product
    # Shape: (n_paths, goal_year)
    ann_rets = np.dot(batch_rets, w_pre)
    
    # 3. Calculate capital trajectory and market multiplier
    # Shape: (n_paths, goal_year)
    market_mults = np.cumprod(1.0 + ann_rets, axis=1)
    
    # 4. Handle goal withdrawal
    start_cap = profile['start_cap']
    goal_amt  = profile['goals'].get(goal_year, 0.0)
    
    # Terminal capital BEFORE withdrawal
    term_cap_pre = start_cap * market_mults[:, -1]
    final_cap    = term_cap_pre - goal_amt
    
    bankrupt = final_cap < 0
    actual_wds = np.where(bankrupt, np.maximum(term_cap_pre, 0.0), goal_amt)
    
    # 5. MDD calculation (pure NumPy)
    running_peak = np.maximum.accumulate(market_mults, axis=1)
    running_peak = np.maximum(running_peak, 1.0) # start peak is 1.0
    drawdowns = (running_peak - market_mults) / running_peak
    max_drawdowns = np.max(drawdowns, axis=1)
    
    # 6. Aggregate results
    gfr = np.mean(~bankrupt)
    etv = np.mean(final_cap)
    aw  = np.mean(actual_wds)
    mean_market_mult = float(np.mean(market_mults[:, -1]))

    if cvar_percentile is not None:
        sorted_mdds = np.sort(max_drawdowns)
        n_cvar = max(1, int(len(sorted_mdds) * (cvar_percentile / 100.0)))
        mdd = float(np.mean(sorted_mdds[-n_cvar:]))
    else:
        mdd = float(np.mean(max_drawdowns))
        
    path_vols = np.std(ann_rets, axis=1)
    mean_annual_vol = float(np.mean(path_vols))
    
    metrics = {
        "GFR": float(gfr), "ETV": float(etv), "MDD": mdd, "AW": float(aw),
        "Annual_Vol": mean_annual_vol,
        "Mean_Market_Mult": mean_market_mult,
        "GoalFails": int(np.sum(bankrupt)), "MarketFails": 0,
        "Total_Simulations": bootstrap_indices.shape[0],
        "Active_Assets": int(np.sum(w_pre > 0))
    }
    reward = _compute_reward(metrics, profile)
    return reward, metrics


# ══════════════════════════════════════════════════════════════════════════════
# Feature Engineering
# ══════════════════════════════════════════════════════════════════════════════

def _encode_user_condition(user_profile):
    """
    Encode a single-goal user profile into an 8-dim condition vector.
    """
    # 1. Sensitivity Dimensions (1-10 normalized to [0, 1])
    dd_sens    = float(user_profile.get("drawdown_sensitivity", 5.0)) / PREFERENCE_NORMALIZER
    vol_sens   = float(user_profile.get("volatility_sensitivity", 5.0)) / PREFERENCE_NORMALIZER
    goal_flex  = float(user_profile.get("goal_flexibility", 5.0)) / PREFERENCE_NORMALIZER
    conc_pref  = float(user_profile.get("concentration_pref", 5.0)) / PREFERENCE_NORMALIZER

    # 2. Financial Dimensions
    start_cap  = float(user_profile.get("start_cap", 100000.0))
    
    if "goal_year" in user_profile:
        goal_year  = float(user_profile["goal_year"]) / GOAL_YEAR_NORMALIZER
        goal_ratio = float(user_profile["goal_amount"]) / start_cap
    else:
        goals = user_profile.get("goals", {})
        goal_year  = float(max(goals.keys())) / GOAL_YEAR_NORMALIZER if goals else 0.0
        goal_ratio = float(sum(goals.values())) / start_cap if goals else 0.0

    # 3. Explicit Required CAGR (Normalized: 20% CAGR = 1.0)
    raw_goal_year = float(user_profile.get("goal_year", 30))
    # Required multiplier to hit goal
    required_mult = goal_ratio
    # CAGR = (mult ^ (1/yrs)) - 1
    required_cagr = (max(required_mult, 1.0) ** (1.0 / max(raw_goal_year, 1))) - 1.0
    required_cagr_norm = min(max(required_cagr, 0.0) / CAGR_NORMALIZER, 1.0)

    # Full 8-dim vector
    return [dd_sens, vol_sens, goal_flex, conc_pref, 
            start_cap / CAPITAL_NORMALIZER, goal_year, goal_ratio, required_cagr_norm]


def _get_static_feature_columns(master_df):
    """Return the list of dummy-encoded categorical feature columns."""
    return [c for c in master_df.columns
            if any(c.startswith(prefix + "_") for prefix in _CATEGORICAL_PREFIXES)]


def _precompute_rl_features(dataset, config):
    """
    Pre-compute invariant feature components ONCE at startup.

    Returns (emb_matrix, static_matrix, tickers) where:
        emb_matrix:    (N_assets, emb_dim)    — dynamic embeddings (fixed for training run)
        static_matrix: (N_assets, static_dim) — one-hot categorical features (never change)
        tickers:       list of N_assets ticker strings (stable ordering)
    """
    master_df  = dataset["master_df"]
    embeddings = dataset.get("dynamic_embeddings", {})
    if not embeddings:
        raise ValueError("Embeddings source is empty. Run Member A first.")

    static_feat_cols = _get_static_feature_columns(master_df)

    tickers = []
    emb_rows = []
    static_rows = []
    for ticker, emb in embeddings.items():
        if ticker not in master_df.index:
            continue
        emb_rows.append(np.array(emb, dtype=np.float32))
        static_rows.append(master_df.loc[ticker, static_feat_cols].astype(np.float32).values)
        tickers.append(ticker)

    if not tickers:
        raise ValueError("No valid assets found for RL training.")

    emb_matrix    = np.stack(emb_rows)       # (N, emb_dim)
    static_matrix = np.stack(static_rows)    # (N, static_dim)
    return emb_matrix, static_matrix, tickers


def _build_rl_input_fast(emb_matrix, static_matrix, user_profile):
    """
    Assemble the input tensor from cached matrices + user profile.
    Layout per asset: [embedding | user_condition(8) | static_features]
    Returns tensor of shape (1, N_assets, input_dim) on DEVICE.
    """
    user_vec   = np.array(_encode_user_condition(user_profile), dtype=np.float32)
    N          = emb_matrix.shape[0]
    user_tiled = np.tile(user_vec, (N, 1))   # (N, 8)

    # [emb | user_condition | static]
    full = np.concatenate([emb_matrix, user_tiled, static_matrix], axis=1)
    return torch.tensor(full[np.newaxis, :, :], dtype=torch.float32, device=DEVICE)


def build_rl_dataset(dataset, user_profile, config, specific_embeddings=None):
    """
    Assemble the input tensor for the Transformer (legacy per-call version).
    Each asset row = [Embedding | UserCondition(8) | StaticFeatures].
    Kept for walk-forward mode where embeddings change per snapshot.
    """
    master_df  = dataset["master_df"]
    embeddings = specific_embeddings or dataset.get("dynamic_embeddings", {})
    if not embeddings:
        raise ValueError("Embeddings source is empty. Run Member A first.")

    user_condition   = _encode_user_condition(user_profile)
    static_feat_cols = _get_static_feature_columns(master_df)

    tickers = []
    feature_rows = []
    for ticker, emb in embeddings.items():
        if ticker not in master_df.index:
            continue
        static_feats = master_df.loc[ticker, static_feat_cols].astype(float).tolist()
        feature_rows.append(list(emb) + user_condition + static_feats)
        tickers.append(ticker)

    if not feature_rows:
        return torch.tensor([], device=DEVICE), []

    tensor_x = torch.tensor([feature_rows], dtype=torch.float32, device=DEVICE)
    return tensor_x, tickers


# ══════════════════════════════════════════════════════════════════════════════
# Training Loop
# ══════════════════════════════════════════════════════════════════════════════

def _run_episode_walkforward(agent, dataset, profile, config, wf_snapshots, debug_path=False, sim_cache_bundle=None):
    """
    One Walk-Forward episode: sample historical snapshots, compute reward per regime.
    Returns (mean_reward, total_log_prob) for REINFORCE update.
    """
    all_dates     = list(wf_snapshots.keys())
    sampled_dates = np.random.choice(
        all_dates, size=min(WF_SNAPSHOTS_PER_EPISODE, len(all_dates)), replace=False
    )

    rewards   = []
    log_probs = []
    last_metrics = None
    last_weights = None
    last_tickers = []

    for date_str in sampled_dates:
        pit_x, pit_tickers = build_rl_dataset(
            dataset, profile, config, specific_embeddings=wf_snapshots[date_str]
        )
        if pit_x.numel() == 0:
            continue

        mu, sigma = agent(pit_x)
        weights, lp = sample_portfolio_weights(mu, sigma)

        reward, metrics = simulate_rl_environment_step(
            weights.detach().cpu().numpy(), pit_tickers,
            dataset, profile, config, start_date=date_str, debug_path=debug_path,
            sim_cache_bundle=sim_cache_bundle
        )
        rewards.append(reward)
        log_probs.append(lp.sum())
        last_metrics = metrics
        last_weights = weights.detach().cpu().numpy()
        last_tickers = pit_tickers

    return rewards, log_probs, last_metrics, last_weights, last_tickers


def _greedy_evaluate_fast(agents, emb_matrix, static_matrix, tickers, user_profile):
    """Run a deterministic (greedy) evaluation using ensemble average. Returns weight dict."""
    if not isinstance(agents, list):
        agents = [agents]
        
    for a in agents:
        a.eval()
        
    with torch.no_grad():
        tensor_x = _build_rl_input_fast(emb_matrix, static_matrix, user_profile)
        mu_sum = 0
        for a in agents:
            mu, _ = a(tensor_x)
            mu_sum += mu
            
        mu_avg = mu_sum / len(agents)
        greedy = F.softmax(mu_avg, dim=-1)
        
    weights_dict = {tickers[i]: float(greedy[0, i]) for i in range(len(tickers))}
    # Return twice for compatibility with multi-horizon callers
    return weights_dict, weights_dict


def get_rl_transformer_filename_base(config, input_dim):
    """Generate a unique identifier for the RL agent's architecture."""
    dm = config.get("rl_d_model", 64)
    nh = config.get("rl_nhead", 4)
    lr = config.get("rl_learning_rate", 0.0001)
    lr_clean = str(lr).replace("0.", "").replace(".", "")
    return f"rl_v2_dm{dm}_nh{nh}_lr{lr_clean}_id{input_dim}"


def _agent_process_entry(agent_idx, data_pickle_path, config, input_dim, log_dir):
    """
    Top-level entry point for each agent process. Fully self-contained:
    loads data from a shared pickle file, builds its own sim cache,
    redirects all output to a per-agent log file, trains, and saves checkpoint.
    """
    import sys, io, pickle, traceback

    # ── Redirect stdout/stderr to per-agent log file ──────────────────────
    log_path = os.path.join(log_dir, f"agent_{agent_idx}.log")
    log_file = open(log_path, "w", buffering=1)  # line-buffered
    sys.stdout = log_file
    sys.stderr = log_file

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[{time.strftime('%H:%M:%S')}] [Agent {agent_idx}] Process started (PID {os.getpid()})")

        # ── Load shared data from pickle ──────────────────────────────────
        print(f"  Loading data from {data_pickle_path}...")
        with open(data_pickle_path, "rb") as f:
            shared = pickle.load(f)
        emb_matrix    = shared["emb_matrix"]
        static_matrix = shared["static_matrix"]
        all_tickers   = shared["all_tickers"]
        base_returns  = shared["base_returns"]
        print(f"  Data loaded. {len(all_tickers)} assets.")

        # ── Build Simulation Cache (each process builds its own) ──────────
        print(f"  Building simulation cache (1-Year Blocks)...")
        cache_array, start_idx_to_pos, clean_returns, column_to_idx, max_sim_years = build_simulation_cache(base_returns, max_horizon_years=30)
        sim_cache = (cache_array, start_idx_to_pos)
        sim_start_pool = np.array(_resolve_simulation_starts(clean_returns))
        
        # Precompute decay probabilities for the entire pool
        start_dates = clean_returns.index[sim_start_pool]
        decay_probs = calculate_decay_probabilities(start_dates, half_life_years=3.0)
        
        num_columns = len(clean_returns.columns)
        valid_col_indices = np.array([column_to_idx.get(t, -1) for t in all_tickers])
        valid_ticker_mask = valid_col_indices >= 0
        is_active_matrix = base_returns.notna().values
        clean_returns_values = clean_returns.values
        
        # Precompute Asset Stats for Logging (Annualized)
        asset_means = cache_array.mean(axis=0)
        asset_vols  = cache_array.std(axis=0)
        
        print(f"  Cache ready. {len(sim_start_pool)} start dates, {num_columns} columns.")

        # ── Initialize agent ──────────────────────────────────────────────
        agent = PortfolioTransformerRL(input_dim, config).to(device)
        lr = config.get("rl_learning_rate", 0.0001)
        optimizer = optim.Adam(agent.parameters(), lr=lr)
        iterations = config.get("rl_episodes", 100)

        # ── Checkpoint Logic ──────────────────────────────────────────────
        cache_dir = config.get("rl_cache_dir", "cache")
        os.makedirs(cache_dir, exist_ok=True)
        fname_base = get_rl_transformer_filename_base(config, input_dim)
        checkpoint_path = os.path.join(cache_dir, f"checkpoint_{fname_base}_agent_{agent_idx}.pt")

        force_rebuild = config.get("rl_force_rebuild", False)
        start_iter = 0
        history = []
        baseline = 0.0
        best_rolling_reward = -float('inf')
        patience_counter = 0
        converged = False

        if force_rebuild and os.path.exists(checkpoint_path):
            print(f"  Restart Mode: Deleting checkpoint...")
            os.remove(checkpoint_path)
        elif not force_rebuild and os.path.exists(checkpoint_path):
            print(f"  Resume Mode: Attempting to load {os.path.basename(checkpoint_path)}...")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                agent.load_state_dict(checkpoint['model_state'])
                optimizer.load_state_dict(checkpoint['optimizer_state'])
                start_iter = checkpoint.get('episode', 0)
                history = checkpoint.get('history', [])
                baseline = checkpoint.get('baseline', 0.0)
                best_rolling_reward = checkpoint.get('best_rolling_reward', -float('inf'))
                patience_counter = checkpoint.get('patience_counter', 0)
                print(f"  Resumed from iteration {start_iter}.")
            except Exception as e:
                print(f"  WARNING: Failed to load checkpoint: {e}. Starting from scratch.")

        n_paths       = config.get("rl_paths_per_step", 5)
        verbose_every = 100
        check_freq    = 100
        conv_window   = config.get("rl_convergence_window", 500)
        conv_patience = config.get("rl_convergence_patience", 5000)
        conv_min_delta = config.get("rl_convergence_min_delta", 0.5)
        early_stopping = config.get("rl_early_stopping", False)

        single_goal_profiles = decompose_profiles(TEST_PROFILES, max_horizon=max_sim_years)

        subset_str = f"{RL_ASSET_SUBSET_SIZE} assets/iter" if RL_ASSET_SUBSET_SIZE else f"{len(all_tickers)} assets"
        print(f"  STOCHASTIC REINFORCE | {len(single_goal_profiles)} profiles | {subset_str} | {n_paths} paths/step | {iterations} iters")
        print(f"  {'='*60}")

        # ── Main Training Loop ────────────────────────────────────────────
        t0 = time.time()
        rng = np.random.default_rng()
        all_starts = sim_start_pool

        for it in range(start_iter, start_iter + iterations):
            profile = single_goal_profiles[np.random.randint(len(single_goal_profiles))]

            agent.train()
            if RL_ASSET_SUBSET_SIZE is not None and RL_ASSET_SUBSET_SIZE < len(all_tickers):
                subset_idx = np.sort(rng.choice(len(all_tickers), size=RL_ASSET_SUBSET_SIZE, replace=False))
                sub_emb, sub_static = emb_matrix[subset_idx], static_matrix[subset_idx]
                sub_col_indices, sub_ticker_mask = valid_col_indices[subset_idx], valid_ticker_mask[subset_idx]
            else:
                subset_idx = np.arange(len(all_tickers))
                sub_emb, sub_static = emb_matrix, static_matrix
                sub_col_indices, sub_ticker_mask = valid_col_indices, valid_ticker_mask

            base_x   = _build_rl_input_fast(sub_emb, sub_static, profile)
            tensor_x = base_x.expand(1, -1, -1)

            # Sample bootstrapping indices for all paths in this step
            # bootstrap_indices shape: (n_paths, max_sim_years)
            bootstrap_indices = rng.choice(len(sim_start_pool), size=(n_paths, max_sim_years), replace=True, p=decay_probs)

            # For masking, we use the first year of each path (approximate)
            path_starts = sim_start_pool[bootstrap_indices[:, 0]]
            safe_col_indices = np.where(sub_ticker_mask, sub_col_indices, 0)
            src_key_padding_mask = ~torch.from_numpy(
                is_active_matrix[path_starts[0:1]][:, safe_col_indices]
            ).to(device)

            mu, sigma = agent(tensor_x, src_key_padding_mask=src_key_padding_mask)
            dist  = torch.distributions.Normal(mu, F.softplus(sigma))
            samp  = dist.rsample()

            weights  = _softmax_normalize_top_k(samp, sub_ticker_mask)
            w_np  = weights[0].detach().cpu().numpy()

            def _build_full_weights(w_np):
                vi = np.where(sub_ticker_mask)[0]
                if len(vi) == 0: return np.zeros(num_columns)
                
                # Dynamic top K selection
                valid_w = w_np[vi]
                max_w = np.max(valid_w)
                threshold = max_w * RELATIVE_WEIGHT_THRESHOLD
                
                # Identify candidates above threshold
                rel_vi = vi[valid_w >= threshold]
                
                # Sort candidates by weight descending
                sorted_idx = np.argsort(w_np[rel_vi])[::-1]
                top_vi = rel_vi[sorted_idx[:MAX_DYNAMIC_ASSETS]]
                
                # Normalize the selection
                top_w = w_np[top_vi].copy()
                if top_w.sum() > 0: top_w /= top_w.sum()
                
                # Apply MIN_ACTIVE_WEIGHT
                keep = top_w > MIN_ACTIVE_WEIGHT
                if not keep.any(): keep[:] = True
                
                sel = top_vi[keep]
                sel_w = top_w[keep]
                
                cp = sub_col_indices[sel]
                fw = np.zeros(num_columns)
                fw[cp] = sel_w
                
                if fw.sum() > 0: fw /= fw.sum()
                return fw

            full_pre  = _build_full_weights(w_np)

            cvar_pct = config.get("sim_cvar_percentile")
            reward, metrics = _run_agent_simulations_batch(
                full_pre, bootstrap_indices, sim_cache, clean_returns_values, profile,
                cvar_percentile=cvar_pct
            )

            if it == start_iter:
                baseline = float(reward)
            else:
                baseline = 0.99 * baseline + 0.01 * float(reward)

            lp_sum = dist.log_prob(samp).mean()
            loss = -lp_sum * (reward - baseline)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ep_reward = float(reward)
            history.append({
                "episode": it, "reward": ep_reward,
                "GFR": metrics.get("GFR", 0.0),
                "ETV": metrics.get("ETV", 0.0),
                "MDD": metrics.get("MDD", 0.0),
            })

            # Convergence
            if len(history) >= conv_window:
                recent = [h["reward"] for h in history[-conv_window:]]
                rolling = sum(recent) / len(recent)
                if rolling > best_rolling_reward + conv_min_delta:
                    best_rolling_reward = rolling
                    patience_counter = 0
                else:
                    patience_counter += 1

            # Checkpoint
            should_ckpt = (it + 1) % check_freq == 0 or it == start_iter + iterations - 1
            if should_ckpt or (patience_counter >= conv_patience and early_stopping):
                # Atomic Save: Write to .tmp first then replace to prevent corruption on crash
                temp_path = checkpoint_path + ".tmp"
                torch.save({
                    'episode': it + 1,
                    'model_state': agent.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'history': history,
                    'baseline': baseline,
                    'best_rolling_reward': best_rolling_reward,
                    'patience_counter': patience_counter,
                }, temp_path)
                # Retry loop for Windows file-lock conflicts (OneDrive, serving process, etc.)
                for _attempt in range(3):
                    try:
                        os.replace(temp_path, checkpoint_path)
                        break
                    except PermissionError:
                        if _attempt < 2:
                            time.sleep(1.0 * (2 ** _attempt))  # 1s, 2s backoff
                        else:
                            # Last resort: save alongside with a unique name so we don't lose progress
                            fallback_path = checkpoint_path.replace(".pt", f"_iter{it+1}.pt")
                            try:
                                os.replace(temp_path, fallback_path)
                                print(f"  WARNING: Saved to fallback {fallback_path} due to file lock.")
                            except Exception:
                                print(f"  WARNING: Could not save checkpoint at iter {it+1}. Training continues.")

            # Early stopping
            if patience_counter >= conv_patience:
                if not converged:
                    converged = True
                    print(f"\n  CONVERGENCE DETECTED at iteration {it+1:,}")
                if early_stopping:
                    break

            # Logging
            if it % verbose_every == 0 or it == start_iter + iterations - 1:
                if it > 0: print("\n" + "-"*80 + "\n")
                elapsed = time.time() - t0
                ips = (it - start_iter + 1) / elapsed if elapsed > 0 else 0
                gs = f"Start: ${profile.get('start_cap',0):,.0f} -> Y{profile.get('goal_year','?')}: ${profile.get('goal_amount',0):,.0f}"
                print(f"  [Iter {it+1:7d}] Reward: {ep_reward:7.3f} | Baseline: {baseline:7.3f} | {ips:.1f} it/s | {profile['profile_name']} [{gs}]")
                
                # Detailed Metrics Breakdown
                m = metrics
                rc = m.get("reward_components", {})
                
                annual_return = rc.get("annual_return", 0)
                # ---- Return component ----
                ret_detail = f"Ret = ({annual_return:.4%}) * {CAGR_REWARD_SCALE:.1f} = {rc.get('return_score',0):+.2f}"
                # ---- GFR component ----
                gfr_val = m.get('GFR',0)
                gfr_contrib = rc.get('gfr_contrib',0)
                gfr_detail = f"GFR = {gfr_val:.4f} -> contrib {gfr_contrib:+.2f}"
                # ---- MDD component ----
                mdd_detail = f"MDD = {m.get('MDD',0):.4f} * {MDD_PENALTY_SCALE:.1f} * {profile.get('drawdown_sensitivity',5):.1f} / {rc.get('desperation_factor',1.0):.1f} = {rc.get('mdd_penalty',0):.2f}"
                # ---- Volatility component ----
                vol_detail = f"Vol = {m.get('Annual_Vol',0):.4f} * {VOL_PENALTY_SCALE:.1f} * {profile.get('volatility_sensitivity',5):.1f} / {rc.get('desperation_factor',1.0):.1f} = {rc.get('vol_penalty',0):.2f}"
                # ---- Diversity component ----
                div_detail = f"Div = {rc.get('diversity_penalty',0):.2f}"
                # ---- Assemble reward equation ----
                reward_eq = f"Reward = {rc.get('return_score',0):+.2f} (Ret) + {gfr_contrib:+.2f} (GFR) - {rc.get('mdd_penalty',0):.2f} (MDD) - {rc.get('vol_penalty',0):.2f} (Vol) - {rc.get('diversity_penalty',0):.2f} (Div)"
                print(f"    -> Math: {reward_eq} (Desp: {rc.get('desperation_factor',1.0):.1f})")
                print(f"       Details: {ret_detail} | {gfr_detail} | {mdd_detail} | {vol_detail} | {div_detail}")
                print(f"    -> Metric: ETV=${m.get('ETV',0):,.0f} | GFR={m.get('GFR',0):.4f} | MDD={m.get('MDD',0):.4f} | Vol={m.get('Annual_Vol',0):.4f}")
                
                # Top Assets
                def _get_top_str(fw):
                    idx = np.where(fw > 0)[0]
                    if len(idx) == 0: return "None"
                    top_idx = idx[np.argsort(fw[idx])][::-1] # All non-zero, sorted
                    tickers_list = clean_returns.columns
                    return " | ".join([f"{tickers_list[i]} ({fw[i]*100:.1f}%, mu={asset_means[i]:.1%}, vol={asset_vols[i]:.1%})" for i in top_idx])
                
                print(f"    -> Allocation: {_get_top_str(full_pre)}")

        elapsed = time.time() - t0
        print(f"\n  {'='*60}")
        print(f"  Training complete. {len(history)} iterations in {elapsed:.0f}s")
        print(f"  Final checkpoint: {checkpoint_path}")

    except Exception:
        traceback.print_exc()
    finally:
        log_file.close()


def train_rl_agent(dataset, user_profile, config, existing_agent=None, verbose=True):
    """
    Orchestrates the training of an ensemble of RL agents in parallel processes.

    Each agent runs in its own process with zero IPC overhead:
    1. Shared data is saved to a temp pickle file (one-time)
    2. Each process loads from the pickle, builds its own sim cache, trains
    3. Each process writes logs to logs/agent_N.log
    4. After completion, models are loaded from per-agent checkpoint files
    """
    import multiprocessing
    import pickle

    # ── Pre-compute invariant features (ONCE) ──────────────────────────────
    emb_matrix, static_matrix, all_tickers = _precompute_rl_features(dataset, config)
    user_cond_dim = len(_encode_user_condition(TEST_PROFILES[0]))
    input_dim = emb_matrix.shape[1] + user_cond_dim + static_matrix.shape[1]

    ensemble_size = config.get("rl_ensemble_size", 1)
    
    # ── Prepare shared data pickle ─────────────────────────────────────────
    base_returns = dataset.get("drip_daily_returns") if dataset.get("drip_daily_returns") is not None else dataset["daily_returns"]
    cache_dir = config.get("rl_cache_dir", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    data_pickle_path = os.path.join(cache_dir, "_rl_shared_data.pkl")
    
    if verbose:
        print(f"[{time.strftime('%H:%M:%S')}] [Member D] Saving shared data to {data_pickle_path}...")
    with open(data_pickle_path, "wb") as f:
        pickle.dump({
            "emb_matrix": emb_matrix,
            "static_matrix": static_matrix,
            "all_tickers": all_tickers,
            "base_returns": base_returns,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # ── Prepare log directory ──────────────────────────────────────────────
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    # Clear existing log files
    for i in range(ensemble_size):
        log_path = os.path.join(log_dir, f"agent_{i}.log")
        if os.path.exists(log_path):
            os.remove(log_path)
    
    if verbose:
        print(f"[{time.strftime('%H:%M:%S')}] [Member D] Launching {ensemble_size} independent agent processes...")
        print(f"  -> Logs: {log_dir}/agent_{{0..{ensemble_size-1}}}.log")
        print(f"  -> Monitor with: Get-Content logs/agent_0.log -Wait")
    
    # ── Spawn processes ────────────────────────────────────────────────────
    processes = []
    for i in range(ensemble_size):
        p = multiprocessing.Process(
            target=_agent_process_entry,
            args=(i, data_pickle_path, config, input_dim, log_dir),
            name=f"Agent-{i}"
        )
        p.start()
        processes.append(p)
        if verbose:
            print(f"  -> Agent {i} started (PID {p.pid})")
    
    # ── Wait for all to finish ─────────────────────────────────────────────
    for p in processes:
        p.join()
    
    if verbose:
        print(f"[{time.strftime('%H:%M:%S')}] [Member D] All {ensemble_size} agents finished.")
    
    # Check for failures
    for i, p in enumerate(processes):
        if p.exitcode != 0:
            log_path = os.path.join(log_dir, f"agent_{i}.log")
            print(f"  WARNING: Agent {i} exited with code {p.exitcode}. Check {log_path}")

    # ── Load trained models from per-agent checkpoints ─────────────────────
    fname_base = get_rl_transformer_filename_base(config, input_dim)
    agents = []
    history = []
    for i in range(ensemble_size):
        ckpt_path = os.path.join(cache_dir, f"checkpoint_{fname_base}_agent_{i}.pt")
        agent = PortfolioTransformerRL(input_dim, config).to(DEVICE)
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            agent.load_state_dict(ckpt['model_state'])
            if i == 0:
                history = ckpt.get('history', [])
            if verbose:
                ep = ckpt.get('episode', 0)
                print(f"  -> Agent {i}: loaded checkpoint at iteration {ep}")
        else:
            if verbose:
                print(f"  WARNING: No checkpoint found for agent {i} at {ckpt_path}")
        agents.append(agent)
    
    # ── Clean up temp pickle ───────────────────────────────────────────────
    try:
        os.remove(data_pickle_path)
    except OSError:
        pass

    # ── Final Greedy Evaluation ────────────────────────────────────────────
    pre_weights, post_weights = _greedy_evaluate_fast(agents, emb_matrix, static_matrix, all_tickers, user_profile)

    return {
        "portfolio_weights": pre_weights,
        "post_goal_weights": post_weights,
        "training_history": history,
        "agent": agents,
    }

