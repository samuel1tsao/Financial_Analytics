"""
_rl_worker.py
─────────────
Member D: RL-Driven Transformer Portfolio Optimizer.

Architecture:
    PortfolioTransformerRL — Transformer with dual Gaussian policy heads:
        - Pre-goal head:  allocation before the goal withdrawal year
        - Post-goal head: growth-phase allocation after the goal is met/missed
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
    RISK_NORMALIZER,
    GOAL_YEAR_NORMALIZER,
    SIM_TERMINAL_HORIZON,
    WF_SNAPSHOTS_PER_EPISODE,
    MIN_ACTIVE_WEIGHT,
    TOP_K_ASSETS,
    RL_ASSET_SUBSET_SIZE,
    decompose_profiles,
)
from _sim_worker import (
    simulate_rl_environment_step,
    build_simulation_cache,
    _resolve_simulation_starts,
    _run_single_sim_path,
    _aggregate_path_results,
    _compute_reward,
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
    Transformer with dual Gaussian policy heads for two-phase allocation.

    Both heads share the same transformer backbone (cross-asset attention),
    but produce independent (mu, sigma) outputs:
        - Pre-goal head:  conservative allocation to fund near-term goals
        - Post-goal head: aggressive growth allocation after goal withdrawal
    """
    def __init__(self, input_dim, config):
        super().__init__()

        self.d_model = config.get("rl_d_model", 64)
        nhead          = config.get("rl_nhead", 4)
        num_layers     = config.get("rl_num_encoder_layers", 2)
        dim_feedforward = config.get("rl_dim_feedforward", 256)
        dropout        = config.get("rl_dropout", 0.1)

        # Project input features to transformer dimension
        self.input_proj = nn.Linear(input_dim, self.d_model)

        # Transformer encoder for cross-asset attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        initial_bias  = config.get("rl_initial_mu_bias", -0.1)
        initial_sigma = config.get("rl_initial_sigma_bias", 1.0)

        # Pre-goal policy head (may learn conservative allocations for short horizons)
        self.mu_head_pre    = nn.Linear(self.d_model, 1)
        self.sigma_head_pre = nn.Linear(self.d_model, 1)
        nn.init.constant_(self.mu_head_pre.bias, initial_bias)
        nn.init.constant_(self.sigma_head_pre.bias, initial_sigma)

        # Post-goal policy head (growth-phase allocation after goal is met/missed)
        self.mu_head_post    = nn.Linear(self.d_model, 1)
        self.sigma_head_post = nn.Linear(self.d_model, 1)
        nn.init.constant_(self.mu_head_post.bias, initial_bias)
        nn.init.constant_(self.sigma_head_post.bias, initial_sigma)

    def forward(self, x, src_key_padding_mask=None):
        """
        x: (batch, num_assets, input_dim)
        Returns: (mu_pre, sigma_pre), (mu_post, sigma_post)
            each mu/sigma is (batch, num_assets)
        """
        h   = F.relu(self.input_proj(x))
        out = self.transformer(h, src_key_padding_mask=src_key_padding_mask)

        mu_pre    = self.mu_head_pre(out).squeeze(-1)
        sigma_pre = F.softplus(self.sigma_head_pre(out).squeeze(-1)) + 0.5

        mu_post    = self.mu_head_post(out).squeeze(-1)
        sigma_post = F.softplus(self.sigma_head_post(out).squeeze(-1)) + 0.5

        return (mu_pre, sigma_pre), (mu_post, sigma_post)


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


def _run_agent_simulations_batch(w_pre, w_post, path_indices, sim_cache, clean_returns, profile):
    """
    Joblib worker function to run all requested simulation paths for a single agent,
    aggregate the results, and return just the final metrics and reward.
    This dramatically reduces IPC serialization overhead by passing the weight 
    vectors only once per agent, rather than once per path.
    """
    agent_res = []
    goal_year = profile.get('goal_year')
    for idx in path_indices:
        # Note: clean_returns here is a NumPy array (clean_returns.values)
        res = _run_single_sim_path(
            idx, w_pre, sim_cache, clean_returns, profile['start_cap'], 
            profile['goals'], "loop", SIM_TERMINAL_HORIZON, False, "",
            post_goal_weights=w_post, goal_year=goal_year
        )
        agent_res.append(res)
        
    gfr, etv, mdd, aw, goal_f, market_f = _aggregate_path_results(agent_res)
    metrics = {
        "GFR": gfr, "ETV": etv, "MDD": mdd, "AW": aw,
        "GoalFails": goal_f, "MarketFails": market_f,
        "Total_Simulations": len(path_indices)
    }
    reward = _compute_reward(metrics, profile)
    return reward, metrics


# ══════════════════════════════════════════════════════════════════════════════
# Feature Engineering
# ══════════════════════════════════════════════════════════════════════════════

def _encode_user_condition(user_profile):
    """
    Encode a single-goal user profile into a 4-dim condition vector.
    [risk_normalized, capital_normalized, goal_year_normalized, goal_ratio]

    For single-goal profiles (from decompose_profiles): uses goal_year and goal_amount.
    For legacy multi-goal profiles: uses the latest goal year and total goal amount.
    """
    risk_val  = float(user_profile.get("risk_tolerance", 5.0)) / RISK_NORMALIZER
    start_cap = float(user_profile.get("start_cap", 100000.0))

    # Single-goal profile (preferred path)
    if "goal_year" in user_profile:
        goal_year   = float(user_profile["goal_year"]) / GOAL_YEAR_NORMALIZER
        goal_ratio  = float(user_profile["goal_amount"]) / start_cap
    else:
        # Legacy fallback: use max goal year and total amount
        goals = user_profile.get("goals", {})
        goal_year   = float(max(goals.keys())) / GOAL_YEAR_NORMALIZER if goals else 0.0
        goal_ratio  = float(sum(goals.values())) / start_cap if goals else 0.0

    return [risk_val, start_cap / CAPITAL_NORMALIZER, goal_year, goal_ratio]


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
    Layout per asset: [embedding | user_condition(33) | static_features]
    Returns tensor of shape (1, N_assets, input_dim) on DEVICE.
    """
    user_vec   = np.array(_encode_user_condition(user_profile), dtype=np.float32)
    N          = emb_matrix.shape[0]
    user_tiled = np.tile(user_vec, (N, 1))   # (N, 33)

    # [emb | user_condition | static] — matches original build_rl_dataset ordering
    full = np.concatenate([emb_matrix, user_tiled, static_matrix], axis=1)
    return torch.tensor(full[np.newaxis, :, :], dtype=torch.float32, device=DEVICE)


def build_rl_dataset(dataset, user_profile, config, specific_embeddings=None):
    """
    Assemble the input tensor for the Transformer (legacy per-call version).
    Each asset row = [Embedding | UserCondition(33) | StaticFeatures].
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
    Updated for dual-head architecture (pre-goal + post-goal weights).
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

        (mu_pre, sigma_pre), (mu_post, sigma_post) = agent(pit_x)
        weights_pre, lp_pre   = sample_portfolio_weights(mu_pre, sigma_pre)
        weights_post, lp_post = sample_portfolio_weights(mu_post, sigma_post)

        reward, metrics = simulate_rl_environment_step(
            weights_pre.detach().cpu().numpy(), pit_tickers,
            dataset, profile, config, start_date=date_str, debug_path=debug_path,
            sim_cache_bundle=sim_cache_bundle
        )
        rewards.append(reward)
        log_probs.append(lp_pre.sum() + lp_post.sum())
        last_metrics = metrics
        last_weights = weights_pre.detach().cpu().numpy()
        last_tickers = pit_tickers

    return rewards, log_probs, last_metrics, last_weights, last_tickers


def _greedy_evaluate_fast(agents, emb_matrix, static_matrix, tickers, user_profile):
    """Run a deterministic (greedy) evaluation using ensemble average. Returns both phase dicts."""
    if not isinstance(agents, list):
        agents = [agents]
        
    for a in agents:
        a.eval()
        
    with torch.no_grad():
        tensor_x = _build_rl_input_fast(emb_matrix, static_matrix, user_profile)
        mu_pre_sum, mu_post_sum = 0, 0
        for a in agents:
            (mu_pre, _), (mu_post, _) = a(tensor_x)
            mu_pre_sum += mu_pre
            mu_post_sum += mu_post
            
        mu_pre_avg = mu_pre_sum / len(agents)
        mu_post_avg = mu_post_sum / len(agents)
        
        greedy_pre  = F.softmax(mu_pre_avg, dim=-1)
        greedy_post = F.softmax(mu_post_avg, dim=-1)
        
    pre_dict  = {tickers[i]: float(greedy_pre[0, i]) for i in range(len(tickers))}
    post_dict = {tickers[i]: float(greedy_post[0, i]) for i in range(len(tickers))}
    return pre_dict, post_dict


def get_rl_transformer_filename_base(config, input_dim):
    """Generate a unique identifier for the RL agent's architecture (v2 = dual-head)."""
    dm = config.get("rl_d_model", 64)
    nh = config.get("rl_nhead", 4)
    lr = config.get("rl_learning_rate", 0.0001)
    lr_clean = str(lr).replace("0.", "").replace(".", "")
    return f"rl_v2_dm{dm}_nh{nh}_lr{lr_clean}_id{input_dim}"


def train_rl_agent(dataset, user_profile, config, existing_agent=None, verbose=True):
    """
    Stochastic REINFORCE training with per-step gradient updates.

    Each iteration:
        1. Sample a random user profile
        2. Build input tensor from cached features (fast numpy concat)
        3. Forward pass through transformer (all assets)
        4. Sample portfolio weights from Gaussian policy
        5. Run a few random sim paths → compute reward
        6. Gradient update (loss.backward + optimizer.step)

    Key optimizations over the old per-episode loop:
        - Feature matrices pre-computed once (eliminates pandas .loc per ticker per step)
        - Sim start pool pre-resolved once (eliminates pandas resample per step)
        - Dense weight mapping pre-built (eliminates dict↔array conversion per step)
        - Fewer sim paths per gradient update (rl_paths_per_step, default 5)
    """
    # ── Pre-compute invariant features (ONCE) ──────────────────────────────
    emb_matrix, static_matrix, all_tickers = _precompute_rl_features(dataset, config)
    user_cond_dim = len(_encode_user_condition(TEST_PROFILES[0]))
    input_dim = emb_matrix.shape[1] + user_cond_dim + static_matrix.shape[1]

    # Initialize or warm-start ensemble of agents
    ensemble_size = config.get("rl_ensemble_size", 1)
    agents = []
    optimizers = []
    
    if existing_agent is not None:
        agents = existing_agent if isinstance(existing_agent, list) else [existing_agent]
        while len(agents) < ensemble_size:
            agents.append(PortfolioTransformerRL(input_dim, config).to(DEVICE))
    else:
        agents = [PortfolioTransformerRL(input_dim, config).to(DEVICE) for _ in range(ensemble_size)]
        
    lr = config.get("rl_learning_rate", 0.0001)
    for a in agents:
        optimizers.append(optim.Adam(a.parameters(), lr=lr))
        
    iterations = config.get("rl_episodes", 100)

    # ── Checkpoint Logic ───────────────────────────────────────────────────
    cache_dir = config.get("rl_cache_dir", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    fname_base = get_rl_transformer_filename_base(config, input_dim)
    checkpoint_path = os.path.join(cache_dir, f"checkpoint_{fname_base}.pt")

    force_rebuild = config.get("rl_force_rebuild", False)
    start_iter = 0
    history = []
    baseline = 0.0
    best_rolling_reward = -float('inf')
    patience_counter = 0
    converged = False

    if force_rebuild and os.path.exists(checkpoint_path):
        if verbose:
            print(f"  [Member D] Restart Mode: Deleting existing checkpoint '{checkpoint_path}'...")
        os.remove(checkpoint_path)
    elif not force_rebuild and existing_agent is None and os.path.exists(checkpoint_path):
        if verbose:
            print(f"  [Member D] Resume Mode: Loading checkpoint from '{checkpoint_path}'...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
            if 'agents_state' in checkpoint:
                for i in range(min(len(agents), len(checkpoint['agents_state']))):
                    agents[i].load_state_dict(checkpoint['agents_state'][i])
                    optimizers[i].load_state_dict(checkpoint['optimizers_state'][i])
            else:
                agents[0].load_state_dict(checkpoint['model_state'])
                optimizers[0].load_state_dict(checkpoint['optimizer_state'])
            start_iter = checkpoint.get('episode', 0)
            history = checkpoint.get('history', [])
            baseline = checkpoint.get('baseline', 0.0)
            best_rolling_reward = checkpoint.get('best_rolling_reward', -float('inf'))
            patience_counter = checkpoint.get('patience_counter', 0)
            if verbose:
                print(f"  [Member D] Successfully resumed from iteration {start_iter}.")
        except Exception as e:
            if verbose:
                print(f"  [WARNING] Failed to load checkpoint: {e}. Starting from scratch.")

    # ── Build Simulation Cache (ONCE) ──────────────────────────────────────
    base_returns = dataset.get("drip_daily_returns") if dataset.get("drip_daily_returns") is not None else dataset["daily_returns"]
    cache_array, start_idx_to_pos, clean_returns, column_to_idx = build_simulation_cache(base_returns, max_horizon_years=30)
    sim_cache = (cache_array, start_idx_to_pos)

    # Pre-resolve simulation start pool (ONCE)
    sim_start_pool = np.array(_resolve_simulation_starts(clean_returns))

    # Pre-build dense weight column mapping (ONCE)
    num_columns     = len(clean_returns.columns)
    valid_col_indices = np.array([column_to_idx.get(t, -1) for t in all_tickers])
    valid_ticker_mask = valid_col_indices >= 0
    
    # Pre-compute existence matrix (True if asset exists/traded on that day)
    is_active_matrix = base_returns.notna().values

    n_paths      = config.get("rl_paths_per_step", 5)
    verbose_every = config.get("rl_verbose_every", 10)
    check_freq   = config.get("rl_checkpoint_frequency", 10)

    # Convergence detection parameters
    conv_window    = config.get("rl_convergence_window", 500)
    conv_patience  = config.get("rl_convergence_patience", 5000)
    conv_min_delta = config.get("rl_convergence_min_delta", 0.5)
    early_stopping = config.get("rl_early_stopping", False)

    # Optionally load walk-forward snapshots (legacy path)
    wf_enabled   = config.get("wf_enabled", False)
    wf_snapshots = {}
    if wf_enabled:
        from _ml_worker import load_all_walkforward_snapshots
        wf_snapshots = load_all_walkforward_snapshots(config)

    # Decompose multi-goal profiles into single-goal episodes
    single_goal_profiles = decompose_profiles(TEST_PROFILES)

    if verbose:
        mode       = "WALK-FORWARD" if wf_enabled else "STOCHASTIC REINFORCE (TWO-PHASE)"
        subset_str = f"{RL_ASSET_SUBSET_SIZE} assets/iter (sampled)" if RL_ASSET_SUBSET_SIZE else f"{len(all_tickers)} assets"
        print(f"[{time.strftime('%H:%M:%S')}] [Member D] {mode} training | "
              f"{len(single_goal_profiles)} single-goal profiles (from {len(TEST_PROFILES)} base) "
              f"| {subset_str} | {n_paths} paths/step | {iterations} iterations")

    # ── Main Training Loop ─────────────────────────────────────────────────
    t0 = time.time()
    rng = np.random.default_rng()
    all_starts = sim_start_pool

    for it in range(start_iter, start_iter + iterations):
        # 1. Sample a random single-goal profile
        profile = single_goal_profiles[np.random.randint(len(single_goal_profiles))]
        goal_year = profile["goal_year"]

        # ── Walk-Forward path ──────────
        if wf_enabled and wf_snapshots:
            sim_cache_bundle = (sim_cache, clean_returns, column_to_idx)
            ensemble_rewards = []
            ensemble_all_metrics = []
            ensemble_all_weights = []
            
            n_cores = config.get("rl_parallel_cores", -1)
            from joblib import Parallel, delayed
            
            def run_agent_wf(a_idx):
                agent = agents[a_idx]
                return _run_episode_walkforward(
                    agent, dataset, profile, config, wf_snapshots,
                    sim_cache_bundle=sim_cache_bundle
                )

            try:
                wf_results = Parallel(n_jobs=n_cores)(delayed(run_agent_wf)(i) for i in range(len(agents)))
            except (OSError, Exception) as e:
                # Fallback for WinError 1450 or PicklingError
                if verbose: print(f"  [Parallel] Multiprocessing failed: {e}. Falling back to threading...")
                wf_results = Parallel(n_jobs=n_cores, backend="threading")(delayed(run_agent_wf)(i) for i in range(len(agents)))

            for agent_idx, (rewards, log_probs, metrics, weights_np, ep_tickers) in enumerate(wf_results):
                # rewards is list of floats, log_probs is list of tensors
                avg_reward = np.mean(rewards)
                lp_sum = torch.stack(log_probs).sum()
                loss = -lp_sum * avg_reward
                
                optimizers[agent_idx].zero_grad()
                loss.backward()
                optimizers[agent_idx].step()
                
                ensemble_rewards.append(float(avg_reward))
                ensemble_all_metrics.append(metrics)
                ensemble_all_weights.append(weights_np)
                
            ep_reward = np.mean(ensemble_rewards) if ensemble_rewards else 0.0
            log_ensemble_rewards = ensemble_rewards
            log_ensemble_metrics = ensemble_all_metrics
            log_ensemble_weights = ensemble_all_weights
            metrics = log_ensemble_metrics[0] if log_ensemble_metrics else {}

        # ── Standard path (fast, batched dynamic masking) ──────────────────
        else:
            for a in agents: a.train()
            # 2.5 Randomly subsample the asset universe
            if RL_ASSET_SUBSET_SIZE is not None and RL_ASSET_SUBSET_SIZE < len(all_tickers):
                subset_idx       = np.sort(rng.choice(len(all_tickers), size=RL_ASSET_SUBSET_SIZE, replace=False))
                sub_emb, sub_static = emb_matrix[subset_idx], static_matrix[subset_idx]
                sub_col_indices, sub_ticker_mask = valid_col_indices[subset_idx], valid_ticker_mask[subset_idx]
            else:
                subset_idx, sub_emb, sub_static = np.arange(len(all_tickers)), emb_matrix, static_matrix
                sub_col_indices, sub_ticker_mask = valid_col_indices, valid_ticker_mask

            # 3. Build batched input tensor
            base_x   = _build_rl_input_fast(sub_emb, sub_static, profile)
            tensor_x = base_x.expand(1, -1, -1)
            
            # Mask generation for Agent 0 (shared mask for all agents in this step)
            path_indices = rng.choice(all_starts, size=config.get("rl_paths_per_step", 5), replace=False)
            safe_col_indices = np.where(sub_ticker_mask, sub_col_indices, 0)
            # Use only first path for attention mask simplification (shared regime)
            src_key_padding_mask = ~torch.from_numpy(is_active_matrix[path_indices[0:1]][:, safe_col_indices]).to(DEVICE)

            # 5. Ensemble Forward Pass and Task Preparation
            all_agent_inputs = []
            ensemble_rewards = []
            ensemble_all_metrics = []
            ensemble_all_weights = []

            for agent_idx, agent in enumerate(agents):
                (mu_pre, sigma_pre), (mu_post, sigma_post) = agent(tensor_x, src_key_padding_mask=src_key_padding_mask)
                dist_pre  = torch.distributions.Normal(mu_pre,  F.softplus(sigma_pre))
                dist_post = torch.distributions.Normal(mu_post, F.softplus(sigma_post))
                samp_pre  = dist_pre.rsample()
                samp_post = dist_post.rsample()
                
                weights_pre  = _softmax_normalize_top_k(samp_pre,  sub_ticker_mask)
                weights_post = _softmax_normalize_top_k(samp_post, sub_ticker_mask)
                
                # Single weight array for simulation/logging (shape 1, N_subset -> N_subset)
                w_pre_np  = weights_pre[0].detach().cpu().numpy()
                w_post_np = weights_post[0].detach().cpu().numpy()
                
                # Helper to prune to TOP_K and apply MIN_ACTIVE_WEIGHT
                # Weight array must be sized to num_columns (clean_returns width)
                # to match the simulation cache precomputed_returns shape.
                def _build_full_weights(w_np):
                    valid_indices = np.where(sub_ticker_mask)[0]
                    if len(valid_indices) == 0:
                        return np.zeros(num_columns)
                    k = min(TOP_K_ASSETS, len(valid_indices))
                    top_k_local = valid_indices[np.argsort(w_np[valid_indices])[-k:]]
                    
                    # Normalized Top-K
                    top_w = w_np[top_k_local].copy()
                    if top_w.sum() > 0:
                        top_w /= top_w.sum()
                    
                    # Thresholding
                    keep = top_w > MIN_ACTIVE_WEIGHT
                    if not keep.any():
                        keep[:] = True
                    
                    selected = top_k_local[keep]
                    selected_w = top_w[keep]
                    # Map: subset local idx → sub_col_indices → column in clean_returns
                    col_positions = sub_col_indices[selected]
                    fw = np.zeros(num_columns)
                    fw[col_positions] = selected_w
                    if fw.sum() > 0:
                        fw /= fw.sum()
                    return fw

                full_pre  = _build_full_weights(w_pre_np)
                full_post = _build_full_weights(w_post_np)
                
                # Gradients (mean across sample paths if P>1, here P=1)
                lp_sum = dist_pre.log_prob(samp_pre).mean() + dist_post.log_prob(samp_post).mean()
                all_agent_inputs.append((full_pre, full_post, lp_sum, w_pre_np))

                if agent_idx == 0:
                    log_ticker_mask  = sub_ticker_mask
                    log_subset_idx   = subset_idx

            # 6. Parallel Simulation (Per-Agent Batching)
            from joblib import Parallel, delayed
            n_cores = config.get("rl_parallel_cores", -1)
            
            try:
                agent_results = Parallel(n_jobs=n_cores)(
                    delayed(_run_agent_simulations_batch)(
                        all_agent_inputs[a_idx][0], # w_pre
                        all_agent_inputs[a_idx][1], # w_post
                        path_indices, sim_cache, clean_returns.values, profile
                    ) for a_idx in range(len(agents))
                )
            except (OSError, Exception) as e:
                if verbose: print(f"  [Parallel] Simulation failed: {e}. Falling back to threading...")
                agent_results = Parallel(n_jobs=n_cores, backend="threading")(
                    delayed(_run_agent_simulations_batch)(
                        all_agent_inputs[a_idx][0], # w_pre
                        all_agent_inputs[a_idx][1], # w_post
                        path_indices, sim_cache, clean_returns.values, profile
                    ) for a_idx in range(len(agents))
                )

            # 7. Collect Rewards and Apply Gradients
            for a_idx in range(len(agents)):
                reward, metrics = agent_results[a_idx]

                
                # Global baseline update
                if it == start_iter and a_idx == 0:
                    baseline = float(reward)
                else:
                    baseline = 0.99 * baseline + 0.01 * float(reward)

                # Gradient Step
                lp_sum = all_agent_inputs[a_idx][2]
                loss = -lp_sum * (reward - baseline)
                optimizers[a_idx].zero_grad()
                loss.backward()
                optimizers[a_idx].step()
                
                ensemble_rewards.append(float(reward))
                ensemble_all_metrics.append(metrics)
                ensemble_all_weights.append(all_agent_inputs[a_idx][3])
            
            ep_reward = np.mean(ensemble_rewards) if ensemble_rewards else 0.0
            log_ensemble_rewards = ensemble_rewards
            log_ensemble_metrics = ensemble_all_metrics
            log_ensemble_weights = ensemble_all_weights
            metrics = log_ensemble_metrics[0] if log_ensemble_metrics else {}

        history.append({
            "episode": it, "reward": ep_reward,
            "GFR": metrics.get("GFR", 0.0),
            "ETV": metrics.get("ETV", 0.0),
            "MDD": metrics.get("MDD", 0.0),
        })

        # ── Convergence Detection ──────────────────────────────────────────
        if len(history) >= conv_window:
            recent_rewards = [h["reward"] for h in history[-conv_window:]]
            rolling_reward = sum(recent_rewards) / len(recent_rewards)
            if rolling_reward > best_rolling_reward + conv_min_delta:
                best_rolling_reward = rolling_reward
                patience_counter = 0
            else:
                patience_counter += 1

        # ── Periodic Checkpoint ────────────────────────────────────────────
        should_ckpt = (it + 1) % check_freq == 0 or it == start_iter + iterations - 1
        if should_ckpt or (patience_counter >= conv_patience and early_stopping):
            torch.save({
                'episode': it + 1,
                'agents_state': [a.state_dict() for a in agents],
                'optimizers_state': [opt.state_dict() for opt in optimizers],
                'history': history,
                'baseline': baseline,
                'best_rolling_reward': best_rolling_reward,
                'patience_counter': patience_counter,
            }, checkpoint_path)

        # ── Early Stopping ─────────────────────────────────────────────────
        if patience_counter >= conv_patience:
            if not converged:
                converged = True
                if verbose:
                    elapsed = time.time() - t0
                    print(f"\n  {'='*64}")
                    print(f"  CONVERGENCE DETECTED at iteration {it+1:,}")
                    print(f"    Rolling reward ({conv_window}-window): {rolling_reward:.3f}")
                    print(f"    Best rolling reward:  {best_rolling_reward:.3f}")
                    print(f"    No improvement for {conv_patience:,} iterations")
                    print(f"    Elapsed: {elapsed:.0f}s")
                    print(f"  {'='*64}")
            if early_stopping:
                if verbose:
                    print(f"  [Member D] Early stopping enabled — training halted.\n")
                break

        # ── Periodic Logging ───────────────────────────────────────────────
        if verbose and (it % verbose_every == 0 or it == start_iter + iterations - 1):
            elapsed = time.time() - t0
            iter_per_sec = (it - start_iter + 1) / elapsed if elapsed > 0 else 0

            goals_str = f"Y{profile.get('goal_year', '?')}: ${profile.get('goal_amount', 0):,.0f}"
            ens_str = ", ".join([f"{r:.2f}" for r in log_ensemble_rewards])
            print(f"\n  [Iter {it+1:7d}] Avg Reward: {ep_reward:7.3f} | Baseline: {baseline:7.3f} | "
                  f"{iter_per_sec:.1f} it/s")
            print(f"    -> Ensemble Rewards: [{ens_str}] | Profile: {profile['profile_name']}")
            print(f"    -> User Info: Risk={profile.get('risk_tolerance', 5.0)}/9.0 "
                  f"| Start Cap=${profile.get('start_cap', 0):,.0f} | Goal: [{goals_str}] | Horizon: 30y")

            for a_idx in range(len(agents)):
                a_reward  = log_ensemble_rewards[a_idx]
                a_metrics = log_ensemble_metrics[a_idx]
                a_weights = log_ensemble_weights[a_idx]

                if not wf_enabled:
                    log_active_mask = (a_weights > MIN_ACTIVE_WEIGHT) & log_ticker_mask
                    if log_active_mask.any():
                        comp = a_metrics.get("reward_components", {})

                        # Map subset indices back to full ticker list for display
                        sub_tickers = [all_tickers[i] for i in log_subset_idx]
                        active_assets = [(sub_tickers[i], a_weights[i])
                                         for i in range(len(sub_tickers)) if log_active_mask[i]]
                        active_assets.sort(key=lambda x: x[1], reverse=True)
                        top_str = ", ".join([f"{t}: {w:.2%}" for t, w in active_assets[:5]])
                        top_str += f" | (Total Active: {len(active_assets)})"

                        print(f"    [Agent {a_idx}] Allocation: {top_str}")
                        n_sims     = a_metrics.get('Total_Simulations', 1)
                        goal_f     = a_metrics.get('GoalFails', 0)
                        market_f   = a_metrics.get('MarketFails', 0)
                        fail_parts = []
                        if goal_f   > 0: fail_parts.append(f"{goal_f}/{n_sims} Goal Bankrupt")
                        if market_f > 0: fail_parts.append(f"{market_f}/{n_sims} Market Collapse")
                        fail_str = f" | Failures: {', '.join(fail_parts)}" if fail_parts else ""
                        print(f"      -> Simulation: ETV=${a_metrics.get('ETV',0):,.0f} "
                              f"| MDD={a_metrics.get('MDD',0):.1%} | GFR={a_metrics.get('GFR',0):.1%}{fail_str}")
                        gfr_part = f"+ GFR Bonus ({comp.get('gfr_bonus',0):.3f})" if comp.get('gfr_bonus',0) > 0 else f"- GFR Penalty ({comp.get('gfr_penalty',0):.3f})"
                        print(f"      -> Reward: Return ({comp.get('return_score',0):.3f}) "
                              f"- MDD ({a_metrics.get('MDD',0):.1%} drop → {comp.get('mdd_penalty',0):.3f}) "
                              f"{gfr_part} = {a_reward:.3f}")
                    else:
                        comp = a_metrics.get("reward_components", {})
                        gfr_part = f"+ GFR Bonus ({comp.get('gfr_bonus',0):.3f})" if comp.get('gfr_bonus',0) > 0 else f"- GFR Penalty ({comp.get('gfr_penalty',0):.3f})"
                        print(f"    [Agent {a_idx}] Allocation: ALL ASSETS PRUNED (Empty Portfolio)")
                        print(f"      -> Reward: Return ({comp.get('return_score',0):.3f}) "
                              f"- MDD (0.0% drop → {comp.get('mdd_penalty',0):.3f}) "
                              f"{gfr_part} = {a_reward:.3f}")
                else:
                    # Walk-forward logging simplified
                    print(f"    [Agent {a_idx}] Reward: {a_reward:.3f} | Metrics: {a_metrics}")

    # ── Final Greedy Evaluation ────────────────────────────────────────────
    pre_weights, post_weights = _greedy_evaluate_fast(agents, emb_matrix, static_matrix, all_tickers, user_profile)

    return {
        "portfolio_weights": pre_weights,
        "post_goal_weights": post_weights,
        "training_history": history,
        "agent": agents,
    }
