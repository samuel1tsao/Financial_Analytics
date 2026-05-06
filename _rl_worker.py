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
        print(f"  Building simulation cache...")
        cache_array, start_idx_to_pos, clean_returns, column_to_idx = build_simulation_cache(base_returns, max_horizon_years=30)
        sim_cache = (cache_array, start_idx_to_pos)
        sim_start_pool = np.array(_resolve_simulation_starts(clean_returns))
        num_columns = len(clean_returns.columns)
        valid_col_indices = np.array([column_to_idx.get(t, -1) for t in all_tickers])
        valid_ticker_mask = valid_col_indices >= 0
        is_active_matrix = base_returns.notna().values
        clean_returns_values = clean_returns.values
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
            print(f"  Resume Mode: Loading checkpoint...")
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
        verbose_every = config.get("rl_verbose_every", 10)
        check_freq    = config.get("rl_checkpoint_frequency", 10)
        conv_window   = config.get("rl_convergence_window", 500)
        conv_patience = config.get("rl_convergence_patience", 5000)
        conv_min_delta = config.get("rl_convergence_min_delta", 0.5)
        early_stopping = config.get("rl_early_stopping", False)

        single_goal_profiles = decompose_profiles(TEST_PROFILES)

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

            path_indices = rng.choice(all_starts, size=n_paths, replace=False)
            safe_col_indices = np.where(sub_ticker_mask, sub_col_indices, 0)
            src_key_padding_mask = ~torch.from_numpy(
                is_active_matrix[path_indices[0:1]][:, safe_col_indices]
            ).to(device)

            (mu_pre, sigma_pre), (mu_post, sigma_post) = agent(tensor_x, src_key_padding_mask=src_key_padding_mask)
            dist_pre  = torch.distributions.Normal(mu_pre,  F.softplus(sigma_pre))
            dist_post = torch.distributions.Normal(mu_post, F.softplus(sigma_post))
            samp_pre  = dist_pre.rsample()
            samp_post = dist_post.rsample()

            weights_pre  = _softmax_normalize_top_k(samp_pre,  sub_ticker_mask)
            weights_post = _softmax_normalize_top_k(samp_post, sub_ticker_mask)
            w_pre_np  = weights_pre[0].detach().cpu().numpy()
            w_post_np = weights_post[0].detach().cpu().numpy()

            def _build_full_weights(w_np):
                vi = np.where(sub_ticker_mask)[0]
                if len(vi) == 0: return np.zeros(num_columns)
                k = min(TOP_K_ASSETS, len(vi))
                top_k_local = vi[np.argsort(w_np[vi])[-k:]]
                top_w = w_np[top_k_local].copy()
                if top_w.sum() > 0: top_w /= top_w.sum()
                keep = top_w > MIN_ACTIVE_WEIGHT
                if not keep.any(): keep[:] = True
                sel = top_k_local[keep]; sel_w = top_w[keep]
                cp = sub_col_indices[sel]
                fw = np.zeros(num_columns); fw[cp] = sel_w
                if fw.sum() > 0: fw /= fw.sum()
                return fw

            full_pre  = _build_full_weights(w_pre_np)
            full_post = _build_full_weights(w_post_np)

            reward, metrics = _run_agent_simulations_batch(
                full_pre, full_post, path_indices, sim_cache, clean_returns_values, profile
            )

            if it == start_iter:
                baseline = float(reward)
            else:
                baseline = 0.99 * baseline + 0.01 * float(reward)

            lp_sum = dist_pre.log_prob(samp_pre).mean() + dist_post.log_prob(samp_post).mean()
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
                torch.save({
                    'episode': it + 1,
                    'model_state': agent.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'history': history,
                    'baseline': baseline,
                    'best_rolling_reward': best_rolling_reward,
                    'patience_counter': patience_counter,
                }, checkpoint_path)

            # Early stopping
            if patience_counter >= conv_patience:
                if not converged:
                    converged = True
                    print(f"\n  CONVERGENCE DETECTED at iteration {it+1:,}")
                if early_stopping:
                    break

            # Logging
            if it % verbose_every == 0 or it == start_iter + iterations - 1:
                elapsed = time.time() - t0
                ips = (it - start_iter + 1) / elapsed if elapsed > 0 else 0
                gs = f"Y{profile.get('goal_year','?')}: ${profile.get('goal_amount',0):,.0f}"
                print(f"  [Iter {it+1:7d}] Reward: {ep_reward:7.3f} | Baseline: {baseline:7.3f} | {ips:.1f} it/s | {profile['profile_name']} [{gs}]")
                
                # Detailed Metrics Breakdown
                m = metrics
                rc = m.get("reward_components", {})
                math_str = (
                    f"Return: {rc.get('return_score',0):+.2f} | "
                    f"MDD_Pen: {rc.get('mdd_penalty',0):.2f} | "
                    f"GFR_Contrib: {rc.get('gfr_bonus',0)-rc.get('gfr_penalty',0):+.2f}"
                )
                print(f"    -> Math: {math_str} | GFR={m.get('GFR',0):.4f}, MDD={m.get('MDD',0):.4f}")
                
                # Top Assets
                def _get_top_str(fw):
                    idx = np.where(fw > 0)[0]
                    if len(idx) == 0: return "None"
                    top_idx = idx[np.argsort(fw[idx])[-5:]][::-1] # Top 5
                    tickers_list = clean_returns.columns
                    return ", ".join([f"{tickers_list[i]}: {fw[i]*100:.1f}%" for i in top_idx])
                
                print(f"    -> Assets (Pre):  {_get_top_str(full_pre)}")
                print(f"    -> Assets (Post): {_get_top_str(full_post)}")

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

