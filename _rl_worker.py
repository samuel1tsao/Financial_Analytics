"""
_rl_worker.py
─────────────
Member D: RL-Driven Transformer Portfolio Optimizer.

Architecture:
    PortfolioTransformerRL — Transformer that outputs Gaussian policy (mu, sigma)
    train_rl_agent         — Stochastic REINFORCE with per-step gradient updates

Optimizations:
    - Feature matrices pre-computed once (no per-step pandas lookups)
    - Simulation start pool pre-resolved once
    - Dense weight array built vectorized (no dict conversion)
    - Fewer sim paths per gradient update (rl_paths_per_step)
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
    GOAL_TIMELINE_SLOTS,
    CAPITAL_NORMALIZER,
    RISK_NORMALIZER,
    WF_SNAPSHOTS_PER_EPISODE,
    MIN_ACTIVE_WEIGHT,
    TOP_K_ASSETS,
    RL_ASSET_SUBSET_SIZE,
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
    Transformer that contextualizes all assets via self-attention,
    then outputs a Gaussian policy (mu, sigma) for each asset weight.
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

        # Policy heads
        self.mu_head    = nn.Linear(self.d_model, 1)
        self.sigma_head = nn.Linear(self.d_model, 1)
        initial_bias    = config.get("rl_initial_mu_bias", -0.1)
        initial_sigma   = config.get("rl_initial_sigma_bias", 1.0)
        nn.init.constant_(self.mu_head.bias, initial_bias)
        nn.init.constant_(self.sigma_head.bias, initial_sigma)

    def forward(self, x, src_key_padding_mask=None):
        """x: (batch, num_assets, input_dim) -> (mu, sigma) each (batch, num_assets)"""
        h   = F.relu(self.input_proj(x))
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


# ══════════════════════════════════════════════════════════════════════════════
# Feature Engineering
# ══════════════════════════════════════════════════════════════════════════════

def _encode_user_condition(user_profile):
    """
    Encode a user profile into a fixed-size condition vector (33 dims).
    [risk_normalized, capital_normalized, goal_timeline_0..30]
    """
    risk_val  = float(user_profile.get("risk_tolerance", 5.0)) / RISK_NORMALIZER
    start_cap = float(user_profile.get("start_cap", 100000.0))

    # Goal timeline: one slot per year (0–30), value = goal_amount / starting_capital
    goal_vec = [0.0] * GOAL_TIMELINE_SLOTS
    for year, amount in user_profile.get("goals", {}).items():
        if 0 <= year < GOAL_TIMELINE_SLOTS:
            goal_vec[int(year)] = float(amount / start_cap)

    return [risk_val, start_cap / CAPITAL_NORMALIZER] + goal_vec


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
        weights, log_prob = sample_portfolio_weights(mu, sigma)

        reward, metrics = simulate_rl_environment_step(
            weights.detach().cpu().numpy(), pit_tickers,
            dataset, profile, config, start_date=date_str, debug_path=debug_path,
            sim_cache_bundle=sim_cache_bundle
        )
        rewards.append(reward)
        log_probs.append(log_prob.sum())
        last_metrics = metrics
        last_weights = weights.detach().cpu().numpy()
        last_tickers = pit_tickers

    return rewards, log_probs, last_metrics, last_weights, last_tickers


def _greedy_evaluate_fast(agent, emb_matrix, static_matrix, tickers, user_profile):
    """Run a deterministic (greedy) evaluation using cached features."""
    agent.eval()
    with torch.no_grad():
        tensor_x = _build_rl_input_fast(emb_matrix, static_matrix, user_profile)
        mu, _ = agent(tensor_x)
        greedy_weights = F.softmax(mu, dim=-1)
    return {tickers[i]: float(greedy_weights[0, i]) for i in range(len(tickers))}


def get_rl_transformer_filename_base(config, input_dim):
    """Generate a unique identifier for the RL agent's architecture."""
    dm = config.get("rl_d_model", 64)
    nh = config.get("rl_nhead", 4)
    lr = config.get("rl_learning_rate", 0.0001)
    lr_clean = str(lr).replace("0.", "").replace(".", "")
    return f"rl_dm{dm}_nh{nh}_lr{lr_clean}_id{input_dim}"


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

    if verbose:
        print(f"  [Member D] Pre-computed features: {len(all_tickers)} assets, "
              f"input_dim={input_dim} "
              f"(emb={emb_matrix.shape[1]} + user={user_cond_dim} + static={static_matrix.shape[1]})")

    # Initialize or warm-start agent
    agent = existing_agent if existing_agent is not None else \
            PortfolioTransformerRL(input_dim, config).to(DEVICE)
    optimizer = optim.Adam(agent.parameters(), lr=config.get("rl_learning_rate", 0.0001))
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

    if force_rebuild and os.path.exists(checkpoint_path):
        if verbose:
            print(f"  [Member D] Restart Mode: Deleting existing checkpoint '{checkpoint_path}'...")
        os.remove(checkpoint_path)
    elif not force_rebuild and existing_agent is None and os.path.exists(checkpoint_path):
        if verbose:
            print(f"  [Member D] Resume Mode: Loading checkpoint from '{checkpoint_path}'...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            agent.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            start_iter = checkpoint.get('episode', 0)
            history = checkpoint.get('history', [])
            baseline = checkpoint.get('baseline', 0.0)
            if verbose:
                print(f"  [Member D] Successfully resumed from iteration {start_iter}.")
        except Exception as e:
            if verbose:
                print(f"  [WARNING] Failed to load checkpoint: {e}. Starting from scratch.")

    # ── Build Simulation Cache (ONCE) ──────────────────────────────────────
    base_returns = dataset.get("drip_daily_returns") if dataset.get("drip_daily_returns") is not None else dataset["daily_returns"]
    sim_cache, clean_returns, column_to_idx = build_simulation_cache(base_returns, max_horizon_years=30)

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

    # Optionally load walk-forward snapshots (legacy path)
    wf_enabled   = config.get("wf_enabled", False)
    wf_snapshots = {}
    if wf_enabled:
        from _ml_worker import load_all_walkforward_snapshots
        wf_snapshots = load_all_walkforward_snapshots(config)

    if verbose:
        mode       = "WALK-FORWARD" if wf_enabled else "STOCHASTIC REINFORCE"
        subset_str = f"{RL_ASSET_SUBSET_SIZE} assets/iter (sampled)" if RL_ASSET_SUBSET_SIZE else f"{len(all_tickers)} assets"
        print(f"[{time.strftime('%H:%M:%S')}] [Member D] {mode} training | {len(TEST_PROFILES)} profiles "
              f"| {subset_str} | {n_paths} paths/step | {iterations} iterations")

    # ── Main Training Loop ─────────────────────────────────────────────────
    t0 = time.time()

    for it in range(start_iter, start_iter + iterations):
        agent.train()
        optimizer.zero_grad()

        # 1. Sample a random user profile
        profile = TEST_PROFILES[np.random.randint(len(TEST_PROFILES))]

        # ── Walk-Forward path (legacy, uses old build_rl_dataset) ──────────
        if wf_enabled and wf_snapshots:
            sim_cache_bundle = (sim_cache, clean_returns, column_to_idx)
            rewards, log_probs, metrics, weights_np, ep_tickers = _run_episode_walkforward(
                agent, dataset, profile, config, wf_snapshots,
                debug_path=False, sim_cache_bundle=sim_cache_bundle
            )
            if rewards:
                loss = sum(-r * lp for r, lp in zip(rewards, log_probs)) / len(rewards)
                loss.backward()
                optimizer.step()
                ep_reward = float(np.mean(rewards))
            else:
                ep_reward = 0.0
                metrics = {}

        # ── Standard path (fast, batched dynamic masking) ──────────────────
        else:
            # 2. Sample N random sim paths
            path_indices = np.random.choice(
                sim_start_pool, size=min(n_paths, len(sim_start_pool)), replace=False
            )

            # 2.5 Randomly subsample the asset universe for this iteration.
            # This gives 13× stronger gradient signal per asset and forces the transformer
            # to learn general feature patterns rather than memorizing specific tickers.
            if RL_ASSET_SUBSET_SIZE is not None and RL_ASSET_SUBSET_SIZE < len(all_tickers):
                subset_idx       = np.random.choice(len(all_tickers), size=RL_ASSET_SUBSET_SIZE, replace=False)
                subset_idx       = np.sort(subset_idx)                   # keep asset order stable
                sub_emb          = emb_matrix[subset_idx]
                sub_static       = static_matrix[subset_idx]
                sub_col_indices  = valid_col_indices[subset_idx]         # map into clean_returns columns
                sub_ticker_mask  = valid_ticker_mask[subset_idx]         # which subset assets are valid
            else:
                subset_idx       = np.arange(len(all_tickers))
                sub_emb          = emb_matrix
                sub_static       = static_matrix
                sub_col_indices  = valid_col_indices
                sub_ticker_mask  = valid_ticker_mask

            # 3. Build batched input tensor (P, N_subset, input_dim)
            base_x   = _build_rl_input_fast(sub_emb, sub_static, profile)
            tensor_x = base_x.expand(len(path_indices), -1, -1)

            # 4. Extract existence mask for these P paths (True = NOT active/mask it)
            safe_col_indices = np.where(sub_ticker_mask, sub_col_indices, 0)
            active_mask_np   = is_active_matrix[path_indices][:, safe_col_indices]
            active_mask_np[:, ~sub_ticker_mask] = False
            src_key_padding_mask = ~torch.from_numpy(active_mask_np).to(DEVICE)

            # 5. Forward pass + sample weights (BATCHED over subset)
            mu, sigma = agent(tensor_x, src_key_padding_mask=src_key_padding_mask)
            weights, log_prob = sample_portfolio_weights(mu, sigma, mask=src_key_padding_mask)

            # 6. Run sims and collect rewards
            results = []
            max_horizon = max(profile['goals'].keys()) if profile['goals'] else 1
            
            for i, idx in enumerate(path_indices):
                weights_np_i = weights[i].detach().cpu().numpy()
                full_weights  = np.zeros(num_columns)

                # Two-stage selection over the SUBSET valid assets.
                # sub_ticker_mask tells us which subset assets are valid (IPO'd) for any path.
                # Per-path IPO status comes from src_key_padding_mask; here we use the
                # conservative global sub_ticker_mask for weight mapping.
                valid_indices    = np.where(sub_ticker_mask)[0]          # indices into subset space
                if len(valid_indices) == 0:
                    results.append({"bankrupt": True, "bankrupt_reason": "empty",
                                    "terminal_value": 0.0, "max_drawdown": 0.0,
                                    "actual_withdrawals": 0.0, "log": ""})
                    continue

                k                = min(TOP_K_ASSETS, len(valid_indices))
                top_k_local      = valid_indices[np.argsort(weights_np_i[valid_indices])[-k:]]
                cand_weights     = weights_np_i[top_k_local]
                cand_weights     = cand_weights / cand_weights.sum()
                keep             = cand_weights > MIN_ACTIVE_WEIGHT
                if not keep.any():
                    keep[:] = True
                selected         = top_k_local[keep]
                full_weights[sub_col_indices[selected]] = weights_np_i[selected]
                full_weights    /= full_weights.sum()

                date_str = str(clean_returns.index[idx].date())
                result = _run_single_sim_path(
                    idx, full_weights, sim_cache, clean_returns,
                    profile['start_cap'], profile['goals'], "loop",
                    max_horizon, False, date_str
                )
                results.append(result)

            # 7. Compute reward
            gfr, etv, mdd, aw, goal_fails, market_fails = _aggregate_path_results(results)
            metrics = {"GFR": gfr, "ETV": etv, "MDD": mdd, "AW": aw,
                       "GoalFails": goal_fails, "MarketFails": market_fails,
                       "Total_Simulations": len(results)}
            reward = _compute_reward(metrics, profile)

            # Update Baseline
            if it == start_iter:
                baseline = float(reward)
            else:
                baseline = 0.99 * baseline + 0.01 * float(reward)

            # 8. Gradient update (REINFORCE)
            loss = -(reward - baseline) * log_prob.mean()
            loss.backward()
            optimizer.step()
            ep_reward = float(reward)
            
            # Store for logging (subset-space weights + subset context)
            weights_np       = weights[0].detach().cpu().numpy()
            log_ticker_mask  = sub_ticker_mask
            log_subset_idx   = subset_idx

        history.append({"episode": it, "reward": ep_reward})

        # ── Periodic Checkpoint ────────────────────────────────────────────
        if (it + 1) % check_freq == 0 or it == start_iter + iterations - 1:
            torch.save({
                'episode': it + 1,
                'model_state': agent.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'history': history,
                'baseline': baseline
            }, checkpoint_path)

        # ── Periodic Logging ───────────────────────────────────────────────
        if verbose and (it % verbose_every == 0 or it == start_iter + iterations - 1):
            elapsed = time.time() - t0
            iter_per_sec = (it - start_iter + 1) / elapsed if elapsed > 0 else 0

            goals_str = ", ".join([f"Yr {y}: ${amt:,.0f}" for y, amt in profile.get('goals', {}).items()])
            print(f"\n  [Iter {it+1:7d}] Reward: {ep_reward:7.3f} | Baseline: {baseline:7.3f} | "
                  f"{iter_per_sec:.1f} it/s | Profile: {profile['profile_name']}")
            print(f"    -> User Info: Risk={profile.get('risk_tolerance', 5.0)}/9.0 "
                  f"| Start Cap=${profile.get('start_cap', 0):,.0f} | Goals: [{goals_str}]")

            if not wf_enabled:
                log_active_mask = (weights_np > MIN_ACTIVE_WEIGHT) & log_ticker_mask
                if log_active_mask.any():
                    comp = metrics.get("reward_components", {})

                    # Map subset indices back to full ticker list for display
                    sub_tickers = [all_tickers[i] for i in log_subset_idx]
                    active_assets = [(sub_tickers[i], weights_np[i])
                                     for i in range(len(sub_tickers)) if log_active_mask[i]]
                    active_assets.sort(key=lambda x: x[1], reverse=True)
                    top_str = ", ".join([f"{t}: {w:.2%}" for t, w in active_assets[:5]])
                    top_str += f" | (Total Active: {len(active_assets)})"

                    print(f"    -> Agent Allocation: {top_str}")
                    n_sims     = metrics.get('Total_Simulations', 1)
                    goal_f     = metrics.get('GoalFails', 0)
                    market_f   = metrics.get('MarketFails', 0)
                    fail_parts = []
                    if goal_f   > 0: fail_parts.append(f"{goal_f}/{n_sims} Goal Bankrupt")
                    if market_f > 0: fail_parts.append(f"{market_f}/{n_sims} Market Collapse")
                    fail_str = f" | Failures: {', '.join(fail_parts)}" if fail_parts else ""
                    print(f"    -> Simulation Metrics: ETV=${metrics.get('ETV',0):,.0f} "
                          f"| MDD={metrics.get('MDD',0):.1%} | GFR={metrics.get('GFR',0):.1%}{fail_str}")
                    bonus_str = f" + Bonus ({comp.get('gfr_bonus',0):.3f})" if comp.get('gfr_bonus',0) > 0 else ""
                    print(f"    -> Reward Math: Return ({comp.get('return_score',0):.3f}) "
                          f"- MDD ({metrics.get('MDD',0):.1%} drop → {comp.get('mdd_penalty',0):.3f}) "
                          f"- GFR ({1.0 - metrics.get('GFR',0):.1%} miss → {comp.get('gfr_penalty',0):.3f}){bonus_str} = {ep_reward:.3f}")
                else:
                    comp = metrics.get("reward_components", {})
                    bonus_str = f" + Bonus ({comp.get('gfr_bonus',0):.3f})" if comp.get('gfr_bonus',0) > 0 else ""
                    print(f"    -> Agent Allocation: ALL ASSETS PRUNED (Empty Portfolio)")
                    print(f"    -> Reward Math: Return ({comp.get('return_score',0):.3f}) "
                          f"- MDD (0.0% drop → {comp.get('mdd_penalty',0):.3f}) "
                          f"- GFR (100.0% miss → {comp.get('gfr_penalty',0):.3f}){bonus_str} = {ep_reward:.3f}")

    # ── Final Greedy Evaluation ────────────────────────────────────────────
    recommendations = _greedy_evaluate_fast(agent, emb_matrix, static_matrix, all_tickers, user_profile)

    return {
        "portfolio_weights": recommendations,
        "training_history": history,
        "agent": agent,
    }
