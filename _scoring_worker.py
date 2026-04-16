"""
_scoring_worker.py
──────────────────
Member B: Asset Scoring, Dynamic K Selection & Portfolio Allocation.

Architecture:
    Phase 1 (current):  Composite scoring via cosine similarity + horizon-weighted
                        predicted return + volatility penalty. This is a PLACEHOLDER.
    Phase 2 (planned):  Replace composite with XGBoost/regression model that takes
                        autoencoder embeddings + user features → asset match score → ReLU threshold.

Public API:
    build_user_preference_vector(dataset, user_profile, config) → np.ndarray
    recommend_and_allocate_member_b(dataset, user_profile, user_vector, config) → dict
"""

import math
import numpy as np
import pandas as pd


# ── Utility Functions ─────────────────────────────────────────────────────────

def cosine_similarity(v1, v2):
    """Cosine similarity between two vectors. Returns 0.0 on NaN input."""
    if np.any(np.isnan(v1)) or np.any(np.isnan(v2)):
        return 0.0
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9
    return float(np.dot(v1, v2) / norm_product)


def vol_to_vscore(raw_vol, anchor=0.50):
    """
    Map raw annualized volatility to a linear V_score in [0, 10].

    Formula:  V_score = min(10, (raw_vol / anchor) * 10)
    Any volatility >= anchor is treated as maximum risk (score 10).
    """
    if anchor <= 0:
        anchor = 0.50
    score = (raw_vol / anchor) * 10.0
    return min(10.0, max(0.0, score))


# ── User Preference Vector ────────────────────────────────────────────────────

def build_user_preference_vector(dataset, user_profile, config):
    """
    Build a user embedding vector from the trained asset embedding space.

    Strategy:
        1. Compute a multi-goal risk budget via exponential glide-path decay.
        2. Filter assets whose V_score is within the budget ceiling (+10% tolerance).
        3. Select the top-30 safe archetypes by return.
        4. Average their embeddings to form the user preference vector.

    Returns:
        np.ndarray of shape (embedding_dim,)
    """
    embeddings    = dataset["dynamic_embeddings"]
    daily_returns = dataset.get("drip_daily_returns") or dataset["daily_returns"]
    tau           = config.get("sim_glide_path_tau", 8.0)
    v_anchor      = config.get("sim_v_score_anchor", 0.50)

    risk_user = user_profile["risk_tolerance"]
    goals     = user_profile["goals"]

    # Step 1: Multi-goal risk budget (glide-path decay)
    total_cap = sum(goals.values())
    risk_budget = 0.0
    for years, amount in goals.items():
        w      = amount / max(total_cap, 1)
        decay  = 1.0 - math.exp(-years / tau)
        g_risk = risk_user * decay
        risk_budget += g_risk * w

    user_profile["_risk_budget_score"] = risk_budget
    user_profile["_v_anchor"] = v_anchor

    # Step 2: Compute per-asset volatility scores
    asset_metrics = {}
    for ticker in embeddings:
        if ticker not in daily_returns.columns:
            continue
        ann_vol = daily_returns[ticker].std() * np.sqrt(252)
        ann_ret = (1 + daily_returns[ticker].mean()) ** 252 - 1
        if not np.isnan(ann_vol) and ann_vol > 0:
            v_score = vol_to_vscore(ann_vol, v_anchor)
            asset_metrics[ticker] = {"return": ann_ret, "v_score": v_score}

    if not asset_metrics:
        embed_dim = config.get("ml_embedding_dim", 8)
        return np.zeros(embed_dim)

    metrics_df = pd.DataFrame(asset_metrics).T

    # Step 3: Filter to safe assets (within budget + 10% tolerance)
    budget_ceil = risk_budget * 1.1
    safe = metrics_df[metrics_df["v_score"] <= budget_ceil]
    if len(safe) == 0:
        safe = metrics_df.nsmallest(20, "v_score")

    # Step 4: Average top-30 safe archetypes' embeddings
    archetypes = safe.nlargest(30, "return").index
    vecs = [embeddings[t] for t in archetypes if t in embeddings]
    return np.nan_to_num(np.nanmean(vecs, axis=0), nan=0.0)


# ── Performance Score ─────────────────────────────────────────────────────────

def compute_performance_score(ticker, asset_predictions, horizons, horizon_weights, target_metrics):
    """
    Horizon-weighted average of the model's predicted forward return for a ticker.

    Uses the asset_predictions[ticker][horizon][metric] dict produced by the
    autoencoder's decoder head in _ml_worker.py.
    """
    weighted_ret = 0.0
    total_w = 0.0
    for h in horizons:
        preds_h = asset_predictions.get(ticker, {}).get(h, {})
        if "return" not in preds_h:
            continue
        w = horizon_weights.get(h, 1.0)
        weighted_ret += w * preds_h["return"]
        total_w += w
    return weighted_ret / max(total_w, 1e-9)


# ── Composite Scoring & Allocation ────────────────────────────────────────────
# NOTE: This is a PLACEHOLDER composite scoring function.
# The architecture is designed so that an XGBoost or regression model can
# replace the linear combination below.  The model would take:
#   Input:  [user_embedding, asset_embedding, asset_features, user_features]
#   Output: match_score (scalar)
#   Filter: ReLU threshold on match_score
# ──────────────────────────────────────────────────────────────────────────────

def recommend_and_allocate_member_b(dataset, user_profile, user_vector, config):
    """
    Score every asset and allocate a portfolio.

    Composite Score (placeholder — future: XGBoost):
        S_i = w_sim * cosine_sim(user_vec, asset_vec)
            + w_perf * perf_score_i
            - w_penalty * (overage / danger_zone)^2

    Weight Interpolation:
        t = clamp(risk_budget / 10.0, 0, 1)
        w_perf    = lerp(perf_low, perf_high, t)     — higher risk → more perf weight
        w_penalty = lerp(penalty_high, penalty_low, t) — lower risk → heavier penalty

    Dynamic K:
        Assets with S_i >= theta enter the portfolio, clamped to [min_k, max_k].

    Allocation:
        Temperature-scaled softmax over selected scores.

    Returns:
        dict with portfolio_weights, scores, score_components, and metadata.
    """
    master_df       = dataset["master_df"]
    daily_returns   = dataset.get("drip_daily_returns") or dataset["daily_returns"]
    embeddings      = dataset["dynamic_embeddings"]
    asset_preds     = dataset.get("asset_predictions", {})

    risk_budget     = user_profile.get("_risk_budget_score", 5.0)
    v_anchor        = user_profile.get("_v_anchor", config.get("sim_v_score_anchor", 0.50))

    # Scoring hyperparameters from config
    w_sim           = config.get("scoring_sim_weight", 1.0)
    perf_range      = config.get("scoring_perf_weight_range", [0.2, 2.0])
    penalty_range   = config.get("scoring_penalty_weight_range", [2.0, 0.2])
    theta           = config.get("sim_dynamic_k_theta", 0.5)
    min_k           = config.get("portfolio_min_k", 5)
    max_k           = config.get("portfolio_max_k", 30)

    horizons        = config.get("ml_target_horizons", [1, 3, 5, 10, 15])
    horizon_weights = config.get("ml_horizon_weights", {1: 1.0, 3: 0.8, 5: 0.6, 10: 0.4, 15: 0.2})
    target_metrics  = config.get("ml_target_metrics", ["return", "volatility", "volume"])

    # Dynamic weight interpolation based on risk budget
    t = min(max(risk_budget / 10.0, 0.0), 1.0)
    w_perf    = perf_range[0] + t * (perf_range[1] - perf_range[0])
    w_penalty = penalty_range[0] + t * (penalty_range[1] - penalty_range[0])
    danger_zone = max(10.0 - risk_budget, 0.1)

    # Score every asset
    scores = {}
    score_components = {}

    for ticker, asset_vector in embeddings.items():
        sim_score  = cosine_similarity(user_vector, asset_vector)
        perf_score = compute_performance_score(
            ticker, asset_preds, horizons, horizon_weights, target_metrics
        )

        # Get annualized volatility
        ann_vol = 0.25  # default
        if ticker in daily_returns.columns:
            v = daily_returns[ticker].std() * np.sqrt(252)
            if pd.notna(v) and v > 0:
                ann_vol = float(v)

        v_score  = vol_to_vscore(ann_vol, v_anchor)
        overage  = max(0.0, v_score - risk_budget)
        penalty  = (overage / danger_zone) ** 2

        final_score = (w_sim * sim_score) + (w_perf * perf_score) - (w_penalty * penalty)

        scores[ticker] = final_score
        score_components[ticker] = {
            "similarity":       sim_score,
            "perf_score":       perf_score,
            "w_perf":           w_perf,
            "w_penalty":        w_penalty,
            "raw_vol":          ann_vol,
            "V_score":          v_score,
            "risk_budget":      risk_budget,
            "overage":          overage,
            "relative_penalty": penalty,
            "final_score":      final_score,
        }

    # Dynamic K selection: keep assets with score >= theta, clamped to [min_k, max_k]
    filtered = {t: s for t, s in scores.items() if s >= theta}
    ranked   = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    if len(filtered) < min_k:
        filtered = dict(ranked[:min_k])
    elif len(filtered) > max_k:
        filtered = dict(sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:max_k])

    if len(filtered) == 0:
        filtered = dict(ranked[:5])

    # Temperature-scaled softmax allocation
    risk_user = user_profile["risk_tolerance"]
    T = max(0.1, (11.0 - risk_user) / 2.0)
    exp_scores = {t: math.exp(min(20, s / T)) for t, s in filtered.items()}
    sum_exp = sum(exp_scores.values())
    weights = {t: v / sum_exp for t, v in exp_scores.items()}

    return {
        "portfolio_weights": weights,
        "scores":            filtered,
        "all_scores":        scores,
        "score_components":  score_components,
        "temperature":       T,
        "theta":             theta,
        "risk_budget":       risk_budget,
        "total_universe":    len(scores),
        "k_selected":        len(filtered),
    }
