"""
_sim_worker.py
──────────────
Flexible backtesting engine for portfolio-goal evaluation.

Core function:
    backtest_portfolio(user_config, portfolio_composition, asset_daily_returns,
                       baseline_daily_returns, start_i) -> SimResult

Loss / training label:
    signed_squared_relative_delta = sign(delta) * delta**2
    where delta = (pers_terminal - base_terminal) / |base_terminal|

This is the label fed to the two-tower recommendation model.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List

from _constants import (
    TRADING_DAYS_PER_YEAR,
    SIM_YEARS,
    LOAN_DAILY_RATE,
    CASH_BUCKET_ANNUAL_RATE,
    DAILY_RETURN_CLAMP,
    GFR_OBJECTIVE_THRESHOLD,
    CAGR_REWARD_SCALE,
    MDD_PENALTY_SCALE,
    VOL_PENALTY_SCALE,
    GFR_BRACKETS,
    DIVERSITY_PENALTY_SCALE,
    DIVERSITY_PENALTY_THRESHOLD,
    MIN_ACTIVE_WEIGHT,
    RELATIVE_WEIGHT_THRESHOLD,
    MAX_DYNAMIC_ASSETS,
    SIM_MONTHLY_START_YEAR,
    SIM_TERMINAL_HORIZON_CAP,
    DEBUG_LOG_MAX_YEARS,
)


# ── Result container ──────────────────────────────────────────────────────────
@dataclass
class SimResult:
    base_terminal:              float = 0.0
    pers_terminal:              float = 0.0
    base_failures:              int   = 0
    pers_failures:              int   = 0
    base_utility:               float = 0.0   # signed log(base_terminal)
    pers_utility:               float = 0.0   # signed log(pers_terminal)
    signed_squared_relative_delta: float = 0.0  # ← training label for two-tower model

    # Raw delta (useful for inspection)
    relative_delta:             float = 0.0   # (pers - base) / |base|


# ── Helpers ───────────────────────────────────────────────────────────────────
def signed_log(x: float) -> float:
    """Signed log utility — handles negative terminal wealth."""
    if x >= 0:
        return float(np.log1p(x))
    return float(-np.log1p(abs(x)))


def signed_squared_delta(pers_terminal: float, base_terminal: float) -> tuple:
    """
    Return (relative_delta, signed_squared_relative_delta).

    Formula:
        delta = (pers - base) / |base|
        label = sign(delta) * delta**2

    Properties:
        • Relative (normalised across starting capitals)
        • Quadratic: large deviations from baseline are emphasised
        • Signed: positive = outperforms baseline, negative = underperforms
    """
    denom = abs(base_terminal) if abs(base_terminal) > 1 else 1.0
    delta = (pers_terminal - base_terminal) / denom
    label = float(np.sign(delta) * delta ** 2)
    return float(delta), label


# ── Core simulation ───────────────────────────────────────────────────────────
def _run_sim_core(
    start_i: int,
    n_days: int,
    portfolio_daily_returns: np.ndarray,   # 1-D array of blended daily returns
    user_config: dict,
    initial_alloc_cash: float,
    initial_alloc_growth: float,           # everything non-Cash goes here
):
    """
    Run one daily simulation path.

    Two-account model:
        cash_bucket  — flat 3% annual, used for immediate goals
        growth_bucket — tracks portfolio_daily_returns, used for all other goals and contributions

    Returns (terminal_net, n_failures).
    """
    goals         = user_config["goals"]         # {year_int: amount_float}
    daily_contrib = user_config["monthly_contrib"] / 21.0

    cash_bucket   = float(initial_alloc_cash)
    growth_bucket = float(initial_alloc_growth)
    debt          = 0.0
    n_failures    = 0

    end_i = min(start_i + n_days, len(portfolio_daily_returns))

    for day_off in range(end_i - start_i):
        i = start_i + day_off

        # 1. Compound debt
        debt *= (1 + LOAN_DAILY_RATE)

        # 2. Apply daily returns
        cash_bucket   *= (1 + CASH_BUCKET_ANNUAL_RATE / TRADING_DAYS_PER_YEAR)
        growth_ret     = float(portfolio_daily_returns[i])
        # Clip extreme daily moves to ±50% to prevent blow-ups from bad data
        growth_ret     = max(-DAILY_RETURN_CLAMP, min(DAILY_RETURN_CLAMP, growth_ret))
        growth_bucket *= (1 + growth_ret)

        # 3. Service debt with daily contribution first; remainder invests
        if debt > 0:
            payment       = min(daily_contrib, debt)
            debt         -= payment
            growth_bucket += daily_contrib - payment
        else:
            growth_bucket += daily_contrib

        # 4. Goal liquidation at annual boundaries
        year_idx = day_off // TRADING_DAYS_PER_YEAR + 1
        if day_off > 0 and day_off % TRADING_DAYS_PER_YEAR == 0 and year_idx in goals:
            needed = float(goals[year_idx])
            # Waterfall: cash first, then growth
            for bucket_name in ("cash", "growth"):
                if needed <= 0:
                    break
                if bucket_name == "cash":
                    drawn        = min(cash_bucket, needed)
                    cash_bucket -= drawn
                else:
                    drawn          = min(growth_bucket, needed)
                    growth_bucket -= drawn
                needed -= drawn
            if needed > 0:
                debt      += needed      # shortfall → loan at 10% APR
                n_failures += 1

    terminal_net = (cash_bucket + growth_bucket) - debt
    return float(terminal_net), n_failures


def _run_sim_with_trajectory(
    start_i: int,
    n_days: int,
    portfolio_daily_returns: np.ndarray,
    user_config: dict,
    initial_alloc_cash: float,
    initial_alloc_growth: float,
) -> tuple:
    """
    Same as _run_sim_core but also returns a list of (year, net_value) snapshots
    taken at each annual boundary — used for trajectory visualisation.
    Returns (terminal_net, n_failures, annual_snapshots).
    """
    goals         = user_config["goals"]
    daily_contrib = user_config["monthly_contrib"] / 21.0

    cash_bucket   = float(initial_alloc_cash)
    growth_bucket = float(initial_alloc_growth)
    debt          = 0.0
    n_failures    = 0
    snapshots     = [(0, cash_bucket + growth_bucket)]   # year 0 = starting value

    end_i = min(start_i + n_days, len(portfolio_daily_returns))

    for day_off in range(end_i - start_i):
        i = start_i + day_off
        debt *= (1 + LOAN_DAILY_RATE)
        cash_bucket   *= (1 + CASH_BUCKET_ANNUAL_RATE / TRADING_DAYS_PER_YEAR)
        growth_ret     = float(portfolio_daily_returns[i])
        growth_ret     = max(-DAILY_RETURN_CLAMP, min(DAILY_RETURN_CLAMP, growth_ret))
        growth_bucket *= (1 + growth_ret)

        if debt > 0:
            payment        = min(daily_contrib, debt)
            debt          -= payment
            growth_bucket += daily_contrib - payment
        else:
            growth_bucket += daily_contrib

        year_idx = day_off // TRADING_DAYS_PER_YEAR + 1
        if day_off > 0 and day_off % TRADING_DAYS_PER_YEAR == 0:
            if year_idx in goals:
                needed = float(goals[year_idx])
                for bucket_name in ("cash", "growth"):
                    if needed <= 0:
                        break
                    if bucket_name == "cash":
                        drawn        = min(cash_bucket, needed)
                        cash_bucket -= drawn
                    else:
                        drawn          = min(growth_bucket, needed)
                        growth_bucket -= drawn
                    needed -= drawn
                if needed > 0:
                    debt      += needed
                    n_failures += 1
            snapshots.append((year_idx, (cash_bucket + growth_bucket) - debt))

    terminal_net = (cash_bucket + growth_bucket) - debt
    return float(terminal_net), n_failures, snapshots


# ── Portfolio return blender ───────────────────────────────────────────────────
def blend_portfolio_returns(
    portfolio_composition: Dict[str, float],
    asset_daily_returns: Dict[str, np.ndarray],
    n_days: int,
) -> np.ndarray:
    """
    Compute blended daily returns for a {ticker: weight} portfolio.

    portfolio_composition  — {ticker: weight}, weights should sum to ~1.0
    asset_daily_returns    — {ticker: 1-D np.ndarray of daily returns}
    n_days                 — clip to this length
    """
    weights        = np.array(list(portfolio_composition.values()), dtype=np.float64)
    weights       /= weights.sum()                      # normalise to sum = 1
    tickers        = list(portfolio_composition.keys())

    # Find common length across all assets
    available      = [asset_daily_returns[t] for t in tickers if t in asset_daily_returns]
    if not available:
        return np.zeros(n_days)

    min_len        = min(len(a) for a in available)
    clip           = min(min_len, n_days)

    matrix         = np.vstack([asset_daily_returns[t][:clip] for t in tickers
                                 if t in asset_daily_returns]).T   # (days, assets)
    blended        = matrix @ weights[:matrix.shape[1]]
    # Pad with zeros if needed
    if len(blended) < n_days:
        blended = np.concatenate([blended, np.zeros(n_days - len(blended))])
    return blended


# ── Allocator ─────────────────────────────────────────────────────────────────
def allocate(
    start_cap: float,
    goals: Dict[int, float],
    short_max_yr: int,
    medium_max_yr: int,
) -> tuple:
    """
    Allocate starting capital into (cash_bucket, growth_bucket).

    Goals ≤ short_max_yr  → cash_bucket  (capital-protected)
    Goals > short_max_yr  → growth_bucket (market-exposed)

    short_max_yr controls what we consider "near-term" — the larger it is,
    the more capital we protect in cash, the lower the growth exposure.

    Returns (cash_alloc, growth_alloc).
    """
    cash_alloc   = 0.0
    growth_alloc = 0.0
    remaining    = float(start_cap)

    for yr in sorted(goals):
        if remaining <= 0:
            break
        funding = min(float(goals[yr]), remaining)
        if yr <= short_max_yr:        # ← uses actual boundary, not hardcoded 1
            cash_alloc += funding
        else:
            growth_alloc += funding
        remaining -= funding

    growth_alloc += remaining    # surplus → long-term growth
    return cash_alloc, growth_alloc


# ── Public API ─────────────────────────────────────────────────────────────────
def backtest_portfolio(
    user_config: dict,
    portfolio_composition: Dict[str, float],
    asset_daily_returns: Dict[str, np.ndarray],
    baseline_daily_returns: np.ndarray,
    start_i: int = 0,
    boundary: tuple = (5, 15),           # (short_max_yr, medium_max_yr)
) -> SimResult:
    """
    Evaluate a portfolio against a user's financial goals and a baseline.

    Parameters
    ----------
    user_config
        {start_cap, monthly_contrib, goals: {year_int: amount_float}}
    portfolio_composition
        {ticker: weight}  — the candidate portfolio to evaluate
    asset_daily_returns
        {ticker: np.ndarray}  — pre-loaded daily returns for each asset
    baseline_daily_returns
        np.ndarray  — daily returns for the baseline portfolio (e.g. S&P500)
    start_i
        Index into the daily returns arrays for the historical start date
    boundary
        (short_max_yr, medium_max_yr) used by the allocator

    Returns
    -------
    SimResult with signed_squared_relative_delta as the training label
    """
    goals       = user_config["goals"]
    start_cap   = float(user_config["start_cap"])
    n_days      = TRADING_DAYS_PER_YEAR * SIM_YEARS
    short_max, medium_max = boundary

    # ── Blend portfolio into a single daily-return series ──────────────────────
    portfolio_returns = blend_portfolio_returns(portfolio_composition, asset_daily_returns, len(baseline_daily_returns))

    # ── Allocate starting capital ───────────────────────────────────────────────
    cash_p, growth_p = allocate(start_cap, goals, short_max, medium_max)
    # Baseline: 100% in growth (tracks baseline returns)
    cash_b, growth_b = 0.0, start_cap

    # ── Run both sims ───────────────────────────────────────────────────────────
    pers_term, pers_fails = _run_sim_core(
        start_i, n_days, portfolio_returns,  user_config, cash_p, growth_p
    )
    base_term, base_fails = _run_sim_core(
        start_i, n_days, baseline_daily_returns, user_config, cash_b, growth_b
    )

    # ── Compute training label ─────────────────────────────────────────────────
    rel_delta, label = signed_squared_delta(pers_term, base_term)

    return SimResult(
        base_terminal               = base_term,
        pers_terminal               = pers_term,
        base_failures               = base_fails,
        pers_failures               = pers_fails,
        base_utility                = signed_log(base_term),
        pers_utility                = signed_log(pers_term),
        signed_squared_relative_delta = label,
        relative_delta              = rel_delta,
    )


# ── Batch job for joblib parallelism ──────────────────────────────────────────
def simulate_batch(
    start_i: int,
    sp_ret: np.ndarray,
    bond_ret: np.ndarray,
    all_configs: list,
    boundary_configs: dict,
) -> list:
    """
    Process one historical start date across all user configs and boundary strategies.

    portfolio_composition defaults to a hybrid {SP500: stock_ratio, VBTIX: bond_ratio}
    using the pre-loaded sp_ret and bond_ret arrays — ready to swap in real ticker
    weights from scoring_df once the asset embedding pipeline is wired up.
    """
    asset_rets = {"^GSPC": sp_ret, "VBTIX": bond_ret}
    results    = []

    for cfg in all_configs:
        for strategy_name, (boundary, ratio) in boundary_configs.items():
            # Dynamic portfolio composition based on boundary + ratio
            # Medium-term goals get (ratio)% SP500 / (1-ratio)% bonds
            composition = {"^GSPC": ratio, "VBTIX": 1.0 - ratio}

            result = backtest_portfolio(
                user_config            = cfg,
                portfolio_composition  = composition,
                asset_daily_returns    = asset_rets,
                baseline_daily_returns = sp_ret,
                start_i                = start_i,
                boundary               = boundary,
            )

            res_dict = {
                "start_i":        start_i,
                "strategy":       strategy_name,
                "start_cap":      cfg["start_cap"],
                "monthly_contrib":cfg["monthly_contrib"],
                "base_terminal":  result.base_terminal,
                "base_failures":  result.base_failures,
                "base_utility":   result.base_utility,
                "pers_terminal":  result.pers_terminal,
                "pers_failures":  result.pers_failures,
                "pers_utility":   result.pers_utility,
                "relative_delta": result.relative_delta,
                "label":          result.signed_squared_relative_delta,
            }
            if "meta_cap_ratio" in cfg:
                res_dict["meta_cap_ratio" ] = cfg["meta_cap_ratio"]
                res_dict["meta_inc_ratio" ] = cfg["meta_inc_ratio"]
                res_dict["meta_dist"      ] = cfg["meta_dist"]
                
            results.append(res_dict)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Precision Monthly Simulator (Member D / Universal Policy)
# ══════════════════════════════════════════════════════════════════════════════



# ── Sim Helpers ───────────────────────────────────────────────────────────────

def _extract_yearly_chunk(daily_returns, start_idx, year_number):
    """Extract a 252-day trading chunk for a given simulation year, wrapping cyclically."""
    n_days = len(daily_returns)
    chunk_start = (start_idx + (year_number - 1) * TRADING_DAYS_PER_YEAR) % n_days
    chunk_end   = (start_idx + year_number * TRADING_DAYS_PER_YEAR) % n_days

    if hasattr(daily_returns, "iloc"):
        return daily_returns.iloc[chunk_start:chunk_end]
    else:
        # NumPy path
        return daily_returns[chunk_start:chunk_end]


def _update_drawdown(capital, peak, max_drawdown):
    """Track peak capital and update the maximum drawdown. Returns (new_peak, new_max_drawdown)."""
    if capital > peak:
        return capital, max_drawdown
    if peak > 0:
        drawdown = (peak - capital) / peak
        return peak, max(max_drawdown, drawdown)
    return peak, max_drawdown


def _process_goal_withdrawal(capital, year, goals):
    """
    Withdraw goal amount if this year has a goal.
    Returns (new_capital, is_bankrupt, log_suffix).
    """
    if year not in goals:
        return capital, False, ""

    target = goals[year]
    capital -= target

    if capital >= 0:
        return capital, False, f" | GOAL -${target:,.0f} -> ${capital:,.0f} ✅"
    return capital, True, f" | GOAL -${target:,.0f} -> ${capital:,.0f} ❌"


def _blend_portfolio_returns_safe(base_returns, ticker_weights):
    # Deprecated: Kept for backwards compatibility if needed outside RL loop
    weight_series = pd.Series(ticker_weights)
    weight_series /= weight_series.sum()
    asset_returns = base_returns[list(ticker_weights.keys())]

    valid_mask     = asset_returns.notna()
    daily_weights  = valid_mask * weight_series
    weight_sums    = daily_weights.sum(axis=1)
    norm_weights   = daily_weights.div(weight_sums, axis=0).fillna(0.0)

    return (asset_returns.fillna(0.0) * norm_weights).sum(axis=1)
    
def build_simulation_cache(base_returns, max_horizon_years=30):
    """
    Precompute annual returns for every asset, for every standard simulation start index.
    Returns: (numpy 2D array cache_array, dict start_idx_to_pos, dataframe clean_returns, dict column_to_idx, int max_sim_years)
    """
    import time
    print(f"[{time.strftime('%H:%M:%S')}] [Member C] Building Annual Simulation Cache (1-Year Blocks)...")
    
    clean_returns = base_returns.fillna(0.0)
    num_assets = len(clean_returns.columns)
    column_to_idx = {col: i for i, col in enumerate(clean_returns.columns)}
    
    start_indices = _resolve_simulation_starts(clean_returns)
    num_starts = len(start_indices)
    start_idx_to_pos = {idx: i for i, idx in enumerate(start_indices)}
    
    # Use float32 to save memory while maintaining precision for returns
    # SHAPE CHANGE: (num_starts, num_assets)
    cache_array = np.zeros((num_starts, num_assets), dtype=np.float32)
    
    # ── HYPER-FAST NUMPY CONVERSION ──
    n_days = len(clean_returns)
    one_plus_ret = (1.0 + clean_returns.values).astype(np.float32)
    
    for i, start_idx in enumerate(start_indices):
        chunk_start = start_idx
        chunk_end   = (start_idx + TRADING_DAYS_PER_YEAR) % n_days
        
        if chunk_end > chunk_start:
            chunk = one_plus_ret[chunk_start:chunk_end]
        else:
            chunk = np.vstack((one_plus_ret[chunk_start:], one_plus_ret[:chunk_end]))
            
        cache_array[i, :] = chunk.prod(axis=0) - 1.0
            
    # Winsorize the 1-year returns to prevent single massive anomalies from dominating
    # Cap maximum 1-year return at +500% (5.0x growth)
    MAX_ANNUAL_RETURN = 5.0
    cache_array = np.clip(cache_array, a_min=-1.0, a_max=MAX_ANNUAL_RETURN)

    # Compute data-derived max horizon
    from _constants import SIM_HORIZON_BUFFER_YEARS
    total_span_years = len(clean_returns) // TRADING_DAYS_PER_YEAR
    max_sim_years = max(1, total_span_years - SIM_HORIZON_BUFFER_YEARS)

    print(f"[{time.strftime('%H:%M:%S')}] [Member C] Cache built: {cache_array.shape} (starts x assets) | Data Horizon: {max_sim_years}y")
    return cache_array, start_idx_to_pos, clean_returns, column_to_idx, max_sim_years


def calculate_decay_probabilities(dates, half_life_years=3.0):
    """
    Calculate exponential decay probabilities for a list of dates.
    Recent dates get higher probabilities.
    """
    if len(dates) == 0:
        return np.array([])
        
    dates = pd.to_datetime(dates)
    max_date = dates.max()
    
    # Time diff in years
    diff_years = (max_date - dates).total_seconds() / (365.25 * 24 * 3600)
    
    # Exponential decay: 2^(-diff / half_life)
    weights = 2.0 ** (-diff_years / half_life_years)
    
    # Ensure numpy array for calculation
    w_np = np.array(weights)
    return w_np / w_np.sum()


def _resolve_simulation_starts(portfolio_returns, start_date=None, start_year_idx=None):
    """
    Determine which indices in the daily return series to start simulations from.
    Priority: exact start_date > start_year_idx > default monthly-from-2015.
    """
    if start_date:
        idx = portfolio_returns.index.get_indexer([pd.to_datetime(start_date)], method='pad')[0]
        return [idx]

    if start_year_idx is not None:
        years = portfolio_returns.index.year.unique()
        year_val = years[start_year_idx % len(years)]
        return [(portfolio_returns.index.year == year_val).argmax()]

    # Default: all trading days from SIM_MONTHLY_START_YEAR onward to allow high-fidelity sampling
    recent = portfolio_returns[portfolio_returns.index.year >= SIM_MONTHLY_START_YEAR]
    if len(recent) > 0:
        indices = np.where(portfolio_returns.index >= recent.index[0])[0]
    else:
        indices = np.arange(len(portfolio_returns))
    return indices.tolist()


def _aggregate_path_results(results, cvar_percentile=None):
    """Compute GFR, ETV, MDD, and Actual Withdrawals from a list of path result dicts."""
    total        = len(results)
    successes    = sum(1 for r in results if not r["bankrupt"])
    goal_fails   = sum(1 for r in results if r["bankrupt"] and r.get("bankrupt_reason") == "goal")
    market_fails = sum(1 for r in results if r["bankrupt"] and r.get("bankrupt_reason") == "market")
    all_tvs      = [r["terminal_value"] for r in results]
    all_mdds     = [r["max_drawdown"] for r in results]
    all_wds      = [r.get("actual_withdrawals", 0.0) for r in results]
    all_mults    = [r.get("market_multiplier", 1.0) for r in results]

    gfr = successes / total if total > 0 else 0
    etv = np.mean(all_tvs) if all_tvs else 0.0
    wds = np.mean(all_wds) if all_wds else 0.0
    
    if all_mults:
        safe_mults = np.maximum(all_mults, 1e-9)
        mean_market_mult = float(np.exp(np.mean(np.log(safe_mults))))
    else:
        mean_market_mult = 1.0

    # MDD Aggregation: Mean or CVaR (Tail Risk)
    if all_mdds:
        if cvar_percentile is not None:
            sorted_mdds = np.sort(all_mdds)
            n_cvar = max(1, int(len(sorted_mdds) * (cvar_percentile / 100.0)))
            mdd = float(np.mean(sorted_mdds[-n_cvar:]))
        else:
            mdd = float(np.mean(all_mdds))
    else:
        mdd = 1.0

    return gfr, etv, mdd, wds, mean_market_mult, goal_fails, market_fails


def _compute_reward(metrics, user_profile):
    """
    Compute the multi-faceted RL reward score for a simulation outcome.
    Incorporates per-user penalty tuning via 4 sensitivity dimensions.
    """
    horizon = float(user_profile.get("goal_year", SIM_TERMINAL_HORIZON_CAP))
    
    # 0. Per-user penalty sensitivities (1-10 scale, higher = more penalty)
    dd_sens   = float(user_profile.get("drawdown_sensitivity", 5.0))
    vol_sens  = float(user_profile.get("volatility_sensitivity", 5.0))
    goal_flex = float(user_profile.get("goal_flexibility", 5.0))
    conc_pref = float(user_profile.get("concentration_pref", 5.0))

    # desperation_factor suppresses penalties if aggressive growth is required
    start_cap = user_profile.get('start_cap', 1.0)
    goal_amt = user_profile['goals'].get(int(horizon), start_cap)
    if start_cap > 0 and horizon > 0:
        required_cagr = (goal_amt / start_cap) ** (1.0 / horizon) - 1.0
    else:
        required_cagr = 0.0
    desperation_factor = max(1.0, required_cagr / 0.10)

    # 1. CAGR Score (Return Component)
    mean_market_mult = float(metrics.get("Mean_Market_Mult", 1.0))
    safe_mult = max(mean_market_mult, 1e-6)
    annual_return = (safe_mult ** (1.0 / horizon)) - 1.0
    # Winsorize upside at +500% to avoid outlier explosion
    annual_return = min(annual_return, 5.0)
    return_score = annual_return * CAGR_REWARD_SCALE

    # 2. MDD Penalty (Drawdown Component)
    mdd_val = float(metrics.get("MDD", 0.0))
    raw_mdd_penalty = mdd_val * MDD_PENALTY_SCALE * dd_sens
    mdd_penalty = raw_mdd_penalty / desperation_factor

    # 3. Volatility Penalty (Consistency Component)
    annual_vol = float(metrics.get("Annual_Vol", 0.0))
    raw_vol_penalty = annual_vol * VOL_PENALTY_SCALE * vol_sens
    vol_penalty = raw_vol_penalty / desperation_factor

    # 4. GFR Contribution (Goal Success Rate)
    gfr = float(metrics.get("GFR", 0.0))
    x_coords = [b[0] for b in GFR_BRACKETS]
    y_coords = [b[1] for b in GFR_BRACKETS]
    idx_sort = np.argsort(x_coords)
    gfr_raw_contrib = float(np.interp(gfr, np.array(x_coords)[idx_sort], np.array(y_coords)[idx_sort]))
    
    # flex=1 -> severity=1.0 (strict), flex=10 -> severity=0.1 (lenient)
    gfr_severity = (11.0 - goal_flex) / 10.0
    gfr_contrib = gfr_raw_contrib * gfr_severity

    # 5. Diversity Penalty (Concentration Component)
    active_assets = int(metrics.get("Active_Assets", 0))
    div_threshold = max(3, int(11 - conc_pref))
    diversity_penalty = max(0, active_assets - div_threshold) * DIVERSITY_PENALTY_SCALE

    total_reward = return_score + gfr_contrib - mdd_penalty - vol_penalty - diversity_penalty

    # Return breakdown for detailed logging
    metrics["reward_components"] = {
        "return_score": return_score,
        "annual_return": annual_return,
        "mdd_penalty": mdd_penalty,
        "vol_penalty": vol_penalty,
        "gfr_contrib": gfr_contrib,
        "diversity_penalty": diversity_penalty,
        "desperation_factor": desperation_factor,
        "div_threshold": div_threshold,
        "gfr_severity": gfr_severity
    }

    return total_reward


# ── Core Path Runner ──────────────────────────────────────────────────────────

def _run_single_sim_path(weight_array, sim_cache, clean_returns, start_capital, goals, mode,
                         max_horizon_years, debug_path, start_date_str,
                         post_goal_weights=None, goal_year=None,
                         bootstrap_indices=None, start_idx=None,
                         monthly_contrib=0.0):
    """
    Simulate one trajectory using FAST precomputed annual dot products.
    If bootstrap_indices is provided, uses block bootstrapping.
    If start_idx is provided, uses chronological paths (with on-the-fly calculation if needed).
    """
    capital           = start_capital
    market_multiplier = 1.0
    horizon_market_multiplier = 1.0
    target_horizon    = goal_year if goal_year is not None else max_horizon_years
    mi_peak           = 1.0
    max_drawdown      = 0.0
    trail             = []
    phase_switched    = False
    any_bankrupt      = False
    bankrupt_reason   = None
    
    cache_array, start_idx_to_pos = sim_cache
    
    # Pre-build precomputed_returns for this path
    precomputed_returns = np.zeros((max_horizon_years, clean_returns.shape[1]))
    
    if bootstrap_indices is not None:
        # Bootstrapped mode: just pull from cache
        for year in range(1, max_horizon_years + 1):
            pos = bootstrap_indices[year - 1]
            precomputed_returns[year - 1] = cache_array[pos]
    elif start_idx is not None:
        # Chronological mode
        n_days = len(clean_returns)
        one_plus_ret = (1.0 + clean_returns.values).astype(np.float32)
        
        for year in range(1, max_horizon_years + 1):
            chunk_start = (start_idx + (year - 1) * TRADING_DAYS_PER_YEAR) % n_days
            if chunk_start in start_idx_to_pos:
                precomputed_returns[year - 1] = cache_array[start_idx_to_pos[chunk_start]]
            else:
                chunk_end = (chunk_start + TRADING_DAYS_PER_YEAR) % n_days
                if chunk_end > chunk_start:
                    chunk = one_plus_ret[chunk_start:chunk_end]
                else:
                    chunk = np.vstack((one_plus_ret[chunk_start:], one_plus_ret[:chunk_end]))
                precomputed_returns[year - 1] = chunk.prod(axis=0) - 1.0
    else:
        raise ValueError("Must provide either bootstrap_indices or start_idx")
        
    actual_withdrawals = 0.0
    contrib_annual = monthly_contrib * 12.0
    
    for year in range(1, max_horizon_years + 1):
        # Two-phase weight selection: pre-goal vs post-goal

        if post_goal_weights is not None and goal_year is not None and year > goal_year:
            w = post_goal_weights
            if not phase_switched and debug_path:
                trail.append(f"Y{year}: → GROWTH PHASE")
                phase_switched = True
        else:
            w = weight_array

        # FAST VECTOR DOT PRODUCT
        yr_return = np.dot(w, precomputed_returns[year - 1])
        capital           *= (1 + yr_return)
        capital           += contrib_annual
        market_multiplier *= (1 + yr_return)
        
        if year == target_horizon:
            horizon_market_multiplier = market_multiplier

        # Track drawdown based on MARKET performance, not cash balance
        mi_peak, max_drawdown = _update_drawdown(market_multiplier, mi_peak, max_drawdown)

        # Process goal withdrawal (if any)
        goal_log = ""
        if year in goals:
            target = goals[year]
            if capital >= target:
                actual_withdrawals += target
                capital -= target
                goal_log = f" | GOAL -${target:,.0f} -> ${capital:,.0f} ✅"
            else:
                withdrawn = max(capital, 0.0)
                actual_withdrawals += withdrawn
                shortfall = target - withdrawn
                capital = 0.0
                any_bankrupt = True
                if bankrupt_reason is None:
                    bankrupt_reason = "goal" if withdrawn > 0 else "market"
                goal_log = f" | GOAL -${target:,.0f} -> $0 (Shortfall: -${shortfall:,.0f}) ❌"

        # Build debug trail
        phase_tag = " [POST]" if (post_goal_weights is not None and goal_year is not None and year > goal_year) else ""
        step_str = f"Y{year}: ${capital:,.0f} ({yr_return:+.1%}){goal_log}{phase_tag}"
        if debug_path and year <= DEBUG_LOG_MAX_YEARS:
            trail.append(step_str)

    log_msg = ""
    if debug_path:
        log_msg = f"    Start: {start_date_str} | Path: {' -> '.join(trail)} " + ("❌" if any_bankrupt else "✅")

    return {"bankrupt": any_bankrupt, "bankrupt_reason": bankrupt_reason, "terminal_value": capital,
            "max_drawdown": max_drawdown,
            "actual_withdrawals": actual_withdrawals, "market_multiplier": horizon_market_multiplier, "log": log_msg}


# ── Portfolio Evaluator ───────────────────────────────────────────────────────

def evaluate_portfolio_member_c(dataset, recommendations, user_profile, config,
                                debug_path=False, start_year_idx=None, start_date=None,
                                sim_cache_bundle=None):
    """
    High-precision monthly-shifted portfolio evaluator utilizing FAST Annual Rebalancing cache.
    Supports two-phase allocation: if recommendations contains 'post_goal_weights',
    the sim switches to post-goal weights after the goal year.
    """
    import joblib
    weights = recommendations["portfolio_weights"]
    post_weights_dict = recommendations.get("post_goal_weights")  # None for legacy single-phase
    mode    = config.get("sim_horizon_mode", "loop")

    base_returns = dataset.get("drip_daily_returns") if dataset.get("drip_daily_returns") is not None else dataset["daily_returns"]
    
    # Retrieve or build cache
    if sim_cache_bundle is not None:
        cache_data, clean_returns, column_to_idx = sim_cache_bundle
        # If cache_data is a tuple (array, pos_dict), it's the new format
        sim_cache = cache_data
    else:
        cache_array, start_idx_to_pos, clean_returns, column_to_idx, max_sim_years = build_simulation_cache(base_returns)
        sim_cache = (cache_array, start_idx_to_pos)

    # Convert weights dict -> sorted candidates -> two-stage top-K + normalized threshold
    # Mirrors the exact selection logic used in the RL training loop for train/eval consistency.
    def _build_weight_array(w_dict):
        all_candidates = [(t, w) for t, w in w_dict.items() if t in column_to_idx and w > 0]
        if not all_candidates:
            return None
        
        # 1. Sort by raw weight
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 2. Find max weight and apply relative threshold
        max_w = all_candidates[0][1]
        threshold = max_w * RELATIVE_WEIGHT_THRESHOLD
        
        # 3. Filter by relative threshold and hard ceiling
        dynamic_candidates = [c for c in all_candidates if c[1] >= threshold]
        dynamic_candidates = dynamic_candidates[:MAX_DYNAMIC_ASSETS]
        
        # 4. Normalize and apply MIN_ACTIVE_WEIGHT
        total_w = sum(w for _, w in dynamic_candidates)
        norm_candidates = [(t, w / total_w) for t, w in dynamic_candidates]
        active = [(t, w) for t, w in norm_candidates if w >= MIN_ACTIVE_WEIGHT]
        
        if not active:
            active = norm_candidates
            
        num_assets = len(clean_returns.columns)
        wa = np.zeros(num_assets)
        for t, w in active:
            wa[column_to_idx[t]] = w
        
        if wa.sum() > 0:
            wa /= wa.sum()
            
        return wa

    weight_array = _build_weight_array(weights)
    if weight_array is None:
        return {"GFR": 0, "ETV": 0, "MDD": 1.0, "Objective_Function_Score": 0, "Total_Simulations": 0,
                "GoalFails": 0, "MarketFails": 0}

    post_weight_array = _build_weight_array(post_weights_dict) if post_weights_dict else None
    goal_year = user_profile.get("goal_year")  # None for legacy multi-goal profiles

    # Step 3: Determine simulation start indices
    sim_starts  = _resolve_simulation_starts(clean_returns, start_date, start_year_idx)
    max_horizon_requested = goal_year if goal_year else (max(user_profile['goals'].keys()) if user_profile['goals'] else 1)
    max_horizon = min(max_horizon_requested, max_sim_years)
    start_cap   = user_profile['start_cap']
    monthly_contrib = float(user_profile.get('monthly_contrib', 0.0))
    
    # Calculate decay probabilities for the entire cache pool (for bootstrapping)
    cache_array, start_idx_to_pos = sim_cache
    num_starts = cache_array.shape[0]
    all_start_dates = clean_returns.index[list(start_idx_to_pos.keys())]
    decay_probs = calculate_decay_probabilities(all_start_dates, half_life_years=3.0)

    is_chronological = (start_date is not None or start_year_idx is not None)
    
    if debug_path and not is_chronological:
        print(f"  [SIMULATOR] Bootstrapping Mode | "
              f"Starts: {num_starts} | Horizon: {max_horizon}y"
              f"{' | Two-Phase @ Y' + str(goal_year) if goal_year else ''}")
    elif debug_path and is_chronological:
        print(f"  [SIMULATOR] Chronological Mode | "
              f"Start: {start_date or start_year_idx} | Horizon: {max_horizon}y")

    # 4️⃣ Run paths
    results = []
    
    if is_chronological:
        # Chronological mode: run a single path from the specified start date
        actual_idx = sim_starts[0]
        date_str = str(clean_returns.index[actual_idx].date())
        results.append(
            _run_single_sim_path(
                weight_array, sim_cache, clean_returns, start_cap,
                user_profile['goals'], mode, max_horizon, debug_path, date_str,
                post_goal_weights=post_weight_array, goal_year=goal_year,
                start_idx=actual_idx, monthly_contrib=monthly_contrib
            )
        )
    else:
        # Bootstrapping mode: run multiple Monte Carlo paths
        max_paths = config.get("sim_paths_per_episode", 100)
        num_paths = min(max_paths, 500) # Cap for responsiveness
        rng = np.random.default_rng()
        
        for _ in range(num_paths):
            bootstrap_indices = rng.choice(num_starts, size=max_horizon, replace=True, p=decay_probs)
            
            results.append(
                _run_single_sim_path(
                    weight_array, sim_cache, clean_returns, start_cap,
                    user_profile['goals'], mode, max_horizon, debug_path=False, start_date_str="Bootstrapped",
                    post_goal_weights=post_weight_array, goal_year=goal_year,
                    bootstrap_indices=bootstrap_indices, monthly_contrib=monthly_contrib
                )
            )


    # Step 5: Aggregate metrics
    cvar_pct = config.get("sim_cvar_percentile")
    gfr, etv, mdd, aw, mean_market_mult, goal_fails, market_fails = _aggregate_path_results(results, cvar_percentile=cvar_pct)
    score = etv if gfr >= GFR_OBJECTIVE_THRESHOLD else etv * (gfr / GFR_OBJECTIVE_THRESHOLD)

    metrics = {"GFR": gfr, "ETV": etv, "MDD": mdd, "AW": aw,
               "Mean_Market_Mult": mean_market_mult,
               "Objective_Function_Score": score, "Total_Simulations": len(results),
               "GoalFails": goal_fails, "MarketFails": market_fails}
               
    if debug_path and results:
        sample = next((r for r in results if r["bankrupt"]), results[0])
        metrics["sample_path_log"] = sample.get("log", "")

    return metrics


# ── RL Environment Interface ─────────────────────────────────────────────────

def simulate_rl_environment_step(weights, tickers, dataset, user_profile, config,
                                 debug_path=False, start_year_idx=None, start_date=None,
                                 sim_cache_bundle=None):
    """
    Bridge between the RL Agent and the Simulator.
    Converts raw weight tensor -> portfolio -> simulation -> scalar reward.
    """
    # Convert weight array to {ticker: weight} dict
    portfolio_weights = {tickers[i]: float(weights[0, i]) for i in range(len(tickers))}

    # Run full simulation
    metrics = evaluate_portfolio_member_c(
        dataset, {"portfolio_weights": portfolio_weights}, user_profile, config,
        debug_path=debug_path, start_year_idx=start_year_idx, start_date=start_date,
        sim_cache_bundle=sim_cache_bundle
    )

    # Compute scalar reward
    reward = _compute_reward(metrics, user_profile)
    return float(reward), metrics
