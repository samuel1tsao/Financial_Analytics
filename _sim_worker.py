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
    TRADING_DAYS_PER_YEAR as TRADING_DAYS,
    SIM_YEARS,
    LOAN_DAILY_RATE,
    CASH_BUCKET_ANNUAL_RATE,
    DAILY_RETURN_CLAMP,
    GFR_OBJECTIVE_THRESHOLD,
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
        cash_bucket   *= (1 + CASH_BUCKET_ANNUAL_RATE / TRADING_DAYS)
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
        year_idx = day_off // TRADING_DAYS + 1
        if day_off > 0 and day_off % TRADING_DAYS == 0 and year_idx in goals:
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
        cash_bucket   *= (1 + CASH_BUCKET_ANNUAL_RATE / TRADING_DAYS)
        growth_ret     = float(portfolio_daily_returns[i])
        growth_ret     = max(-DAILY_RETURN_CLAMP, min(DAILY_RETURN_CLAMP, growth_ret))
        growth_bucket *= (1 + growth_ret)

        if debt > 0:
            payment        = min(daily_contrib, debt)
            debt          -= payment
            growth_bucket += daily_contrib - payment
        else:
            growth_bucket += daily_contrib

        year_idx = day_off // TRADING_DAYS + 1
        if day_off > 0 and day_off % TRADING_DAYS == 0:
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
    n_days      = TRADING_DAYS * SIM_YEARS
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

from _constants import (
    TRADING_DAYS_PER_YEAR,
    GFR_OBJECTIVE_THRESHOLD,
    MIN_ACTIVE_WEIGHT,
    TOP_K_ASSETS,
    SIM_MONTHLY_START_YEAR,
    SIM_TERMINAL_HORIZON,
    DEBUG_LOG_MAX_YEARS,
    CAGR_REWARD_SCALE,
    MDD_PENALTY_SCALE,
    MAX_RISK_OFFSET,
    GFR_BRACKETS,
    GFR_MISS_FLAT_PENALTY,
    GFR_PERFECT_REWARD,
)


# ── Sim Helpers ───────────────────────────────────────────────────────────────

def _extract_yearly_chunk(daily_returns, start_idx, year_number):
    """Extract a 252-day trading chunk for a given simulation year, wrapping cyclically."""
    n_days = len(daily_returns)
    chunk_start = (start_idx + (year_number - 1) * TRADING_DAYS_PER_YEAR) % n_days
    chunk_end   = (start_idx + year_number * TRADING_DAYS_PER_YEAR) % n_days

    if hasattr(daily_returns, "iloc"):
        if chunk_end > chunk_start:
            return daily_returns.iloc[chunk_start:chunk_end]
        # Wrap around
        return pd.concat([daily_returns.iloc[chunk_start:], daily_returns.iloc[:chunk_end]])
    else:
        # NumPy path
        if chunk_end > chunk_start:
            return daily_returns[chunk_start:chunk_end]
        # Wrap around
        return np.vstack((daily_returns[chunk_start:], daily_returns[:chunk_end]))


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
    Precompute annual returns for every asset, for every standard simulation start index, for every year.
    Returns: (numpy 3D array cache_array, dict start_idx_to_pos, dataframe clean_returns, dict column_to_idx)
    """
    import time
    print(f"[{time.strftime('%H:%M:%S')}] [Member C] Building Annual Simulation Cache...")
    
    clean_returns = base_returns.fillna(0.0)
    num_assets = len(clean_returns.columns)
    column_to_idx = {col: i for i, col in enumerate(clean_returns.columns)}
    
    start_indices = _resolve_simulation_starts(clean_returns)
    num_starts = len(start_indices)
    start_idx_to_pos = {idx: i for i, idx in enumerate(start_indices)}
    
    # Use float32 to save memory while maintaining precision for returns
    cache_array = np.zeros((num_starts, max_horizon_years, num_assets), dtype=np.float32)
    
    # ── HYPER-FAST NUMPY CONVERSION ──
    from _constants import TRADING_DAYS_PER_YEAR
    n_days = len(clean_returns)
    one_plus_ret = (1.0 + clean_returns.values).astype(np.float32)
    
    for i, start_idx in enumerate(start_indices):
        for year in range(1, max_horizon_years + 1):
            chunk_start = (start_idx + (year - 1) * TRADING_DAYS_PER_YEAR) % n_days
            chunk_end   = (start_idx + year * TRADING_DAYS_PER_YEAR) % n_days
            
            if chunk_end > chunk_start:
                chunk = one_plus_ret[chunk_start:chunk_end]
            else:
                chunk = np.vstack((one_plus_ret[chunk_start:], one_plus_ret[:chunk_end]))
                
            cache_array[i, year - 1, :] = chunk.prod(axis=0) - 1.0
            
    print(f"[{time.strftime('%H:%M:%S')}] [Member C] Cache built: {cache_array.shape} (starts x years x assets)")
    return cache_array, start_idx_to_pos, clean_returns, column_to_idx


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

    # Default: one start per month from SIM_MONTHLY_START_YEAR onward
    recent = portfolio_returns[portfolio_returns.index.year >= SIM_MONTHLY_START_YEAR]
    month_ends = recent.resample('ME').last().index
    # Use 'pad' to snap calendar month-ends to the nearest prior trading day
    indices = portfolio_returns.index.get_indexer(month_ends, method='pad')
    return [i for i in indices if i >= 0]


def _aggregate_path_results(results):
    """Compute GFR, ETV, MDD, and Actual Withdrawals from a list of path result dicts."""
    total        = len(results)
    successes    = sum(1 for r in results if not r["bankrupt"])
    goal_fails   = sum(1 for r in results if r["bankrupt"] and r.get("bankrupt_reason") == "goal")
    market_fails = sum(1 for r in results if r["bankrupt"] and r.get("bankrupt_reason") == "market")
    all_tvs      = [r["terminal_value"] for r in results]
    all_mdds     = [r["max_drawdown"] for r in results]
    all_wds      = [r.get("actual_withdrawals", 0.0) for r in results]

    gfr = successes / total if total > 0 else 0
    etv = np.mean(all_tvs) if all_tvs else 0.0
    mdd = np.mean(all_mdds) if all_mdds else 1.0
    wds = np.mean(all_wds) if all_wds else 0.0
    return gfr, etv, mdd, wds, goal_fails, market_fails


def _compute_reward(metrics, user_profile):
    """
    3-part reward: Annual return score – risk-scaled MDD penalty – softened GFR penalty.
    Horizon is always SIM_TERMINAL_HORIZON (30y) since post-goal capital keeps compounding.
    All scaling constants live in _constants.py for easy tuning.
    """
    start_cap  = float(user_profile["start_cap"])
    horizon    = SIM_TERMINAL_HORIZON
    risk_tol   = float(user_profile["risk_tolerance"])
    actual_wds = metrics.get("AW", sum(user_profile["goals"].values())) # Fallback for old logs

    # 1. Annual Simple Return Score — smooth gradient, avoids division by zero
    total_profit = metrics["ETV"] + actual_wds - start_cap
    annual_return = (total_profit / start_cap) / horizon
    return_score = annual_return * CAGR_REWARD_SCALE

    # 2. Risk-Scaled MDD Penalty
    mdd_penalty = metrics["MDD"] * MDD_PENALTY_SCALE * (MAX_RISK_OFFSET - risk_tol)

    # 3. GFR Bracketed Contribution (Success Rate over N paths)
    gfr = float(metrics["GFR"])
    x_coords = [b[0] for b in GFR_BRACKETS]
    y_coords = [b[1] for b in GFR_BRACKETS]
    
    # Sort for interpolation
    idx_sort = np.argsort(x_coords)
    x_sorted = np.array(x_coords)[idx_sort]
    y_sorted = np.array(y_coords)[idx_sort]
    
    gfr_contrib = float(np.interp(gfr, x_sorted, y_sorted))
    gfr_penalty = -min(0, gfr_contrib)
    gfr_bonus   = max(0, gfr_contrib)

    total_reward = return_score + gfr_contrib - mdd_penalty
    
    metrics["reward_components"] = {
        "return_score": return_score,
        "mdd_penalty": mdd_penalty,
        "gfr_penalty": gfr_penalty,
        "gfr_bonus": gfr_bonus,
        "annual_return": annual_return
    }

    return total_reward


# ── Core Path Runner ──────────────────────────────────────────────────────────

def _run_single_sim_path(start_idx, weight_array, sim_cache, clean_returns, start_capital, goals, mode,
                         max_horizon_years, debug_path, start_date_str,
                         post_goal_weights=None, goal_year=None):
    """
    Simulate one trajectory using FAST precomputed annual dot products.

    Two-phase allocation:
        - Years 1..goal_year use `weight_array` (the pre-goal allocation)
        - Years goal_year+1..max_horizon use `post_goal_weights` (the growth-phase allocation)
        If post_goal_weights is None, weight_array is used for the entire horizon (legacy behavior).
    """
    capital           = start_capital
    market_multiplier = 1.0
    mi_peak           = 1.0
    max_drawdown      = 0.0
    trail             = []
    phase_switched    = False
    
    # On-the-fly computation if missing from cache (e.g. custom Walk-Forward dates)
    if isinstance(sim_cache, tuple):
        cache_array, start_idx_to_pos = sim_cache
        if start_idx in start_idx_to_pos:
            precomputed_returns = cache_array[start_idx_to_pos[start_idx]]
        else:
            # Fallback for indices not in the precomputed cache array
            precomputed_returns = np.zeros((max_horizon_years, clean_returns.shape[1]))
            for year in range(1, max_horizon_years + 1):
                chunk = _extract_yearly_chunk(clean_returns, start_idx, year)
                if hasattr(chunk, "values"):
                    precomputed_returns[year - 1, :] = (1 + chunk.values).prod(axis=0) - 1
                else:
                    precomputed_returns[year - 1, :] = (1 + chunk).prod(axis=0) - 1
    elif start_idx in sim_cache:
        precomputed_returns = sim_cache[start_idx]
    else:
        precomputed_returns = np.zeros((max_horizon_years, len(clean_returns.columns)))
        for year in range(1, max_horizon_years + 1):
            chunk = _extract_yearly_chunk(clean_returns, start_idx, year)
            precomputed_returns[year - 1, :] = (1 + chunk).prod() - 1
        sim_cache[start_idx] = precomputed_returns

    actual_withdrawals = 0.0
    
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
        market_multiplier *= (1 + yr_return)

        # Track drawdown based on MARKET performance, not cash balance
        mi_peak, max_drawdown = _update_drawdown(market_multiplier, mi_peak, max_drawdown)

        # Track actual withdrawals before it's deducted
        if year in goals:
            target = goals[year]
            if capital >= target:
                actual_withdrawals += target
            else:
                actual_withdrawals += max(capital, 0.0)

        # Process goal withdrawal (if any)
        capital_before = capital
        capital, bankrupt, goal_log = _process_goal_withdrawal(capital, year, goals)

        # Build debug trail
        phase_tag = " [POST]" if (post_goal_weights is not None and goal_year is not None and year > goal_year) else ""
        step_str = f"Y{year}: ${capital:,.0f} ({yr_return:+.1%}){goal_log}{phase_tag}"
        if bankrupt:
            trail.append(step_str)
            reason = "goal" if capital_before > 0 else "market"
            reported_mdd = max_drawdown if reason == "goal" else 1.0
            return {"bankrupt": True, "bankrupt_reason": reason, "terminal_value": 0.0,
                    "max_drawdown": reported_mdd,
                    "actual_withdrawals": actual_withdrawals, "log": " -> ".join(trail)}
        if debug_path and year <= DEBUG_LOG_MAX_YEARS:
            trail.append(step_str)

    log_msg = ""
    if debug_path:
        log_msg = f"    Start: {start_date_str} | Path: {' -> '.join(trail)} ✅"

    return {"bankrupt": False, "terminal_value": capital, "actual_withdrawals": actual_withdrawals,
            "max_drawdown": max_drawdown, "log": log_msg}


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
        cache_array, start_idx_to_pos, clean_returns, column_to_idx = build_simulation_cache(base_returns)
        sim_cache = (cache_array, start_idx_to_pos)

    # Convert weights dict -> sorted candidates -> two-stage top-K + normalized threshold
    # Mirrors the exact selection logic used in the RL training loop for train/eval consistency.
    def _build_weight_array(w_dict):
        all_candidates = [(t, w) for t, w in w_dict.items() if t in column_to_idx and w > 0]
        if not all_candidates:
            return None
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = all_candidates[:TOP_K_ASSETS]
        total_w        = sum(w for _, w in top_candidates)
        norm_candidates = [(t, w / total_w) for t, w in top_candidates]
        active = [(t, w) for t, w in norm_candidates if w > MIN_ACTIVE_WEIGHT]
        if not active:
            active = norm_candidates
        num_assets   = len(clean_returns.columns)
        wa = np.zeros(num_assets)
        for t, w in active:
            wa[column_to_idx[t]] = w
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
    max_horizon = SIM_TERMINAL_HORIZON if goal_year else (max(user_profile['goals'].keys()) if user_profile['goals'] else 1)
    start_cap   = user_profile['start_cap']

    if debug_path and len(sim_starts) > 1:
        print(f"  [SIMULATOR] Cyclic Mode: {mode.upper()} | "
              f"Starts: {len(sim_starts)} | Horizon: {max_horizon}y"
              f"{' | Two-Phase @ Y' + str(goal_year) if goal_year else ''}")

    # ---------------------------------------------------------------
    # 4️⃣ Subsample start‑indices if the user wants fewer paths per episode
    # ---------------------------------------------------------------
    max_paths = config.get("sim_paths_per_episode")
    if max_paths is not None and max_paths < len(sim_starts):
        rng = np.random.default_rng()
        sim_starts = rng.choice(sim_starts, size=max_paths, replace=False).tolist()

    # ---------------------------------------------------------------
    # 5️⃣ Run all selected paths **sequentially** – no joblib overhead
    # ---------------------------------------------------------------
    results = []
    for idx in sim_starts:
        date_str = str(clean_returns.index[idx].date())
        results.append(
            _run_single_sim_path(
                idx, weight_array, sim_cache, clean_returns, start_cap,
                user_profile['goals'], mode, max_horizon, debug_path, date_str,
                post_goal_weights=post_weight_array, goal_year=goal_year
            )
        )


    # Step 5: Aggregate metrics
    gfr, etv, mdd, aw, goal_fails, market_fails = _aggregate_path_results(results)
    score = etv if gfr >= GFR_OBJECTIVE_THRESHOLD else etv * (gfr / GFR_OBJECTIVE_THRESHOLD)

    metrics = {"GFR": gfr, "ETV": etv, "MDD": mdd, "AW": aw,
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
