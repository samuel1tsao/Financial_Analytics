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
import joblib
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
                res_dict["meta_cap_ratio"] = cfg["meta_cap_ratio"]
                res_dict["meta_inc_ratio"] = cfg["meta_inc_ratio"]
                res_dict["meta_dist"]      = cfg["meta_dist"]
                
            results.append(res_dict)

    return results


def _run_single_year_path(year_idx, start_year_idx, num_years, annual_returns, start_capital, goals, mode, max_horizon_years, debug_path):
    """
    Helper function for parallel execution. Simulates a single historical start-year path.
    """
    actual_start_date = annual_returns.index[start_year_idx].year
    current_balance = start_capital
    bankrupt = False
    trail = []
    
    # Decide how many years this specific simulation will run
    sim_horizon = max_horizon_years if mode == "loop" else (num_years - start_year_idx)

    for step_year_idx in range(sim_horizon):
        # Calculate data index (supports looping via modulo)
        data_idx = (start_year_idx + step_year_idx) % num_years
        yr_return = annual_returns.iloc[data_idx]
        
        # Update balance
        current_balance = current_balance * (1 + yr_return)
        actual_year_in_sim = step_year_idx + 1
        
        step_str = f"Y{actual_year_in_sim}: ${current_balance:,.0f} ({yr_return:+.1%})"
        
        # Check for goals
        if actual_year_in_sim in goals:
            withdrawal = goals[actual_year_in_sim]
            current_balance -= withdrawal
            step_str += f" | GOAL -${withdrawal:,.0f} -> ${current_balance:,.0f}"
            if current_balance < 0:
                bankrupt = True
                step_str += " [BANKRUPT] ❌"
                trail.append(step_str)
                break 
        
        trail.append(step_str)

    log_str = ""
    if debug_path:
        status = "✅" if not bankrupt else "❌"
        log_str = f"    Path {actual_start_date} (Horizon {sim_horizon}): {' -> '.join(trail)} {status}"
        
    return {
        "bankrupt": bankrupt,
        "terminal_value": current_balance if not bankrupt else None,
        "log": log_str
    }


# ── Lightweight Annual Evaluation ─────────────────────────────────────────────
# Complements backtest_portfolio (daily-granularity training-label generator).
# This function uses annual rolling windows for quick GFR/ETV assessment.

def evaluate_portfolio_member_c(dataset, recommendations, user_profile, config, debug_path=False):
    """
    Member C: Portfolio evaluation via historical rolling-window backtesting.
    Restored with NaN-safe returns and configurable horizon modes.
    """
    weights = recommendations["portfolio_weights"]
    mode = config.get("sim_horizon_mode", "ignore") # 'ignore' or 'loop'

    # Use DRIP returns if available, otherwise price-only
    base_returns = dataset.get("drip_daily_returns") or dataset["daily_returns"]

    # Filter to simulation date window
    sim_start = pd.Timestamp(config.get("simulation_start_date", "2015-01-01"))
    sim_end   = pd.Timestamp(config.get("simulation_end_date", "2026-12-31"))

    # FIX: NaN-Safe Returns. Only include tickers with non-zero WEIGHTS in the math.
    active_tickers = [t for t in weights.keys() if weights[t] > 0.0001 and t in base_returns.columns]
    
    if not active_tickers:
        if debug_path: print(f"  [DEBUG] No active tickers with data found.")
        return {"GFR": 0, "ETV": 0, "Objective_Function_Score": 0, "Total_Simulations": 0}

    sim_returns = base_returns[active_tickers].loc[sim_start:sim_end]

    # Compute portfolio-weighted daily returns
    weight_series = pd.Series({t: weights[t] for t in active_tickers})
    weight_series = weight_series / weight_series.sum()  # ensure sum=1.0
    
    # FIX: Dynamically reapportion weights to assets that are actively trading.
    # If a stock hasn't IPO'd yet (NaN return), its weight shifts proportionally to public stocks.
    valid_returns_mask = sim_returns.notna()
    daily_raw_weights = valid_returns_mask * weight_series 
    daily_weight_sums = daily_raw_weights.sum(axis=1)
    
    # Normalize daily weights to 1.0, fallback to 0.0 if NO stocks are trading
    daily_normalized_weights = daily_raw_weights.div(daily_weight_sums, axis=0).fillna(0.0)
    
    # Dot product is now NaN-safe and mathematically accurate for available universe
    portfolio_returns = (sim_returns.fillna(0.0) * daily_normalized_weights).sum(axis=1)
    annual_returns = portfolio_returns.resample('YE').apply(lambda x: (1+x).prod() - 1)

    max_horizon_years = max(user_profile['goals'].keys()) if user_profile['goals'] else 1
    start_capital = user_profile['start_cap']
    successful_simulations, total_simulations = 0, 0
    terminal_values = []

    if debug_path:
        print(f"\n  [SIMULATOR DEBUG] Parallel Mode: ON | Cores: {joblib.cpu_count()} | Available Data: {len(annual_returns)} years | Goal Horizon: {max_horizon_years} years")

    # Rolling window: Parallelized execution across cores
    num_years = len(annual_returns)
    
    results = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(_run_single_year_path)(
            i, i, num_years, annual_returns, start_capital, 
            user_profile['goals'], mode, max_horizon_years, debug_path
        )
        for i in range(num_years)
        if not (mode == "ignore" and (num_years - i) < min(max_horizon_years, 5))
    )

    successful_simulations = 0
    total_simulations = len(results)
    terminal_values = []
    
    for res in results:
        if not res["bankrupt"]:
            successful_simulations += 1
            terminal_values.append(res["terminal_value"])
        
        if debug_path and res["log"]:
            print(res["log"])

    GFR = successful_simulations / total_simulations if total_simulations > 0 else 0
    ETV = np.median(terminal_values) if len(terminal_values) > 0 else 0
    
    # Scoring: weight ETV by GFR to penalize risky portfolios
    from _constants import GFR_OBJECTIVE_THRESHOLD
    alpha = 1.0
    objective_score = (alpha * ETV) if GFR >= GFR_OBJECTIVE_THRESHOLD else (alpha * ETV) * (GFR / GFR_OBJECTIVE_THRESHOLD)

    max_terminal = np.max(terminal_values) if len(terminal_values) > 0 else 0
    min_terminal = np.min(terminal_values) if len(terminal_values) > 0 else 0

    return {
        "GFR": GFR,
        "ETV": ETV,
        "max_terminal": max_terminal,
        "min_terminal": min_terminal,
        "Objective_Function_Score": objective_score,
        "Total_Simulations": total_simulations
    }

# ── RL Environment Step ───────────────────────────────────────────────────────

def simulate_rl_environment_step(weights, tickers, dataset, user_profile, config, debug_path=False):
    """
    Phase 4: The Black-Box Environment.
    Evaluates sampled portfolio weights through the non-differentiable simulator
    and returns a single scalar Reward Score.
    """
    # 1. Map sampled tensor weights to ticker dictionary
    portfolio_weights = {tickers[i]: float(weights[0, i]) for i in range(len(tickers))}
    
    # 2. Evaluate portfolio using existing non-differentiable rolling window logic
    # This evaluates how the portfolio handles real historical market paths.
    metrics = evaluate_portfolio_member_c(
        dataset, 
        {"portfolio_weights": portfolio_weights}, 
        user_profile, 
        config,
        debug_path=debug_path
    )
    
    # 3. Retrieve Reward Hyperparameters & Dynamic Target
    # Dynamic target based on "net" value of portfolio minus goals, compounded over horizon.
    start_cap = user_profile.get("start_cap", 100000.0)
    goals = user_profile.get("goals", {})
    total_goal_cost = sum(goals.values())
    max_horizon = max(goals.keys()) if goals else 10

    #USING RISK TO ADJUST PENALTY AND REWARDS FOR DIFFERENT PROFILES
    risk = float(user_profile.get("risk_tolerance", 5.0))
    risk01 = np.clip(risk / 10.0, 0.0, 1.0)   # map to [0, 1]
    
    net_starting_value = max(start_cap - total_goal_cost, 10000.0)
    baseline_growth = config.get("reward_baseline_growth", 0.06)  # 6% baseline growth
    dynamic_term_target = net_starting_value * ((1.0 + baseline_growth) ** max_horizon)
    
    # Allow config override, else use the logical dynamic target
    term_target = config.get("reward_terminal_target", None)
    if term_target is None or term_target <= 0.0:
        term_target = dynamic_term_target
        
    # Base defaults from config (used as anchors)
    base_term_k = config.get("reward_terminal_k", 2.0)
    base_penalty_rate = config.get("reward_goal_penalty_rate", 0.5)
    base_lambda_term = config.get("reward_lambda_terminal", 1.0)
    base_lambda_goal = config.get("reward_lambda_goal", 1.0)

    # TEMPORARY CHANGE BUT CAN ADD THESE CONSTANTS TO USER_PROFILES INSTEAD IF THIS CHANGE IS GOOD
    # Conservative -> stronger failure aversion
    # Aggressive   -> stronger upside preference
    term_k = base_term_k * (0.85 + 0.55 * risk01)          # ~1.7 to ~2.8 if base=2.0
    penalty_rate = base_penalty_rate * (1.4 - 0.8 * risk01) # ~0.66 to ~0.30 if base=0.5
    lambda_term = base_lambda_term * (0.75 + 0.90 * risk01) # ~0.84 to ~1.56 if base=1.0
    lambda_goal = base_lambda_goal * (1.7 - 1.1 * risk01)   # ~1.59 to ~0.60 if base=1.0
    
    ETV = metrics["ETV"]
    GFR = metrics["GFR"] # 0.0 to 1.0 (1.0 means all goals funded without bankruptcy)
    
    # 4. Compute Inverse-Exponential Terminal Reward
    # Sigmoid-like scaling: heavily penalizes small balances, rewards large balances exponentially until saturation.
    # We normalize ETV to the target to keep exponents numerically stable.
    normalized_balance = (ETV - term_target) / max(term_target, 1.0)
    terminal_reward = 1.0 / (1.0 + np.exp(-term_k * normalized_balance))
    
    # 5. Compute Goal Penalty 
    # (1 - GFR) represents the fraction of simulations that went bankrupt or missed goals
    goal_penalty = penalty_rate * (1.0 - GFR)
    
    # 6. Final Scalar Reward
    reward = (lambda_term * terminal_reward) - (lambda_goal * goal_penalty)
    
    return float(reward), metrics
