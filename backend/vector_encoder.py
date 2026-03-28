"""
Vector Encoder: Transforms structured questionnaire JSON into actionable portfolio parameters.

The encoder maps user preferences into:
1. Target equity/bond split based on risk tolerance
2. FOMO adjustment factor for momentum-tilted assets
3. Hard constraint carve-outs
4. Short-term vs long-term goal segregation (cash-out logic)
"""
import numpy as np
from typing import Any


# ─── Asset Universe ──────────────────────────────────────────────────────────
# ETFs and their approximate characteristics for the MVP
EQUITY_UNIVERSE = {
    "VOO":  {"name": "S&P 500",           "risk": 0.5, "category": "large_cap"},
    "QQQ":  {"name": "Nasdaq 100",         "risk": 0.7, "category": "tech"},
    "VTI":  {"name": "Total Market",       "risk": 0.5, "category": "broad"},
    "VXUS": {"name": "International",      "risk": 0.6, "category": "intl"},
    "VGT":  {"name": "Info Tech",          "risk": 0.8, "category": "sector_tech"},
    "ARKK": {"name": "Innovation ETF",     "risk": 0.95, "category": "speculative"},
    "VNQ":  {"name": "Real Estate",        "risk": 0.6, "category": "reit"},
    "VWO":  {"name": "Emerging Markets",   "risk": 0.75, "category": "emerging"},
}

BOND_UNIVERSE = {
    "BND":  {"name": "Total Bond Market",  "risk": 0.1, "category": "bond"},
    "SGOV": {"name": "Short Treasury",     "risk": 0.02, "category": "stbond"},
    "TLT":  {"name": "20+ Year Treasury",  "risk": 0.25, "category": "lt_bond"},
    "TIPS": {"name": "Inflation Protected", "risk": 0.1, "category": "tips"},
}

# Expected annual returns (simplified, for MVP simulation)
EXPECTED_RETURNS = {
    "VOO": 0.10, "QQQ": 0.12, "VTI": 0.10, "VXUS": 0.07,
    "VGT": 0.13, "ARKK": 0.15, "VNQ": 0.08, "VWO": 0.09,
    "BND": 0.04, "SGOV": 0.045, "TLT": 0.035, "TIPS": 0.038,
}

# Annual volatility (std dev of returns)
VOLATILITIES = {
    "VOO": 0.16, "QQQ": 0.20, "VTI": 0.16, "VXUS": 0.17,
    "VGT": 0.22, "ARKK": 0.35, "VNQ": 0.20, "VWO": 0.22,
    "BND": 0.04, "SGOV": 0.01, "TLT": 0.14, "TIPS": 0.05,
}

# Simplified correlation matrix (pairwise)
# For MVP we use broad categories rather than full NxN matrix
CATEGORY_CORRELATIONS = {
    ("equity", "equity"): 0.75,
    ("equity", "bond"): -0.15,
    ("bond", "bond"): 0.60,
}


def classify_asset(ticker: str) -> str:
    """Classify ticker as 'equity' or 'bond' category."""
    if ticker in BOND_UNIVERSE:
        return "bond"
    return "equity"


def build_covariance_matrix(tickers: list[str]) -> np.ndarray:
    """Build a variance-covariance matrix for the given tickers."""
    n = len(tickers)
    cov = np.zeros((n, n))
    for i, t1 in enumerate(tickers):
        for j, t2 in enumerate(tickers):
            vol1 = VOLATILITIES.get(t1, 0.15)
            vol2 = VOLATILITIES.get(t2, 0.15)
            cat1 = classify_asset(t1)
            cat2 = classify_asset(t2)
            if i == j:
                cov[i][j] = vol1 ** 2
            else:
                pair = tuple(sorted([cat1, cat2]))
                corr = CATEGORY_CORRELATIONS.get(pair, 0.5)
                cov[i][j] = corr * vol1 * vol2
    return cov


def encode_questionnaire(answers: dict) -> dict:
    """
    Transform questionnaire answers into portfolio parameters.

    Returns:
        {
          "weights": {"VOO": 0.40, "BND": 0.20, ...},
          "goals": [{"name": ..., "amount": ..., "years": ..., "is_short_term": bool}],
          "risk_score": float,
          "fomo_score": int,
        }
    """
    risk = answers.get("risk_tolerance", 50)
    fomo = answers.get("fomo_tendency", 5)
    goals = answers.get("goals", [])
    hard_constraints = answers.get("hard_constraints", [])

    # ─── 1. Base equity/bond split from risk tolerance ─────────────────────
    # risk=1 → 20% equity, risk=100 → 95% equity
    equity_pct = 0.20 + (risk / 100) * 0.75
    bond_pct = 1.0 - equity_pct

    # ─── 2. FOMO adjustment: high FOMO → tilt toward speculative/tech ─────
    # fomo 1-3: conservative equity basket, 4-7: balanced, 8-10: aggressive
    fomo_tilt = max(0, (fomo - 3) / 7)  # 0 to 1 scale

    # ─── 3. Separate short-term goals (≤5 years) to route to bonds/SGOV ──
    enriched_goals = []
    short_term_capital_ratio = 0.0
    for g in goals:
        years = g.get("years", 10)
        is_short = years <= 5
        enriched_goals.append({**g, "is_short_term": is_short})
        if is_short:
            # Approximate how much of the portfolio should be near-cash
            short_term_capital_ratio += 0.15  # Each short-term goal reserves ~15%

    short_term_capital_ratio = min(short_term_capital_ratio, 0.50)

    # Adjust: move some equity → bond for short-term safety
    equity_pct = equity_pct * (1 - short_term_capital_ratio * 0.5)
    bond_pct = 1.0 - equity_pct

    # ─── 4. Carve out hard constraints ────────────────────────────────────
    hard_total = 0.0
    hard_weights = {}
    for c in hard_constraints:
        ticker = c.get("ticker", "").upper()
        pct = c.get("pct", 0) / 100.0
        if ticker and pct > 0:
            hard_weights[ticker] = pct
            hard_total += pct

    remaining = max(0, 1.0 - hard_total)
    equity_pct *= remaining
    bond_pct *= remaining

    # ─── 5. Distribute equity allocation across universe ──────────────────
    weights = dict(hard_weights)

    # Core equity positions
    if equity_pct > 0:
        # Base: 50% VOO/VTI, rest distributed by fomo_tilt
        core_equity = equity_pct * (1 - fomo_tilt * 0.4)
        speculative_equity = equity_pct * fomo_tilt * 0.4

        weights["VOO"] = weights.get("VOO", 0) + core_equity * 0.45
        weights["VTI"] = weights.get("VTI", 0) + core_equity * 0.25
        weights["VXUS"] = weights.get("VXUS", 0) + core_equity * 0.15
        weights["VNQ"] = weights.get("VNQ", 0) + core_equity * 0.15

        # Speculative tilt from FOMO
        if speculative_equity > 0:
            weights["QQQ"] = weights.get("QQQ", 0) + speculative_equity * 0.45
            weights["VGT"] = weights.get("VGT", 0) + speculative_equity * 0.30
            weights["ARKK"] = weights.get("ARKK", 0) + speculative_equity * 0.15
            weights["VWO"] = weights.get("VWO", 0) + speculative_equity * 0.10

    # Bond positions
    if bond_pct > 0:
        if short_term_capital_ratio > 0:
            # Heavy into SGOV for short-term safety
            weights["SGOV"] = weights.get("SGOV", 0) + bond_pct * 0.50
            weights["BND"] = weights.get("BND", 0) + bond_pct * 0.30
            weights["TIPS"] = weights.get("TIPS", 0) + bond_pct * 0.20
        else:
            weights["BND"] = weights.get("BND", 0) + bond_pct * 0.55
            weights["TLT"] = weights.get("TLT", 0) + bond_pct * 0.25
            weights["TIPS"] = weights.get("TIPS", 0) + bond_pct * 0.20

    # Normalize to ensure sum = 1.0 and remove tiny positions (<1%)
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    weights = {k: round(v, 4) for k, v in weights.items() if v >= 0.01}

    # Re-normalize after removing tiny positions
    total = sum(weights.values())
    if total > 0:
        weights = {k: round(v / total, 4) for k, v in weights.items()}

    return {
        "weights": weights,
        "goals": enriched_goals,
        "risk_score": risk,
        "fomo_score": fomo,
    }


def simulate_portfolio(
    weights: dict,
    goals: list[dict],
    initial_investment: float = 100000,
    years: int = 30,
    num_std: float = 2.0,
) -> dict:
    """
    Run a variance-covariance Monte Carlo-style projection.

    Returns expected path, upper/lower bounds (±2σ), and cash-out events.
    """
    tickers = list(weights.keys())
    w = np.array([weights[t] for t in tickers])

    # Portfolio expected return (weighted sum)
    mu_p = sum(weights[t] * EXPECTED_RETURNS.get(t, 0.08) for t in tickers)

    # Portfolio variance from var-covar matrix
    cov = build_covariance_matrix(tickers)
    sigma_p_sq = float(w @ cov @ w)
    sigma_p = float(np.sqrt(sigma_p_sq))

    # ─── Build year-by-year projection ────────────────────────────────────
    expected_path = [initial_investment]
    upper_bound = [initial_investment]
    lower_bound = [initial_investment]
    cash_out_events = []

    current_value = initial_investment

    # Sort goals by year for cash-out processing
    sorted_goals = sorted(
        [g for g in goals if g.get("is_short_term", False)],
        key=lambda g: g.get("years", 99),
    )

    for year in range(1, years + 1):
        # Grow the portfolio
        expected_val = current_value * (1 + mu_p)
        upper_val = current_value * (1 + mu_p + num_std * sigma_p)
        lower_val = current_value * max(0.01, (1 + mu_p - num_std * sigma_p))

        # Process cash-out events for this year
        for goal in sorted_goals:
            goal_year = goal.get("years", 99)
            if goal_year == year:
                cashout = goal.get("amount", 0)
                expected_val = max(0, expected_val - cashout)
                upper_val = max(0, upper_val - cashout)
                lower_val = max(0, lower_val - cashout)
                cash_out_events.append({
                    "year": year,
                    "goal_name": goal.get("name", "Goal"),
                    "amount": cashout,
                    "remaining_expected": round(expected_val, 2),
                })

        expected_path.append(round(expected_val, 2))
        upper_bound.append(round(upper_val, 2))
        lower_bound.append(round(max(0, lower_val), 2))
        current_value = expected_val

    # Build year labels
    year_labels = list(range(0, years + 1))

    # Goal annotations for the chart
    goal_annotations = []
    for g in goals:
        y = g.get("years", 10)
        if y <= years:
            goal_annotations.append({
                "year": y,
                "label": g.get("name", "Goal"),
                "amount": g.get("amount", 0),
                "is_short_term": g.get("is_short_term", False),
            })

    return {
        "years": year_labels,
        "expected_path": expected_path,
        "upper_bound": upper_bound,
        "lower_bound": lower_bound,
        "cash_out_events": cash_out_events,
        "goal_annotations": goal_annotations,
        "portfolio_stats": {
            "expected_annual_return": round(mu_p * 100, 2),
            "annual_volatility": round(sigma_p * 100, 2),
            "sharpe_ratio": round((mu_p - 0.04) / sigma_p, 2) if sigma_p > 0 else 0,
            "initial_investment": initial_investment,
            "projected_final_expected": expected_path[-1],
            "projected_final_upper": upper_bound[-1],
            "projected_final_lower": lower_bound[-1],
        },
    }
