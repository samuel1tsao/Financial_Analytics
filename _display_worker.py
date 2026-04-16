"""
_display_worker.py
──────────────────
Visualization and reporting functions for portfolio recommendations.

Public API:
    display_dynamic_top_k(dataset, recommendations, user_profile, user_vector) → dict
    plot_theta_sensitivity(profiles, data_cache, config, theta_grid)           → None
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _scoring_worker import build_user_preference_vector, recommend_and_allocate_member_b
from _constants import DEFAULT_THETA_GRID


# ═══════════════════════════════════════════════════════════════════════
# DYNAMIC TOP-K RECOMMENDATION DISPLAY
# ═══════════════════════════════════════════════════════════════════════

def display_dynamic_top_k(dataset, recommendations, user_profile, user_vector):
    """
    Full diagnostic display for Dynamic Top-K recommendations.
    Uses logarithmic V_score (0-10) in all outputs for interpretability.

    Generates:
        1. Score distribution histogram with theta cutoff
        2. V_score vs similarity scatter plot
        3. Sector allocation pie chart
        4. Portfolio weight bar chart
        5. Selected and rejected asset tables
        6. Portfolio summary statistics

    Returns:
        dict with k, weighted_volatility, weighted_vscore, hhi, effective_n, rejected_count
    """
    master_df   = dataset["master_df"]
    all_scores  = recommendations["all_scores"]
    selected    = recommendations["scores"]
    weights     = recommendations["portfolio_weights"]
    theta       = recommendations["theta"]
    T           = recommendations["temperature"]
    components  = recommendations["score_components"]
    risk_budget = recommendations["risk_budget"]
    n_k         = len(selected)

    # —— Header ——
    _print_report_header(user_profile, risk_budget, theta, T, recommendations, n_k)

    # —— Visualizations ——
    _plot_recommendation_charts(
        all_scores, selected, weights, theta, T, risk_budget,
        components, n_k, master_df
    )

    # —— Selected Asset Table ——
    _print_selected_assets(selected, components, weights, n_k)

    # —— Rejected Asset Table ——
    rej = _print_rejected_assets(all_scores, selected, components, theta)

    # —— Portfolio Summary Stats ——
    stats = _compute_portfolio_stats(selected, weights, components, risk_budget, rej)

    return stats


# ═══════════════════════════════════════════════════════════════════════
# THETA SENSITIVITY ANALYSIS PLOT
# ═══════════════════════════════════════════════════════════════════════

def plot_theta_sensitivity(profiles, data_cache, config, evaluate_fn, theta_grid=None):
    """
    Plot how Dynamic K threshold (theta) affects recommendations for each profile.

    Args:
        profiles:     list of user profile dicts
        data_cache:   DATA_CACHE dict from training pipeline
        config:       PIPELINE_CONFIG dict
        evaluate_fn:  evaluate_portfolio_member_c function
        theta_grid:   list of theta values to test (default: DEFAULT_THETA_GRID)
    """
    if theta_grid is None:
        theta_grid = DEFAULT_THETA_GRID

    fig, axes = plt.subplots(1, len(profiles), figsize=(6 * len(profiles), 5))
    fig.suptitle("Threshold (\u03b8) Sensitivity: Dynamic K vs GFR vs ETV",
                 fontsize=14, fontweight='bold')

    if len(profiles) == 1:
        axes = [axes]

    for ax_idx, profile in enumerate(profiles):
        user_vec = build_user_preference_vector(data_cache, profile, config)

        k_vals, gfr_vals, etv_vals = [], [], []

        for theta_val in theta_grid:
            test_config = config.copy()
            test_config["sim_dynamic_k_theta"] = theta_val

            recs = recommend_and_allocate_member_b(
                dataset=data_cache,
                user_profile=profile,
                user_vector=user_vec,
                config=test_config
            )
            metrics = evaluate_fn(data_cache, recs, profile, config=config)

            k_vals.append(recs["k_selected"])
            gfr_vals.append(metrics["GFR"])
            etv_vals.append(metrics["ETV"])

        ax = axes[ax_idx]
        ax2 = ax.twinx()

        l1 = ax.plot(theta_grid, k_vals, 'o-', color='#2196F3', linewidth=2,
                     markersize=6, label='K (assets)')
        l2 = ax2.plot(theta_grid, gfr_vals, 's--', color='#4CAF50', linewidth=2,
                      markersize=6, label='GFR')
        l3 = ax2.plot(theta_grid,
                      [e / max(max(etv_vals, default=1), 1) for e in etv_vals],
                      '^:', color='#FF9800', linewidth=2, markersize=6,
                      label='ETV (norm)')

        ax.set_xlabel('Threshold (\u03b8)')
        ax.set_ylabel('Dynamic K (# assets)', color='#2196F3')
        ax2.set_ylabel('GFR / Normalized ETV')
        ax.set_title(f'{profile["profile_name"][:25]}\nRisk={profile["risk_tolerance"]}')
        ax.grid(alpha=0.3)

        lines = l1 + l2 + l3
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, fontsize=7, loc='upper right')

    plt.tight_layout()
    plt.show()
    print("\nThreshold Analysis Complete.")


# ═══════════════════════════════════════════════════════════════════════
# HELPERS — Called only by display_dynamic_top_k (flat, no nesting)
# ═══════════════════════════════════════════════════════════════════════

def _print_report_header(user_profile, risk_budget, theta, T, recommendations, n_k):
    """Print the report header block."""
    print("\n" + "\u2588" * 90)
    print("\u2588  DYNAMIC TOP-K ASSET RECOMMENDATION REPORT")
    print("\u2588" * 90)
    print(f"\u2588  Profile:              {user_profile.get('profile_name', 'Custom User')}")
    print(f"\u2588  Risk Tolerance:       {user_profile['risk_tolerance']}/10")
    print(f"\u2588  Risk Budget (Glide):  {risk_budget:.3f}/10 (Linear Scale)"
          f"(V_score ceiling after multi-goal decay)")
    print(f"\u2588  Threshold (\u03b8):        {theta}  (S_i >= \u03b8 to enter portfolio)")
    print(f"\u2588  Temperature (T):      {T:.2f}  "
          f"{'(diversified — high T)' if T > 2 else '(concentrated — low T)' if T < 1 else '(balanced)'}")
    print(f"\u2588  Universe Scored:      {recommendations['total_universe']} assets")
    print(f"\u2588  Dynamic K:            {n_k} assets passed \u03b8  "
          f"({n_k/max(recommendations['total_universe'],1):.1%} of universe)")
    print("\u2588" * 90)


def _plot_recommendation_charts(all_scores, selected, weights, theta, T,
                                 risk_budget, components, n_k, master_df):
    """Generate the 4-panel recommendation visualization."""
    fig_height = max(12, min(32, 8 + n_k * 0.18))
    hratio = [1, max(1.5, n_k * 0.05)]

    fig = plt.figure(figsize=(17, fig_height))
    gs  = fig.add_gridspec(2, 2, height_ratios=hratio)
    ax1 = fig.add_subplot(gs[0, 0])   # score histogram
    ax4 = fig.add_subplot(gs[0, 1])   # V_score vs similarity scatter
    ax3 = fig.add_subplot(gs[1, 0])   # sector pie
    ax2 = fig.add_subplot(gs[1, 1])   # weight bar chart

    fig.suptitle(
        f"Dynamic Top-K \u2014 K={n_k} | Risk Budget {risk_budget:.2f}/10 | T={T:.1f}",
        fontsize=14, fontweight="bold"
    )

    # Plot 1: Score distribution
    all_sv   = [v for v in all_scores.values() if np.isfinite(v)]
    sel_sv   = [v for v in selected.values() if np.isfinite(v)]
    if all_sv:
        ax1.hist(all_sv, bins=min(60, len(all_sv)), color="#2196F3", alpha=0.4,
                 label=f"All ({len(all_sv)})", edgecolor="white")
    if sel_sv:
        ax1.hist(sel_sv, bins=min(40, len(sel_sv)), color="#FF5722", alpha=0.7,
                 label=f"Selected K={n_k}", edgecolor="white")
    ax1.axvline(theta, color="#E91E63", linestyle="--", lw=2, label=f"\u03b8={theta}")
    ax1.set_xlabel("Composite Score S_i")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Score Distribution & Dynamic K Threshold")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot 2: Weight bar chart (all K assets)
    sw  = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    tpl = [t for t, _ in sw]
    wpl = [w for _, w in sw]
    bc  = plt.cm.viridis(np.linspace(0.2, 0.9, n_k))
    bars = ax2.barh(range(n_k), wpl, color=bc, edgecolor="white")
    ax2.set_yticks(range(n_k))
    ax2.set_yticklabels(tpl, fontsize=max(5, min(8, 200 // max(n_k, 1))))
    ax2.set_xlabel("Portfolio Weight")
    ax2.set_title(f"All {n_k} Portfolio Weights")
    ax2.invert_yaxis()
    for bar, wv in zip(bars, wpl):
        ax2.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                 f"{wv:.2%}", va="center", fontsize=max(5, min(7, 200 // max(n_k, 1))))
    ax2.grid(axis="x", alpha=0.3)

    # Plot 3: Sector pie
    sector_map = {}
    for ticker in selected:
        if ticker not in master_df.index:
            continue
        sector = "Unknown"
        if "sector" in master_df.columns:
            s = str(master_df.loc[ticker, "sector"])
            if s != "nan":
                sector = s
        if sector == "Unknown":
            sc2 = [c for c in master_df.columns if c.startswith("sector_")]
            if sc2:
                hv = master_df.loc[ticker, sc2]
                hot = hv[hv == 1.0]
                if len(hot):
                    sector = hot.index[0].replace("sector_", "")
        sector_map[sector] = sector_map.get(sector, 0) + weights.get(ticker, 0)
    if sector_map:
        ss = sorted(sector_map.items(), key=lambda x: x[1], reverse=True)
        sl = [s[:18] for s, _ in ss]
        sv2 = [v for _, v in ss]
        pc = plt.cm.Set3(np.linspace(0, 1, len(sl)))
        ax3.pie(sv2, labels=sl, autopct="%1.1f%%", colors=pc,
                startangle=90, pctdistance=0.8,
                textprops={"fontsize": 7})
        ax3.set_title("Sector Allocation (by weight)")
    else:
        ax3.text(0.5, 0.5, "No sector data", ha="center", va="center")

    # Plot 4: V_score vs Similarity scatter — budget line on x-axis
    sel_t = list(selected.keys())
    uns_t = [t for t in all_scores if t not in selected]
    if uns_t:
        ux = [components[t]["V_score"] for t in uns_t if t in components]
        uy = [components[t]["similarity"] for t in uns_t if t in components]
        ax4.scatter(ux, uy, c="#BDBDBD", alpha=0.25, s=8, label="Below \u03b8")
    if sel_t:
        sx = [components[t]["V_score"] for t in sel_t if t in components]
        sy = [components[t]["similarity"] for t in sel_t if t in components]
        sw2 = [weights.get(t, 0) for t in sel_t if t in components]
        sc = ax4.scatter(sx, sy, c=sw2, cmap="hot_r", s=45,
                         edgecolors="black", lw=0.4, label="Selected (K)", zorder=5)
        plt.colorbar(sc, ax=ax4, label="Portfolio Weight", shrink=0.8)
        top5 = sorted(sel_t, key=lambda t: weights.get(t, 0), reverse=True)[:5]
        for t in top5:
            if t in components:
                ax4.annotate(t, (components[t]["V_score"], components[t]["similarity"]),
                             fontsize=7, fontweight="bold",
                             xytext=(4, 4), textcoords="offset points")
    ax4.axvline(risk_budget, color="red", linestyle=":", lw=1.5,
                label=f"Risk Budget ({risk_budget:.2f})")
    ax4.set_xlabel("V_score (Linear Capped 0-10)")
    ax4.set_ylabel("Cosine Similarity to User Vector")
    ax4.set_title("V_score vs Similarity — Constraint Check")
    ax4.set_xlim(0, 10)
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def _print_selected_assets(selected, components, weights, n_k):
    """Print the selected asset table sorted by score."""
    sorted_sel = sorted(selected.items(), key=lambda x: x[1], reverse=True)
    W = 130
    print("\n" + "=" * W)
    print(f"  SELECTED ASSETS (K={n_k}) \u2014 Sorted by Score")
    print("=" * W)
    print(f"  {'Rank':<5} {'Ticker':<8} {'Similarity':>11} "
          f"{'Raw Vol':>9} {'V_score':>8} {'Budget':>8} "
          f"{'Penalty%':>9} {'Score':>9} | {'Weight':>8}")
    print("  " + "-" * (W - 2))
    for rank, (ticker, score) in enumerate(sorted_sel, 1):
        c = components.get(ticker, {})
        w = weights.get(ticker, 0)
        print(
            f"  {rank:<5} {ticker:<8} "
            f"{c.get('similarity', 0):>11.4f} "
            f"{c.get('raw_vol', 0):>8.1%} "
            f"{c.get('V_score', 0):>8.2f} "
            f"{c.get('risk_budget', 0):>8.2f} "
            f"{c.get('relative_penalty', 0):>8.1%} "
            f"{score:>9.4f} | {w:>8.2%}"
        )


def _print_rejected_assets(all_scores, selected, components, theta):
    """Print the rejected asset table and return the sorted list."""
    rej = sorted(
        [(t, s) for t, s in all_scores.items() if t not in selected],
        key=lambda x: x[1], reverse=True
    )
    W = 130
    print("\n" + "=" * W)
    print(f"  REJECTED ASSETS (Below \u03b8={theta}) \u2014 Sorted Greatest-to-Least")
    print("=" * W)
    print(f"  {'Rank':<5} {'Ticker':<8} {'Similarity':>11} "
          f"{'Raw Vol':>9} {'V_score':>8} {'Budget':>8} "
          f"{'Penalty%':>9} {'Score':>9}")
    print("  " + "-" * (W - 2))
    for rank, (ticker, score) in enumerate(rej, 1):
        c = components.get(ticker, {})
        print(
            f"  {rank:<5} {ticker:<8} "
            f"{c.get('similarity', 0):>11.4f} "
            f"{c.get('raw_vol', 0):>8.1%} "
            f"{c.get('V_score', 0):>8.2f} "
            f"{c.get('risk_budget', 0):>8.2f} "
            f"{c.get('relative_penalty', 0):>8.1%} "
            f"{score:>9.4f}"
        )
    print("=" * W + "\n")
    return rej


def _compute_portfolio_stats(selected, weights, components, risk_budget, rej):
    """Compute and print portfolio summary statistics."""
    wtd_vol = sum(
        weights.get(t, 0) * components[t].get("raw_vol", 0.25)
        for t in selected if t in components
    )
    wtd_vscore = sum(
        weights.get(t, 0) * components[t].get("V_score", 5.0)
        for t in selected if t in components
    )
    hhi = sum(w ** 2 for w in weights.values()) * 10000
    eff_n = 1.0 / max(sum(w ** 2 for w in weights.values()), 1e-9)

    n_k = len(selected)
    print("=" * 65)
    print("  PORTFOLIO SUMMARY")
    print("=" * 65)
    print(f"  Weighted Avg Raw Vol:    {wtd_vol:.2%}")
    print(f"  Weighted Avg V_score:    {wtd_vscore:.2f} / 10  "
          f"(budget {risk_budget:.2f})")
    print(f"  HHI Concentration:       {hhi:.0f}  "
          f"{'(diversified)' if hhi < 1500 else '(moderate)' if hhi < 2500 else '(concentrated)'}")
    print(f"  Effective # Assets:      {eff_n:.1f}")
    print("=" * 65 + "\n")

    return {
        "k":                   n_k,
        "weighted_volatility": wtd_vol,
        "weighted_vscore":     wtd_vscore,
        "hhi":                 hhi,
        "effective_n":         eff_n,
        "rejected_count":      len(rej),
    }
