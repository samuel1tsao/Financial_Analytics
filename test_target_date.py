import pandas as pd
import numpy as np

dates = pd.date_range(end='2020-01-01', periods=10, freq='Y')
returns = pd.DataFrame({
    'A': [np.nan, np.nan, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    'B': [0.05]*10,
    'SPY': [0.02]*10
}, index=dates)

segments = [
    {"horizon_years": [0, 5], "goal_name": "Phase 1", "weights": {"A": 0.8, "B": 0.2}},
    {"horizon_years": [5, 30], "goal_name": "Phase 2", "weights": {"A": 0.2, "B": 0.8}}
]

# Get all unique assets across all segments
all_assets = set()
for seg in segments:
    all_assets.update(seg["weights"].keys())

active_assets = [t for t in all_assets if t in returns.columns]
if not active_assets:
    active_assets = ["SPY"]

port_returns = returns[active_assets]

# Build the target_weights DataFrame
target_df = pd.DataFrame(0.0, index=dates, columns=active_assets)
phase_labels = pd.Series("Unknown", index=dates)

for i, dt in enumerate(dates):
    years_ago = (dates[-1] - dt).days / 365.25
    
    # Find matching segment
    chosen_seg = segments[-1] # default to last
    for seg in segments:
        h_min, h_max = seg.get("horizon_years", [0, 99])
        if h_min <= years_ago < h_max:
            chosen_seg = seg
            break
            
    phase_labels.iloc[i] = chosen_seg.get("goal_name", f"Phase {segments.index(chosen_seg)+1}")
    for t, w in chosen_seg.get("weights", {}).items():
        if t in active_assets:
            target_df.loc[dt, t] = w

active_mask = port_returns.notna()
daily_weights = active_mask * target_df
daily_weight_sums = daily_weights.sum(axis=1)

spy_returns = returns["SPY"].fillna(0.0) if "SPY" in returns.columns else pd.Series(0.0, index=dates)
daily_weights = daily_weights.div(daily_weight_sums.where(daily_weight_sums > 0, 1.0), axis=0)
port_daily_rets = (port_returns.fillna(0.0) * daily_weights).sum(axis=1)
daily_rets = np.where(daily_weight_sums > 0, port_daily_rets, spy_returns)

print(daily_weights)
print("Phases:", phase_labels.values)
