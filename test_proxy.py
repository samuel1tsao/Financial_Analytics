import pandas as pd
import numpy as np

dates = pd.date_range('2020-01-01', periods=5)
returns = pd.DataFrame({
    'A': [np.nan, np.nan, 0.1, 0.1, 0.1], # IPOs on day 3
    'B': [0.05, 0.05, 0.05, 0.05, 0.05],
    'SPY': [0.02, 0.02, 0.02, 0.02, 0.02]
}, index=dates)

target_df = pd.DataFrame({
    'A': [0.5]*5,
    'B': [0.5]*5
}, index=dates)

active_mask = returns[['A', 'B']].notna()

# Unnormalized weights (just what is active)
daily_weights = active_mask * target_df

# The missing weight goes to SPY!
# The sum of target_df is 1.0. The sum of daily_weights is what is currently active.
missing_weight = 1.0 - daily_weights.sum(axis=1)

port_daily_rets = (returns[['A', 'B']].fillna(0.0) * daily_weights).sum(axis=1)
spy_returns = returns['SPY']

daily_rets = port_daily_rets + (spy_returns * missing_weight)

print("Daily Weights:")
print(daily_weights)
print("Missing Weight to SPY:")
print(missing_weight)
print("Total Return:")
print(daily_rets)
