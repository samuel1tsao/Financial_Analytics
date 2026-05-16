import pandas as pd
import numpy as np

# Mock data
dates = pd.date_range('2020-01-01', '2020-01-05')
returns = pd.DataFrame({
    'AAPL': [0.01, -0.01, 0.02, 0.01, -0.02],
    'ZETX': [np.nan, np.nan, np.nan, 0.05, 0.01]
}, index=dates)

weights_dict = {'AAPL': 0.2, 'ZETX': 0.8}
target_series = pd.Series(weights_dict)

active_mask = returns.notna()

# Multiply DataFrame by Series (aligns on columns)
daily_weights = active_mask * target_series
daily_weight_sums = daily_weights.sum(axis=1)
daily_weights = daily_weights.div(daily_weight_sums.where(daily_weight_sums > 0, 1.0), axis=0)

port_daily_rets = (returns.fillna(0.0) * daily_weights).sum(axis=1)
spy_returns = pd.Series(0.0, index=dates)

daily_rets = np.where(daily_weight_sums > 0, port_daily_rets, spy_returns)

print(daily_weights)
print("Daily Rets:")
print(daily_rets)
