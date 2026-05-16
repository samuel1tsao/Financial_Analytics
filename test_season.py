import pandas as pd
import numpy as np

dates = pd.date_range('2020-01-01', periods=10)
returns = pd.DataFrame({
    'A': [np.nan, np.nan, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1] # IPOs on day 3
}, index=dates)

# Find IPO date (first non-NaN)
# Then shift the valid mask by N days
active_mask = returns.notna()

seasoning_days = 3
for col in returns.columns:
    first_valid = returns[col].first_valid_index()
    if first_valid is not None:
        # Find integer location of first_valid
        loc = returns.index.get_loc(first_valid)
        # Set the first N days of trading to False
        end_loc = min(loc + seasoning_days, len(returns))
        active_mask.iloc[loc:end_loc, active_mask.columns.get_loc(col)] = False

print(active_mask)
