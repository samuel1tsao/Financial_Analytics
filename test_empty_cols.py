import pandas as pd
import numpy as np

# Create empty df with RangeIndex columns
df = pd.DataFrame() 
print(f"Columns type: {type(df.columns)}")
print(f"Columns: {df.columns}")

try:
    print(f"Is 'VOO' in columns? {'VOO' in df.columns}")
except Exception as e:
    print(f"Crash on 'in': {e}")

try:
    print(f"Column 0: {df.columns[0]}")
except Exception as e:
    print(f"Crash on [0]: {e}")
