import _rl_worker
import _constants
import pandas as pd

config = _constants.DEFAULT_PIPELINE_CONFIG.copy()
master_df = pd.read_csv('sp1500_master_research_dataset.csv', index_col=0)
static_cols = _rl_worker._get_static_feature_columns(master_df)
static_dim = len(static_cols)
emb_dim = config.get("ml_embedding_dim", 8)
user_cond_dim = 8
input_dim = emb_dim + user_cond_dim + static_dim

print(f"Static Dim: {static_dim}")
print(f"Emb Dim: {emb_dim}")
print(f"User Cond Dim: {user_cond_dim}")
print(f"Calculated Input Dim: {input_dim}")
