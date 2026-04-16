import sys
import os

from _constants import DEFAULT_PIPELINE_CONFIG, TEST_PROFILES, DataSyncMode
from _data_worker import fetch_macro_universe, generate_dataset_member_a
from _ml_worker import train_pytorch_embedding_model
from _rl_worker import train_rl_agent

def main():
    config = DEFAULT_PIPELINE_CONFIG.copy()
    config["data_source_mode"] = DataSyncMode.OFFLINE_CSV_ONLY
    config["rl_episodes"] = 5  # Small number of episodes for validation
    
    print("Loading Dataset...")
    tickers = []
    # Try to load local dataset
    master_df, price_matrix, volume_matrix, daily_returns, drip_returns = generate_dataset_member_a(tickers, config)
    
    print("Training Autoencoder (Fast)...")
    config["ml_epochs"] = 1 # Force fast training
    dataset_cache = train_pytorch_embedding_model(
        master_df, price_matrix, volume_matrix, daily_returns,
        config,
        drip_daily_returns=drip_returns
    )
    
    print("Checking Test Profile 0...")
    profile = TEST_PROFILES[0]
    print(f"Profile: {profile['profile_name']}")
    
    print("Starting RL Agent Training Loop...")
    result = train_rl_agent(dataset_cache, profile, config, verbose=True)
    
    print("RL Validation Successful!")
    print(f"Final Reward: {result['final_reward']}")
    print(f"Selected Assets: {len(result['portfolio_weights'])}")

if __name__ == '__main__':
    main()
