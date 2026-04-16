"""
_rl_worker.py
─────────────
Member D: RL-Driven Transformer Portfolio Optimizer.

Implements the RL pipeline:
Phase 1: Ingestion & Concatenation
Phase 2: Core Engine (Transformer)
Phase 3: Action Generation (Gaussian Policy)
Phase 5: Policy Gradient Update
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, Tuple

class PortfolioTransformerRL(nn.Module):
    def __init__(self, input_dim, config):
        """
        End-to-End RL-Driven Transformer.
        """
        super(PortfolioTransformerRL, self).__init__()
        
        self.d_model = config.get("rl_d_model", 64)
        nhead = config.get("rl_nhead", 4)
        num_layers = config.get("rl_num_encoder_layers", 2)
        dim_feedforward = config.get("rl_dim_feedforward", 256)
        dropout = config.get("rl_dropout", 0.1)
        
        # Initial projection to d_model
        self.input_proj = nn.Linear(input_dim, self.d_model)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Gaussian Policy Head
        self.mu_head = nn.Linear(self.d_model, 1)
        self.sigma_head = nn.Linear(self.d_model, 1)
        
    def forward(self, x):
        """
        x shape: (batch_size, num_assets, input_dim)
        We typically run with batch_size=1 (the entire universe at once).
        """
        h = self.input_proj(x)
        h = F.relu(h)
        
        # Self-Attention contextualizes the entire market without pre-filtering
        out = self.transformer(h)
        
        # Output mu and sigma for each asset
        mu = self.mu_head(out).squeeze(-1) # shape: (batch, num_assets)
        sigma = F.softplus(self.sigma_head(out)).squeeze(-1) + 1e-5 # softplus ensures sigma > 0
        
        # Apply ReLU to mu to allow absolute 0.0 allocations before normalization
        mu = F.relu(mu)
        
        return mu, sigma

def get_action_and_log_prob(mu, sigma):
    """
    Samples weights from the Gaussian Policy and returns them along with log probability.
    
    Terminology Guardrail: $\sigma$ strictly represents the RL agent's exploration noise—how wildly
    it searches for better weights around $\mu$. It is NOT historical market volatility.
    """
    dist = torch.distributions.Normal(mu, sigma)
    # Sample action
    raw_action = dist.sample()
    
    # Enforce non-negativity for valid long-only portfolio weights
    action = F.relu(raw_action)
    
    # Normalize to 100% (prevent divide-by-zero if all actions sampled below 0)
    action_sum = action.sum(dim=-1, keepdim=True) + 1e-9
    normalized_weights = action / action_sum
    
    # Log probability of the original sampled action
    log_prob = dist.log_prob(raw_action).sum(dim=-1)
    
    return normalized_weights, log_prob

def build_rl_dataset(dataset, user_profile, config):
    """
    Phase 1: Ingestion & Concatenation.
    Build the concatenated feature matrix [Frozen_Emb, User_Features, Static_Features]
    for all assets.
    
    Returns:
        tensor_x: The input tensor of shape (1, num_assets, feature_dim)
        tickers: List of original order of tickers
    """
    master_df = dataset["master_df"]
    embeddings = dataset.get("dynamic_embeddings", {})
    
    if not embeddings:
        raise ValueError("Dynamic embeddings not found. Please run Member A first.")
        
    # Get categorical features directly from master_df (dummy encoded previously)
    categorical_cols = ["sector", "industry", "state", "quoteType", "exchange"]
    all_feature_cols = [c for c in master_df.columns if any(c.startswith(cat + "_") for cat in categorical_cols)]
    
    # User Profile Features
    risk_tolerance = float(user_profile.get("risk_tolerance", 5.0))
    budget_score = user_profile.get("_risk_budget_score", risk_tolerance)
    
    tickers = []
    feature_vectors = []
    
    for ticker, emb in embeddings.items():
        if ticker not in master_df.index:
            continue
            
        static_features = master_df.loc[ticker, all_feature_cols].astype(float).tolist()
        
        # Input Vector structure: [Autoencoder_Embedding, User_Risk, User_Budget, Static_Identity...]
        full_vec = list(emb) + [risk_tolerance, budget_score] + static_features
        
        tickers.append(ticker)
        feature_vectors.append(full_vec)
        
    tensor_x = torch.tensor([feature_vectors], dtype=torch.float32)
    return tensor_x, tickers

from _sim_worker import simulate_rl_environment_step

def train_rl_agent(dataset, user_profile, config, verbose=True):
    """
    Phase 5: The Policy Gradient Update.
    Instantiates the RL agent, runs interactions with the simulator, and updates via REINFORCE.
    """
    # 1. Build Ingestion Dataset
    tensor_x, tickers = build_rl_dataset(dataset, user_profile, config)
    
    if tensor_x.numel() == 0:
        return {"portfolio_weights": {}, "training_history": []}
        
    input_dim = tensor_x.shape[-1]
    agent = PortfolioTransformerRL(input_dim, config)
    
    # 2. Setup Optimizer
    lr = config.get("rl_learning_rate", 0.0001)
    episodes = config.get("rl_episodes", 100)
    optimizer = optim.Adam(agent.parameters(), lr=lr)
    
    # 3. RL Training Loop
    if verbose:
        print(f"[{time.strftime('%H:%M:%S')}] [Member D] Starting RL Training Loop for {episodes} episodes...")
        
    history = []
    
    for ep in range(episodes):
        agent.train()
        optimizer.zero_grad()
        
        # Forward pass (Phase 2 & 3)
        mu, sigma = agent(tensor_x)
        
        # Sample action and get log probability
        action_weights, log_prob = get_action_and_log_prob(mu, sigma)
        
        # Phase 4: Environment interaction (Simulator)
        # Detach action for the environment, it's a black box.
        action_numpy = action_weights.detach().numpy()
        
        reward_scalar, metrics = simulate_rl_environment_step(action_numpy, tickers, dataset, user_profile, config)
        
        # Phase 5: Policy Gradient Update
        # Loss = -Reward * log_prob
        # We want to maximize reward, so we minimize -reward * log_prob
        # log_prob shape is (1,), sum makes it a scalar.
        loss = -float(reward_scalar) * log_prob.sum()
        
        # Backprop
        loss.backward()
        optimizer.step()
        
        history.append({
            "episode": ep,
            "reward": reward_scalar,
            "loss": loss.item(),
            "ETV": metrics.get("ETV", 0),
            "GFR": metrics.get("GFR", 0)
        })
        
        if verbose and (ep + 1) % 10 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] Episode {ep+1}/{episodes} | Reward: {reward_scalar:.4f} | ETV: ${metrics.get('ETV',0):,.0f} | GFR: {metrics.get('GFR',0):.2f}")
            
    # Final Evaluation (greedy, no sampling noise)
    agent.eval()
    with torch.no_grad():
        mu, _ = agent(tensor_x)
        action_weights = F.relu(mu)
        action_sum = action_weights.sum(dim=-1, keepdim=True) + 1e-9
        final_weights = action_weights / action_sum
        
    portfolio_weights = {tickers[i]: float(final_weights[0, i]) for i in range(len(tickers))}
    selected_assets = {k: v for k, v in portfolio_weights.items() if v > 0.001}
    
    # Normalize selected assets
    total_selected = sum(selected_assets.values())
    if total_selected > 0:
        selected_assets = {k: v / total_selected for k, v in selected_assets.items()}
    
    return {
        "portfolio_weights": selected_assets,
        "training_history": history,
        "final_reward": history[-1]["reward"] if history else 0.0
    }
