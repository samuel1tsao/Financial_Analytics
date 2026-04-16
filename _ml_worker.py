import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import time
import os

from _constants import TRADING_DAYS_PER_YEAR

class AssetEmbeddingNet(nn.Module):
    def __init__(self, input_dim, config, output_dim):
        super(AssetEmbeddingNet, self).__init__()
        
        hidden_layers = config.get("ml_hidden_layers", [128, 64, 32])
        emb_dim = config.get("ml_embedding_dim", 8)
        
        # Encoder
        enc_layers = []
        in_d = input_dim
        for h in hidden_layers:
            enc_layers.append(nn.Linear(in_d, h))
            enc_layers.append(nn.ReLU())
            enc_layers.append(nn.LayerNorm(h)) # Stable training
            in_d = h
            
        enc_layers.append(nn.Linear(in_d, emb_dim))
        self.encoder = nn.Sequential(*enc_layers)
        
        # Decoder
        dec_layers = []
        in_d = emb_dim
        for h in reversed(hidden_layers):
            dec_layers.append(nn.Linear(in_d, h))
            dec_layers.append(nn.ReLU())
            in_d = h
            
        dec_layers.append(nn.Linear(in_d, output_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        emb = self.encoder(x)
        out = self.decoder(emb)
        return emb, out

class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def masked_weighted_mse_loss(pred, target, weights):
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    
    diff = pred[mask] - target[mask]
    w = weights[mask]
    loss = (w * (diff ** 2)).mean()
    return loss

def train_pytorch_embedding_model(master_df, price_matrix, volume_matrix, daily_returns, config, drip_daily_returns=None, verbose=True):
    print(f"[{time.strftime('%H:%M:%S')}] [Member A] Initializing Top-K Time-Decayed Embedding Network...")
    
    # 1. Base Setup
    horizons = config.get("ml_target_horizons", [1, 3, 5, 10, 15])
    horizon_weights = config.get("ml_horizon_weights", {1: 1.0, 3: 0.8, 5: 0.6, 10: 0.4, 15: 0.2})
    target_metrics = config.get("ml_target_metrics", ["return", "volatility", "volume"])
    output_dim = len(horizons) * len(target_metrics) # e.g. 5 * 3 = 15
    
    # 2. Compute Return & Volatility Snapshots
    if drip_daily_returns is not None:
        annual_returns = drip_daily_returns.resample('YE').apply(lambda x: (1 + x).prod() - 1)
        annual_vols = drip_daily_returns.resample('YE').std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    else:
        annual_returns = daily_returns.resample('YE').apply(lambda x: (1 + x).prod() - 1)
        annual_vols = daily_returns.resample('YE').std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        
    daily_log_vol = np.log1p(volume_matrix).diff().dropna(how='all')
    annual_log_vol = daily_log_vol.resample('YE').sum().reindex(index=annual_returns.index, columns=price_matrix.columns).fillna(0)
    
    metric_sources = {
        "return": annual_returns,
        "volatility": annual_vols,
        "volume": annual_log_vol
    }

    # 3. Assemble Dataset
    features = config.get("ml_training_features", ["hist_momentum", "hist_volatility", "hist_volume", "sector", "industry", "state", "quoteType", "exchange"])
    
    categorical_cols = ["sector", "industry", "state", "quoteType", "exchange"]
    numerical_cols = ["hist_momentum", "hist_volatility", "hist_volume"]
    
    master_encoded = pd.get_dummies(master_df, columns=[c for c in categorical_cols if c in master_df.columns], drop_first=True)
    all_feature_cols = [c for c in master_encoded.columns if any(c.startswith(cat + "_") for cat in categorical_cols)]
    
    if "hist_momentum" not in master_encoded.columns:
        master_encoded["hist_momentum"] = 0.0
        master_encoded["hist_volatility"] = 0.0
        master_encoded["hist_volume"] = 0.0
    
    X_samples, Y_samples, Y_weights = [], [], []
    valid_years = annual_returns.index
    
    half_life = config.get("ml_time_decay_half_life", 10)
    max_year = valid_years[-1].year if len(valid_years) > 0 else 2026
    
    for i in range(1, len(valid_years) - 1):
        for ticker in master_df.index:
            if ticker not in annual_returns.columns: continue
            
            p_prior = price_matrix[ticker].loc[:valid_years[i]].dropna()
            v_prior = volume_matrix[ticker].loc[:valid_years[i]].dropna()
            
            if len(p_prior) < TRADING_DAYS_PER_YEAR: continue
            
            mom = (p_prior.iloc[-1] / p_prior.iloc[-TRADING_DAYS_PER_YEAR]) - 1
            vol = p_prior.pct_change().tail(TRADING_DAYS_PER_YEAR).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
            logv = np.log1p(v_prior.tail(TRADING_DAYS_PER_YEAR).sum())
            
            # Point-in-time features
            feat_vec = master_encoded.loc[ticker, all_feature_cols].astype(float).tolist()
            # prepend numerical
            feat_vec = [float(mom), float(vol), float(logv)] + feat_vec
            
            y_vec = []
            y_w_vec = []
            
            # Time decay weighting based on the current year
            time_w = 1.0
            if half_life is not None:
                time_w = 2.0 ** (-(max_year - valid_years[i].year) / half_life)
                
            has_valid = False
            for h in horizons:
                if i + h < len(valid_years):
                    for metric in target_metrics:
                        val = metric_sources[metric][ticker].iloc[i+1 : i+h+1].mean()
                        y_vec.append(val if pd.notna(val) else np.nan)
                        y_w_vec.append(horizon_weights[h] * time_w)
                        if pd.notna(val):
                            has_valid = True
                else:
                    y_vec.extend([np.nan] * len(target_metrics))
                    y_w_vec.extend([horizon_weights[h] * time_w] * len(target_metrics))
                    
            if has_valid:
                X_samples.append(feat_vec)
                Y_samples.append(y_vec)
                Y_weights.append(y_w_vec)
                
    X_tensor = torch.tensor(X_samples, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_samples, dtype=torch.float32)
    W_tensor = torch.tensor(Y_weights, dtype=torch.float32)
    
    # Normalize
    X_mean = X_tensor.mean(dim=0, keepdim=True)
    X_std  = X_tensor.std(dim=0, keepdim=True) + 1e-8
    X_norm = (X_tensor - X_mean) / X_std
    
    # Safe Y normalization
    Y_mean = torch.nanmean(Y_tensor, dim=0, keepdim=True)
    
    Y_std = []
    for col in range(Y_tensor.shape[1]):
        col_data = Y_tensor[:, col]
        col_data = col_data[~torch.isnan(col_data)]
        if len(col_data) > 1:
            Y_std.append(col_data.std().item() + 1e-8)
        else:
            Y_std.append(1.0)
    Y_std = torch.tensor(Y_std, dtype=torch.float32).unsqueeze(0)
    
    Y_norm = (Y_tensor - Y_mean) / Y_std

    model = AssetEmbeddingNet(input_dim=X_norm.shape[1], config=config, output_dim=output_dim)
    optimizer = optim.Adam(model.parameters(), lr=config.get("ml_learning_rate", 0.001))
    
    epochs = config.get("ml_epochs", 150)
    batch_size = config.get("ml_batch_size", 64)
    early_stopping = EarlyStopping(patience=10, min_delta=1e-4)

    # 80/20 train/val split
    num_samples = X_norm.shape[0]
    indices = torch.randperm(num_samples)
    val_size = int(0.2 * num_samples)
    
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    train_X, train_Y, train_W = X_norm[train_indices], Y_norm[train_indices], W_tensor[train_indices]
    val_X, val_Y, val_W = X_norm[val_indices], Y_norm[val_indices], W_tensor[val_indices]
    
    dataset = torch.utils.data.TensorDataset(train_X, train_Y, train_W)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    if verbose:
        print(f"[{time.strftime('%H:%M:%S')}] [Member A] Training {len(train_indices)} samples / Validating {len(val_indices)} samples. Output dim: {output_dim}")

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for b_x, b_y, b_w in loader:
            optimizer.zero_grad()
            _, pred = model(b_x)
            loss = masked_weighted_mse_loss(pred, b_y, b_w)
            if loss.requires_grad and not torch.isnan(loss) and loss.item() > 0:
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        model.eval()
        with torch.no_grad():
            _, val_pred = model(val_X)
            val_loss = masked_weighted_mse_loss(val_pred, val_Y, val_W).item()
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(loader):.4f} | Val Loss: {val_loss:.4f}")
            
        early_stopping(val_loss)
        best_val_loss = early_stopping.best_loss
        if early_stopping.early_stop:
            if verbose:
                print(f"[{time.strftime('%H:%M:%S')}] Early stopping triggered at epoch {epoch+1}")
            break

    # Extract Embeddings
    model.eval()
    embeddings = {}
    asset_predictions = {}
    
    with torch.no_grad():
        for ticker in master_df.index:
            if ticker not in annual_returns.columns: continue
            p_prior = price_matrix[ticker].dropna()
            v_prior = volume_matrix[ticker].dropna()
            
            if len(p_prior) < TRADING_DAYS_PER_YEAR: continue
            
            mom = (p_prior.iloc[-1] / p_prior.iloc[-TRADING_DAYS_PER_YEAR]) - 1
            vol = p_prior.pct_change().tail(TRADING_DAYS_PER_YEAR).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
            logv = np.log1p(v_prior.tail(TRADING_DAYS_PER_YEAR).sum())
            
            feat_vec = master_encoded.loc[ticker, all_feature_cols].astype(float).tolist()
            feat_vec = [float(mom), float(vol), float(logv)] + feat_vec
            
            x_t = torch.tensor([feat_vec], dtype=torch.float32)
            x_norm = (x_t - X_mean) / X_std
            
            emb, pred = model(x_norm)
            embeddings[ticker] = emb.squeeze(0).numpy()
            
            # Denormalize predictions
            pred_raw = (pred.squeeze(0) * Y_std.squeeze(0) + Y_mean.squeeze(0)).numpy()
            preds = {}
            idx = 0
            for h in horizons:
                preds[h] = {}
                for metric in target_metrics:
                    preds[h][metric] = float(pred_raw[idx])
                    idx += 1
            asset_predictions[ticker] = preds
            
    print(f"[{time.strftime('%H:%M:%S')}] [Member A] Generated dynamic embeddings & predictions for {len(embeddings)} assets.")
    
    return {
        "dynamic_embeddings": embeddings,
        "asset_predictions": asset_predictions,
        "master_df": master_df,
        "price_matrix": price_matrix,
        "volume_matrix": volume_matrix,
        "daily_returns": daily_returns,
        "drip_daily_returns": drip_daily_returns,
        "val_loss": best_val_loss
    }

def run_ml_grid_search(master_df, price_matrix, volume_matrix, daily_returns, config, drip_daily_returns=None):
    """
    Runs a grid search over specified hyperparameters in PIPELINE_CONFIG.
    Returns a pandas DataFrame of results.
    """
    print(f"[{time.strftime('%H:%M:%S')}] Starting Grid Search...")
    
    # Expand lists or single values into iterations
    lrs = config.get("grid_lrs", [config.get("ml_learning_rate", 0.001)])
    batch_sizes = config.get("grid_batch_sizes", [config.get("ml_batch_size", 64)])
    hidden_dims = config.get("grid_hidden_layers", [config.get("ml_hidden_layers", [128, 64, 32])])
    
    results = []
    
    for lr in lrs:
        for bs in batch_sizes:
            for hd in hidden_dims:
                run_config = config.copy()
                run_config["ml_learning_rate"] = lr
                run_config["ml_batch_size"] = bs
                run_config["ml_hidden_layers"] = hd
                
                print(f"\\n--- Testing Config: LR={lr}, BatchSize={bs}, HiddenDims={hd} ---")
                
                cache = train_pytorch_embedding_model(
                    master_df, price_matrix, volume_matrix, daily_returns, 
                    run_config, 
                    drip_daily_returns=drip_daily_returns,
                    verbose=False
                )
                
                results.append({
                    "LR": lr,
                    "Batch Size": bs,
                    "Hidden Dims": str(hd),
                    "Val MSE": cache["val_loss"]
                })
                
    df = pd.DataFrame(results).sort_values("Val MSE").reset_index(drop=True)
    return df
