import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import time
import math
import os
import pickle
from tqdm import tqdm

from _constants import TRADING_DAYS_PER_YEAR

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class AssetTransformerNet(nn.Module):
    def __init__(self, input_dim, config, output_macro_dim, output_ar_dim):
        super(AssetTransformerNet, self).__init__()
        self.d_model = config.get("ml_d_model", 64)
        self.max_seq_len = config.get("ml_max_seq_len", 3780)
        
        self.input_proj = nn.Linear(input_dim, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=max(self.max_seq_len, 5000))
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config.get("ml_nhead", 4),
            dim_feedforward=config.get("ml_dim_feedforward", 128),
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=config.get("ml_num_encoder_layers", 2)
        )
        
        self.emb_dim = config.get("ml_embedding_dim", 8)
        self.bottleneck = nn.Sequential(
            nn.Linear(self.d_model, self.emb_dim),
            nn.ReLU()
        )
        
        self.head_macro = nn.Linear(self.emb_dim, output_macro_dim)
        self.head_ar = nn.Linear(self.d_model, output_ar_dim)

    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(device)

    def forward(self, src, src_key_padding_mask=None):
        x = self.input_proj(src)
        x = self.pos_encoder(x)
        
        seq_len = src.size(1)
        mask = self.generate_square_subsequent_mask(seq_len, src.device)
        
        out = self.transformer_encoder(x, mask=mask, src_key_padding_mask=src_key_padding_mask)
        
        ar_preds = self.head_ar(out)
        
        if src_key_padding_mask is not None:
            lengths = (~src_key_padding_mask).sum(dim=1) - 1
            lengths = torch.clamp(lengths, min=0)
            idx = lengths.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.d_model)
            last_tokens = torch.gather(out, 1, idx).squeeze(1)
        else:
            last_tokens = out[:, -1, :]
            
        emb = self.bottleneck(last_tokens)
        macro_preds = self.head_macro(emb)
        
        return ar_preds, macro_preds, emb

class TransformerDataset(torch.utils.data.Dataset):
    def __init__(self, samples, ticker_data, X_mean, X_std, max_len):
        self.samples = samples
        self.ticker_data = ticker_data
        self.X_mean = X_mean
        self.X_std = X_std
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ticker, target_date, y_vec, y_w_vec = self.samples[idx]
        info = self.ticker_data[ticker]
        
        dates = info['dates']
        feats = info['feats']
        target_np64 = np.datetime64(target_date)
        idx_end = np.searchsorted(dates, target_np64, side='right')
        
        hist_feats = feats[max(0, idx_end - self.max_len):idx_end]
        L = len(hist_feats)
        
        static_feats = np.tile(info['static'], (L, 1))
        seq = np.concatenate([hist_feats, static_feats], axis=1)
        
        seq_norm = (torch.tensor(seq, dtype=torch.float32) - self.X_mean) / self.X_std
        return seq_norm, torch.tensor(y_vec, dtype=torch.float32), torch.tensor(y_w_vec, dtype=torch.float32)

def collate_fn(batch):
    seqs, ys, yws = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs])
    padded_seqs = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
    ys = torch.stack(ys)
    yws = torch.stack(yws)
    
    L_max = padded_seqs.shape[1]
    mask = torch.arange(L_max)[None, :] >= lengths[:, None]
    
    return padded_seqs, ys, yws, mask

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
    return (w * (diff ** 2)).mean()

def train_pytorch_embedding_model(master_df, price_matrix, volume_matrix, daily_returns, config, drip_daily_returns=None, verbose=True):
    print(f"[{time.strftime('%H:%M:%S')}] [Member A] Initializing Dual-Head Sequence Transformer...")
    
    # Base Setup
    horizons = config.get("ml_target_horizons", [1, 3, 5, 10, 15])
    horizon_weights = config.get("ml_horizon_weights", {1: 1.0, 3: 0.8, 5: 0.6, 10: 0.4, 15: 0.2})
    target_metrics = config.get("ml_target_metrics", ["return", "volatility", "volume"])
    output_macro_dim = len(horizons) * len(target_metrics)
    output_ar_dim = 3 # (return, volatility, log_volume)
    max_seq_len = config.get("ml_max_seq_len", 3780)
    
    if drip_daily_returns is not None:
        annual_returns = drip_daily_returns.resample('YE').apply(lambda x: (1 + x).prod() - 1)
        annual_vols = drip_daily_returns.resample('YE').std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        daily_ret = drip_daily_returns.copy()
    else:
        annual_returns = daily_returns.resample('YE').apply(lambda x: (1 + x).prod() - 1)
        annual_vols = daily_returns.resample('YE').std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        daily_ret = daily_returns.copy()
        
    daily_log_vol = np.log1p(volume_matrix).diff().dropna(how='all')
    annual_log_vol = daily_log_vol.resample('YE').sum().reindex(index=annual_returns.index, columns=price_matrix.columns).fillna(0)
    
    # Feature engineering for sequences
    daily_logv = np.log1p(volume_matrix)
    daily_vol = daily_ret.rolling(21, min_periods=1).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    daily_vol = daily_vol.fillna(0.0)
    daily_ret = daily_ret.fillna(0.0)
    
    metric_sources = {
        "return": annual_returns,
        "volatility": annual_vols,
        "volume": annual_log_vol
    }

    categorical_cols = ["sector", "industry", "state", "quoteType", "exchange"]
    master_encoded = pd.get_dummies(master_df, columns=[c for c in categorical_cols if c in master_df.columns], drop_first=True)
    all_feature_cols = [c for c in master_encoded.columns if any(c.startswith(cat + "_") for cat in categorical_cols)]
    
    ticker_data = {}
    for ticker in master_df.index:
        if ticker not in daily_ret.columns: continue
        
        df_t = pd.DataFrame({
            'ret': daily_ret[ticker],
            'vol': daily_vol[ticker],
            'logv': daily_logv[ticker]
        }).dropna()
        
        if len(df_t) < 252: continue
        
        feat_vec = master_encoded.loc[ticker, all_feature_cols].astype(float).values
        ticker_data[ticker] = {
            'dates': df_t.index.values,
            'feats': df_t.values,
            'static': feat_vec
        }
    
    # 1. Filter Tickers with < 20 Years of History (as per User Request)
    # -------------------------------------------------------------------------
    min_history_years = config.get("ml_validation_horizon_years", 20)
    required_days = min_history_years * TRADING_DAYS_PER_YEAR
    
    eligible_ticker_data = {}
    for ticker, info in ticker_data.items():
        if len(info['feats']) >= required_days:
            eligible_ticker_data[ticker] = info
        elif verbose:
            print(f"  [Member A] Excluding '{ticker}': Insufficient history ({len(info['feats'])} days < {required_days})")
            
    ticker_data = eligible_ticker_data
    if not ticker_data:
        raise ValueError(f"No tickers found with at least {min_history_years} years of historical data.")

    # Compute Normalization Stats efficiently
    all_dyn = np.vstack([info['feats'] for info in ticker_data.values()])
    all_static = np.vstack([info['static'] for info in ticker_data.values()])
    
    dyn_mean = all_dyn.mean(axis=0)
    dyn_std  = all_dyn.std(axis=0) + 1e-8
    static_mean = all_static.mean(axis=0)
    static_std  = all_static.std(axis=0) + 1e-8
    
    X_mean = torch.tensor(np.concatenate([dyn_mean, static_mean]), dtype=torch.float32)
    X_std  = torch.tensor(np.concatenate([dyn_std, static_std]), dtype=torch.float32)
    
    input_dim = X_mean.shape[0]

    # Sample Generation
    samples = []
    Y_samples, Y_weights = [], []
    valid_years = sorted(annual_returns.index)
    
    half_life = config.get("ml_time_decay_half_life", 10)
    max_year = valid_years[-1].year if len(valid_years) > 0 else 2026
    
    for i in range(1, len(valid_years) - 1):
        target_date = valid_years[i]
        for ticker, info in ticker_data.items():
            first_valid_date = pd.Timestamp(info['dates'][0])
            if target_date < first_valid_date + pd.Timedelta(days=365):
               continue
               
            y_vec = []
            y_w_vec = []
            
            time_w = 1.0
            if half_life is not None:
                time_w = 2.0 ** (-(max_year - target_date.year) / half_life)
                
            has_valid = False
            for h in horizons:
                if i + h < len(valid_years):
                    for metric in target_metrics:
                        val = metric_sources[metric][ticker].iloc[i+1 : i+h+1].mean()
                        y_vec.append(val if pd.notna(val) else np.nan)
                        y_w_vec.append(horizon_weights[h] * time_w)
                        if pd.notna(val): has_valid = True
                else:
                    y_vec.extend([np.nan] * len(target_metrics))
                    y_w_vec.extend([horizon_weights[h] * time_w] * len(target_metrics))
                    
            if has_valid:
                samples.append((ticker, target_date, y_vec, y_w_vec))
                Y_samples.append(y_vec)
    
    Y_tensor = torch.tensor(Y_samples, dtype=torch.float32)
    Y_mean = torch.nanmean(Y_tensor, dim=0, keepdim=True)
    Y_std = []
    for col in range(Y_tensor.shape[1]):
        col_data = Y_tensor[:, col]
        col_data = col_data[~torch.isnan(col_data)]
        Y_std.append(col_data.std().item() + 1e-8 if len(col_data) > 1 else 1.0)
    Y_std = torch.tensor(Y_std, dtype=torch.float32).unsqueeze(0)
    
    # Store normalized targets based on TEMPORAL SPLIT (as per User Request)
    # -------------------------------------------------------------------------
    val_mode = config.get("ml_validation_mode", "absolute")
    val_years = config.get("ml_validation_horizon_years", 20)
    val_pct = config.get("ml_validation_percent", 0.20)
    
    # Identify the global timeline cutoff
    max_dt = pd.Timestamp(valid_years[-1])
    if val_mode == "absolute":
        cutoff_date = max_dt - pd.DateOffset(years=val_years)
    else:
        # Proportional: Find the date at the Nth percentile of the total year-range
        timeline_len = (max_dt - pd.Timestamp(valid_years[0])).days
        cutoff_date = pd.Timestamp(valid_years[0]) + pd.Timedelta(days=int(timeline_len * (1.0 - val_pct)))
        
    train_samples = []
    val_samples = []
    
    for (ticker, td, yv, ywv) in samples:
        # Normalize target
        n_yv = (torch.tensor(yv) - Y_mean.squeeze(0)) / Y_std.squeeze(0)
        sample_tuple = (ticker, td, n_yv.tolist(), ywv)
        
        if pd.Timestamp(td) < cutoff_date:
            train_samples.append(sample_tuple)
        else:
            val_samples.append(sample_tuple)
            
    if verbose:
        print(f"  [Member A] Temporal Split Cutoff: {cutoff_date.date()}")
        print(f"  [Member A] Train Samples: {len(train_samples)} (Pre-{cutoff_date.year})")
        print(f"  [Member A] Val Samples:   {len(val_samples)} ({cutoff_date.year}+)")

    train_ds = TransformerDataset(train_samples, ticker_data, X_mean, X_std, max_seq_len)
    val_ds   = TransformerDataset(val_samples, ticker_data, X_mean, X_std, max_seq_len)
    
    batch_size = config.get("ml_batch_size", 32)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    model = AssetTransformerNet(input_dim=input_dim, config=config, output_macro_dim=output_macro_dim, output_ar_dim=output_ar_dim)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.get("ml_learning_rate", 0.001))
    epochs = config.get("ml_epochs", 150)
    early_stopping = EarlyStopping(patience=10, min_delta=1e-4)

    if verbose:
        print(f"[{time.strftime('%H:%M:%S')}] [Member A] Training {len(train_ds)} limit: {max_seq_len} tokens / Valid {len(val_ds)}. (Device: {device})")

    best_val_loss = float('inf')
    start_epoch = 0
    resume_mode = config.get("ml_resume_mode", "auto")
    cache_dir = config.get("ml_cache_dir", "cache")
    
    # Ensure cache directory exists before training starts
    os.makedirs(cache_dir, exist_ok=True)
    
    # DYNAMIC CHECKPOINT NAMING
    fname_base = get_transformer_filename_base(config)
    checkpoint_path = os.path.join(cache_dir, f"checkpoint_{fname_base}.pt")
    
    if resume_mode == "restart" and os.path.exists(checkpoint_path):
        if verbose: print(f"  [Member A] Restart Mode: Deleting existing checkpoint '{checkpoint_path}'...")
        os.remove(checkpoint_path)
    
    if resume_mode != "restart" and os.path.exists(checkpoint_path):
        if verbose: print(f"  [Member A] Resume Mode: Loading checkpoint for {fname_base} from '{checkpoint_path}'...")
        try:
            # Check for standard model vs JIT
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            early_stopping.best_loss = best_val_loss
            early_stopping.counter = checkpoint.get('early_stop_counter', 0)
            if verbose: print(f"  [Member A] Successfully resumed from Epoch {start_epoch+1}. Previous Best Val Loss: {best_val_loss:.4f}")
        except Exception as e:
            print(f"  [WARNING] Failed to load checkpoint: {e}. Starting from scratch.")

    for epoch in range(start_epoch, epochs):
        # Manual stop flag: create this file to stop training cleanly
        stop_flag_path = os.path.join(cache_dir, "STOP_TRAINING")

        if os.path.exists(stop_flag_path):
            if verbose:
                print(f"[{time.strftime('%H:%M:%S')}] [Member A] Stop flag detected. Saving checkpoint and exiting training loop.")
            torch.save({
                'epoch': max(epoch - 1, start_epoch - 1),
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'early_stop_counter': early_stopping.counter,
                'X_mean': X_mean,
                'X_std': X_std,
                'Y_mean': Y_mean,
                'Y_std': Y_std
            }, checkpoint_path)
            try:
                os.remove(stop_flag_path)
            except Exception:
                pass
            break
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", disable=not verbose, leave=False)
        for b_x, b_y, b_w, b_mask in pbar:
            b_x, b_y, b_w, b_mask = b_x.to(device), b_y.to(device), b_w.to(device), b_mask.to(device)
            optimizer.zero_grad()
            
            ar_preds, macro_preds, _ = model(b_x, src_key_padding_mask=b_mask)
            
            valid_ar_mask = ~b_mask[:, 1:] 
            target_ar = b_x[:, 1:, :3] 
            diff_ar = ar_preds[:, :-1, :][valid_ar_mask] - target_ar[valid_ar_mask]
            
            ar_loss = (diff_ar ** 2).mean() if valid_ar_mask.sum() > 0 else 0.0
            macro_loss = masked_weighted_mse_loss(macro_preds, b_y, b_w)
            
            loss = ar_loss + macro_loss
            if loss.requires_grad and not torch.isnan(loss) and loss.item() > 0:
                loss.backward()
                optimizer.step()
                curr_loss = loss.item()
                total_loss += curr_loss
                pbar.set_postfix({"loss": f"{curr_loss:.4f}"})
                
        model.eval()
        val_loss_sum = 0
        vbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", disable=not verbose, leave=False)
        with torch.no_grad():
            for b_x, b_y, b_w, b_mask in vbar:
                b_x, b_y, b_w, b_mask = b_x.to(device), b_y.to(device), b_w.to(device), b_mask.to(device)
                ar_preds, macro_preds, _ = model(b_x, src_key_padding_mask=b_mask)
                valid_ar_mask = ~b_mask[:, 1:] 
                target_ar = b_x[:, 1:, :3] 
                diff_ar = ar_preds[:, :-1, :][valid_ar_mask] - target_ar[valid_ar_mask]
                ar_loss = (diff_ar ** 2).mean() if valid_ar_mask.sum() > 0 else 0.0
                macro_loss = masked_weighted_mse_loss(macro_preds, b_y, b_w)
                v_loss = (ar_loss + macro_loss).item()
                val_loss_sum += v_loss
                vbar.set_postfix({"v_loss": f"{v_loss:.4f}"})
                
        val_loss = val_loss_sum / max(1, len(val_loader))
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] Epoch {epoch+1}/{epochs} | Train Dual-Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")
            
        
        early_stopping(val_loss)
        best_val_loss = early_stopping.best_loss
        
        # --- CHECKPOINTING ---
        check_freq = config.get("ml_checkpoint_frequency", 1)
        if (epoch + 1) % check_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'early_stop_counter': early_stopping.counter,
                'X_mean': X_mean,
                'X_std': X_std,
                'Y_mean': Y_mean,
                'Y_std': Y_std
            }, checkpoint_path)
        
        if early_stopping.early_stop:
            if verbose:
                print(f"[{time.strftime('%H:%M:%S')}] Early stopping triggered at epoch {epoch+1}")
            break

    # Extract Current Embeddings for all assets
    model.eval()
    embeddings = {}
    asset_predictions = {}
    
    with torch.no_grad():
        for ticker, info in ticker_data.items():
            feats = info['feats'][-max_seq_len:]
            L = len(feats)
            static_feats = np.tile(info['static'], (L, 1))
            seq = np.concatenate([feats, static_feats], axis=1)
            
            x_t = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
            x_norm = (x_t - X_mean) / X_std
            x_norm = x_norm.to(device)
            
            # Predict
            mask = torch.zeros((1, L), dtype=torch.bool).to(device)
            _, pred, emb = model(x_norm, src_key_padding_mask=mask)
            embeddings[ticker] = emb.squeeze(0).cpu().numpy()
            
            pred_raw = (pred.squeeze(0).cpu() * Y_std.squeeze(0) + Y_mean.squeeze(0)).numpy()
            preds = {}
            idx = 0
            for h in horizons:
                preds[h] = {}
                for metric in target_metrics:
                    preds[h][metric] = float(pred_raw[idx])
                    idx += 1
            asset_predictions[ticker] = preds
            
    print(f"[{time.strftime('%H:%M:%S')}] [Member A] Generated dynamic sequence embeddings for {len(embeddings)} assets.")
    
    result_cache = {
        "dynamic_embeddings": embeddings,
        "asset_predictions": asset_predictions,
        "master_df": master_df,
        "price_matrix": price_matrix,
        "volume_matrix": volume_matrix,
        "daily_returns": daily_returns,
        "drip_daily_returns": drip_daily_returns,
        "val_loss": best_val_loss,
        "model_state": model.state_dict(),
        "X_mean": X_mean,
        "X_std": X_std,
        "Y_mean": Y_mean,
        "Y_std": Y_std,
        "input_dim": input_dim,
        "output_macro_dim": output_macro_dim,
        "output_ar_dim": output_ar_dim,
        "ml_embedding_dim": config.get("ml_embedding_dim"),
        "ml_d_model": config.get("ml_d_model"),
        "ml_learning_rate": config.get("ml_learning_rate"),
        "epoch": epoch
    }

    # Always save final artifacts at end of training, including early-stop exits
    save_embedding_cache(result_cache, folder=cache_dir)

    return {
        "dynamic_embeddings": embeddings,
        "asset_predictions": asset_predictions,
        "master_df": master_df,
        "price_matrix": price_matrix,
        "volume_matrix": volume_matrix,
        "daily_returns": daily_returns,
        "drip_daily_returns": drip_daily_returns,
        "val_loss": best_val_loss,
        "model_state": model.state_dict(),
        "X_mean": X_mean,
        "X_std": X_std,
        "Y_mean": Y_mean,
        "Y_std": Y_std,
        "input_dim": input_dim,
        "output_macro_dim": output_macro_dim,
        "output_ar_dim": output_ar_dim
    }

def run_ml_grid_search(master_df, price_matrix, volume_matrix, daily_returns, config, drip_daily_returns=None):
    print(f"[{time.strftime('%H:%M:%S')}] Starting Sequence Transformer Grid Search...")
    lrs = config.get("grid_lrs", [config.get("ml_learning_rate", 0.001)])
    batch_sizes = config.get("grid_batch_sizes", [config.get("ml_batch_size", 32)])
    d_models = config.get("grid_d_models", [config.get("ml_d_model", 64)])
    emb_dims = config.get("grid_embedding_dims", [config.get("ml_embedding_dim", 8)])
    
    results = []
    for lr in lrs:
        for bs in batch_sizes:
            for dm in d_models:
                for ed in emb_dims:
                    run_config = config.copy()
                    run_config["ml_learning_rate"] = lr
                    run_config["ml_batch_size"] = bs
                    run_config["ml_d_model"] = dm
                    run_config["ml_embedding_dim"] = ed
                    
                    print(f"\\n--- Testing Config: LR={lr}, BatchSize={bs}, d_model={dm}, emb_dim={ed} ---")
                    cache = train_pytorch_embedding_model(
                        master_df, price_matrix, volume_matrix, daily_returns, 
                        run_config, drip_daily_returns=drip_daily_returns, verbose=False
                    )
                    results.append({
                        "LR": lr, 
                        "Batch Size": bs, 
                        "d_model": dm, 
                        "Emb Dim": ed, 
                        "Val MSE": cache["val_loss"]
                    })
                
    df = pd.DataFrame(results).sort_values("Val MSE").reset_index(drop=True)
    return df

def get_transformer_filename_base(config):
    """
    Generate a unique, clean filename base for the current model architecture.
    Example: ed8_dm64_lr001
    """
    ed = config.get("ml_embedding_dim", 8)
    dm = config.get("ml_d_model", 64)
    lr = config.get("ml_learning_rate", 0.001)
    
    # Clean LR: 0.001 -> 001
    lr_clean = str(lr).replace("0.", "").replace(".", "")
    return f"ed{ed}_dm{dm}_lr{lr_clean}"

def save_embedding_cache(cache_data, folder="cache"):
    """
    Saves only the essential embedding results and model state to disk.
    Uses dynamic naming based on the model's architecture.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Use config info from cache if available, otherwise default naming
    fname_base = get_transformer_filename_base(cache_data)
    model_path = os.path.join(folder, f"checkpoint_{fname_base}.pt")
    final_path = os.path.join(folder, f"final_{fname_base}.pt")
    data_path  = os.path.join(folder, f"results_{fname_base}.pkl")
    
    # 1. Save results dict (embeddings & predictions)
    save_dict = {
        "dynamic_embeddings": cache_data.get("dynamic_embeddings"),
        "asset_predictions": cache_data.get("asset_predictions"),
        "val_loss": cache_data.get("val_loss"),
        "config_snapshot": {
            "ed": cache_data.get("ml_embedding_dim"),
            "dm": cache_data.get("ml_d_model"),
            "lr": cache_data.get("ml_learning_rate")
        }
    }
    with open(data_path, "wb") as f:
        pickle.dump(save_dict, f)
        
    # 2. Save model state and normalization stats
    if "model_state" in cache_data:
        payload = {
            "model_state": cache_data["model_state"],
            "X_mean": cache_data["X_mean"],
            "X_std": cache_data["X_std"],
            "Y_mean": cache_data["Y_mean"],
            "Y_std": cache_data["Y_std"],
            "input_dim": cache_data.get("input_dim"),
            "output_macro_dim": cache_data.get("output_macro_dim"),
            "output_ar_dim": cache_data.get("output_ar_dim"),
            "epoch": cache_data.get("epoch", 999) # If saving final
        }
        torch.save(payload, final_path)
    
    print(f"[{time.strftime('%H:%M:%S')}] [Member A] Persistence: Results and model saved to '{folder}/{fname_base}'")

def load_embedding_cache(master_df, price_matrix, volume_matrix, daily_returns, config, drip_daily_returns=None, folder="cache"):
    """
    Loads saved model and embeddings from disk if they exist for the SPECIFIC config.
    """
    fname_base = get_transformer_filename_base(config)
    final_path = os.path.join(folder, f"final_{fname_base}.pt")
    data_path  = os.path.join(folder, f"results_{fname_base}.pkl")
    
    if not os.path.exists(final_path) or not os.path.exists(data_path):
        return None
        
    print(f"[{time.strftime('%H:%M:%S')}] [Member A] Persistence: Found existing model for architecture {fname_base}. Loading...")
    
    try:
        with open(data_path, "rb") as f:
            data = pickle.load(f)
            
        checkpoint = torch.load(final_path, map_location='cpu')
        
        cache = {
            "dynamic_embeddings": data["dynamic_embeddings"],
            "asset_predictions": data["asset_predictions"],
            "master_df": master_df,
            "price_matrix": price_matrix,
            "volume_matrix": volume_matrix,
            "daily_returns": daily_returns,
            "drip_daily_returns": drip_daily_returns,
            "val_loss": data.get("val_loss"),
            "model_checkpoint": checkpoint # Store full state for later inference
        }
        return cache
    except Exception as e:
        print(f"  [WARNING] Failed to load cache for {fname_base}: {e}")
        return None

# HELPER FUNCTIONS IF WE NEED - NOT CURRENTLY USING THESE
def save_training_checkpoint(
    model,
    optimizer,
    epoch,
    best_val_loss,
    early_stopper,
    X_mean,
    X_std,
    Y_mean,
    Y_std,
    input_dim,
    output_macro_dim,
    output_ar_dim,
    path="cache/asset_transformer_checkpoint.pt"
):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "early_stopping_counter": early_stopper.counter,
        "early_stopping_best_loss": early_stopper.best_loss,
        "early_stopping_triggered": early_stopper.early_stop,
        "X_mean": X_mean,
        "X_std": X_std,
        "Y_mean": Y_mean,
        "Y_std": Y_std,
        "input_dim": input_dim,
        "output_macro_dim": output_macro_dim,
        "output_ar_dim": output_ar_dim,
    }, path)

    print(f"[Member A] Checkpoint saved to {path}")


def load_training_checkpoint(path, config, device):
    if not os.path.exists(path):
        return None

    ckpt = torch.load(path, map_location=device)

    model = AssetTransformerNet(
        input_dim=ckpt["input_dim"],
        config=config,
        output_macro_dim=ckpt["output_macro_dim"],
        output_ar_dim=ckpt["output_ar_dim"]
    ).to(device)

    model.load_state_dict(ckpt["model_state"])

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get("ml_learning_rate", 0.001)
    )
    optimizer.load_state_dict(ckpt["optimizer_state"])

    early_stopper = EarlyStopping(
        patience=config.get("ml_patience", 5),
        min_delta=config.get("ml_min_delta", 1e-4)
    )
    early_stopper.counter = ckpt.get("early_stopping_counter", 0)
    early_stopper.best_loss = ckpt.get("early_stopping_best_loss", None)
    early_stopper.early_stop = ckpt.get("early_stopping_triggered", False)

    return {
        "model": model,
        "optimizer": optimizer,
        "start_epoch": ckpt["epoch"] + 1,
        "best_val_loss": ckpt["best_val_loss"],
        "early_stopper": early_stopper,
        "X_mean": ckpt["X_mean"],
        "X_std": ckpt["X_std"],
        "Y_mean": ckpt["Y_mean"],
        "Y_std": ckpt["Y_std"],
    }