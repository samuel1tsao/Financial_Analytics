"""
Vector Encoder: Transforms structured questionnaire JSON into actionable portfolio parameters.
Enhanced with RL-Transformer multi-horizon recommendation engine.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import logging
import json
import os
import sys
from typing import Any, Dict, List

# Add root to sys.path to access root modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from _constants import (
    DEFAULT_PIPELINE_CONFIG, RELATIVE_WEIGHT_THRESHOLD, MIN_ACTIVE_WEIGHT, MAX_DYNAMIC_ASSETS,
    SIM_TERMINAL_HORIZON_CAP,
    CAPITAL_NORMALIZER, GOAL_YEAR_NORMALIZER,
    TRADING_DAYS_PER_YEAR, CASH_BUCKET_ANNUAL_RATE, DAILY_RETURN_CLAMP, LOAN_DAILY_RATE,
    CACHE_DIR
)
from _rl_worker import PortfolioTransformerRL, _softmax_normalize_top_k, _encode_user_condition
from _data_worker import generate_dataset_member_a
from _ml_worker import load_embedding_cache
from _sim_worker import build_simulation_cache, _resolve_simulation_starts

logger = logging.getLogger(__name__)

# Singleton instance for the RL Recommender to avoid reloading data on every request
_RECOMMENDER_INSTANCE = None

# ─── Asset Universe (Legacy / Fallback) ──────────────────────────────────────
EQUITY_UNIVERSE = {
    "VOO":  {"name": "S&P 500",           "risk": 0.5, "category": "large_cap"},
    "QQQ":  {"name": "Nasdaq 100",         "risk": 0.7, "category": "tech"},
    "VTI":  {"name": "Total Market",       "risk": 0.5, "category": "broad"},
    "VXUS": {"name": "International",      "risk": 0.6, "category": "intl"},
    "VGT":  {"name": "Info Tech",          "risk": 0.8, "category": "sector_tech"},
    "ARKK": {"name": "Innovation ETF",     "risk": 0.95, "category": "speculative"},
    "VNQ":  {"name": "Real Estate",        "risk": 0.6, "category": "reit"},
    "VWO":  {"name": "Emerging Markets",   "risk": 0.75, "category": "emerging"},
}

BOND_UNIVERSE = {
    "BND":  {"name": "Total Bond Market",  "risk": 0.1, "category": "bond"},
    "SGOV": {"name": "Short Treasury",     "risk": 0.02, "category": "stbond"},
    "TLT":  {"name": "20+ Year Treasury",  "risk": 0.25, "category": "lt_bond"},
    "TIPS": {"name": "Inflation Protected", "risk": 0.1, "category": "tips"},
}

# ─── Fallback Constants (used when DB has no market data) ────────────────────
FALLBACK_EXPECTED_RETURNS = {
    "VOO": 0.10, "QQQ": 0.12, "VTI": 0.10, "VXUS": 0.07,
    "VGT": 0.13, "ARKK": 0.15, "VNQ": 0.08, "VWO": 0.09,
    "BND": 0.04, "SGOV": 0.045, "TLT": 0.035, "TIPS": 0.038,
}

FALLBACK_VOLATILITIES = {
    "VOO": 0.16, "QQQ": 0.20, "VTI": 0.16, "VXUS": 0.17,
    "VGT": 0.22, "ARKK": 0.35, "VNQ": 0.20, "VWO": 0.22,
    "BND": 0.04, "SGOV": 0.01, "TLT": 0.14, "TIPS": 0.05,
}

FALLBACK_CATEGORY_CORRELATIONS = {
    ("equity", "equity"): 0.75,
    ("equity", "bond"): -0.15,
    ("bond", "bond"): 0.60,
}

# ─── RL Recommender Service ──────────────────────────────────────────────────

class RLRecommender:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(RLRecommender, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def _load_state_dict_with_compat(self, model, state_dict):
        """Maps old dual-head keys (pre/post) to the new single-head architecture.
        Uses strict=False to handle architecture migrations (e.g., old flat model -> new adapter model).
        """
        new_state_dict = {}
        for k, v in state_dict.items():
            # 1. Ignore post-goal heads (they are redundant in the single-head architecture)
            if k.startswith("mu_head_post") or k.startswith("sigma_head_post"):
                continue
            # 2. Map pre-goal heads to the new single heads
            elif k.startswith("mu_head_pre"):
                new_state_dict[k.replace("mu_head_pre", "mu_head")] = v
            elif k.startswith("sigma_head_pre"):
                new_state_dict[k.replace("sigma_head_pre", "sigma_head")] = v
            # 3. Keep standard names if they already exist (single-head v3+)
            elif k.startswith("mu_head") or k.startswith("sigma_head"):
                new_state_dict[k] = v
            # 4. Keep all other layers (transformer, input_proj, adapters)
            else:
                new_state_dict[k] = v
        
        # Pre-filter: remove keys whose shapes don't match the current model
        # This handles architecture transitions (e.g., old input_proj [64,254] -> new [64,128])
        model_state = model.state_dict()
        compatible_state = {}
        skipped_keys = []
        for k, v in new_state_dict.items():
            if k in model_state and model_state[k].shape == v.shape:
                compatible_state[k] = v
            else:
                skipped_keys.append(k)
        
        if skipped_keys:
            logger.info(f"Compat load - skipped {len(skipped_keys)} shape-mismatched/legacy keys: {skipped_keys}")
        
        result = model.load_state_dict(compatible_state, strict=False)
        if result.missing_keys:
            # Adapter keys are expected to be missing when loading old checkpoints
            unexpected_missing = [k for k in result.missing_keys if 'adapter' not in k and k not in skipped_keys]
            if unexpected_missing:
                logger.warning(f"Compat load - unexpected missing keys: {unexpected_missing}")

    def __init__(self):
        if self._initialized:
            return
        
        self.config = DEFAULT_PIPELINE_CONFIG.copy()
        self.device = torch.device('cpu')
        
        # 1. Load Data
        from _constants import DataSyncMode
        self.config["data_source_mode"] = DataSyncMode.OFFLINE_CSV_ONLY
        
        try:
            res = generate_dataset_member_a([], self.config)
            self.master_df = res[0]
            self.price_matrix = res[1]
            self.volume_matrix = res[2]
            self.daily_returns = res[3]
            self.drip_daily_returns = res[4]
            
            emb_cache = load_embedding_cache(
                self.master_df, self.price_matrix, self.volume_matrix, self.daily_returns, self.config, 
                drip_daily_returns=self.drip_daily_returns, folder=CACHE_DIR
            )
            if emb_cache:
                self.dynamic_embeddings = emb_cache["dynamic_embeddings"]
                self.X_mean = emb_cache["model_checkpoint"]["X_mean"]
                self.X_std = emb_cache["model_checkpoint"]["X_std"]
                logger.info(f"RL Embeddings loaded successfully from {CACHE_DIR}")
                
                # Load ML Model for Dynamic PIT Recalculation
                try:
                    from _ml_worker import AssetTransformerNet
                    horizons = self.config.get("ml_target_horizons", [1, 3, 5, 10, 15])
                    target_metrics = self.config.get("ml_target_metrics", ["return", "volatility", "volume"])
                    output_macro_dim = len(horizons) * len(target_metrics)
                    output_ar_dim = 3
                    input_dim = self.X_mean.shape[0]
                    
                    self.ml_model = AssetTransformerNet(input_dim, self.config, output_macro_dim, output_ar_dim)
                    self.ml_model.load_state_dict(emb_cache["model_checkpoint"]["model_state"])
                    self.ml_model.eval()
                    logger.info("Successfully loaded Phase 1 ML model for dynamic embedding recalculation.")
                except Exception as ex:
                    logger.warning(f"Could not load ML model for dynamic embedding recalculation: {ex}")
                    self.ml_model = None
            else:
                self.dynamic_embeddings = {}
                self.ml_model = None
                logger.error("Failed to load Phase 1 embeddings")
        except Exception as e:
            logger.error(f"Error loading dataset for RL: {e}")
            self.dynamic_embeddings = {}
            self.ml_model = None

        # 2. Dynamic Model Discovery (Ensemble)
        # Calculate expected input_dim dynamically to match current features
        try:
            from _rl_worker import _get_static_feature_columns, _encode_user_condition, TEST_PROFILES
            static_cols = _get_static_feature_columns(self.master_df)
            static_dim = len(static_cols)
            
            # emb_dim is usually 8, user_cond_dim is 8
            emb_dim = len(next(iter(self.dynamic_embeddings.values()))) if self.dynamic_embeddings else 8
            user_cond_dim = len(_encode_user_condition(TEST_PROFILES[0]))
            
            target_input_dim = emb_dim + user_cond_dim + static_dim
            logger.info(f"RL Recommender: Calculated target_input_dim={target_input_dim} (Emb:{emb_dim}, User:{user_cond_dim}, Static:{static_dim})")
        except Exception as e:
            logger.warning(f"Could not calculate target_input_dim dynamically, using fallback: {e}")
            target_input_dim = 230 # Current production default

        self.models = []
        self.loaded_input_dim = target_input_dim
        
        # 1. Helper to find agents for a specific dimension
        import glob
        import re
        def find_valid_agents_for_dim(dim):
            p1 = os.path.join(CACHE_DIR, f"checkpoint_rl_v2_*_id{dim}_agent_*.pt")
            p2 = os.path.join(CACHE_DIR, f"checkpoint_rl_v2_*_id{dim}.pt")
            files = glob.glob(p1)
            if not files:
                files = glob.glob(p2)
            return [f for f in files if os.path.getsize(f) > 0]
            
        agent_files = find_valid_agents_for_dim(target_input_dim)
        
        # 2. Dynamic Fallback: if no valid agents for target_input_dim, fallback to highest available dimension
        if not agent_files:
            all_files = glob.glob(os.path.join(CACHE_DIR, "checkpoint_rl_v2_*_id*.pt"))
            available_dims = []
            for f in all_files:
                if os.path.getsize(f) > 0:
                    m = re.search(r'_id(\d+)(?:_agent)?', f)
                    if m:
                        available_dims.append(int(m.group(1)))
            
            if available_dims:
                fallback_dim = max(set(available_dims)) # take the most recent/highest dimension
                logger.warning(f"No valid RL agents found for target_input_dim {target_input_dim}. Falling back to {fallback_dim}.")
                self.loaded_input_dim = fallback_dim
                agent_files = find_valid_agents_for_dim(fallback_dim)

        if agent_files:
            for epath in sorted(agent_files):
                try:
                    ecp = torch.load(epath, map_location=self.device, weights_only=False)
                    m = PortfolioTransformerRL(self.loaded_input_dim, self.config).to(self.device)
                    
                    # Handle different checkpoint formats (single agent vs ensemble master)
                    if 'model_state' in ecp:
                        self._load_state_dict_with_compat(m, ecp['model_state'])
                        m.eval()
                        self.models.append(m)
                    elif 'agents_state' in ecp and isinstance(ecp['agents_state'], list):
                        # If it's a master file with multiple states, load all of them
                        for state in ecp['agents_state']:
                            m_sub = PortfolioTransformerRL(self.loaded_input_dim, self.config).to(self.device)
                            self._load_state_dict_with_compat(m_sub, state)
                            m_sub.eval()
                            self.models.append(m_sub)
                    
                    logger.info(f"Successfully loaded agent(s) from {os.path.basename(epath)}")
                except Exception as e:
                    logger.error(f"Failed to load agent file {os.path.basename(epath)}: {e}")
            
            if self.models:
                logger.info(f"Ensemble ready with {len(self.models)} total agents.")
            else:
                logger.warning(f"No valid RL agents found for input_dim {self.loaded_input_dim}.")
        else:
            logger.warning(f"No RL checkpoints found matching any recent input_dim in {CACHE_DIR}.")

        self._initialized = True

    def _recalculate_point_in_time_embeddings(self, tickers: list, as_of_date: str) -> dict:
        """
        Dynamically computes the historical sequence embeddings up to as_of_date
        to completely eliminate look-ahead bias and data poisoning.
        """
        if not hasattr(self, "ml_model") or self.ml_model is None:
            return {}
            
        import pandas as pd
        import numpy as np
        
        # 1. Truncate returns and volume up to as_of_date
        dr = self.daily_returns[self.daily_returns.index <= as_of_date] if self.daily_returns is not None else None
        if dr is None or dr.empty:
            return {}
            
        # Re-compute dynamic features up to as_of_date
        daily_ret = dr.copy()
        
        # Rolling 21-day volatility
        daily_vol = daily_ret.rolling(21, min_periods=1).std() * np.sqrt(252)
        daily_vol = daily_vol.fillna(0.0)
        daily_ret = daily_ret.fillna(0.0)
        
        # Log volume
        daily_logv = np.log1p(self.volume_matrix)
        daily_logv = daily_logv[daily_logv.index <= as_of_date]
        daily_logv = daily_logv.fillna(0.0)
        
        # Categorical / static features
        categorical_cols = ["sector", "industry", "state", "quoteType", "exchange"]
        master_encoded = pd.get_dummies(self.master_df, columns=[c for c in categorical_cols if c in self.master_df.columns], drop_first=True)
        all_feature_cols = [c for c in master_encoded.columns if any(c.startswith(cat + "_") for cat in categorical_cols)]
        
        max_seq_len = self.config.get("ml_max_seq_len", 3780)
        pit_embeddings = {}
        
        self.ml_model.eval()
        with torch.no_grad():
            for t in tickers:
                if t not in daily_ret.columns or t not in daily_vol.columns or t not in daily_logv.columns:
                    continue
                if t not in master_encoded.index:
                    continue
                    
                df_t = pd.DataFrame({
                    'ret': daily_ret[t],
                    'vol': daily_vol[t],
                    'logv': daily_logv[t]
                }).dropna()
                
                if len(df_t) < 5:
                    continue
                    
                # Truncate to max_seq_len
                df_t = df_t.tail(max_seq_len)
                L = len(df_t)
                
                feat_vec = master_encoded.loc[t, all_feature_cols].astype(float).values
                static_feats = np.tile(feat_vec, (L, 1))
                
                seq = np.concatenate([df_t.values, static_feats], axis=1)
                
                x_t = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
                x_norm = (x_t - self.X_mean) / self.X_std
                
                # Check for NaNs or Infs and fill with 0
                x_norm = torch.nan_to_num(x_norm, nan=0.0, posinf=0.0, neginf=0.0)
                
                mask = torch.zeros((1, L), dtype=torch.bool)
                
                _, _, emb = self.ml_model(x_norm, src_key_padding_mask=mask)
                pit_embeddings[t] = emb.squeeze(0).cpu().numpy()
                
        return pit_embeddings

    def get_weights(self, user_profile: dict, tickers: list, as_of_date: str = None) -> tuple:
        """
        Inference: Returns (pre_goal_weights, post_goal_weights) as dictionaries.
        If as_of_date is provided, volatility and consistency filters use data up to that date.
        """
        if not self.models or not self.dynamic_embeddings:
            return {}, {}

        # 1. Build input tensor
        from _rl_worker import _get_static_feature_columns
        static_cols = _get_static_feature_columns(self.master_df)
        
        emb_rows = []
        static_rows = []
        valid_tickers = []
        
        # Fundamental Volatility Filter: Automatically weed out strictly broken assets
        # We keep this broad to avoid "market timing", relying on RL hold policy instead
        vol_sens = float(user_profile.get("volatility_sensitivity", 5))
        max_allowed_vol = 1.50 - ((vol_sens - 1) / 9.0) * 1.25
        
        # Recalculate embeddings dynamically in memory if as_of_date is provided
        pit_embeddings = {}
        if as_of_date:
            try:
                pit_embeddings = self._recalculate_point_in_time_embeddings(tickers, as_of_date)
                logger.info(f"Dynamically recalculated {len(pit_embeddings)} point-in-time embeddings for {as_of_date}.")
            except Exception as e:
                logger.warning(f"Failed to recalculate point-in-time embeddings: {e}")

        # 1. Filter data by date if requested
        dr = self.daily_returns
        if as_of_date and self.daily_returns is not None:
            dr = self.daily_returns[self.daily_returns.index <= as_of_date]
        
        # First pass: collect all eligible candidates with their volatility scores
        all_candidates = []
        all_candidates_dict = {}
        for t in tickers:
            if t in self.dynamic_embeddings and t in self.master_df.index:
                recent_vol = None
                hist_vol = None
                if dr is not None and t in dr.columns:
                    recent_vol = dr[t].tail(30).std() * np.sqrt(252)
                    hist_vol = dr[t].tail(252).std() * np.sqrt(252)
                all_candidates.append((t, recent_vol, hist_vol))
                all_candidates_dict[t] = recent_vol
        
        # Second pass: apply volatility and consistency filters
        for t, recent_vol, hist_vol in all_candidates:
            passed = True
            if recent_vol is not None and pd.notna(recent_vol):
                # 1. Check if RECENT volatility exceeds the user's tolerance
                if recent_vol > max_allowed_vol:
                    passed = False
                # 2. Check for Consistency: Reject assets whose volatility has recently spiked >50%
                elif hist_vol is not None and pd.notna(hist_vol) and hist_vol > 0:
                    if (recent_vol / hist_vol) > 1.5:
                        passed = False
            
            if passed:
                emb = None
                if as_of_date:
                    emb = pit_embeddings.get(t)
                    if emb is None:
                        try:
                            from _ml_worker import load_walkforward_embedding
                            emb = load_walkforward_embedding(t, as_of_date, self.config)
                        except Exception:
                            pass
                if emb is None:
                    emb = self.dynamic_embeddings[t]
                emb_rows.append(emb)
                static_rows.append(self.master_df.loc[t, static_cols].values)
                valid_tickers.append(t)
        
        # Fallback: if vol filter eliminated ALL candidates, relax and take the lowest-vol assets
        if not valid_tickers and all_candidates:
            logger.warning(f"Volatility filter (max={max_allowed_vol:.2f}) eliminated all {len(all_candidates)} candidates. "
                           f"Relaxing to lowest-volatility assets.")
            # Sort by recent_vol (ascending), NaN last
            sorted_candidates = sorted(all_candidates, key=lambda x: x[1] if (x[1] is not None and pd.notna(x[1])) else 999.0)
            # Take the top 30 lowest-volatility assets as fallback
            for t, _, _ in sorted_candidates[:30]:
                emb = None
                if as_of_date:
                    emb = pit_embeddings.get(t)
                    if emb is None:
                        try:
                            from _ml_worker import load_walkforward_embedding
                            emb = load_walkforward_embedding(t, as_of_date, self.config)
                        except Exception:
                            pass
                if emb is None:
                    emb = self.dynamic_embeddings[t]
                emb_rows.append(emb)
                static_rows.append(self.master_df.loc[t, static_cols].values)
                valid_tickers.append(t)
        
        if not valid_tickers:
            return {}, {}

        emb_matrix = np.stack(emb_rows).astype(np.float32)
        static_matrix = np.stack(static_rows).astype(np.float32)
        
        # Build user condition
        user_vec = np.array(_encode_user_condition(user_profile), dtype=np.float32)
        N = emb_matrix.shape[0]
        user_tiled = np.tile(user_vec, (N, 1))
        
        # Handle Dimensional Drift: dynamically pad/truncate static_matrix if model expects different dimension
        expected_static_dim = getattr(self, "loaded_input_dim", 230) - emb_matrix.shape[1] - user_tiled.shape[1]
        if static_matrix.shape[1] > expected_static_dim:
            static_matrix = static_matrix[:, :expected_static_dim] # Truncate new features
        elif static_matrix.shape[1] < expected_static_dim:
            pad_width = expected_static_dim - static_matrix.shape[1]
            static_matrix = np.pad(static_matrix, ((0, 0), (0, pad_width)), mode='constant')
        
        full_input = np.concatenate([emb_matrix, user_tiled, static_matrix], axis=1)
        x = torch.tensor(full_input[np.newaxis, :, :], dtype=torch.float32).to(self.device)

        # 2. Forward pass (Ensemble Average)
        if not self.models:
            return {}, {}

        with torch.no_grad():
            mu_sum = 0
            for m in self.models:
                mu, _ = m(x)
                mu_sum += mu
            
            # Average the logits across all agents
            mu_avg = mu_sum / len(self.models)
            
            # Duplicate for compatibility (single head now handles both pre/post logic via condition)
            mu_pre = mu_avg
            mu_post = mu_avg

            # ── Dynamic Logit Standardization ──
            # The raw RL logits have near-zero variance. We MUST standardize them to a target standard deviation
            # to blow up the microscopic differences into meaningful Softmax percentages.
            conc_pref = float(user_profile.get("concentration_pref", 5))
            
            # High conc_pref (9) -> target_std = 1.2 (Blows up differences -> Concentrated Softmax)
            # Low conc_pref (1)  -> target_std = 0.2 (Keeps differences tiny -> Flat/Diversified Softmax)
            dynamic_target_std = 0.2 + ((conc_pref - 1) / 8.0) * 1.0

            def sharpen_logits(m, target_std=0.5):
                std = m.std(dim=-1, keepdim=True)
                std = torch.clamp(std, min=1e-6)
                return (m - m.mean(dim=-1, keepdim=True)) / std * target_std

            mu_pre_sharp = sharpen_logits(mu_pre, target_std=dynamic_target_std)
            mu_post_sharp = sharpen_logits(mu_post, target_std=dynamic_target_std)

            w_pre = F.softmax(mu_pre_sharp, dim=-1)[0].cpu().numpy()
            w_post = F.softmax(mu_post_sharp, dim=-1)[0].cpu().numpy()
            
        # 3. Post-process (Dynamic Top-K and thresholding)
        def _finalize_weights(w_np):
            if w_np.sum() == 0: return {}
            
            # 1. Dynamic Top-K: Keep assets >= RELATIVE_WEIGHT_THRESHOLD of max
            max_w = np.max(w_np)
            threshold = max_w * RELATIVE_WEIGHT_THRESHOLD
            
            # 2. Sort and filter
            idx_sorted = np.argsort(w_np)[::-1]
            dynamic_idx = [i for i in idx_sorted if w_np[i] >= threshold]
            
            # 3. Hard ceiling for safety
            top_idx = dynamic_idx[:MAX_DYNAMIC_ASSETS]
            
            final_w = np.zeros_like(w_np)
            final_w[top_idx] = w_np[top_idx]
            
            # 4. Normalize
            if final_w.sum() > 0:
                final_w /= final_w.sum()
            
            # 5. Apply min threshold
            keep = final_w >= MIN_ACTIVE_WEIGHT
            if not keep.any(): keep[:] = True
            final_w[~keep] = 0
            if final_w.sum() > 0:
                final_w /= final_w.sum()
                
            return {valid_tickers[i]: float(final_w[i]) for i in range(len(valid_tickers)) if final_w[i] > 0}

        return _finalize_weights(w_pre), _finalize_weights(w_post)

    def _finalize_weights_ui(self, weights: Dict[str, float], rationales: Dict[str, str]):
        """Normalize weights and ensure rationales exist for all."""
        total = sum(weights.values())
        if total <= 1e-9: return {}, {}
        
        # 1. Prune tiny weights (< 0.5% relative to total) to keep UI clean
        # But don't prune if it leaves us with nothing
        pruned = {t: w for t, w in weights.items() if (w/total) >= 0.005}
        if not pruned: pruned = weights
        
        # 2. Re-normalize to exactly 1.0
        final_total = sum(pruned.values())
        norm_w = {t: round(w/final_total, 4) for t, w in pruned.items()}
        
        # 3. Ensure rationales exist
        final_r = {}
        for t in norm_w:
            rat = rationales.get(t)
            if not rat or rat == "RL-optimized allocation based on user profile.":
                # Add dynamic rationale based on asset type if missing
                if t in BOND_UNIVERSE:
                    rat = "Selected for capital preservation and yield stability."
                else:
                    rat = "Selected for growth potential aligned with horizon risk tolerance."
            final_r[t] = rat
            
        return norm_w, final_r

# ─── Multi-Horizon Encoding ──────────────────────────────────────────────────

def encode_multi_horizon(answers: dict, require_historical_years: int = None, as_of_date: str = None) -> dict:
    """
    Given a multi-goal questionnaire, returns the 'pre' weights for each contiguous segment.
    Handles 'reserved assets' by fixing their weights and scaling the RL recommendation.
    If require_historical_years is provided, only assets with that many years of historical data are recommended.
    If as_of_date is provided, the recommendation is made as of that point in history.
    """
    global _RECOMMENDER_INSTANCE
    if _RECOMMENDER_INSTANCE is None:
        logger.info("Initializing global RLRecommender instance...")
        _RECOMMENDER_INSTANCE = RLRecommender()
    
    recommender = _RECOMMENDER_INSTANCE
    dr = recommender.daily_returns
    if as_of_date and dr is not None:
        dr = dr[dr.index <= as_of_date]
    
    start_cap = float(answers.get("start_cap") or 100000)
    monthly_contrib = float(answers.get("monthly_contrib") or 500)
    goals = answers.get("goals", [])
    
    # NEW: Handle multiple hard constraints from frontend schema
    hard_constraints = answers.get("hard_constraints", [])
    reserved_weights = {}
    for c in hard_constraints:
        ticker = c.get("ticker")
        pct_val = c.get("pct")
        if ticker and pct_val is not None and str(pct_val).strip() != "":
            try:
                reserved_weights[ticker.upper()] = float(pct_val) / 100.0
            except (ValueError, TypeError):
                continue
            
    total_reserved_ratio = sum(reserved_weights.values())
    
    if total_reserved_ratio >= 1.0:
        # User reserved 100% or more (oops). Cap it at 95% to allow some recommendation or return just constraints.
        total_reserved_ratio = 1.0
        scale = 0.0
    else:
        scale = 1.0 - total_reserved_ratio
    
    if not goals:
        # Default 30 year horizon if no goals, but we'll clamp it
        goals = [{"name": "Retirement", "amount": 1000000, "years": 30}]

    # Determine max horizon from data
    data_years = len(recommender.price_matrix) // TRADING_DAYS_PER_YEAR
    max_horizon = min(data_years - 1, SIM_TERMINAL_HORIZON_CAP)
        
    sorted_goals = sorted(goals, key=lambda g: g.get("years", 30))
    horizons = [min(g.get("years", 30), max_horizon) for g in sorted_goals]
    
    segments = []
    # Segment 1: Start to Goal 1
    # Segment 2: Goal 1 to Goal 2
    # ...
    # Segment N: Goal N-1 to Goal N
    
    prev_yr = 0
    for i, g in enumerate(sorted_goals):
        curr_yr_requested = g.get("years", 30)
        curr_yr = min(curr_yr_requested, max_horizon)
        goal_amount = g.get("amount", 0)
        
        # Condition RL agent for this specific goal horizon
        profile = {
            "drawdown_sensitivity": answers.get("drawdown_sensitivity", 5),
            "volatility_sensitivity": answers.get("volatility_sensitivity", 5),
            "goal_flexibility": answers.get("goal_flexibility", 5),
            "concentration_pref": answers.get("concentration_pref", 5),
            "start_cap": start_cap,
            "monthly_contrib": monthly_contrib,
            "goal_year": curr_yr,
            "goal_amount": goal_amount
        }
        
        # Ensure we only recommend assets that have existed for at least required years as of the target date
        valid_tickers = list(recommender.dynamic_embeddings.keys())
        if dr is not None:
            required_years = require_historical_years if require_historical_years is not None else 0
            min_required_days = int((required_years * TRADING_DAYS_PER_YEAR) * 0.95)
            valid_tickers = [
                t for t in valid_tickers 
                if t in dr.columns 
                and dr[t].count() >= min_required_days
            ]
            if not valid_tickers:
                # Fallback: Relax constraint, but still strictly require the asset to have existed as of as_of_date
                # (i.e., has at least 5 returns in dr, which is already sliced <= as_of_date)
                valid_tickers = [
                    t for t in recommender.dynamic_embeddings.keys()
                    if t in dr.columns 
                    and dr[t].count() >= 5
                ]
                logger.info(f"Fallback triggered: relaxed required history but strictly limited to {len(valid_tickers)} assets active as of {as_of_date}")
                
        # Use RL agent to get "Pre-goal" weights for this segment
        # In a multi-horizon setup, we use the pre-goal head for the active goal segment.
        w_pre, _ = recommender.get_weights(profile, valid_tickers, as_of_date=as_of_date)
        
        # Integrate reserved assets
        # Weights = sum(reserved) + (1 - sum(reserved_ratio)) * w_pre
        combined_weights = reserved_weights.copy()
        for t, wt in w_pre.items():
            combined_weights[t] = combined_weights.get(t, 0) + wt * scale
            
        # Generate rationales
        rationales = {}
        for t, wt in combined_weights.items():
            if t in reserved_weights:
                rationales[t] = "User-defined custom preference."
                continue
                
            vol = None
            if recommender and hasattr(recommender, 'master_df') and t in recommender.master_df.index:
                row = recommender.master_df.loc[t]
                if 'hist_volatility' in row:
                    vol = row['hist_volatility']
            
            if wt >= 0.20:
                prefix = "High-conviction primary holding"
            elif wt >= 0.10:
                prefix = "Core portfolio allocation"
            elif wt >= 0.05:
                prefix = "Strategic diversifier"
            else:
                prefix = "Tactical exposure"
                
            if vol is not None:
                if vol > 0.4:
                    reason = "selected for high-variance growth potential to overcome funding shortfalls."
                elif vol > 0.25:
                    reason = "selected for aggressive capital appreciation."
                elif vol < 0.15:
                    reason = "providing crucial downside protection and stability."
                else:
                    reason = "offering balanced risk-adjusted compounding."
            else:
                reason = "optimized by the RL policy for this time horizon."
                
            rationales[t] = f"{prefix} {reason}"
            
        final_w, final_r = recommender._finalize_weights_ui(combined_weights, rationales)
        segments.append({
            "horizon_years": (prev_yr, curr_yr),
            "goal_name": g.get("name", f"Goal {i+1}"),
            "weights": final_w,
            "rationales": final_r
        })
        prev_yr = curr_yr

    # NEW: Add Terminal Growth Phase (Phase N+1)
    # Uses the 'post-goal' weights from the RL model for the last goal's profile
    if prev_yr < max_horizon:
        # Re-fetch weights to get mu_post from the last goal context
        _, w_post = recommender.get_weights(profile, valid_tickers if require_historical_years is not None else list(recommender.dynamic_embeddings.keys()), as_of_date=as_of_date)
        
        combined_post_weights = reserved_weights.copy()
        for t, wt in w_post.items():
            combined_post_weights[t] = combined_post_weights.get(t, 0) + wt * scale
            
        post_rationales = {}
        for t, wt in combined_post_weights.items():
            if t in reserved_weights:
                post_rationales[t] = "User-defined custom preference."
                continue
            
            if wt >= 0.20:
                post_rationales[t] = "High-conviction primary holding optimized for long-term terminal growth."
            elif wt >= 0.10:
                post_rationales[t] = "Core allocation selected for sustained long-term compounding."
            else:
                post_rationales[t] = "Diversifying asset to balance terminal horizon risk."
            
        final_w, final_r = recommender._finalize_weights_ui(combined_post_weights, post_rationales)
        segments.append({
            "horizon_years": (prev_yr, max_horizon),
            "goal_name": "Growth Phase",
            "weights": final_w,
            "rationales": final_r
        })
        
    # Calculate an aggregate risk score for the profile
    ds = float(answers.get("drawdown_sensitivity", 5))
    vs = float(answers.get("volatility_sensitivity", 5))
    risk = (ds + vs) / 2.0

    return {
        "risk_score": risk,
        "start_cap": start_cap,
        "monthly_contrib": monthly_contrib,
        "segments": segments,
        "goals": sorted_goals,
        "hard_constraints": hard_constraints
    }


# ─── Multi-Horizon Simulation ───────────────────────────────────────────────

def simulate_multi_horizon_portfolio(
    segments: list,
    projection_years: int = 30,
) -> dict:
    """
    Run a year-by-year simulation of portfolio returns, switching weights at segment boundaries.
    """
    recommender = RLRecommender()
    if not recommender._initialized or recommender.daily_returns is None or len(recommender.daily_returns.columns) == 0:
        # Fallback if no data
        return {"error": "Market data unavailable for simulation"}

    total_years = projection_years

    # 1. Identify active tickers across all segments
    active_tickers = set()
    for seg in segments:
        if isinstance(seg.get("weights"), dict):
            for t in seg["weights"].keys():
                active_tickers.add(t)
    
    active_tickers = [t for t in active_tickers if t in recommender.daily_returns.columns]
    if not active_tickers:
        # Fallback to S&P 500 if no valid tickers found (to avoid crash)
        cols = recommender.daily_returns.columns
        if "VOO" in cols:
            active_tickers = ["VOO"]
        elif "SPY" in cols:
            active_tickers = ["SPY"]
        elif len(cols) > 0:
            active_tickers = [cols[0]]
        else:
            return {"error": "No valid assets found in market data for simulation"}

    # 2. Extract only required daily returns
    base_returns = recommender.drip_daily_returns if recommender.drip_daily_returns is not None else recommender.daily_returns
    filtered_returns = base_returns[active_tickers]
    
    # 3. Build a Mini-Cache (Fast!)
    cache_array, start_idx_to_pos, clean_returns, column_to_idx, max_sim_years = build_simulation_cache(filtered_returns, max_horizon_years=total_years)
    
    # Clamp total years to data-derived max for simulation
    total_years = min(total_years, max_sim_years)

    # 4. Determine start dates for Monte Carlo (use high-fidelity starts)
    sim_starts = _resolve_simulation_starts(clean_returns)
    if len(sim_starts) == 0:
        return {"error": "Insufficient historical data for simulation paths"}

    from _sim_worker import calculate_decay_probabilities
    all_start_dates = clean_returns.index[sim_starts]
    decay_probs = calculate_decay_probabilities(all_start_dates, half_life_years=3.0)

    # Take 1000 paths for high fidelity
    rng = np.random.default_rng(42)
    num_paths = 1000
    
    # 3. Pre-process segment weights into year-indexed arrays for speed
    num_assets = len(clean_returns.columns)
    year_weight_arrays = []
    for yr in range(1, total_years + 1):
        # Find active segment for this year
        active_w_dict = {}
        for seg in segments:
            start, end = seg["horizon_years"]
            if start < yr <= end:
                active_w_dict = seg["weights"]
                break
        if not active_w_dict:
            # Fallback to last segment if beyond
            active_w_dict = segments[-1]["weights"]
            
        wa = np.zeros(num_assets)
        for t, wt in active_w_dict.items():
            if t in column_to_idx:
                wa[column_to_idx[t]] = wt
        if wa.sum() > 0: wa /= wa.sum()
        year_weight_arrays.append(wa)
    
    year_weight_arrays = np.array(year_weight_arrays)

    # 4. Vectorized Block Bootstrapping
    num_starts = cache_array.shape[0]
    path_returns = np.zeros((num_paths, total_years))
    
    for p in range(num_paths):
        # Sample random 1-year blocks for each projection year
        bootstrap_indices = rng.choice(num_starts, size=total_years, replace=True, p=decay_probs)
        for yr in range(total_years):
            cache_row = cache_array[bootstrap_indices[yr]]
            path_returns[p, yr] = np.dot(year_weight_arrays[yr], cache_row)

    # 6. Prepare raw paths for frontend percentile calculation
    raw_paths = np.round(path_returns, 6).tolist()
    
    # Still send a basic expected return array just in case it's needed for fallback
    expected_annual_returns = [round(float(np.mean(path_returns[:, yr])), 6) for yr in range(total_years)]

    # 6. Calculate Per-Segment Stats
    segment_stats = []
    for seg in segments:
        start_yr, end_yr = seg["horizon_years"]
        seg_returns = path_returns[:, start_yr:end_yr]
        if seg_returns.size > 0:
            mean_ret = float(np.mean(seg_returns))
            std_ret = float(np.std(seg_returns))
            seg_sharpe = (mean_ret - 0.02) / std_ret if std_ret > 0 else 0
            segment_stats.append({
                "expected_return": round(mean_ret * 100, 2),
                "volatility": round(std_ret * 100, 2),
                "sharpe_ratio": round(float(seg_sharpe), 2)
            })
        else:
            segment_stats.append({"expected_return": 0, "volatility": 0, "sharpe_ratio": 0})

    return {
        "raw_paths": raw_paths,
        "expected_annual_returns": expected_annual_returns,
        "segment_stats": segment_stats,
    }

# ─── Legacy Wrappers ─────────────────────────────────────────────────────────

def classify_asset(ticker: str) -> str:
    if ticker in BOND_UNIVERSE: return "bond"
    return "equity"

def build_covariance_matrix(tickers: list[str]) -> np.ndarray:
    try:
        from market_data import get_covariance_matrix as _get_real_cov
        return _get_real_cov(tickers)
    except Exception as e:
        logger.warning(f"Could not build covariance from DB, using fallback: {e}")
    # Fallback: category-based approximation
    n = len(tickers)
    cov = np.zeros((n, n))
    for i, t1 in enumerate(tickers):
        for j, t2 in enumerate(tickers):
            vol1 = FALLBACK_VOLATILITIES.get(t1, 0.15)
            vol2 = FALLBACK_VOLATILITIES.get(t2, 0.15)
            cat1 = classify_asset(t1)
            cat2 = classify_asset(t2)
            if i == j:
                cov[i][j] = vol1 ** 2
            else:
                pair = tuple(sorted([cat1, cat2]))
                corr = FALLBACK_CATEGORY_CORRELATIONS.get(pair, 0.5)
                cov[i][j] = corr * vol1 * vol2
    return cov

def encode_questionnaire(answers: dict) -> dict:
    # Legacy heuristic logic - keep for backward compat
    # ... (omitted for brevity, or we can replace it with the RL one)
    # Actually, let's just make it return the RL one if possible.
    return encode_multi_horizon(answers)

def simulate_portfolio(weights, goals, initial_investment, years):
    # Legacy wrapper for old API
    # We'll adapt it to the multi-horizon format
    segments = [{"horizon_years": (0, years), "weights": weights}]
    res = simulate_multi_horizon_portfolio(segments, years)
    return {
        "expected_path": res.get("expected_annual_returns", []),
        "years": list(range(years)),
        "step_balances": []
    }

# Eagerly initialize the recommender to reduce first-request latency
try:
    logger.info("Eagerly warming up RLRecommender and Market Data...")
    _RECOMMENDER_INSTANCE = RLRecommender()
except Exception as e:
    logger.error(f"Failed to eager-load RLRecommender: {e}")
