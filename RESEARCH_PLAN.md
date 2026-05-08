# Multi-Goal Financial Asset Recommender System
## Project Plan & Research Specification

> **Focus:** Hybrid Recommendation & Portfolio Optimization  
> **Course:** CMPE 256  
> **Team:** Samuel Tsao, Ianna Duran, Christian Oh

---

## 📋 Table of Contents
1. [Objective](#1-objective)
2. [Dataset](#2-dataset)
3. [Methodology](#3-methodology)
4. [MVP Scope & Implementation](#4-mvp-scope--implementation)
5. [Novelty & Data Mining Significance](#5-novelty--data-mining-significance)
6. [Evaluation Metrics](#6-evaluation-metrics)
7. [4-Week Execution Plan](#7-4-week-execution-plan)
8. [MVP Design Specification](#8-mvp-design-specification)
9. [Key Design Considerations & Trade-offs](#9-key-design-considerations--trade-offs)
10. [Stretch Goals & Future Enhancements](#10-stretch-goals--future-enhancements)
11. [Architectural Alternatives & Embedding Considerations](#11-architectural-alternatives--embedding-considerations)

---

## 1. Objective
We propose a personalized stock and ETF recommender system that integrates user-specific goals with **Modern Portfolio Theory (MPT)**. While standard recommenders focus on item similarity, our system treats financial assets as "items" to be matched with a user’s unique risk profile, sector interest, and temporal goals (short vs. long term), transforming a simple ranking task into a constrained optimization problem.

## 2. Dataset
We synthesize data from Yahoo Finance and Wikipedia to create a rich, safety-classified feature set for our S&P 1500 universe:
*   **Market Data:** Daily prices, returns, and volatility via Yahoo Finance API for ~1,500 equities and ETFs.
*   **Company Identity (SAFE_STATIC):** Sector, industry, state, exchange, quoteType — time-invariant structural features safe for historical backtesting.
*   **Stable Fundamentals (SLOW_CHANGING):** Employee count, governance risk scores — change gradually, acceptable for embedding training with acknowledged approximation.
*   **Excluded (POINT_IN_TIME):** Price-derived metrics (P/E, Market Cap, Debt-to-Equity, beta, etc.) are **deliberately excluded** from embedding training to prevent data poisoning in backtesting (see Section 11.7).

> **Data Poisoning Prevention:** All candidate features are classified by temporal stability using a `COLUMN_SAFETY_REGISTRY` with 150+ annotated yfinance columns. Only `SAFE_STATIC` and `DERIVED` features are permitted as embedding model inputs. See Section 11.7 for full rationale.

## 3. Methodology
The core of the project involves transitioning from early baseline models to an **End-to-End Transformer Architecture trained via Reinforcement Learning (Policy Gradients)**.
*   **Baselines:** Popularity-based (Top Gainers) and Content-Based Filtering using metadata (Sector/Industry) - e.g., S&P index funds.
*   **RL-Driven Transformer Pipeline:**
    1.  **Ingestion:** Concatenates Transformer-derived temporal sequence embeddings (replacing older autoencoders), user features, and static asset features into a dense vector for every asset.
    2.  **Core Engine (Transformer):** Self-Attention contextualizes the entire market without pre-filtering, dynamically calculating cross-asset covariance and user-asset relevance.
    3.  **Action Generation (Gaussian Policy):** Outputs a parameterized normal distribution ($\mu$, $\sigma$) per asset. $\mu$ serves as the target allocation, while $\sigma$ dictates the RL agent's exploration noise.
    4.  **Black-Box Environment:** Executes sampled portfolio weights through a discrete, non-differentiable simulator handling cash flows and goal logic, outputting a scalar Reward.
    5.  **Policy Gradient Update:** Uses the REINFORCE framework to bypass simulator non-differentiability, pulling $\mu$ toward successful allocations and refining $\sigma$.

## 4. MVP Scope & Implementation
The MVP will focus on a **Personalized Allocation Pipeline**:
1.  **Input:** User defines Risk Tolerance (1–10), Sector Preferences, and Time Horizons (e.g., "5-year house down payment" vs. "30-year retirement").
2.  **Processing:** The system filters assets by risk/volatility, then applies a weighted allocation logic.
3.  **Output:** A diversified portfolio recommendation including specific ticker symbols and percentage weights.

### Future Extensions (Stretch Goals)
*   **Time-Decay Rebalancing:** Automated risk reduction as goal dates approach.
*   **Tax-Aware Optimization:** Logic to prioritize long-term capital gains.

## 5. Novelty & Data Mining Significance
*   **Multi-Objective Optimization:** Unlike traditional systems (e.g., Netflix), this system balances competing objectives: **Return vs. Risk vs. Time**.
*   **Context-Aware Recommendations:** The "item" value changes based on the user's temporal context (a stock might be a 'buy' for a 30-year goal but a 'skip' for a 2-year goal).

## 6. Evaluation Metrics
We will use a dual-evaluation strategy:
*   **Recommender Metrics:** NDCG and Precision@K to measure how well the system identifies high-performing assets within the user's preferred sectors.
*   **Financial Metrics:** Back-testing recommended portfolios against a **Sharpe Ratio** benchmark and the S&P 500.
*   **Terminal Wealth Comparison:** Simulating a $10,000 investment over a 20-30 year period against:
    *   **Static Baseline:** 60/40 Stock/Bond split.
    *   **Aggressive Baseline:** 100% S&P 500 Equity.
*   **Goal Success Rate:** Measuring **Max Drawdown** for short/medium-term goals. Success is defined by capital preservation during volatility.

---

## 7. 4-Week Execution Plan

### Team Roles
*   **Ianna Duran (Data & ML Lead):** Data ingestion, latent embeddings, asset scoring, and UI.
*   **Christian Oh (Financial Logic Lead):** Goal modeling, risk decay (Glide Path), dynamic selection, and Monte Carlo generation.
*   **Samuel Tsao (Simulation & Architecture Lead):** Repo structure, simulator, historical backtesting, and optimization loop.

### Weekly Milestones

#### 🗓️ Week 1: Data Infrastructure & Foundation
*   **Data/ML:** Set up `yfinance` pipelines; calculate rolling returns, volatility, MDD; start Hybrid Embedding model.
*   **Finance:** Implement `UserObjective` class; code Nonlinear Risk Scaling (Glide Path).
*   **Sim/Arch:** Initialize repository; build base `Simulator` class; implement core loop with cash flow deductions.
*   🏁 **Milestone:** Data pipeline functional; dummy portfolio passes through simulation with cash deductions.

#### 🗓️ Week 2: Asset Scoring & Dynamic Selection
*   **Data/ML:** Finalize Feature Fusion; build parametric `ScoringFunction`.
*   **Finance:** Implement Dynamic K Thresholding; write unit tests for threshold scaling.
*   **Sim/Arch:** Upgrade Simulator to full Historical Backtester; implement rolling windows; track GFR and ETV.
*   🏁 **Milestone:** System can score the universe, filter candidates, and run a basic historical backtest.

#### 🗓️ Week 3: Portfolio Optimization & Synthesis
*   **Data/ML:** Implement Temperature-scaled Softmax Weighting.
*   **Finance:** Build Monte Carlo Bootstrap overlay for synthetic market paths.
*   **Sim/Arch:** Build Optimizer Loop (Random/Grid Search); connect Scoring and Simulation layers.
*   🏁 **Milestone:** MVP pipeline complete end-to-end; system automatically finds optimal allocation.

#### 🗓️ Week 4: Validation, UI, & Final Delivery
*   **Data/ML:** Wrap backend in Streamlit/Gradio UI.
*   **Finance:** Stress-test against 2000 & 2008 crashes; generate visualizations (Success Probability, Glide Path).
*   **Sim/Arch:** Generate Correlation Matrix; lead final report drafting.
*   🏁 **Milestone:** Code freeze; UI deployable; final report and presentation slides ready.

---

## 8. MVP Design Specification

### Phase 1: Ingestion & Concatenation (The "Item" & "User" Layer)
We map assets into a comprehensive feature space, joining ML-derived behavioral embeddings with static identity and user context.

*   **Transformer Asset Embeddings (Replaces Autoencoder):** High-dimensional embeddings capturing temporal market dynamics.
    *   *Architecture:* A sequence-based Transformer model processing a 5-year sliding context window (1260 trading day tokens) with positional embeddings to understand volatility clustering and regime shifts natively.
    *   *Training:* Persistent caching and temporal data splitting ensure robust, reproducible training pipelines.
*   **Feature Fusion (Safety-Classified):**
    For each asset, the input to the Transformer is concatenataed:
    `Input Vector = [Autoencoder_Embedding (8d), User_Risk_Budget (1d), SAFE_STATIC_Features (~217d)]`
    *(See Section 11.7 for safety classification constraints on features like Market Cap)*

### Phase 2: The Core Engine (Transformer)
Translating the entire market universe into context-aware representations natively.

*   All asset vectors pass through a **Self-Attention** mechanism simultaneously.
*   The Transformer natively calculates cross-asset similarity and user-asset relevance without pre-filtering, bypassing traditional similarity score modules.

### Phase 3: Action Generation (The Gaussian Policy)
Translating Transformer output into normalized portfolio allocations via an RL policy head.

*   A linear projection layer outputs a normal distribution for each asset:
    *   **Mean ($\mu$):** The intended target portfolio weight.
    *   **Std Dev ($\sigma$):** The RL agent's exploration noise (NOT historical volatility).
*   **Zero-Weighting:** Applies ReLU or Sparsemax to $\mu$ allowing explicit 0.0% allocations.
*   **Action Sampling:** Weights are sampled from $N(\mu, \sigma)$ and normalized to 100%.

### Phase 4: The Black-Box Environment (Simulator)
Executes discrete, non-differentiable logic that prevents standard backpropagation.

*   Runs `_sim_worker.py` logic: cascading cash flows, loan generation on missed milestones, bounded daily returns.
*   Returns an **Inverse-Exponential Reward Score** coupling terminal wealth limits and goal penalties.

### Phase 5: Policy Gradient Update (Evaluation & Training)
Closing the loop end-to-end.

*   **Loss Function:** `Loss = -(Reward - Baseline) * log(Probability of Sampled Weights)`
*   **Variance Reduction:** Utilizes an Exponential Moving Average (EMA) baseline to center gradient updates, stabilizing learning across the massive continuous action space.
*   **Stochastic Updates:** Performs ultra-fast gradient updates after every iteration using three key throughput optimizations (see Section 12.8): (1) a pre-computed Annual Simulation Cache replaces per-step Pandas slicing with a single `np.dot` per year, (2) Random Asset Subset Sampling (`RL_ASSET_SUBSET_SIZE = 500`) reduces Transformer attention cost by ~169×, and (3) each gradient update uses only `rl_paths_per_step = 5` simulation paths instead of 50, enabling ~10× more updates per wall-clock second. Combined, these achieve sustained 1+ it/s training throughput.
*   **Optimization:** Gradient flows from the Policy Loss directly through the Gaussian Policy and into the RL Transformer, shifting $\mu$ toward successful weights and shrinking $\sigma$ as confidence grows.

---

## 9. Key Design Considerations & Trade-offs

| Choice | Rationale | Trade-off |
| :--- | :--- | :--- |
| **Simulated Loss vs. Ground Truth** | Lack of user portfolio data. Simulation validates scoring weights. | Computationally expensive; requires vectorized code. |
| **Dynamic K vs. Fixed K** | Avoids forced diversification in bad regimes. | Harder to build static UI (asset count fluctuates). |
| **ETFs + Stocks Matrix** | ETFs provide stability (Beta); Stocks provide growth (Alpha). | Increases data ingestion and sector mapping complexity. |
| **Nonlinear vs. Linear Decay** | Matches professional "Target Date Fund" logic. | Harder to explain math to non-technical users. |
| **Autoencoder vs. GRU/LSTM** | Feedforward autoencoder with per-year snapshots avoids sequence-length sensitivity and trains faster on ~1,500 assets × ~20 years. Milestone horizons capture temporal structure without recurrence. | Loses intra-year sequential patterns; cannot model regime transitions within a year. See 11.8. |
| **SAFE_STATIC features only** | Prevents data poisoning: backtesting uses TODAY's snapshot for all historical years. Only time-invariant features (sector, state, industry) avoid leakage. | Loses potentially useful signals (P/E, beta, margins). Accepted trade-off for backtest integrity. See 11.7. |
| **Block Bootstrapping vs. Chronological** | Bootstrapping 1-year blocks from the full 20-year cache increases regime diversity and prevents the model from overfitting to a specific historical sequence. | Can lose longer-term cyclical dependencies (>1 year). |
| **Recency-Weighted Sampling** | Exponential decay probabilities (10y half-life) ensure the model prioritizes modern market dynamics while still learning from 2008-style crashes. | Reduces the statistical weight of early-2000s data. |
| **Dynamic Top-K Thresholding** | Replaces static limits with a relative threshold (10% of max weight). Allows portfolio size to adapt to market confidence. | Makes UI layout more fluid as asset counts fluctuate. |
| **Configurable Hidden Layers** | `ml_hidden_layers: [128, 64, 32]` allows architecture tuning via dashboard config. Input dimension (~224) requires gradual compression to 8-dim embedding. | Deeper networks risk overfitting on ~30k training samples. Mitigated by masked loss + z-score normalization. |
| **Learnable Embedding Adapter** | A trainable 2-layer MLP inside the RL Transformer expands frozen 8-dim embeddings to 32-dim, allowing RL gradients to learn which embedding aspects are useful for allocation. | Adds ~1,300 parameters. Requires fresh training run (architecture break). Does not update the upstream embeddings themselves. |
| **Static Interpretation Cards** | Replaced real-time jumping text above sliders with fixed-height cards below. Improves readability and eliminates layout jarring. | Requires manual mapping of all 1-10 values to descriptive rationales. |

---

## 10. Stretch Goals & Future Enhancements
*   **9.1 Tax-Aware Liquidation:** Modify simulator to track holding periods and prioritize long-term capital gains/tax-loss harvesting.
*   **9.2 Self-Supervised ML Scoring:** Generate 100k randomized portfolios to train a neural network to predict success probability.
*   **9.3 Reinforcement Learning Allocator:** Formulate allocation as an MDP (State, Action, Reward) using PPO/DDPG.
*   **9.4 Regime-Switching Monte Carlo:** Use Hidden Markov Models (HMM) to simulate prolonged bear/bull regimes.
*   **9.5 Multi-Phase "Waterfall" Allocation:** (Active) Sequence multiple single-goal inferences to handle complex multi-goal users via the dual-head policy (Pre-Goal $W_1 \to$ Pre-Goal $W_2 \to$ Post-Goal $W_n$).

---

## 11. Architectural Alternatives & Embedding Considerations

### 11.1 Forward-Predicting Embeddings vs. "Flipped" Generative Modeling
*   **Rejected:** "Flipped" models can lead to "Value Traps" (cheap stocks that are actually failing).
*   **Chosen:** ML predicts individual outcomes; Optimizer combines them into a portfolio.

### 11.2 Term Structure Loss (Milestone Horizons)
*   **Rejected:** 30-year granularity (too much noise).
*   **Chosen:** Loss calculated at Y1, Y3, Y5, Y10, Y15 with higher weight on near-term predictions.

### 11.3 Time-Decayed Sample Weighting
*   **Chosen:** Exponential Decay Half-Life (configurable, default 10 years) for training samples.
*   **Rationale:** Financial markets undergo structural regime changes (pre/post-2008 regulatory shifts, rise of algorithmic trading, COVID-era monetary policy). A model trained uniformly on 2001–2024 data treats a 2002 sample identically to a 2022 sample, despite fundamentally different market microstructures. Time-decay ensures the model prioritizes modern dynamics while still learning from historical crashes.
*   **Implementation:** Each training sample generated from backtest year $y$ receives a weight or sampling probability:
    $$p(y) \propto 2^{\,-(y_{\max} - y) \,/ \,\lambda}$$
    where $\lambda$ is the half-life. **Aggressive Recency Bias:** The half-life is set to **3 years** to ensure that decade-old anomalies (e.g., 2015 spikes) do not dominate current projections.
*   **Block Bootstrapping:** To maximize statistical variance during training, the simulator constructs each 30-year path by drawing 30 independent 1-year blocks from the cache, weighted by the decay probabilities. This prevents the agent from simply memorizing the exact sequence of 2008 -> 2009.
*   **Configurable via Dashboard:** `ml_time_decay_half_life` parameter (set to `null` to disable and revert to uniform weighting for ablation studies).

### 11.4 Missing Token Imputation & Feature Safety Classification
*   **Rejected:** Deep fundamental data (too many missing values pre-2010, and most are `POINT_IN_TIME` snapshots).
*   **Chosen:** Price-derived temporal signals (`hist_momentum`, `hist_volatility`) computed per-year for unbroken 25-year stress testing, supplemented by time-invariant company identity features.
*   **Data Quality Gate:** A `run_data_diagnostics()` engine validates all configured features at runtime, reporting:
    *   Per-column fill rate (% non-null), dtype, unique value count
    *   Safety classification with data poisoning warnings
    *   Predicted input vector dimensionality and architecture preview
    *   Validation verdict (pass/fail) before training begins

### 11.5 The "Right-Edge Problem"
*   **Solution:** **Frozen Inference & Masked Training**. Frozen models generate today's predictions; Masked training updates weights as horizons elapse.

### 11.6 Layer 3 Hard Constraints (Sanity Check)
*   **Guardrail:** Optimizer acts as a logic gate (e.g., "Reject P/E > 100") regardless of embedding score, blending ML predictions with hard-coded financial logic.

### 11.7 Data Poisoning Prevention & Feature Safety Framework
A critical discovery during implementation: the backtesting training loop iterates through historical years to generate training samples, but **non-temporal features are pulled from today's `master_df` snapshot**. This means any feature that changes with market conditions (price ratios, market cap, volume, analyst targets, etc.) introduces **data leakage** — the model sees 2024 values when "predicting" 2015 forward returns.

*   **Solution:** A `COLUMN_SAFETY_REGISTRY` classifying all 150+ yfinance columns into safety tiers:
    *   🟢 `SAFE_STATIC` — Time-invariant company identity (sector, industry, state, exchange). Rarely changes; safe across all backtest years.
    *   🟡 `SLOW_CHANGING` — Gradual changes (employees, governance scores). Today's snapshot used for all years. Low risk, acknowledged approximation.
    *   🔴 `POINT_IN_TIME` — Current market snapshot (price, P/E, market cap, beta, margins, analyst targets). **Banned from embedding training.**
    *   ⚪ `DERIVED` — Computed per-year inside the training loop (hist_momentum, hist_volatility). Properly time-aligned, zero leakage.
    *   ⬜ `METADATA` / 📝 `TEXT` / 🏷️ `IDENTIFIER` — API plumbing, free text, names. Not useful for ML.

*   **Why Market Cap was removed:** `marketCap = currentPrice × sharesOutstanding`. The numerator is today's stock price, making it a direct function of the quantity the model is trying to predict (future asset performance). Including it is equivalent to giving the model the answer key during training. `fullTimeEmployees` serves as a leakage-free size proxy.

*   **Validation at runtime:** The `run_data_diagnostics()` function prints a full data dictionary with fill rates, validates every selected feature against the registry, and issues explicit warnings if any `POINT_IN_TIME` feature is configured for training.

### 11.8 Pivot from Autoencoder to Transformer Sequence Embeddings
The original plan specified a feedforward autoencoder. During implementation, we pivoted to a **Transformer-based sequence architecture**:

*   **Why we pivoted:** The autoencoder struggled to capture intricate temporal dynamics, sequence of returns, and the "progression of time" inherent in financial data. A Transformer, utilizing a 5-year sliding context window (1260 daily tokens) and positional embeddings, natively understands sequence and volatility clustering.
*   **Performance Trade-off:** While computationally heavier to train initially, the Transformer embeddings proved vastly superior at representing an asset's market behavior. By pairing this with persistent caching, we eliminated the inference bottleneck, passing rich, context-aware dense vectors to the downstream RL agent.

### 11.10 Reward Function: Geometric CAGR vs. Simple Return
*   **Rejected (Simple Average):** Arithmetic mean of multipliers (Terminal Wealth / Start Capital / Horizon) was mathematically unstable. It allowed "lottery ticket" paths (e.g., a 1,000,000x multiplier from sampled spikes) to generate astronomical rewards that dwarfed all risk penalties.
*   **Chosen (Geometric CAGR):** We shifted to the **Compound Annual Growth Rate**: `CAGR = (Mean_Market_Mult ^ (1/Horizon)) - 1`.
*   **Why it works:** This creates a smooth, linear gradient in annualized space. By squashing exponential wealth growth into a stable annual percentage, it ensures that **Volatility Penalties** (scaled to annual standard deviation) can properly balance and reject hyper-volatile anomalies. Critically, it decouples investment performance measurement from cash flow mechanics.

### 11.9 Multi-Horizon Input Feature Expansion & Lifespan Truncation
Instead of providing the autoencoder with just a 1-year trailing snapshot of `hist_momentum` and `hist_volatility`, the model inputs are expanded into **multi-dimensional vectors** that perfectly mirror the `ml_target_horizons` (e.g., [1, 3, 5, 10, 15] years). The model is given the 1-year, 3-year, 5-year, 10-year, and 15-year trailing backward averages for both return and volatility at every historical snapshot.

*   **Rationale:** Giving the model the structural historical context (e.g., "this asset had a bad 1-year run but a stellar 15-year record") provides much deeper behavioral clustering capability than a strictly cross-sectional 1-year view, compensating further for the removal of recurrent sequence models.
*   **The "Short Lifespan" Problem (Truncated Windows):** How do we handle companies that IPO'd 7 years ago when computing a 15-year trailing return?
    *   *Rejected (Padding):* Padding with 0s acts as a punitive drag on the average, misrepresenting a successful young company as a mediocre one.
    *   *Rejected (Dropping):* Dropping samples without 15 years of data would eliminate the vast majority of our S&P 1500 universe from the backtester.
    *   *Chosen (Truncated Windows):* If the required lookback window exceeds the asset's lifespan at that historical snapshot, we compute the mean over its maximum available lifetime `[max(0, i - h + 1) : i + 1]`. A 7-year-old company's "15-year trailing return" simply equals its lifetime 7-year return. This maximizes data utilization while preventing NaN contamination.

---
> [!TIP]
> **Risk Management:** If Hybrid Embeddings take too long, fall back to a simple Historical Correlation Matrix for Week 2 to avoid blocking other tracks.

---

## 12. RL Training Findings & Engineering Considerations

### 12.1 Reward Function Tuning History

| Constant | Original Value | Final Value | Rationale |
| :--- | :--- | :--- | :--- |
| `CAGR_REWARD_SCALE` | 100.0 | 200.0 | Returns were too small compared to penalties; agent became overly risk-averse. |
| `MDD_PENALTY_SCALE` | 4.0 | 2.0 | High-volatility growth assets were being unfairly penalized during exploration. |
| `VOL_PENALTY_SCALE` | 50.0 | 10.0 | Reduced to prevent over-penalizing reasonable market noise in conservative profiles. |
| `GFR_BRACKETS (0.0)` | -30.0 | -100.0 | Increased failure penalty to prevent the agent from "safely failing" to avoid volatility. |
| `rl_learning_rate` | 0.01 | 0.001 | High LR caused gradient explosions; policy head collapsed within first 50 iterations. |
| `DIVERSITY_THRESHOLD` | 0 | 10 | Introduced to allow broad portfolios (up to 10 assets) before overhead penalties apply. |
| `RISK_NORMALIZER` | N/A | 10.0 | Added for normalized aggregation of drawdown and volatility sensitivities across the risk profile. |

### 12.12 Math Observability & Debugging
**Problem:** The RL agent's loss was often opaque. High negative rewards were difficult to debug without seeing the exact interaction between return scores and risk penalties.
**Solution:** Implemented **Step-by-Step Math Logging**. Every training step now prints the exact equation used to compute the reward, including the risk-tolerance multipliers and desperation factor adjustments. This allows for real-time verification of the reward landscape.

### 12.13 Learnable Embedding Adapter (Gradient Flow into Embeddings)
**Problem:** The frozen 8-dim embeddings from Member A constituted only ~3.5% of the RL Transformer's input vector (~229 dims total). The remaining ~96.5% was static one-hot categoricals (sector, industry, exchange). This imbalance meant the RL model could learn effective policies based purely on sector patterns while ignoring the behavioral embeddings entirely.

**Alternatives Considered:**
- **Option B (Live Embedding Model):** Run Member A's full Transformer (3,780 tokens per asset) inside the RL forward pass. Principled but would drop throughput from 25 it/s to <0.1 it/s.
- **Option C (Periodic Fine-Tuning):** Periodically unfreeze Member A and update embeddings using accumulated RL reward signal. Complex implementation with stale gradient issues.

**Solution (Option A):** Added a **Learnable Embedding Adapter** inside `PortfolioTransformerRL` - a 2-layer MLP (`Linear(8->32) -> ReLU -> Linear(32->32)`) that projects the frozen embeddings into a richer representation before concatenation with other features. RL gradients flow through this adapter, learning to amplify or suppress specific embedding dimensions based on what improves portfolio rewards. The embedding's input share increases from ~3.5% to ~13% at a cost of only ~1,300 additional parameters with zero throughput impact.

---

### 12.2 MDD Must Reflect Pure Market Performance

**Problem discovered:** The original `max_drawdown` tracker used `peak_capital` (the running maximum of the cash balance). When a goal withdrawal was executed (e.g., paying out $200k for a retirement milestone), the denominator for MDD shrank, making the exact same market drop look proportionally worse. A portfolio that successfully paid out all goals was paradoxically penalized *more* on MDD than one that paid nothing.

**Solution:** Introduced a parallel `market_multiplier` tracker (starting at 1.0) that compounds only market returns and ignores all cash inflows/outflows. The `max_drawdown` metric is now computed exclusively from `market_multiplier`, decoupling investment performance measurement from cash flow mechanics.

```
market_multiplier *= (1 + yr_return)   # Pure market signal
mi_peak, max_drawdown = _update_drawdown(market_multiplier, mi_peak, max_drawdown)

capital *= (1 + yr_return)             # Real cash balance (used for withdrawals)
capital, bankrupt, goal_log = _process_goal_withdrawal(capital, year, goals)
```

### 12.3 Bankruptcy Type Taxonomy
**Problem discovered:** The simulation treated two fundamentally different failure modes as identical. Only `market` bankruptcies are assigned `max_drawdown = 1.0`. `goal` bankruptcies return the actual market-measured drawdown from `market_multiplier`.

### 12.4 Empty Portfolio Collapse (Policy Pruning Bug)
**Solution:** The training loop now detects all-zero weight arrays before calling `_run_single_sim_path` and immediately appends a max-penalty result without running the simulation.

### 12.5 Non-IPO Asset Exclusion
**Solution:** Implemented `src_key_padding_mask` for the Transformer encoder and `masked_fill(-1e9)` on `raw_action` to ensure non-IPO assets do not influence the policy.

### 12.6 Sigma Floor
**Solution:** Clamped `sigma` output: `sigma = F.softplus(self.sigma_head(out).squeeze(-1)) + 0.5` to preserve exploration.

### 12.7 Reward Math Log Transparency
Logs now include `+ Bonus` indicators for perfect GFR, making performance tracking visual in the training logs.

### 12.8 Training Throughput Optimizations
The simulation pipeline was optimized using `np.dot` caches and `joblib` parallelization, moving from ~0.01 it/s to 25+ it/s.

### 12.9 Dynamic Top-K vs. Empty Portfolio (The DECISIVE Policy)
**Problem discovered:** A static `TOP_K_ASSETS` limit forced the agent to always pick 10 assets, even if it only had confidence in 3.

**Solution:** **Relative Dynamic Thresholding**.
1. Calculate `max_weight` across all assets.
2. Keep any asset where `weight >= 0.1 * max_weight`.
3. Apply `DIVERSITY_PENALTY_SCALE` to the reward.

This creates "Decisiveness Pressure": the agent pays a tax for every asset it adds, driving the agent toward a concentrated, "High-Conviction" variety of assets.

### 12.10 The Desperation Factor (Hail Mary Mechanics)
**Problem:** The agent was mathematically incentivized to choose "Safe Failure" (failing the goal with low volatility) because the Volatility Penalty was larger than the Goal Failure Penalty.
**Solution:** Introduced a **Desperation Multiplier**. If the user's goal requires aggressive growth (Required CAGR > 10%), the agent dynamically suppresses Volatility and MDD penalties. This allows "Hail Mary" strategies where the agent aggressively pursues bursts of growth when it is the *only* path to goal fulfillment.

### 12.11 Upside Winsorization (Outlier Management)
**Problem:** A single historical anomaly (e.g., +10,000% spike from a reverse split) could be sampled multiple times in the bootstrapping engine, creating an "Infinite Money" illusion.
**Solution:** Implemented hard **Upside Winsorization** at **+500%** (6.0x) in the simulation cache. This clips extreme anomalies while preserving legitimate high-growth clusters (e.g., NVDA best years). Consistent "bursters" are now favored over one-off anomalies.

### 12.14 UI Interpretability & Sensitivity Mapping
**Problem:** Users found the questionnaire sliders "jarring" because the interpretation text above the slider jumped and shifted the layout during interaction. Additionally, numeric values were opaque without clear labels.
**Solution:** Migrated detailed interpretations to **Fixed-Height Cards** below the sliders. These cards provide static visual anchors and more detailed financial rationales for each of the 10 sensitivity levels. Added numeric tick marks (1-10) to the slider track to align the visual "feel" with the underlying RL condition vector.

---

## 13. Multi-Phase Waterfall Allocation (Engineering Spec)

To handle users with multiple goals using the dual-head RL model, we implement a **Waterfall Schedule**. This decouples the complex multi-objective optimization into a sequence of atomic, goal-specific strategy shifts.

### 13.1 Inference Flow
For a user with goals at $T_1, T_2, \dots, T_n$:
1.  **Iterative Scoring:** Run the RL Transformer $n$ times, once for each goal condition $(T_i, Amount_i)$.
2.  **Head Selection:** 
    *   For the $i$-th goal period $[T_{i-1}, T_i]$, select the `pre-goal` weights from Inference $i$.
    *   For the final terminal period $[T_n, 30]$, select the `post-goal` weights from Inference $n$.
3.  **Result:** A weight schedule $S = \{ (0, W_{pre,1}), (T_1, W_{pre,2}), \dots, (T_n, W_{post,n}) \}$.

### 13.2 Simulation Support
The `_sim_worker.py` engine is updated to accept a weight dictionary keyed by year. At the start of each simulated year, if `year` exists in the schedule, the active `weight_array` is updated.

---

## 14. Explainable AI & Rationale Generation

To improve user trust and transparency, the inference pipeline includes a **Rationale Engine** that explains the "why" behind every recommendation.

### 14.1 Categorization Logic
Assets are categorized based on their assigned weight:
*   **>20%**: "High-conviction primary holding"
*   **10-20%**: "Core portfolio allocation"
*   **5-10%**: "Strategic diversifier"
*   **<5%**: "Tactical exposure"

### 14.2 Behavioral Reasoning
The engine uses historical volatility (`hist_volatility`) to explain the asset's role in the specific goal horizon:
*   **High Volatility (>40%)**: Selected for "high-variance growth potential to overcome funding shortfalls."
*   **Low Volatility (<15%)**: Providing "crucial downside protection and stability."
*   **Moderate Volatility**: Offering "balanced risk-adjusted compounding."

### 14.3 Multi-Goal Sensitivity
The rationales change based on the waterfall segment. A stock might be a "Strategic diversifier" in a 5-year house goal but shift to a "High-conviction primary holding" in the 30-year Growth Phase, reflecting the RL agent's temporal adaptability.
