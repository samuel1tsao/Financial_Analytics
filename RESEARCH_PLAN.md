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
| **`fullTimeEmployees` vs. `marketCap`** | Market cap = price × shares (changes daily). Employee count is a stable company-size proxy that doesn't leak future price information. | Employee count changes annually, so it's an approximation. Classified `SLOW_CHANGING` with acknowledged risk. |
| **Configurable Hidden Layers** | `ml_hidden_layers: [128, 64, 32]` allows architecture tuning via dashboard config. Input dimension (~224) requires gradual compression to 8-dim embedding. | Deeper networks risk overfitting on ~30k training samples. Mitigated by masked loss + z-score normalization. |

---

## 10. Stretch Goals & Future Enhancements
*   **9.1 Tax-Aware Liquidation:** Modify simulator to track holding periods and prioritize long-term capital gains/tax-loss harvesting.
*   **9.2 Self-Supervised ML Scoring:** Generate 100k randomized portfolios to train a neural network to predict success probability.
*   **9.3 Reinforcement Learning Allocator:** Formulate allocation as an MDP (State, Action, Reward) using PPO/DDPG.
*   **9.4 Regime-Switching Monte Carlo:** Use Hidden Markov Models (HMM) to simulate prolonged bear/bull regimes.
*   **9.5 Multi-Phase "Waterfall" Allocation:** (Active) Sequence multiple single-goal inferences to handle complex multi-goal users via the dual-head policy (Pre-Goal $W_1 \to$ Pre-Goal $W_2 \to$ Post-Goal $W_n$).

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
The `_sim_worker.py` engine is updated to accept a weight dictionary keyed by year. At the start of each simulated year, if `year` exists in the schedule, the active `weight_array` is updated. This allows the backtester to evaluate the "Waterfall" performance on original multi-goal profiles while the model only ever has to learn "single-goal funding" and "general growth."

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
*   **Implementation:** Each training sample generated from backtest year $y$ receives a weight:
    $$w(y) = 2^{\,-(y_{\max} - y) \,/ \,\lambda}$$
    where $\lambda$ is the half-life (default 10 years) and $y_{\max}$ is the most recent year. This means:
    *   Samples from the most recent year: weight = **1.0**
    *   Samples from 10 years ago: weight = **0.5**
    *   Samples from 20 years ago: weight = **0.25**
*   **Integration with Masked Weighted MSE:** The per-sample time-decay weight is multiplied into the existing horizon-weighted loss, creating a compound weighting: `total_weight = horizon_weight × time_decay_weight`. This preserves the near-term prediction emphasis from Section 11.2 while adding temporal recency bias.
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

### 11.10 Reward Function: Annual Simple Return vs. CAGR
*   **Rejected (CAGR):** Compounding Annual Growth Rate proved mathematically unstable for RL gradients. When portfolios had heavy withdrawals or went bankrupt, CAGR suffered from division-by-zero errors or produced massive, exponential reward spikes that destroyed the training gradient.
*   **Chosen (Annualized Simple Return):** We shifted the objective function to `Total Profit = ETV + Actual Withdrawals - Start Capital`, yielding an `Annual Return = (Total Profit / Start Capital) / Horizon`. 
*   **Why it works:** This creates a smooth, linear gradient. Critically, by tracking *actual* cash withdrawals instead of *target* goals, it correctly punishes the agent for going bankrupt early. Combined with exponential Goal Fulfillment Rate (GFR) and Max Drawdown (MDD) penalties, the agent is forced to balance high-growth capital generation with strict risk management.

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

This section documents empirical discoveries made during live RL training that informed architectural and reward-function changes. These are not theoretical trade-offs — they are observed failure modes and their proven fixes.

---

### 12.1 Reward Function Tuning History

| Constant | Original Value | Final Value | Rationale |
| :--- | :--- | :--- | :--- |
| `MDD_PENALTY_SCALE` | 20.0 | 4.0 | At 20.0, even low-MDD (~12%) portfolios generated net-negative rewards, eliminating any positive signal for the agent. |
| `GFR_EXP_SCALE` | 8.0 | 2.0 | The exponential curve was too steep — a 60% GFR would wipe out the entire return component. |
| `GFR_MISS_FLAT_PENALTY` | 0.0 | 10.0 | Added a flat base deterrent so any goal miss carries a fixed cost regardless of how close the agent got. |
| `GFR_PERFECT_REWARD` | 0.0 | +10.0 | Creates a ±20 point swing between perfect and zero GFR, giving the agent a clear signal to chase 100% goal fulfillment. |
| `rl_learning_rate` | 0.01 | 0.001 | High LR caused gradient explosions after early bad samples, collapsing the policy head within the first 50 iterations. |

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

---

### 12.3 Bankruptcy Type Taxonomy

**Problem discovered:** The simulation treated two fundamentally different failure modes as identical — both reported `max_drawdown = 1.0` and `ETV = 0`:

- **Market Collapse:** The portfolio lost its value through market returns alone (e.g., position in a stock that went to zero).
- **Goal Bankrupt:** The market performed fine, but the withdrawal target exceeded the current capital (e.g., a $25k portfolio trying to pay a $30k Year-3 goal).

Setting `max_drawdown = 1.0` for Goal Bankrupt paths was incorrect — the market hadn't collapsed, so the MDD penalty was falsely punishing the agent for what was purely a GFR problem (already captured by the GFR penalty).

**Solution:** Each bankrupt path is now tagged with a `bankrupt_reason` ("goal" or "market"). Only `market` bankruptcies are assigned `max_drawdown = 1.0`. `goal` bankruptcies return the actual market-measured drawdown from `market_multiplier`.

**Logs now display:** `Failures: 5/5 Goal Bankrupt` or `Failures: 3/5 Market Collapse, 2/5 Goal Bankrupt` for full observability.

---

### 12.4 Empty Portfolio Collapse (Policy Pruning Bug)

**Problem discovered:** When the policy's softmax distributes weight across the full ~6,500 asset universe, each individual weight is ~0.015%, well below `MIN_ACTIVE_WEIGHT = 1%`. The `active_mask` was all-False, `full_weights` stayed all-zeros, and the simulator was called with a zero-weight array. Because `np.dot(zeros, returns) = 0`, capital never changed, and the portfolio inevitably went bankrupt at the first withdrawal goal. This manifested as "ALL ASSETS PRUNED (Empty Portfolio)" in the logs but generated a *misleading* negative reward rather than a proper max-penalty signal.

**Solution:** The training loop now detects all-zero weight arrays before calling `_run_single_sim_path` and immediately appends a max-penalty result (`bankrupt=True, MDD=1.0, ETV=0, GFR=0`) without running the simulation. This gives REINFORCE an honest, strong negative gradient specifically tied to the uniform-distribution policy, teaching the agent to concentrate weight.

**Lesson:** Always guard for the degenerate empty-portfolio case *in the training loop*, not just in the evaluator. The training path calling `_run_single_sim_path` directly bypassed the `evaluate_portfolio_member_c` guard that handled this case.

---

### 12.5 Non-IPO Asset Exclusion in the Transformer

**Problem raised:** Should assets that had not yet IPO'd be masked from the Transformer's attention, or is setting their weight to 0 sufficient?

**Finding:** Setting non-IPO weights to 0 alone is insufficient — the Transformer's Self-Attention mechanism still attends to those token positions, allowing them to influence the representation of valid assets. Proper exclusion requires both:
1. A `src_key_padding_mask` passed to the Transformer encoder (marks invalid positions so they are excluded from key/value computation in attention).
2. A `masked_fill(-1e9)` on `raw_action` before softmax (ensures the Gaussian sample for that asset is zeroed out in the final weight vector).

Both are currently implemented in `_rl_worker.py` and confirmed active.

---

### 12.6 Sigma Floor for Exploration Stability

**Problem observed:** With a sufficiently negative reward signal and a high learning rate, the `sigma_head` can be driven to output near-zero variance. Once `sigma → 0`, the Gaussian policy becomes effectively deterministic (always outputs `mu`). If `mu` at that point concentrates on a single asset, the agent locks in a degenerate single-asset portfolio with no way to recover exploration.

**Solution:** The `sigma` output is clamped with a hard floor:
```python
sigma = F.softplus(self.sigma_head(out).squeeze(-1)) + 0.5
```
The `+ 0.5` ensures a minimum exploration noise of 0.5 regardless of what the network learns, preserving the agent's ability to sample diverse portfolios throughout training.

---

### 12.7 Reward Math Log Transparency

All training logs now display the full reward decomposition including raw percentages, making it straightforward to correlate each penalty component with its cause:

```
-> Reward Math: Return (23.54) - MDD (8.2% drop → 3.12) - GFR (0.0% miss → 0.000) + Bonus (10.000) = 30.42
```

The `+ Bonus (10.000)` term only appears when `GFR = 100%`, making perfect-goal-fulfillment runs immediately identifiable in training logs.

---

### 12.8 Training Throughput Optimizations

Three architectural changes, implemented incrementally, combined to increase RL training throughput from ~0.01 it/s (original per-episode loop with full Pandas simulation) to sustained **1+ it/s** — a ~100× improvement. Each optimization is independent and can be toggled via `_constants.py`.

---

#### 12.8.1 Annual Simulation Cache (`build_simulation_cache`)

**File:** `_sim_worker.py`

**Before:** Each simulation path called `_extract_yearly_chunk()` which used Pandas `.iloc` slicing and `.prod()` to compute annual returns on-the-fly for every year × every asset. With 50 paths × 30 years × 6,600 assets, this dominated training time.

**After:** `build_simulation_cache()` runs once at startup and pre-computes a numpy array of annual compound returns for every `(start_index, year, asset)` triple:

```python
one_plus_ret = 1.0 + clean_returns.values   # Convert to numpy once
cache[start_idx] = year_returns              # shape: (30, num_assets)
```

The entire simulation inner loop collapses to:
```python
yr_return = np.dot(weight_array, precomputed_returns[year - 1])
```

**Cache dimensions:** `136 start indices × 30 years × 6,601 assets`

**Impact:** Eliminated all per-step Pandas overhead. Simulation cost per path reduced from ~100ms to <1ms.

---

#### 12.8.2 Random Asset Subset Sampling (`RL_ASSET_SUBSET_SIZE`)

**File:** `_rl_worker.py`, `_constants.py`

**Before:** All ~6,500 assets were fed through the Transformer at every training iteration. Transformer Self-Attention is O(N²), making each forward pass expensive.

**After:** Each iteration randomly samples `RL_ASSET_SUBSET_SIZE = 500` assets. The Transformer processes only this subset. Subset indices are re-drawn every iteration (with `np.sort` to maintain stable ordering), ensuring full universe coverage over time.

```python
if RL_ASSET_SUBSET_SIZE is not None and RL_ASSET_SUBSET_SIZE < len(all_tickers):
    subset_idx = np.random.choice(len(all_tickers), size=RL_ASSET_SUBSET_SIZE, replace=False)
    subset_idx = np.sort(subset_idx)   # keep asset order stable
```

**Impact:**
- Attention compute reduced by ~169× per step (`(6500/500)² ≈ 169`)
- Per-asset gradient signal strengthened ~13× (gradient distributed over 500 vs 6,500 assets)
- Forces generalization: the policy must learn from feature patterns (sector, volatility profile, embedding similarity) rather than memorizing specific ticker positions since the asset set changes every step

**Configuration:** Set `RL_ASSET_SUBSET_SIZE = None` in `_constants.py` to disable and use the full universe.

---

#### 12.8.4 Per-Agent Parallelization Optimization (`joblib`)

**File:** `_rl_worker.py`

**Before:** Simulations were parallelized at the path level ($N$ agents $\times$ $P$ paths = tasks). Because individual simulations are now hyper-fast (<1ms), the IPC (Inter-Process Communication) overhead of serializing the 6,601-element weight vector 30+ times per step was becoming the primary bottleneck.

**After:** Parallel execution is now batched **per agent**. Each of the $N$ agents is a single joblib task. The worker process receives the weight vector once, loops through the $P$ paths internally, and aggregates the results before returning only the final metrics/reward to the main process.

**Impact:** 
- **Drastic Overhead Reduction:** Reduces IPC serialization frequency by $P$ times (e.g., from 30 tasks to 3 tasks).
- **Streamlined Workflow:** Training throughput increased further, enabling larger ensembles without IPC congestion.

---

#### Combined Effect Summary

| Component | Before | After | Speedup |
| :--- | :--- | :--- | :--- |
| Simulation per path | Pandas `.iloc` + `.prod()` (~100ms) | `np.dot` on cached numpy (<1ms) | ~100× |
| Transformer input size | ~6,500 assets (full universe) | 500 assets (random subset) | ~169× (attention) |
| Paths per gradient update | 50 | 10 | ~5× |
| **Ensemble Concurrency** | Sequential (1x) | **Per-Agent Parallel (Nx)** | **~N-core speedup** |
| **Net training throughput** | ~0.01 it/s | **25+ it/s (Estimated)** | **~2500×+** |
