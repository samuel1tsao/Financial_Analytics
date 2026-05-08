# Changelog

## [2026-05-08] - RL Reward Math & Fundamental Volatility Consistency

### FIX — Geometric Path Aggregation (The "MSTX Bias" Cure)
- **Mathematical Correction**: Discovered that aggregating randomized market paths using an **Arithmetic Mean** of terminal multipliers was creating a massive bias toward hyper-volatile "lottery" assets (like MSTX). One extreme +500% outlier path was skewing the average expectation, tricking the RL agent into ignoring the 99% of paths where the asset crashed.
- **Geometric CAGR across Paths**: Switched the simulation aggregator in `_sim_worker.py` to use the **Geometric Mean** across all paths. This correctly captures "volatility drag," ensuring that an asset which spikes and then crashes is correctly penalized.

### FEAT — Fundamental Volatility & Consistency Filter
- **Dynamic Risk Boundary**: Replaced the hard-coded asset exclusion list with a **Fundamental Volatility Filter** in `vector_encoder.py`. The system now dynamically calculates the trailing 30-day volatility of every asset and compares it against a ceiling derived from the user's `volatility_sensitivity` profile.
- **Consistency Ratio**: Implemented a "Structural Stability" check that rejects assets whose recent (30-day) volatility has spiked by more than **50%** relative to their 1-year historical baseline. This automatically weeds out "broken" or erratic assets like MSTX while allowing consistent high-growth stocks like SNDK to pass.
- **Logit Smoothing**: Reduced `target_std` in `sharpen_logits` from 1.0 to 0.5. This prevents the neural network from "locking on" to a single high-logit asset with 80%+ weights, forcing a more balanced and diversified consensus across the ensemble.

## [2026-05-08] - Asset Fundamentals & Sync Stability

### FIX — Missing Asset Fundamentals (MSTX)
- **Broken Import Resolution**: Fixed a critical `ImportError` in `market_routes.py` where the background sync task was attempting to call `sync_historical_deep` (renamed to `sync_historical_financials`). This prevented newly discovered tickers (like MSTX) from being added to the database.
- **Robust Background Sync**: Added explicit `try-except` blocks and logging to `_sync_single_ticker_background` to prevent a single asset's sync failure from crashing the entire background process.
- **MSTX Data Recovery**: Manually triggered a sync for the `MSTX` ticker, restoring access to its fundamentals (Defiance Daily Target 2X Long MSTR ETF).

### FIX — RL Ensemble Loading & Checkpoint Robustness
- **Checkpoint Validation**: Improved the `RLRecommender` singleton to gracefully handle partial ensemble loads. If a specific agent checkpoint (e.g., `agent_2`) is corrupted or 0 bytes, the system now logs a warning and proceeds with the remaining healthy ensemble members instead of crashing the entire recommendation engine.
- **Diagnostic Cleanup**: Removed temporary diagnostic scripts used to trace model loading inconsistencies.

## [2026-05-07] - Learnable Embedding Adapter & Questionnaire UX

### FIX — Questionnaire Validation & State Persistence
- **Strict Step Validation**: Implemented a comprehensive `stepOk` validation engine that prevents users from skipping required questions. Users must now provide starting capital, at least one financial goal, and answer all risk-profile multiple-choice questions before proceeding.
- **Allocation Sanitization**: Added validation to "Specific Stock Preferences" and "Current Portfolio" steps to prevent 0% or negative percentage entries.
- **Row Deletion**: Added removal buttons ("×") to all dynamic row sections (Goals, Stock Preferences, and Current Portfolio), allowing users to easily delete erroneous entries.
- **Answer Persistence**: Expanded the backend schema and frontend pre-fill logic to store and reload raw multiple-choice answers. Users can now return to the questionnaire and see their previously selected responses instead of just the aggregated sensitivity scores.
- **UX Feedback**: Added descriptive error messages for each step to guide users through missing or invalid inputs.

### FEAT — Questionnaire UX Refinement
- **Slider Overhaul**: Updated the Goal Flexibility and Concentration Preference sliders in the questionnaire. Removed the jarring large-text feedback above the slider and replaced it with a sleek value badge and a detailed "Interpretation" card underneath.
- **Dynamic Interpretations**: Added human-readable rationales that explain the financial trade-offs of different sensitivity settings (e.g., alpha focus vs. risk mitigation).
- **Numeric Ticks**: Added 1-10 numeric labels below sliders for improved precision and clarity.

### FEAT — Embedding Adapter Layer in RL Transformer
- **Problem**: The frozen 8-dim embeddings from Member A (the Sequence Transformer) represented only ~3.5% of the RL Transformer's ~229-dim input vector. The remaining ~96.5% was static one-hot categoricals (sector, industry, exchange), meaning the RL model could easily learn to ignore the behavioral embeddings entirely.
- **Solution**: Added a **Learnable Embedding Adapter** — a small 2-layer MLP (`Linear(8→32) → ReLU → Linear(32→32)`) inside `PortfolioTransformerRL` that expands the frozen embeddings to a 32-dim representation before concatenating with user/static features.
- **Gradient Flow**: RL policy gradients now flow through the adapter, allowing the model to learn *which aspects* of the behavioral embedding are useful for portfolio allocation. The frozen embeddings themselves don't change, but their interpretation is trainable.
- **Signal Amplification**: The embedding's share of the input increases from ~3.5% to ~13%, making it significantly harder for the model to ignore.
- **Cost**: Adds only ~1,300 parameters with zero throughput impact.
- **Breaking Change**: The architecture change requires a fresh training run; old checkpoints are incompatible.

### FIX — Backend Dependency & Import Stability
- **Constant Registry Fix**: Resolved critical `ImportError`s caused by missing `RISK_NORMALIZER` and `DIVERSITY_PENALTY_THRESHOLD` constants in `_constants.py`.
- **Import Cleanup**: Removed unused and redundant imports in `vector_encoder.py` to stabilize the API startup process.

### TUNE — Risk Score Aggregation
- **Consolidated Risk Metric**: Implemented an aggregate `risk_score` calculation in the questionnaire encoder that averages drawdown and volatility sensitivities, providing a unified metric for frontend visualizations.

## [2026-05-06] - Reward Stabilization & Math Observability

### FEAT — Rich Reward "Math" Observability
- **Equation-Style Logging**: Overhauled RL training logs to display a complete mathematical breakdown: `Reward = Ret + GFR - MDD - Vol - Div`.
- **Step-by-Step Details**: Added a "Details" line showing exactly how each component was derived (e.g., `Vol = Vol_Metric * Scale * Risk_Multiplier / Desperation`).
- **ASCII Compatibility**: Resolved `UnicodeEncodeError` on Windows terminals by replacing special characters (→) with ASCII equivalents (->).

### TUNE — Volatility Penalty Normalization
- **Penalty Scaling**: Lowered `VOL_PENALTY_SCALE` from **50.0** to **10.0** in `_constants.py`. This prevents moderate volatility from completely erasing high returns in low-risk profiles while maintaining the directional signal.
- **Diversity Thresholding**: Migrated the diversity penalty threshold to a configurable constant `DIVERSITY_PENALTY_THRESHOLD = 10`.

### FIX — Training Loop Crashes
- **Import Integrity**: Fixed a `NameError` in `_rl_worker.py` by ensuring all reward scaling constants are imported from `_constants.py`.
- **Encoding Defense**: Fixed a crash in the logging pipeline caused by unsupported unicode characters in standard Windows CMD/PowerShell environments.

## [2026-05-05] - Geometric CAGR, Desperation Factor & Outlier Clipping

### FIX — Geometric CAGR Reward Math
- **Exponential Reward Suppression**: Transitioned from Simple Average Return to **Geometric CAGR** (`Multiplier ^ (1/Horizon) - 1`) for the return-score calculation. This prevents "lottery ticket" paths (e.g., 1,000,000x multiplier) from generating astronomical, un-penalizable reward scores.
- **Scale Normalization**: Return scores are now properly normalized to a human-readable annualized scale, ensuring that Volatility and MDD penalties remain mathematically significant even for high-growth paths.

### FEAT — The Desperation Factor ("Hail Mary" Logic)
- **Goal-Aware Risk Suppression**: Implemented dynamic suppression of risk penalties (Volatility and MDD) when a user's goal requires aggressive growth.
- **Shortfall Scaling**: If the **Required CAGR** to hit a goal exceeds 10%, the `desperation_factor` progressively weakens risk penalties, allowing the RL agent to prioritize goal fulfillment over smoothness in "impossible" funding scenarios.
- **Harsher Failure Penalties**: Increased the base penalty for 0% Goal Fulfillment Rate (GFR) from `-30` to `-100`, forcing the agent to take the necessary risk to hit the target.

### FEAT — Upside Winsorization (Outlier Clipping)
- **Outlier Management**: Implemented a hard **+500% (6.0x)** ceiling on 1-year annual returns in the simulation cache (`build_simulation_cache`).
- **Anomalous Spike Defense**: This prevents the "Bootstrapping Illusion" where a single historical data error or penny-stock pump-and-dump is sampled multiple times, creating unrealistic expectations. Consistent "bursters" (e.g., +100% multiple times) are favored over one-off +10,000% anomalies.

### TUNE — Strict Recency Bias (3-Year Half-Life)
- **Aggressive Time Decay**: Shortened the exponential sampling half-life from **10.0 years** to **3.0 years**. 
- **Regime Integrity**: Ensures the simulator's expectations are strictly grounded in modern market behavior. Spikes from 10 years ago are now effectively invisible to the training environment, preventing reliance on obsolete market dynamics.

## [2026-05-04] - Block Bootstrapping, Volatility Penalties & Allocation Rationales

### FEAT — Block Bootstrapping with Recency Bias
- **Exponential Decay Probabilities**: Implemented a decay-weighted sampling mechanism (`calculate_decay_probabilities`) for Monte Carlo paths. The simulation now prioritizes recent market regimes (10-year half-life) while still sampling historical stress periods.
- **Bootstrapping Engine**: Replaced chronological-only paths with **Block Bootstrapping**. The simulator now constructs 30-year synthetic trajectories by randomly sampling 1-year historical blocks from the entire 20-year cache, providing much higher regime diversity.
- **2D Simulation Cache**: Optimized `build_simulation_cache` to store returns in 1-year atomic blocks (`num_starts × num_assets`), enabling flexible bootstrapping and reducing memory overhead.

### FEAT — Consistency-Aware Reward (Volatility Penalty)
- **Volatility Penalty (`VOL_PENALTY_SCALE = 50.0`)**: Introduced a new component to the RL reward function that penalizes high annual volatility. 
- **Decisiveness vs. Spikiness**: Complements the Diversification Penalty by ensuring the agent doesn't just pick "spiky" one-hit-wonder assets to maximize CAGR, but instead favors consistent, high-Sharpe growth.

### FEAT — Allocation Rationale Engine
- **Explainable AI (XAI)**: Implemented an automated rationale generator in the inference pipeline. Every recommended asset now includes a human-readable explanation (e.g., "High-conviction primary holding selected for aggressive capital appreciation").
- **Volatility-Aware Reasoning**: Rationales are dynamically generated based on the asset's historical volatility and its assigned weight, providing transparency into the RL agent's "thinking."

### REFACTOR — Dynamic Top-K & Reward Re-Balancing
- **Dynamic Thresholding**: Transitioned from a fixed `TOP_K_ASSETS=10` to a relative threshold (`RELATIVE_WEIGHT_THRESHOLD=0.1`). The portfolio size now dynamically adjusts (up to 20 assets) based on the model's confidence.
- **Reward Scale Reset**: Re-balanced the reward function to prioritize returns (`CAGR_REWARD_SCALE=200.0`) and goal success (+20 bonus) over pure risk avoidance, preventing the "Safe Asset Trap."


## [2026-05-03] - High-Fidelity Monte Carlo & True Wealth Percentiles
### FEAT — High-Fidelity Monte Carlo Simulation
- **1,000 Simulation Paths**: Upgraded from 20 paths to 1,000 paths per portfolio, providing a much more stable and representative statistical sample of historical market behavior.
- **Daily Historical Sampling**: Shifted from monthly start-date sampling to daily sampling since 2006. The simulation now draws from a pool of ~5,100 unique historical scenarios, capturing more granular market regimes.

### REFACTOR — Ticker-Aware "Just-In-Time" Simulation
- **Latency Optimization**: Eliminated the 100+ second "cold start" delay by refactoring the simulation to only process tickers active in the user's specific portfolio or comparison set.
- **Memory Efficiency**: Reduced the simulation memory footprint from 4GB (full market) to <10MB (portfolio subset), enabling instant response times (<1.5s) on standard hardware.

### FEAT — True Wealth Percentiles (River Plot)
- **Statistical Correction**: Replaced synthetic "Standard Deviation of Returns" with **True Wealth Percentiles**. This fixes the visual "collapse" of lower bounds caused by compounding independent extreme annual events.
- **10th - 90th Probability Fan**: Implemented a "Fan Chart" (River Plot) visualization displaying 8 distinct probability zones (10%, 20% ... 90%).
- **Median-Based Projections**: Switched the primary "Expected Path" from the mathematical mean to the 50th percentile (Median), providing a more robust representative outcome.
- **Client-Side Simulation**: Offloaded cashflow compounding to the frontend (running all 1,000 paths in JS) to maintain instant slider interactivity while using high-fidelity raw return data.


## [2026-05-01] - Parallel RL Ensemble Training
### Added
- **Process-Based Parallelism**: Migrated the RL ensemble training from threads to `multiprocessing.Process` to achieve true hardware parallelism and bypass GIL limitations.
- **Per-Agent Logging**: Implemented isolated logging where each ensemble member writes to its own `logs/agent_N.log` file, enabling real-time monitoring of individual agent convergence.
- **File-Based Data Sharing**: Optimized inter-process communication by using a temporary pickle file for large matrices, preventing "Insufficient system resources" errors on Windows.

### Fixed
- Resolved performance bottlenecks in ensemble startup by pre-computing simulation caches and feature matrices once in the orchestrator.
- Improved stability of ensemble reassembly by using a robust checkpoint-loading mechanism after parallel training completion.

All notable changes to the RL portfolio pipeline are documented here.  
Format: `[TYPE]` — one of `FIX`, `FEAT`, `TUNE`, `REFACTOR`, `DOCS`.

---

## 2026-05-01

### REFACTOR — Complete Per-Agent Parallelization
**File:** `_rl_worker.py`  
**Motivation:** Even with batched simulations, running the forward and backward passes sequentially for all agents in the ensemble was bottlenecking throughput. Total synchronization at every step prevented full utilization of multi-core hardware for the neural network portion of the pipeline.
**Change:** 
- **Top-Level Parallelization:** Refactored `train_rl_agent` to spawn $N$ completely independent training processes (one per agent) using `joblib` with the `loky` backend.
- **Process Isolation:** Each agent now runs its own full training loop, computes its own gradients, and manages its own independent checkpoint file (`checkpoint_..._agent_n.pt`).
- **Resource Allocation:** Divided total available CPU cores (`rl_parallel_cores`) among the agents, ensuring each sub-process has its own dedicated simulation workers without oversubscribing the system.
- **Async Consolidation:** The orchestrator waits for all agents to complete their full episode counts before reassembling them for final ensemble evaluation.
**Impact:** 
- **Training Throughput:** Realized a ~3x speedup for a 3-agent ensemble by fully parallelizing the backprop and gradient update cycles.
- **Reliability:** Eliminated per-step synchronization bottlenecks and simplified the inner training loop.

---

## 2026-04-30

### FEAT — Multi-Agent Ensemble RL Training
**File:** `_rl_worker.py`, `_constants.py`  
**Motivation:** Single-agent RL is prone to local optima and high variance. Training an ensemble of agents improves stability, as the final recommendation is the averaged consensus of multiple independent learners.  
**Change:** 
- **Ensemble Initialization:** `train_rl_agent` now initializes $N$ independent `PortfolioTransformerRL` agents (controlled by `rl_ensemble_size`).
- **Concurrent Training:** All $N$ agents are trained in parallel within each iteration.
- **Batched Checkpointing:** The checkpoint file now stores a list of states for all agents and their respective optimizers, allowing seamless resume of the full ensemble.
- **Consensus Evaluation:** `_greedy_evaluate_fast` now averages the policy outputs of all ensemble members to produce a robust final allocation.

---

### REFACTOR — Parallelized Simulation Execution (Joblib)
**File:** `_rl_worker.py`  
**Motivation:** CPU-bound simulations for an ensemble of $N$ agents each running $P$ paths became a major bottleneck when executed sequentially ($N \times P$ simulations).  
**Change:** Refactored the training loop to batch all $N \times P$ simulations into a single task list, executed concurrently using `joblib.Parallel`.  
**Impact:** 
- **Hardware Utilization:** Saturates all available CPU cores (via `rl_parallel_cores`) during the simulation phase.
- **Throughput Boost:** Training speed for a 3-agent ensemble is now nearly identical to a single-agent run on multi-core systems.

---

### FEAT — Robust Per-Agent Observability
**File:** `_rl_worker.py`  
**Change:** Overhauled training logs to display detailed **Allocation**, **Simulation Metrics**, and **Reward Math** for every agent in the ensemble independently.  
**Impact:** Enables visual verification of agent diversity and progress, allowing the user to see if individual ensemble members are exploring different regions of the asset universe.

---

## 2026-04-28
**File:** `_rl_worker.py`, `_sim_worker.py`, `_constants.py`  
**Motivation:** The previous single-head model was forced to find a single static portfolio that satisfied both pre-goal safety and post-goal growth. This often resulted in a "Safe Asset Trap" where the model stayed in conservative assets even after goals were met, missing out on long-term compounding.  
**Change:** 
- **Dual Policy Heads:** The Transformer now outputs two independent Gaussian distributions: `pre-goal` (funding focus) and `post-goal` (growth focus).
- **Simplified State Space:** User condition vector reduced from 33-dim to 4-dim: `[risk, capital, goal_year_normalized, goal_ratio]`.
- **Single-Goal Episodes:** Multi-goal profiles are decomposed into single-goal training episodes via `decompose_profiles()`, providing a cleaner gradient signal for funding-specific strategies.
- **Switching Logic:** The simulator now supports weight switching at the `goal_year` boundary.
- **Terminal Horizon:** All episodes now run for a fixed 30-year horizon to reward post-goal capital growth.
**Impact:** Enables "Waterfall Allocation" where strategies shift automatically from target-funding to growth-maximizing as goal milestones are passed.

---

## 2026-04-28

### FEAT — Random Asset Subset Sampling Per Training Iteration
**File:** `_rl_worker.py`, `_constants.py`  
**Motivation:** Feeding all ~6,500 assets through the Transformer at every training step was prohibitively expensive. Transformer Self-Attention is O(N²), so the full universe created a massive computational bottleneck. Additionally, presenting every asset every step encouraged the policy to memorize ticker-specific patterns rather than learning general feature-based allocation strategies.  
**Change:** Each training iteration now randomly samples `RL_ASSET_SUBSET_SIZE = 500` assets from the full universe. The Transformer processes only this subset, and the policy head outputs weights over the subset. Subset indices are re-drawn every iteration, ensuring full universe coverage over time.  
**Impact:**
- ~169× reduction in attention compute per step (`(6500/500)² ≈ 169`)
- ~13× stronger per-asset gradient signal (gradient distributed over 500 vs 6,500 assets)
- Forces the policy to learn from feature patterns (sector, volatility profile, embedding similarity) rather than memorizing specific ticker positions
- Combined with the Annual Simulation Cache, enables sustained 1+ it/s training throughput

---

### FEAT — Stochastic Per-Step Gradient Updates with Reduced Path Count
**File:** `_rl_worker.py`, `_constants.py`  
**Motivation:** The original training loop ran `sim_paths_per_episode = 50` simulation paths per gradient update, making each iteration expensive. For stochastic REINFORCE, frequent noisy updates converge faster than infrequent precise updates.  
**Change:** Introduced `rl_paths_per_step = 5` — each gradient update uses only 5 randomly sampled simulation start dates instead of 50. The per-step reward is noisier, but the EMA baseline (`0.99 * baseline + 0.01 * reward`) absorbs the variance. The net effect is ~10× more gradient updates per wall-clock second.  
**Trade-off:** Higher per-step variance in reward estimates, mitigated by the EMA baseline variance reduction. Empirically, convergence speed improved significantly despite the noisier signal.

---

### FIX — Empty Portfolio Collapse in Training Loop
**File:** `_rl_worker.py`  
**Root cause:** The training loop called `_run_single_sim_path` directly, bypassing the `evaluate_portfolio_member_c` guard that detects all-zero weight arrays. When the policy spread weight uniformly across ~6,500 assets via softmax, every individual weight (~0.015%) fell below `MIN_ACTIVE_WEIGHT = 1%`, producing an all-zero `full_weights` array. The simulator then computed `yr_return = dot(zeros, returns) = 0` every year, so capital never grew and inevitably went bankrupt at the first goal withdrawal. The resulting negative reward was misleading — REINFORCE couldn't distinguish "bad market" from "no portfolio."  
**Fix:** Detect `active_mask.any() == False` in the training loop before calling the simulator. Short-circuit immediately with a max-penalty result (`bankrupt=True, MDD=1.0, ETV=0`), giving REINFORCE a correct, strong signal to push away from uniform-distribution policies.

---

### FIX — Goal Bankrupt Paths Incorrectly Assigned MDD=100%
**File:** `_sim_worker.py`  
**Root cause:** All bankrupt paths hardcoded `max_drawdown = 1.0` in the return dict, regardless of *why* they went bankrupt. A portfolio that went bankrupt solely because the withdrawal target exceeded its capital (e.g., $25k portfolio meeting a $30k Year-3 goal) was penalized identically to one where the market wiped out the entire investment. This double-penalized goal-miss events (already captured by GFR penalty) with an additional 100% MDD penalty.  
**Fix:** Each bankrupt path is now tagged with `bankrupt_reason`:
- `"goal"` → portfolio had positive capital before withdrawal; return the actual `max_drawdown` from `market_multiplier`.
- `"market"` → portfolio capital was already ≤ 0 before withdrawal; keep `max_drawdown = 1.0`.

---

### FEAT — Bankruptcy Type Taxonomy in Logs
**File:** `_rl_worker.py`, `_sim_worker.py`  
Added per-path tracking of failure mode and aggregation across all simulation paths. The `Simulation Metrics` log line now shows:
```
-> Simulation Metrics: ETV=$0 | MDD=8.2% | GFR=0.0% | Failures: 5/5 Goal Bankrupt
```
Possible failure labels: `Goal Bankrupt`, `Market Collapse`. Both counts and totals are shown.

---

### FIX — MDD Inflated by Goal Withdrawals
**File:** `_sim_worker.py`  
**Root cause:** The original drawdown tracker used the running peak of the *cash balance*. When a goal withdrawal reduced the capital (e.g., -$200k paid to a retirement goal), the denominator for future drawdown calculations shrank, making the same subsequent market drop appear proportionally larger. A portfolio that met all its goals was paradoxically penalized harder on MDD than one that never paid anything.  
**Fix:** Introduced a separate `market_multiplier` (starts at 1.0, compounded only by `yr_return`). Max drawdown is now measured exclusively on this multiplier. Cash withdrawals have zero impact on MDD.

---

### FEAT — GFR Perfect Reward Now Tracked & Displayed
**File:** `_sim_worker.py`, `_rl_worker.py`  
The `GFR_PERFECT_REWARD` (+10.0 bonus for 100% GFR) was being applied to `total_reward` but not stored in `reward_components`, causing the log to show `-0.000` GFR penalty with no indication of the bonus. Added `gfr_bonus` field to `reward_components`. The Reward Math log now conditionally shows `+ Bonus (10.000)` when the bonus is awarded:
```
-> Reward Math: Return (23.54) - MDD (8.2% drop → 3.12) - GFR (0.0% miss → 0.000) + Bonus (10.000) = 30.42
```

---

### FEAT — Reward Math Log Shows Raw Percentages
**File:** `_rl_worker.py`  
Enhanced the `Reward Math` log line to display the raw MDD drop percentage and GFR miss percentage alongside their computed penalty values, making it straightforward to understand why a penalty is large or small:
```
-> Reward Math: Return (X) - MDD (18.1% drop → 6.52) - GFR (40.0% miss → 9.83) = Y
```

---

## 2026-04-27

### TUNE — Reward Function Constants
**File:** `_constants.py`

| Constant | Before | After | Reason |
| :--- | :--- | :--- | :--- |
| `MDD_PENALTY_SCALE` | 20.0 | 4.0 | Portfolios with 12% MDD were generating net-negative rewards, removing all positive signal. |
| `GFR_EXP_SCALE` | 8.0 | 2.0 | Exponential curve too steep; 60% GFR erased the full return score. |
| `GFR_MISS_FLAT_PENALTY` | N/A | 10.0 | Added flat deterrent — any goal miss now carries a fixed base cost. |
| `GFR_PERFECT_REWARD` | N/A | +10.0 | Creates a ±20-point swing between 100% and 0% GFR to motivate goal completion. |
| `rl_learning_rate` | 0.01 | 0.001 | High LR caused gradient explosions; policy head collapsed within first 50 iterations. |

---

### FIX — Sigma Floor for Policy Collapse Prevention
**File:** `_rl_worker.py`  
**Root cause:** High negative rewards combined with a high learning rate caused the `sigma_head` to learn near-zero variance. Once `sigma → 0`, the Gaussian policy became deterministic, concentrating entirely on whichever single asset `mu` pointed to. The policy then had no exploration capacity to recover.  
**Fix:** Added a hard minimum floor to sigma output:
```python
sigma = F.softplus(self.sigma_head(out).squeeze(-1)) + 0.5
```
Guarantees minimum exploration noise of 0.5 throughout training.

---

### FEAT — Non-IPO Asset Masking Confirmed
**File:** `_rl_worker.py`  
Confirmed that pre-IPO assets are properly excluded via:
1. `src_key_padding_mask` passed to the Transformer encoder — invalid positions excluded from attention key/value computation.
2. `raw_action.masked_fill(mask, -1e9)` before softmax — ensures zero weight assigned to pre-IPO assets.

Setting weight to 0 after softmax would be insufficient — the Transformer's self-attention would still contaminate valid asset representations by attending to pre-IPO token positions.

---

### FEAT — Flat GFR Miss Penalty & Perfect GFR Bonus
**File:** `_sim_worker.py`, `_constants.py`  
Added `GFR_MISS_FLAT_PENALTY` (applied whenever any goal is missed) and `GFR_PERFECT_REWARD` (bonus when all goals are met). This creates a sharper incentive gradient around the 100% GFR threshold, encouraging the agent to push through near-misses rather than settling for partial fulfillment.

---

## Earlier Sessions (Pre-2026-04-27)

### REFACTOR — Switched from CAGR to Annualized Simple Return
**File:** `_sim_worker.py`  
CAGR was mathematically unstable under bankruptcy and heavy withdrawal scenarios (division-by-zero, exponential spikes). Replaced with `Annual Return = (ETV + Actual Withdrawals - Start Capital) / Start Capital / Horizon`. Provides a smooth, linear gradient and correctly handles goal-payment cash flows.

### FEAT — Moving Average Baseline for REINFORCE
**File:** `_rl_worker.py`  
Added an Exponential Moving Average (EMA) baseline to center gradient updates: `loss = -(reward - baseline) * log_prob`. Reduces variance in the policy gradient, stabilizing training especially during early high-variance exploration.

### REFACTOR — Autoencoder → Transformer Sequence Embeddings
**File:** `_data_worker.py`, `_rl_worker.py`  
Replaced the original feedforward autoencoder with a sequence-based Transformer architecture processing a 5-year sliding context window (1,260 daily tokens). Transformer embeddings are cached to disk to eliminate inference bottleneck in the RL training loop.

### FEAT — Annual Simulation Cache (Major Throughput Optimization)
**File:** `_sim_worker.py`  
**Motivation:** Each simulation path previously sliced into `portfolio_returns` using Pandas indexing and called `.prod()` per year per path — extremely slow when running hundreds of paths per episode across 6,600+ assets.  
**Change:** `build_simulation_cache()` now pre-computes a numpy array of annual compound returns for every `(start_index, year, asset)` triple at startup: `136 paths × 30 years × 6,601 assets`. The entire simulation inner loop is replaced by a single `np.dot(weight_array, precomputed_returns[year])` call per year.  
**Impact:** Eliminated all per-step Pandas overhead. Combined with asset subset sampling and reduced path count, this enables sustained 1+ it/s training throughput — up from ~0.01 it/s in the original per-episode loop with full Pandas simulation.
