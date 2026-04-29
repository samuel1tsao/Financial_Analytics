# Changelog

All notable changes to the RL portfolio pipeline are documented here.  
Format: `[TYPE]` — one of `FIX`, `FEAT`, `TUNE`, `REFACTOR`, `DOCS`.

---

## [Unreleased] — Active Development

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
