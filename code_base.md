# LOB Simulator — Model Structure Reference

> **Paper**: *Bridging the Reality Gap in Limit Order Book Simulation* (Noble, Rosenbaum, Souilmi, 2026)
> **Stack**: Python 3.11 · NumPy / SciPy (offline) · JAX + Gymnax (online)

---

## Directory layout

```
lob_sim/
│
├── estimate/                  # Offline estimation — pure NumPy/SciPy, runs once
│   ├── preprocess.py
│   ├── mle.py
│   ├── gmm_timing.py
│   └── kernel_nnls.py
│
├── jax_sim/                   # Online simulation — JAX-jitted, GPU/TPU ready
│   ├── types.py
│   ├── params_loader.py
│   ├── book_ops.py
│   ├── kernel.py
│   ├── event_handlers.py
│   ├── sim_loop.py
│   └── gymnax_env.py
│
├── strategies/                # Strategy logic — thin wrappers over gymnax_env
│   ├── hft_imbalance.py
│   └── mft_signal.py
│
├── eval/                      # Validation and sweeps
│   ├── validate_baseline.py
│   └── sweep.py
│
└── data/                      # Artefacts (not source code)
    ├── raw/                   # Databento MBP-10 .dbn files (read-only)
    ├── processed/             # Aggregated event CSVs per ticker
    └── params/                # Estimated .npz tables per ticker
```

---

## `estimate/preprocess.py`

**Purpose**: Convert raw Databento MBP-10 feed into a clean event sequence ready for estimation. This is the only module that touches raw feed files.

### What it must do

**`load_raw_feed(path: str, ticker: str) -> pd.DataFrame`**
- Read a Databento `.dbn` or `.csv` export for one ticker and one date range.
- Keep only these columns: `ts_recv` (int64 nanoseconds), `action` (ADD/CANCEL/TRADE), `side` (BID/ASK), `price` (int64 in price ticks), `size` (int64), and `bid_sz_00..03` / `ask_sz_00..03` (top-4 queue snapshots).
- Discard the first and last 30 minutes of each trading day to exclude open/close artefacts (keep 10:00–15:30 ET).
- Return a dataframe sorted ascending by `ts_recv`.

**`aggregate_trades(df: pd.DataFrame) -> pd.DataFrame`**
- Group consecutive TRADE rows that share the same `ts_recv` value into one row with summed `size`.
- Rationale: a single aggressive order hitting multiple resting orders generates one message per fill; these must be treated as one event for volume estimation.
- Implementation: use `groupby` on a "burst ID" that increments whenever `ts_recv` changes or `action != TRADE`. Take the first row's metadata and sum `size`.

**`aggregate_creates(df: pd.DataFrame) -> pd.DataFrame`**
- Detect "CreateBid" / "CreateAsk" events: an ADD that places a new best bid or ask (i.e. the added price is strictly inside the current spread).
- Aggregate consecutive ADD messages at that same new price into one synthetic CREATE row with combined `size`. Stop aggregation as soon as any other event (different price, cancel, trade) is observed.
- Mark these rows with `action = CREATE_BID` or `CREATE_ASK` and discard the constituent individual ADD rows.
- Rationale: when a new level is created, multiple participants immediately join it; estimating Create volume from the first message alone severely underestimates the true initial queue size.

**`compute_spread_and_imbalance(df: pd.DataFrame) -> pd.DataFrame`**
- After each event, recompute `spread_n` as `(best_ask_price - best_bid_price)` in ticks, clamped to `[1, 5]` (treat 5+ as one bin).
- Compute `imbalance_raw = (bid_sz_00 - ask_sz_00) / (bid_sz_00 + ask_sz_00)`, range `[-1, 1]`.
- Add columns `imb_bin` (int, 0–20 per the 21-bin scheme) and `spread_bin` (int, 0–4).
- Binning rule: 21 bins of width 0.1; bins are right-closed for positive imbalance, left-closed for negative; the exact-zero bin is a point mass. See Section 2.3 of the paper.

**`compute_inter_event_times(df: pd.DataFrame) -> pd.DataFrame`**
- Add column `dt_ns` = `ts_recv.diff()` in nanoseconds (first row gets NaN, drop it).
- Add `log10_dt` = `log10(dt_ns)` for GMM fitting downstream.
- Flag rows where `dt_ns < delta_rtl` as `is_fast = True` (used for race model validation).

**`normalise_volumes(df: pd.DataFrame) -> pd.DataFrame`**
- Compute the Median Event Size (MES) separately for each queue level (q1, q2, q3, q4) and each event type (ADD, CANCEL, TRADE), symmetrised between BID and ASK.
- Add column `vol_norm = size / MES[level, event_type]`, rounded to nearest integer, minimum 1.
- Cap at 50 MES (set `vol_norm = min(vol_norm, 50)`).
- Save MES table to `data/params/{ticker}_mes.npz` for use during simulation (needed when queues shift levels after a price move).

**`save_processed(df: pd.DataFrame, ticker: str)`**
- Write the cleaned, annotated dataframe to `data/processed/{ticker}_events.parquet`.
- Schema: `[ts_recv, action, side, level, price, size, vol_norm, dt_ns, log10_dt, imb_bin, spread_bin, is_fast]`.

---

## `estimate/mle.py`

**Purpose**: Estimate event probabilities, total intensities, and volume distributions from the processed event dataframe. All outputs are NumPy arrays indexed by `(imb_bin, spread_bin)`.

### What it must do

**`estimate_event_probs(df: pd.DataFrame) -> np.ndarray`**
- For each `(imb_bin, spread_bin)` cell, count occurrences of each event type `e` and divide by cell total.
- Event types (10 total): ADD_BID_L1, ADD_BID_L2, ADD_ASK_L1, ADD_ASK_L2, CANCEL_BID_L1, CANCEL_BID_L2, CANCEL_ASK_L1, CANCEL_ASK_L2, TRADE_BID, TRADE_ASK, CREATE_BID, CREATE_ASK.
  - When `spread_bin == 0` (n=1), CREATE events have zero probability by construction; set explicitly to 0.
  - When `spread_bin > 0`, TRADE events may be very rare; allow them but do not artificially zero them.
- Output shape: `(21, 5, 12)` float32. Rows sum to 1.0 along axis 2.
- **Symmetrise**: after estimation, average `p[imb_bin=k, :, BID_event]` with `p[imb_bin=20-k, :, ASK_event]` for all matching bid/ask pairs. This eliminates any residual directional drift in the model.
- Save to `data/params/{ticker}_p_event.npy`.

**`estimate_intensities(df: pd.DataFrame) -> np.ndarray`**
- For each `(imb_bin, spread_bin)` cell, compute `Lambda = 1 / mean(dt_ns)`.
- Output shape: `(21, 5)` float32, units: events per nanosecond.
- Symmetrise the same way as event probs.
- Save to `data/params/{ticker}_lambda.npy`.

**`estimate_volume_cdfs(df: pd.DataFrame) -> np.ndarray`**
- For each `(imb_bin, spread_bin, event_type)` cell, build the empirical CDF of `vol_norm` over the range `[1, 50]`.
- Output shape: `(21, 5, 12, 50)` float32. `cdf[i, j, e, v]` = P(vol ≤ v+1 | phi, event=e).
- For cells with fewer than 30 observations, fall back to the marginal CDF over all imbalance bins for that `(spread_bin, event_type)`.
- Save to `data/params/{ticker}_vol_cdf.npy`.

**`estimate_queue_stationary(df: pd.DataFrame) -> np.ndarray`**
- Estimate the marginal distribution of `vol_norm` at each deep queue level q3, q4 (bid and ask).
- These are sampled during simulation when a price move reveals a previously-unseen deep level.
- Output shape: `(4, 50)` float32 — one CDF per (level, side) pair, symmetrised.
- Save to `data/params/{ticker}_deep_queue_dist.npy`.

**`check_cell_counts(df: pd.DataFrame) -> pd.DataFrame`**
- Return a diagnostic dataframe showing observation count per `(imb_bin, spread_bin)` cell.
- Flag cells below 500 observations as `low_data = True`.
- Print a warning if any cell used for core estimation (n=1, all imb bins) has < 1000 observations.
- This is a data quality gate — if too many cells are sparse, you need more data.

---

## `estimate/gmm_timing.py`

**Purpose**: Fit 5-component Gaussian mixture models on `log10(Δt)` for each `(imb_bin, spread_bin, event_type)` cell. Replace the QR model's exponential clock assumption.

### What it must do

**`fit_gmm_cell(log10_dt: np.ndarray, k: int = 5) -> dict`**
- Fit a k-component GMM to a 1D array of `log10(Δt)` values using `sklearn.mixture.GaussianMixture`.
- Settings: `n_components=k`, `covariance_type='full'`, `n_init=5`, `max_iter=500`.
- Return `{'pi': weights (k,), 'mu': means (k,), 'sigma': std devs (k,)}`.
- For cells with fewer than 200 observations, return the GMM fitted on the pooled `(spread_bin, event_type)` marginal — never fit on fewer than 200 points.

**`fit_all_gmms(df: pd.DataFrame, k: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]`**
- Iterate over all `(imb_bin, spread_bin, event_type)` cells and call `fit_gmm_cell`.
- Output three arrays of shape `(21, 5, 12, k)`: `gmm_pi`, `gmm_mu`, `gmm_sigma`.
- After fitting, verify that the mode of the pooled `log10(Δt)` distribution is near 4.47 (≈29 µs). Warn if the detected mode deviates by more than 0.3 in log10 space — this indicates a data or timestamp issue.
- Save to `data/params/{ticker}_gmm_pi.npy`, `_gmm_mu.npy`, `_gmm_sigma.npy`.

**`select_k_bic(df: pd.DataFrame, k_range: range = range(1, 9)) -> int`**
- Optional diagnostic. For each k, compute total BIC across all cells. Plot ∆BIC(k) − BIC(1).
- Return the k at the elbow. The paper finds k=5; use this to validate your data matches.

**`sample_gmm(pi, mu, sigma, key) -> float`**
- JAX-compatible utility: given GMM parameters (as jnp arrays) and a JAX random key, return one sample in `log10(Δt)` space.
- Implementation: `comp = jax.random.categorical(key, jnp.log(pi))`, then `jax.random.normal` scaled by `mu[comp]`, `sigma[comp]`.
- This function is called inside the JIT-compiled simulation loop, so it must contain no Python control flow.

---

## `estimate/kernel_nnls.py`

**Purpose**: Approximate the power-law decay kernel `G(t) = (1 + t/τ)^{-β}` as a sum of 12 exponentials. Provides O(1)-per-event impact state updates inside JAX.

### What it must do

**`fit_kernel_weights(tau: float = 50.0, beta: float = 1.5, K: int = 12) -> tuple[np.ndarray, np.ndarray]`**
- Construct 12 exponential half-lives `h_i` log-spaced from 0.01 s to 1000 s. Compute decay rates `lambda_i = log(2) / h_i`.
- Build design matrix `X[j, i] = exp(-lambda_i * t_j)` on a log-spaced time grid `t_j` from 0.01 s to 3600 s (100 points).
- Build target vector `y[j] = G(t_j)`.
- Solve `min_{w >= 0} ||Xw - y||^2` using `scipy.optimize.nnls`.
- Return `(weights, lambda_rates)` both of shape `(K,)`.
- Cross-check: print the half-lives with non-zero weights. Should concentrate at ~15 s and ~43 s per Table 1 of the paper.

**`fit_kernel_mle(events_df: pd.DataFrame, K: int = 12) -> tuple[float, float, float]`**
- Optional: fit `(tau, beta, m)` jointly by maximum likelihood directly on observed trade sequences.
- For a grid of `(tau, beta)` values, compute kernel weights via NNLS, update `phi_t` recursively, and evaluate the reduced log-likelihood from Section E of the paper.
- Profile out `m` at each grid point via 1D optimisation (`scipy.optimize.minimize_scalar`).
- Return the `(tau, beta, m)` triple at the minimum negative log-likelihood.
- This is the principled alternative to the default `tau=50, beta=1.5, m=0.036`.

**`calibrate_impact_m(tau, beta, weights, lambdas, target_profile_fn, n_paths: int = 100_000) -> float`**
- Calibrate the impact multiplier `m` by binary search over Monte Carlo metaorder simulations.
- Target profile: TWAP metaorder at 10% of hourly volume, T=10 min execution, observed for 60 min.
- `target_profile_fn(t)` returns the normalised theoretical impact at time `t` (square-root rise then partial reversion).
- Binary search on `m` minimising MSE between simulated average price path and the target.
- Uses the full JAX simulator — import `sim_loop.run_metaorder_mc` here.
- Save `m` to `data/params/{ticker}_impact_m.npy`.

---

## `jax_sim/types.py`

**Purpose**: Define all JAX pytree-compatible dataclasses. Every field must be a jnp array or a Python scalar/enum — no Python lists or dicts inside these classes.

### What it must do

**`LOBState`** — the full mutable simulation state, passed through `lax.scan`:
```
queues:          jnp.ndarray   # (8,) int32  — q[0..3]=ask levels 1-4, q[4..7]=bid levels 1-4
spread:          jnp.ndarray   # ()   int32  — current spread in ticks, range [1,5]
mid_price:       jnp.ndarray   # ()   float32 — mid price in ticks
phi_components:  jnp.ndarray   # (12,) float32 — sum-of-exp impact kernel state
position:        jnp.ndarray   # ()   int32  — strategy net position in shares
cash:            jnp.ndarray   # ()   float32 — cumulative cash flow
pnl:             jnp.ndarray   # ()   float32 — mark-to-market PnL
elapsed_ns:      jnp.ndarray   # ()   int64  — total simulated time in nanoseconds
step:            jnp.ndarray   # ()   int32  — event counter
key:             jnp.ndarray   # (2,) uint32 — JAX PRNG key
```

**`EnvParams`** — frozen parameter tables loaded from `.npz`, never mutated during simulation:
```
p_event:          (21, 5, 12)    float32 — event probabilities per (imb, spread)
lambda_tot:       (21, 5)        float32 — total intensity (events/ns)
gmm_pi:           (21, 5, 12, 5) float32 — GMM weights
gmm_mu:           (21, 5, 12, 5) float32 — GMM means in log10(ns)
gmm_sigma:        (21, 5, 12, 5) float32 — GMM std devs in log10(ns)
vol_cdf:          (21, 5, 12, 50) float32 — volume empirical CDF
deep_queue_cdf:   (4, 50)        float32 — CDF for newly revealed deep levels
kernel_weights:   (12,)          float32 — sum-of-exp weights w_i
kernel_lambdas:   (12,)          float32 — decay rates lambda_i (per ns)
mes_table:        (4, 12)        float32 — MES per (level, event_type) for re-scaling
delta_rtl_ns:     ()             float32 — exchange round-trip latency in ns (~29000)
impact_m:         ()             float32 — impact scaling factor (~0.036)
```

**`StrategyParams`** — hyperparameters swept during optimisation:
```
threshold:    ()  float32  — signal/imbalance threshold for trade trigger
max_inv:      ()  int32    — maximum allowed absolute position
order_size:   ()  int32    — shares per order (capped at best queue depth)
pfill_alpha:  ()  float32  — fill probability slope in race model
```

**`SimConfig`** — static integers defining array sizes, used to set `lax.scan` lengths:
```
n_steps:    int  — number of QR events per trajectory (e.g. 100_000)
n_paths:    int  — number of parallel trajectories for vmap
v_max:      int  — max normalised volume (50)
k_exp:      int  — number of exponential kernel components (12)
```

Define all classes using `chex.dataclass` (preferred) or `flax.struct.dataclass`. Both register as JAX pytrees automatically.

---

## `jax_sim/params_loader.py`

**Purpose**: Load `.npy` files produced by the estimation pipeline and package them into an `EnvParams` instance with correct `jnp` dtypes. The only file that bridges NumPy and JAX.

### What it must do

**`load_params(ticker: str, params_dir: str = "data/params/") -> EnvParams`**
- Load each `.npy` file for the given ticker using `np.load`.
- Cast to the correct dtype (float32 for probabilities and reals, int32 for counts/indices).
- Wrap with `jnp.array(...)`. All downstream JAX code assumes arrays are already on-device.
- Validate shapes match the expected dimensions defined in `types.py`. Raise `ValueError` with a clear message if any shape is wrong.
- Return a fully populated `EnvParams`.

**`validate_params(params: EnvParams) -> None`**
- Check that all `p_event` rows sum to 1.0 ± 1e-5. Any deviation indicates a symmetrisation bug.
- Check that `lambda_tot > 0` everywhere (zero intensity cells would hang the simulation).
- Check that `vol_cdf[:, :, :, -1] == 1.0` (CDFs must reach 1.0 at the cap).
- Check that `kernel_weights.sum() ≈ 1.0` (the kernel should integrate to approximately 1 at t=0).
- Print a summary of cells with `p_event` mass below 1e-4 on any single event type — these are data-sparse cells.

---

## `jax_sim/book_ops.py`

**Purpose**: All operations on the order book state. Pure functions, no side effects, JAX-jittable. This is the most JAX-constraint-sensitive module.

### What it must do

**`project_state(state: LOBState) -> tuple[jnp.ndarray, jnp.ndarray]`**
- Compute imbalance: `imb = (q_bid1 - q_ask1) / (q_bid1 + q_ask1)`.
- Convert to `imb_bin` integer (0–20) using the 21-bin scheme. Use `jnp.digitize`-equivalent logic with no Python conditionals.
- Return `(imb_bin, spread_bin)` both as scalar int32.
- Note: `q_bid1 = state.queues[4]`, `q_ask1 = state.queues[0]` by the index convention in `types.py`.

**`sample_event(key, imb_bin, spread_bin, params: EnvParams) -> jnp.ndarray`**
- Look up `p = params.p_event[imb_bin, spread_bin]` — shape `(12,)`.
- When `spread_bin == 0` (n=1), CREATE events (indices 10, 11) must have zero probability. Apply a static mask rather than a conditional.
- Sample `event = jax.random.categorical(key, jnp.log(p + 1e-12))`.
- Return `event` as int32 scalar.

**`sample_dt(key, imb_bin, spread_bin, event, params: EnvParams) -> jnp.ndarray`**
- Fetch `pi = params.gmm_pi[imb_bin, spread_bin, event]`, `mu`, `sigma` similarly.
- Sample component index: `comp = jax.random.categorical(key_1, jnp.log(pi + 1e-12))`.
- Sample `log10_dt = mu[comp] + sigma[comp] * jax.random.normal(key_2)`.
- Return `dt_ns = jnp.pow(10.0, log10_dt)` as float32.
- No Python control flow. Use `jnp.take` or direct array indexing to extract the chosen component.

**`sample_volume(key, imb_bin, spread_bin, event, params: EnvParams) -> jnp.ndarray`**
- Fetch `cdf = params.vol_cdf[imb_bin, spread_bin, event]` — shape `(50,)`.
- Invert the CDF: draw `u = jax.random.uniform(key)`, return `jnp.searchsorted(cdf, u) + 1` as int32.
- Result is normalised volume in MES units, range `[1, 50]`.

**`apply_add(state: LOBState, side: int, level: int, vol: int) -> LOBState`**
- Add `vol` to the appropriate queue. Side 0 = ask, side 1 = bid; level 0 = best (level 1 in paper notation).
- Queue index: `ask_idx = level`, `bid_idx = level + 4`.
- Use `state.queues.at[idx].add(vol)` (JAX functional update).
- Clamp result to `[1, V_MAX]` to prevent overflow.

**`apply_cancel(state: LOBState, side: int, level: int, vol: int) -> LOBState`**
- Subtract `vol` from the appropriate queue, clamped at 1 (best queues cannot go to zero by cancellation alone — depletion happens only via trades).

**`apply_trade(state: LOBState, side: int, vol: int, params: EnvParams) -> LOBState`**
- Trades hit the best ask (side=0) or best bid (side=1).
- Subtract `vol` from `queues[0]` (or `queues[4]`).
- If the resulting queue volume is ≤ 0, trigger a price move: call `_do_price_move`.
- Update `mid_price` accordingly.
- All branching via `lax.cond(queue_depleted, _do_price_move, _no_price_move, state)`.

**`_do_price_move(state: LOBState, side: int, key, params: EnvParams) -> LOBState`**
- Shift all queues: `new_queues = jnp.roll(queues_on_side, -1)`.
- The newly revealed deep queue (formerly level 4, now level 3) gets sampled: `jnp.searchsorted(params.deep_queue_cdf[side * 2 + level_index], jax.random.uniform(key))`.
- Re-scale volumes when levels shift: multiply by `MES[old_level] / MES[new_level]` using `params.mes_table`. This corrects for MES differing between queue levels.
- Update `spread` to reflect the new best prices.

**`apply_create(state: LOBState, side: int, vol: int) -> LOBState`**
- When `spread > 1`, a CREATE event places a new best bid (or ask) one tick inside the current spread.
- Shift all queues on the relevant side outward by one: `jnp.roll(side_queues, 1)` then set `side_queues[0] = vol`.
- Decrement `spread` by 1, clamped at 1.

---

## `jax_sim/kernel.py`

**Purpose**: Maintain the impact state `phi_t` as a sum of K=12 exponential components. O(K) update per event, fully JAX-compatible.

### What it must do

**`decay_components(phi_components, dt_ns, kernel_lambdas) -> jnp.ndarray`**
- Apply exponential decay to all K components: `phi_components * jnp.exp(-kernel_lambdas * dt_ns)`.
- `dt_ns` is the inter-event time sampled this step. Decay happens *before* the new trade contribution is added.
- Returns updated `phi_components` shape `(K,)`.

**`add_trade_contribution(phi_components, kernel_weights, vol, sign) -> jnp.ndarray`**
- `sign = +1` for ask-side (buy) trades, `-1` for bid-side (sell) trades.
- `phi_components += kernel_weights * sign * jnp.sqrt(vol.astype(float32))`.
- Called only when `event` is a TRADE; otherwise pass through unchanged.
- Use `lax.cond` to conditionally add contribution based on event type — do not use Python `if`.

**`compute_phi_scalar(phi_components, kernel_weights) -> jnp.ndarray`**
- Aggregate to scalar: `phi_t = jnp.dot(kernel_weights, phi_components)`.
- This scalar is used in `event_handlers.py` to bias trade probabilities.

**`bias_trade_probs(p_event, phi_t, impact_m) -> jnp.ndarray`**
- Implements the one-sided biasing from equation (2) of the paper.
- When `phi_t > 0`: multiply bid-trade probabilities by `exp(m * phi_t)`, leave ask unchanged.
- When `phi_t < 0`: multiply ask-trade probabilities by `exp(m * |phi_t|)`, leave bid unchanged.
- Renormalise the full `p_event` vector so it sums to 1.
- All without Python conditionals: use `jnp.where(phi_t > 0, ...)` element-wise.
- This function modifies `p_event` *after* it is looked up from `EnvParams` and *before* categorical sampling in `book_ops.sample_event`.

---

## `jax_sim/event_handlers.py`

**Purpose**: Dispatch from sampled event integer to the correct `book_ops` function. Uses `lax.switch` — the JAX equivalent of a switch/case with no Python branching inside JIT.

### What it must do

**`EVENT_HANDLERS: list[Callable]`**
- A list of 12 functions, one per event type, each with signature `(state, vol, params) -> LOBState`.
- Index 0: ADD_BID_L1 → `book_ops.apply_add(state, side=1, level=0, vol)`
- Index 1: ADD_BID_L2 → `book_ops.apply_add(state, side=1, level=1, vol)`
- Index 2: ADD_ASK_L1 → `book_ops.apply_add(state, side=0, level=0, vol)`
- Index 3: ADD_ASK_L2 → `book_ops.apply_add(state, side=0, level=1, vol)`
- Index 4–7: CANCEL variants, same pattern
- Index 8: TRADE_ASK (buy, aggressor hits ask) → `book_ops.apply_trade(state, side=0, vol)`
- Index 9: TRADE_BID (sell, aggressor hits bid) → `book_ops.apply_trade(state, side=1, vol)`
- Index 10: CREATE_BID → `book_ops.apply_create(state, side=1, vol)`
- Index 11: CREATE_ASK → `book_ops.apply_create(state, side=0, vol)`

**`dispatch(event: jnp.ndarray, state: LOBState, vol: jnp.ndarray, params: EnvParams) -> LOBState`**
- `return jax.lax.switch(event, EVENT_HANDLERS, state, vol, params)`.
- All 12 branches are compiled; the correct one executes based on the runtime value of `event`.
- This is the only correct pattern for dynamic dispatch inside `jax.jit` — never use `if event == X`.

---

## `jax_sim/sim_loop.py`

**Purpose**: Compose all components into the full simulation. The `step` function is the scan body; `rollout` scans over `n_steps`; `run_batch` vmaps over `n_paths`.

### What it must do

**`step(state: LOBState, _, params: EnvParams, strategy_params: StrategyParams) -> tuple[LOBState, Obs]`**
- The `lax.scan` body. Second argument `_` is unused (no per-step input array).
- Sequence:
  1. Split `state.key` into 5 subkeys.
  2. Call `project_state` → `(imb_bin, spread_bin)`.
  3. Compute `phi_t` from `kernel.compute_phi_scalar`.
  4. Call `kernel.bias_trade_probs` to get modified `p_event`.
  5. Call `book_ops.sample_event` → `event`.
  6. Call `book_ops.sample_dt` → `dt_ns`.
  7. Call `book_ops.sample_volume` → `vol`.
  8. Call `kernel.decay_components` with `dt_ns`.
  9. Call `event_handlers.dispatch` → updated `state`.
  10. Call `kernel.add_trade_contribution` if event is a TRADE.
  11. Call `strategy_step` (see `gymnax_env.py`) → updates position/cash/pnl.
  12. Update `elapsed_ns`, `step`, `key`.
- Return `(new_state, obs)` where `obs = compute_obs(new_state)`.

**`compute_obs(state: LOBState) -> Obs`**
- Extract the strategy-visible observation: `(imb_raw, spread, phi_t_scalar, position, pnl)` as a float32 vector of shape `(5,)`.
- This is the `obs` returned to the RL agent at each step.

**`rollout(init_state: LOBState, params: EnvParams, strategy_params: StrategyParams, config: SimConfig) -> tuple[LOBState, jnp.ndarray]`**
- `jax.lax.scan(body, init_state, None, length=config.n_steps)`.
- `body = functools.partial(step, params=params, strategy_params=strategy_params)`.
- Returns `(final_state, obs_trajectory)` where `obs_trajectory` has shape `(n_steps, 5)`.

**`run_batch(keys: jnp.ndarray, params: EnvParams, strategy_params: StrategyParams, config: SimConfig) -> jnp.ndarray`**
- `jax.vmap(lambda key: rollout(reset(key, params, config), params, strategy_params, config))`.
- `keys` shape: `(n_paths, 2)`.
- Returns `obs_trajectories` shape `(n_paths, n_steps, 5)` and `final_pnls` shape `(n_paths,)`.
- Decorate with `@jax.jit`. First call triggers compilation (~30–60 s); subsequent calls are fast.

**`run_metaorder_mc(params, config, n_paths, metaorder_size, metaorder_duration_steps) -> jnp.ndarray`**
- Specialised runner for impact calibration. Injects a TWAP metaorder (fixed child order at every `k`-th step) and records the average price path.
- Used by `kernel_nnls.calibrate_impact_m`.
- Returns average normalised price path shape `(n_steps,)`.

---

## `jax_sim/gymnax_env.py`

**Purpose**: Wrap the simulator as a Gymnax-compatible environment so any Gymnax RL algorithm can optimise strategy parameters.

### What it must do

**`LOBEnv(gymnax.Environment)`**

- **`reset(key, params) -> tuple[Obs, LOBState]`**
  - Initialise `LOBState` with the empirical stationary queue distribution.
  - Sample initial queues from `params.deep_queue_cdf` for all 8 levels.
  - Set `spread = 1`, `position = 0`, `cash = 0`, `phi_components = zeros(12)`.
  - Return `(compute_obs(state), state)`.

- **`step(key, state, action, params) -> tuple[Obs, LOBState, float, bool, dict]`**
  - `action` is int: 0=hold, 1=buy_market, 2=sell_market.
  - Run one QR simulation step (calls `sim_loop.step` internally).
  - Then resolve the strategy action:
    - If `action == 1` and `position < strategy_params.max_inv`: attempt buy. Check race condition via `lax.cond(last_dt < params.delta_rtl_ns, _no_fill, _fill_buy, state)`.
    - Update `position`, `cash` accordingly.
  - `reward = new_pnl - old_pnl` (step PnL).
  - `done = state.step >= config.n_steps`.
  - Return `(obs, new_state, reward, done, {})`.

- **`observation_space(params) -> gymnax.spaces.Box`**
  - Returns `Box(low=-inf, high=inf, shape=(5,), dtype=float32)`.

- **`action_space(params) -> gymnax.spaces.Discrete`**
  - Returns `Discrete(3)` for HFT strategy; `Box` for continuous MFT threshold.

**`strategy_step(state, action, key, params, strategy_params) -> LOBState`**
- Pure function version of the strategy logic for embedding inside `sim_loop.step`.
- Used when running headless Monte Carlo (not RL training).
- Implements the race fill rule: `fill_prob = lax.cond(dt_qr < params.delta_rtl_ns, lambda: strategy_params.pfill_alpha, lambda: 1.0, None)`.

---

## `strategies/hft_imbalance.py`

**Purpose**: The high-frequency imbalance strategy from Section 5.2 of the paper. Reference implementation for validating the race model.

### What it must do

**`HFTPolicy`**
- Fires when `|imbalance| > strategy_params.threshold` (default 0.85).
- Buys at the ask when imbalance is strongly positive (ask side thin); sells at the bid when strongly negative.
- Latency: drawn from `params.delta_rtl_ns` (the empirical exchange round-trip distribution).
- Fill rule: fills if the next QR event `dt_qr > delta`; misses otherwise (simplest version of Section 3.3).
- Respects `max_inventory` and `order_size` caps.

**`run_hft_sweep(params, config, inv_range, size_range, n_paths) -> jnp.ndarray`**
- Sweep over a grid of `(max_inv, order_size)` values using `vmap`.
- Returns `pnl_grid` shape `(len(inv_range), len(size_range))` — the equivalent of Figure 18.
- Run twice: once with `self_impact=True`, once with `self_impact=False` (set `epsilon_k=0` for strategy trades in the kernel update).
- The gap between the two surfaces measures the cost of the strategy's own footprint.

---

## `strategies/mft_signal.py`

**Purpose**: The mid-frequency OU signal strategy from Section 5.1 of the paper.

### What it must do

**`OUSignal`**
- Simulate the OU process `dα = -κ α dt + σ dW` at each QR event step.
- In discrete time: `alpha_new = alpha * exp(-kappa * dt_s) + sigma * sqrt((1 - exp(-2*kappa*dt_s)) / (2*kappa)) * noise`.
- `dt_s = dt_ns * 1e-9` (convert to seconds).
- Store `alpha` in `LOBState` (add the field, or pass as a separate carry in the scan).

**`MFTPolicy`**
- Fires when `|alpha| > strategy_params.threshold`.
- Combined bias term: `b_t = m * phi_t - lambda_signal * alpha_t`. This modifies both the market impact (via `kernel.bias_trade_probs`) and the strategy trigger.
- The signal component tilts trade probabilities in the direction of `alpha`; the impact component opposes it.

**`run_mft_sweep(params, config, threshold_range, max_inv_range, n_paths) -> jnp.ndarray`**
- Equivalent of Figure 17: sweep `(threshold, max_inv)` and return PnL grid.
- Run with and without impact to reproduce the paper's monotone vs. trade-off comparison.

---

## `eval/validate_baseline.py`

**Purpose**: Reproduce the six validation statistics from Figure 9 of the paper. Pass/fail gate before running any strategy analysis.

### What it must do

**`compute_validation_stats(obs_trajectory: jnp.ndarray, final_states: jnp.ndarray) -> dict`**
- From a batch of `n_paths` trajectories:
  1. **Event-type distribution**: fraction of each event type. Target: ~97% ADD+CANCEL, ~2% TRADE.
  2. **Imbalance before trades**: distribution of `imb_bin` at the step immediately before each TRADE event. Should show a U-shape.
  3. **Hourly traded volume**: total TRADE `vol` per simulated hour. Should match empirical mean.
  4. **5-min realised volatility**: `std(last_traded_price_in_bin)` over 5-minute bins. Should match empirical.
  5. **5-min return distribution**: histogram of mid-to-mid returns over 5-min bins.
  6. **5-min returns QQ plot**: quantiles of simulated vs. empirical return distribution.

**`run_validation(ticker: str, params: EnvParams, config: SimConfig, empirical_stats: dict) -> bool`**
- Run `sim_loop.run_batch` for the equivalent of 1000 simulated trading hours.
- Call `compute_validation_stats` on the output.
- Compare each statistic against `empirical_stats` (loaded from the processed data).
- Print a pass/fail report. Return `True` if all six statistics pass tolerance checks.
- Tolerance guidelines: event-type fractions within ±0.5pp; volatility within ±20% of empirical median.

---

## `eval/sweep.py`

**Purpose**: High-throughput parameter sweeps using `vmap` over strategy parameters. The main use case for GPU acceleration.

### What it must do

**`param_sweep(params: EnvParams, config: SimConfig, param_grid: dict, n_paths: int) -> jnp.ndarray`**
- Accept a grid dict like `{'threshold': [0.45, 0.55, ..., 0.80], 'max_inv': [5, 10, ..., 30]}`.
- Construct a `StrategyParams` array for each grid point.
- Use `jax.vmap` over the grid axis (outer) and `lax.scan` over steps (inner).
- Return PnL array shaped `(*grid_shape, n_paths)`.

**`compute_pnl_stats(pnl_array: jnp.ndarray) -> dict`**
- Return mean PnL, std, Sharpe (mean/std), and 5th/95th percentile for each grid point.
- These are the inputs to the parameter-selection plots analogous to Figures 17 and 18.

---

## Key JAX patterns — quick reference

| Problem | Wrong | Right |
|---|---|---|
| Branch on event type | `if event == 0: ...` | `jax.lax.switch(event, handlers, ...)` |
| Branch on queue depletion | `if q <= 0: ...` | `jax.lax.cond(q <= 0, price_move_fn, noop_fn, ...)` |
| Update one queue | `queues[i] = v` | `queues.at[i].set(v)` |
| Dynamic loop length | `while not done:` | `jax.lax.scan(..., length=T)` |
| Random number | `np.random.uniform()` | `jax.random.uniform(key)` then `key, subkey = jax.random.split(key)` |
| Sample from CDF | `np.searchsorted(cdf, u)` | `jnp.searchsorted(cdf, u)` inside jit |
| Parallel trajectories | `for i in range(N): rollout(i)` | `jax.vmap(rollout)(keys)` |
| Conditional assignment | `x = a if cond else b` | `x = jnp.where(cond, a, b)` |

---

## Validation checkpoints

Run these in order. Do not proceed to the next step if a checkpoint fails.

1. `preprocess.py` — verify the 29 µs mode appears in the `log10_dt` histogram of your processed data.
2. `mle.py` — verify `check_cell_counts` shows no cells below 500 obs for n=1.
3. `gmm_timing.py` — verify `select_k_bic` elbow is at k=5.
4. `sim_loop.py` (no strategy) — run `validate_baseline.run_validation`. All 6 stats must pass.
5. `kernel.py` — run a metaorder MC and verify concave impact rise and partial reversion (Figure 15 shape).
6. `strategies/hft_imbalance.py` — verify PnL is higher without self-impact than with (Figure 18 gap is positive).
7. `eval/sweep.py` — verify the `(max_inv, threshold)` PnL surface shows a clear trade-off under impact.