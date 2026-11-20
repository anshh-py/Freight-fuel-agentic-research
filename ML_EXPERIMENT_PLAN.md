# M1 — ML Baselines & Cross-Truck Fuel-Rate Prediction

**Experiment plan / design document (pre-implementation).**
Milestone: **M1**. No models are trained in this document; this is the design
contract. Per project rules, the dataset is formula-generated synthetic
telemetry: **metrics here do not establish real-world predictive accuracy.**

---

## 0. Scope and purpose

- Build reproducible baselines that predict instantaneous `fuel_rate_lph` (L/h).
- Evaluate **cross-truck generalization**: train on 20 trucks, tune on 4
  validation trucks, test on 5 **unseen** trucks.
- Establish the evaluation protocol, feature contract, metrics and risks before
  any training code is written.

**Deliverables in this milestone (files):**
- `ML_EXPERIMENT_PLAN.md` (this file)
- `configs/ml_synthetic_feature_spec.json` (feature allowlist/denylist contract)

---

## 1. Data and split contract

| Item | Value |
| --- | --- |
| Telemetry | `data/synthetic_heavy_truck_research/telemetry_master_963697.csv.gz` (963,697 rows × 24 cols) |
| Truck metadata | `truck_metadata.csv` (29 trucks) |
| Split manifest | `truckwise_split_manifest.csv` (20 / 4 / 5) |
| Sampling | 1 Hz; 540 trips; classes 32T/40T/49T |

**Split rule (must be honored exactly):**
1. Read the split manifest; a truck belongs to exactly one split.
2. Load telemetry and join `truck_id` → split label.
3. Train = rows of the 20 train trucks; validation = 4 val trucks; test = 5 test trucks.
4. **Never** shuffle rows across trucks or reuse a truck in another split.
5. The test set is touched only once, for final evaluation.

Verification against the manifest:
- train trucks (20): `truck_010,007,006,008,002,009,013,018,016,012,021,015,011,019,029,026,024,025,023,022`
- validation trucks (4): `truck_005,003,014,027`
- test trucks (5): `truck_004,001,020,017,028`

All three classes (32T/40T/49T) are present in the test set.

---

## 2. Feature contract (allowlist / denylist)

Full machine-readable spec: `configs/ml_synthetic_feature_spec.json`.

### 2.1 Denylist (never inputs)

| Column | Reason |
| --- | --- |
| `record_id` | Row identifier (identity leakage) |
| `timestamp_utc` | Time/order leakage; not a causal input |
| `truck_id` | Leaks truck identity; defeats cross-truck test |
| `trip_id` | Leaks trip identity; pseudo-random ID |
| `latitude`, `longitude` | Synthetic route/trip coordinates — route identity leakage |
| `fuel_consumed_l_increment` | Target-derived (`= fuel_rate_lph / 3600`) |
| `cumulative_fuel_l_trip` | Target-derived (cumulative sum of target) |
| `synthetic_efficiency_factor` | Per-truck fuel multiplier (proxy of target) → leaks target structure |
| any direct transform of the target | e.g. `log(fuel_rate)` reused as input |
| future rows | any value at a later timestamp |

### 2.2 Allowlist

Two feature sets are defined (both are subsets of the 24 columns).

**A. `full_operational` (primary)**
```
vehicle_class, gross_vehicle_weight_kg, payload_kg, vehicle_mass_kg,
speed_kph, acceleration_mps2, road_grade_pct, ignition_status,
engine_rpm, engine_load_pct, throttle_position_pct, torque_nm,
coolant_temp_c, intake_air_temp_c, ambient_temp_c
```
- Static truck attributes: `vehicle_class`, `gross_vehicle_weight_kg`.
- Trip load: `payload_kg`, `vehicle_mass_kg`.
- Motion state: `speed_kph`, `acceleration_mps2`, `road_grade_pct`.
- Engine state: `engine_rpm`, `engine_load_pct`, `throttle_position_pct`, `torque_nm`.
- Thermal: `coolant_temp_c`, `intake_air_temp_c`, `ambient_temp_c`.
- Binary: `ignition_status`.

**B. `telematics_only` (ablation — driver-feedback explainability)**
```
vehicle_class, gross_vehicle_weight_kg, payload_kg, vehicle_mass_kg,
speed_kph, acceleration_mps2, road_grade_pct, ignition_status,
coolant_temp_c, intake_air_temp_c, ambient_temp_c
```
- Excludes engine-internal signals (`engine_rpm`, `engine_load_pct`,
  `throttle_position_pct`, `torque_nm`) to study how much fuel can be inferred
  from external/operational telematics alone.

### 2.3 Why these are legitimate inputs
- `speed_kph`, `acceleration_mps2`, `road_grade_pct`: physical operating
  conditions the driver/EMS act on and the fuel model depends on.
- `payload_kg` / `vehicle_mass_kg` / `gross_vehicle_weight_kg`: total weight
  drives tractive demand; stable inputs.
- `engine_rpm`, `engine_load_pct`, `throttle_position_pct`, `torque_nm`:
  engine operating point. These are **not** arithmetic transforms of fuel rate;
  they are independent physical signals. We report a full set and a
  telematics-only ablation so the contribution of engine-internal telemetry is
  transparent rather than hidden.
- `coolant_temp_c`, `intake_air_temp_c`, `ambient_temp_c`: thermal state, a
  legitimate exogenous input.
- `ignition_status`: distinguishes engine-off (fuel ≡ 0) from engine-on rows.

### 2.4 Encoding
- `vehicle_class` is low-cardinality categorical → one-hot (drop first) or
  ordinal `[32T=0,40T=1,49T=2]`.
- All other features are continuous/numeric (no imputation needed; the dataset
  is complete).

---

## 3. Models

| Baseline | Purpose | Notes |
| --- | --- | --- |
| Mean / median | Floor, interpretability | Predict constant (global; also report per-class) |
| Ridge regression | Linear reference | Standardized features; L2 penalty |
| Random Forest | Non-linear reference | n_estimators ~ 200; samples for 963K rows |
| XGBoost | Strong gradient-boosting baseline | `xgboost` 3.4.1 is installed; `hist` tree method |

Rules:
- All models use the **same** feature set (full_operational; telematics-only as
  an ablation run).
- Random state fixed (`random_state=20240617`) for reproducibility.
- Training on 963K rows is feasible; for very large row counts use a
  deterministic subsample for exploration during model selection.

---

## 4. Evaluation protocol

### 4.1 Metrics (primary, per model)
- **MAE** (L/h), **RMSE** (L/h), **R²**.
- **MAPE** on `fuel_rate > 1 L/h` records only (engine-off rows are excluded
  from MAPE; see 5.1). Report the number of rows used and the coverage.

### 4.2 Reported breakdowns
- **Per-truck metrics** on validation and test (computed per truck, then
  aggregated across trucks).
- **Per-class (32T / 40T / 49T) metrics**.
- **Residual analysis**: residuals vs. truth, vs. speed, vs. grade, vs. truck;
  per-class bias (mean residual).

### 4.3 Learning protocol
1. Train on the 20 train trucks only.
2. Tune hyperparameters on the 4 validation trucks (never test).
3. Final-selected model evaluated once on the 5 test trucks.
4. Test trucks are never used for fitting or tuning.

### 4.4 Reproducibility
- Fix `random_state=20240617` across models and sampling.
- Store per-run config (feature set, model class, hyperparameters, split sha).
- Record environment versions (numpy/pandas/sklearn/xgboost).

### 4.5 Reporting
- Table of validation vs. test metrics for each model/feature set.
- Per-truck and per-class tables.
- Residual diagnostic (residual vs. predicted and vs. grade).

---

## 5. Analysis notes

### 5.1 Target distribution and metric pitfalls
- `fuel_rate_lph` is ~0 when engine-off and ~1–3 when idle: there is a large
  mass of near-zero values. MAPE is undefined/inflated there, so MAPE uses
  `fuel_rate > 1` (documented, with coverage).
- Fuel rate is long-tailed (heavy-load climbs), so RMSE is sensitive to the tail.

### 5.2 Interpretability / explainability (later milestone)
- SHAP uses the same feature contract. Explainability should reference the
  allowlist (engine-internal vs. telematics), and should never be computed over
  the denylist.

### 5.3 Cross-truck caution
- The model is expected to perform well on training trucks (they drive the fit)
  and less well on unseen test trucks. The **gap** between train and test
  performance is the key generalization signal; we report it explicitly.

---

## 6. Dependencies

| Package | Version | Use |
| --- | --- | --- |
| Python | 3.13 | runtime |
| numpy | 2.3.2 | numeric |
| pandas | 3.1.0 | data loading |
| scipy | 1.18.1 | stats helpers |
| scikit-learn | 1.9.1 | Ridge, RF, metrics, splits |
| xgboost | 3.4.1 | XGBoost |

No training is run in this milestone; the list above is the planned stack.

---

## 7. Implementation steps (next)

1. Implement `src/data/load_synthetic.py` → returns X, y, meta for the selected
   split (telemetry + truck metadata + manifest).
2. Implement `src/features/build_features.py` → encode categoricals, select the
   allowlist set, drop denylist columns, return feature names.
3. Implement `src/models/baselines.py` → mean/median, Ridge, RF, XGBoost with
   fixed seeds and a common `fit/predict` interface.
4. Implement `src/evaluation/metrics.py` → MAE/RMSE/R²/MAPE (with `>1` filter).
5. Implement `src/evaluation/evaluate.py` → per-truck, per-class, residual
   summaries; one report writer.
6. Wire a single reproducible script (e.g. `scripts/run_m1_baselines.py`) that:
   - loads telemetry once (memory-efficient),
   - builds each feature set,
   - runs each model on train → tune on val → eval on test,
   - writes a JSON report + Markdown summary under `reports/m1/`.
7. `git` the feature spec and plan; keep the dataset artifacts untouched.

---

## 8. Risks / open items (before implementation)

1. **Synthetic determinism**: The target is formula-generated, so an ML model
   may fit near-perfectly (high R²). This is expected for the synthetic suite;
   we must report it as engineering validation, not real-world accuracy.
2. **Engine-internal features**: Including `engine_rpm`/`torque_nm`/
   `engine_load_pct` yields very high R² because they are direct physical
   drivers. We report both full and telematics-only sets to avoid overclaiming.
3. **MAPE at near-zero fuel** is unstable — mitigated by the documented
   `fuel_rate > 1` filter.
4. **Truck-level dependence**: Rows within a truck/trip are autocorrelated; we
   report per-truck metrics (not row-level only) to avoid overstating
   significance.
5. **Test class balance**: test has 5 trucks (2×32T, 2×40T, 1×49T); per-class
   test estimates are small-sample and should be read accordingly.
6. **Future leakage**: we forbid future rows and future telemetry; any lag
   features must be strictly past-only and documented.

---

*This plan is a design contract for M1. No model was trained and no dataset
file was modified.*
