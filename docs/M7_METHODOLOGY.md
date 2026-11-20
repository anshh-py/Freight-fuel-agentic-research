# M7.3 — Dataset, Study Design and Methodology

**Project:** Cross-Truck Fuel-Rate Prediction and Explainable Agentic Driver
Feedback for Heavy-Duty Vehicles

**Stage:** Research-paper phase — methodology only. This document describes the
experimental design, dataset, preprocessing, feature sets, splitting/evaluation
protocol, models, explainability, agent, and robustness/ablation methodology.
It contains **no new experimental results**, changes no source code, dataset or
evaluation output, and marks neither M5.5 nor M6.5 complete.

> **Integrity & grounding statement.** Every methodological claim below is
> supported by an existing project artifact (cited in brackets) or is clearly
> identified as a **design decision**. Where documentation is missing,
> contradictory, or under-specified, this is recorded explicitly under
> _Component gaps and contradictions_ rather than silently resolved. The
> telemetry and target labels are **synthetic and formula-generated**; no step
> in this document implies real-world predictive validation.

---

## 1. Study design

The study is a **pipeline-validation study on synthetic data**. It decomposes
into seven linked components; each is defined and distinguished below so that
the model evaluation, the agent evaluation, and the software integration are
never conflated.

### 1.1 Synthetic telemetry generation

A formula-driven generator (`scripts/synthetic_heavy_truck/gen_telemetry.py`,
seed 20240617) produced 963,697 rows of 1 Hz telemetry for 29 synthetic truck
identities across three weight classes (32T/40T/49T). The target
`fuel_rate_lph` is **formula-generated**, not measured. This is a **dataset
construction** component, separate from modelling. (Evidence:
`data/synthetic_heavy_truck_research/dataset_card.json`,
`generation_config.json`, `validation_summary.json`.)

### 1.2 Predictive modelling

Supervised regression of `fuel_rate_lph` on operational features, trained on
20 synthetic trucks, tuned on 4 validation trucks, evaluated once on 5 held-out
synthetic trucks. Four model families compete; XGBoost is selected on
validation MAE. This is the **offline predictive model** component.
(Evidence: `ML_EXPERIMENT_PLAN.md`, `reports/m1/metrics_report.json`,
`reports/m1/test_metrics_report.json`.)

### 1.3 Cross-truck evaluation

The primary result is the model's performance on **unseen truck identities**
(the 5 held-out test trucks). A trip-clustered bootstrap quantifies uncertainty
conditional on these five trucks. This is an **evaluation protocol** component,
separate from training. (Evidence: `reports/m3/cross_truck_generalization_report.json`.)

### 1.4 Feature-set comparison

The same protocol is run on two feature sets — `full_operational` (engine-bus +
telematics + static) and `telematics_only` (engine-bus signals removed). The
outcome is a _difference comparison_ (paired deltas and bootstrap CI), not a
superiority claim. (Evidence: `reports/m3/cross_truck_generalization_report.json`,
`reports/m3/condition_robustness_summary.md`.)

### 1.5 SHAP explanation

Instance-level SHAP attribution is computed for a deterministic per-truck
subsample of the held-out test trucks, with an additive reconstruction check.
This is an **explainability** component, explicitly non-causal. (Evidence:
`reports/m1/explainability_report.json`, `reports/m1/explainability_summary.md`.)

### 1.6 Agent retrieval and response evaluation

A **deterministic**, retrieval-grounded driver-feedback agent is evaluated
against manually authored benchmarks (M2.4, M5.1–M5.5) and a response-validation
guardrail battery. This is an **agent evaluation** component (engineering /
rubric), not a human-subject study and not real-world reliability. (Evidence:
`reports/m2/agent_evaluation_report.md`, `reports/m2/response_generation_report.md`,
`reports/m5/m5_4/summary.md`, `reports/m5/m5_5/summary.md`.)

### 1.7 Software integration

A FastAPI gateway and React/Vite cockpit integrate the saved model artifacts,
SHAP tool, retrieval tool, and agent. This is verified via **integration tests**
(backend pytest, frontend vitest) and **live-browser verification**, all on
synthetic data. This is a separate result class from the model/agent evaluations.
(Evidence: `docs/REPRODUCIBILITY_RESEARCH.md`, `reports/m1/artifacts/`,
`evidence/m6_5_demo/index.md`.)

**Design decision (not an artifact):** the seven components are reported as
distinct result classes; §1.1–1.5 feed the model/explainability sections, §1.6
feeds the agent section, and §1.7 feeds the integration/verification section.

---

## 2. Dataset description and provenance

### 2.1 Origin and generation

- Generator: `scripts/synthetic_heavy_truck/gen_telemetry.py`, fixed seed
  `20240617`.
- Recorded provenance (`dataset_card.json`): _"Synthetically generated … No
  claim that any record represents real measured truck telemetry."_
- Physics model (`generation_config.json`): road-load equation with
  aerodynamic drag, rolling resistance and grade; a **motion-power-limited**
  engine model (power feasibility tolerance 0.02); grade converted via
  `sin(atan(percent/100))`; signed inertia (braking assists and overrun fuel
  drop); deterministic smooth highway grade segments. Class spec constants
  (GVW, rated power, displacement, CdA, CrR, idle/rated RPM) are documented per
  class (32T/40T/49T). Five route profiles (highway/urban/city/mountainous/mixed)
  are each assigned probability 0.2.

### 2.2 Synthetic-data status

All data are **synthetic**. `dataset_card.json` records
`"synthetic_status": "SYNTHETIC. No claim that any record represents real
measured truck telemetry."` There is **no** real truck, live OBD collection, or
validated fuel measurement anywhere in the project (`docs/REPRODUCIBILITY_RESEARCH.md`
§1, `DATA_SOURCES.md` §8, `README.md`). The generator is deterministic and
documented for reproduction.

### 2.3 Distinction from FEAD and other external datasets

- The **modelling dataset is the synthetic suite**
  `data/synthetic_heavy_truck_research/` (963,697 rows). This is the only data
  used for training/evaluation.
- **FEAD** (Liu et al., 2023) is a real EMS/IFM dataset whose provisioning
  journey (EMS-1 archive download, integrity checks) is documented in
  `DATA_SOURCES.md`. **The FEAD archive is not the modelling dataset.** The
  downloaded EMS-1 archive is documented; the synthetic suite is generated
  separately. The two must never be conflated.
- No other external dataset is used.

### 2.4 Dataset dimensions and files

| File                             | Contents                                                                                       |
| -------------------------------- | ---------------------------------------------------------------------------------------------- |
| `telemetry_master_963697.csv.gz` | 963,697 rows × 24 columns; 1 Hz sampling                                                       |
| `truck_metadata.csv`             | 29 truck identities; class, GVW, displacement, rated power, age, `synthetic_efficiency_factor` |
| `trip_metadata.csv`              | trip-level metadata (540 trips)                                                                |
| `truckwise_split_manifest.csv`   | truck → split assignment (20/4/5)                                                              |
| `data_dictionary.csv`            | column-level dictionary (name, dtype, unit, description, role, fill rate, valid range, notes)  |
| `generation_config.json`         | generator configuration, class/route specs, physical constants                                 |
| `dataset_card.json`              | provenance, synthetic status, intended use, limitations                                        |
| `validation_summary.json`        | 28/28 dataset checks (see §3.1)                                                                |
| `rag_knowledge_corpus.jsonl`     | synthetic retrieval corpus (21 documents)                                                      |
| `agent_evaluation_queries.csv`   | 550 grounded agent queries                                                                     |
| `telemetry_replay_manifest.csv`  | replay trip manifest                                                                           |

### 2.5 Truck and trip counts

- 29 truck identities (10×32T, 11×40T, 8×49T in `generation_config.json`).
- 540 trips (`validation_summary.json` replay check: `replay_trips=540
meta_trips=540`). The held-out test split contains 96 trips across 5 trucks
  (`reports/m3/cross_truck_generalization_report.json`).
- Split counts: 20 train / 4 validation / 5 test trucks (rows 657,336 /
  133,072 / 173,289; `reports/m1/metrics_report.json`).

### 2.6 Telemetry sampling and variables

Sampling interval is 1 s within a trip (constant interval; `data_dictionary.csv`).
24 columns: identifiers (record_id, timestamp, truck_id, trip_id), static truck
attrs (vehicle_class, GVW), trip load (payload, mass), motion state (speed,
acceleration, grade), synthetic coordinates (latitude/longitude — not real
geo), ignition status, engine state (RPM, load %, throttle %, torque), thermal
(coolant, intake air, ambient), the target `fuel_rate_lph`, and two excluded
target-derived fields (fuel increment, cumulative fuel).

### 2.7 Target variable and target-derived fields

- **Target:** `fuel_rate_lph` (L/h), instantaneous fuel rate, range 0–220 L/h,
  fill rate 1.0 (`data_dictionary.csv`). Engine-off rows have zero speed/engine/
  fuel.
- **Target-derived fields (excluded as inputs):** `fuel_consumed_l_increment`
  (= `fuel_rate_lph / 3600`) and `cumulative_fuel_l_trip` (cumulative sum per
  trip) — `configs/ml_synthetic_feature_spec.json` `target_derived_excluded`.
- **Per-truck proxy (excluded):** `synthetic_efficiency_factor` (a per-truck
  fuel multiplier) — `proxy_excluded`.

### 2.8 Data dictionary

Full machine-readable dictionary: `data/synthetic_heavy_truck_research/data_dictionary.csv`.
Key entries (units / valid ranges) are quoted in §4's feature table.

### 2.9 Dataset limitations

- Target is formula-generated; metrics do not establish real-world accuracy
  (`dataset_card.json` limitations; `ML_EXPERIMENT_PLAN.md` §8).
- Physics is simplified: steady-state efficiency, no weather/tyre/road-surface
  detail (`dataset_card.json`).
- Synthetic fleet is not a sample of any real truck population; coordinates are
  not geographic reality.
- **No heavy-payload regime by construction:** across all 963,697 rows the load
  factor `payload_kg / gross_vehicle_weight_kg` never exceeds 0.3886, so the
  heavy-payload cell (LF > 0.66) and combined heavy cells are empty by
  construction, not by error (`reports/m3/condition_robustness_summary.md`
  Limitations).

---

## 3. Data preprocessing

### 3.1 Data validation

`validate_dataset.py` produced `validation_summary.json` with **28/28 checks
passing**:

- Row count (963,697) and exact 24-column schema match.
- Record IDs unique; no missing required values.
- Truck IDs/trip IDs consistent with metadata; each trip belongs to a single
  truck.
- Target-derived consistency: `increment == fuel_rate_lph / 3600`; cumulative =
  per-trip cumsum; cumulative resets at trip start.
- Value ranges reasonable; engine-off rows (7,042) consistent (zero
  speed/engine/fuel).
- Trip distance/record-count/payload match trip metadata; one vehicle_class per
  truck.
- Highway grade coverage present (uphill > 1.5% and downhill < −1.5% exist) with
  smooth transitions (max |Δgrade|/s ≤ 0.223 vs. limit 0.5).
- Power feasibility: 0 records exceed rated power × 1.020 (max demand/rated =
  1.0000).
- Leakage checks: target-derived fields and target withheld from inputs; split
  has zero overlap and exact sizes.
- Reproducibility: identical hashes under the fixed seed.

### 3.2 Missing-value handling

No imputation is required: `no_missing_required` passes; the dataset is complete
(`validation_summary.json`). `ML_EXPERIMENT_PLAN.md` §2.4 states "no imputation
needed; the dataset is complete."

### 3.3 Feature transformations

- `vehicle_class` (low-cardinality) is one-hot encoded with `drop='first'`
  (`configs/ml_synthetic_feature_spec.json`; `reports/m1/` report `columns`
  show `vehicle_class_40T`, `vehicle_class_49T`).
- Continuous features are used as-is; the saved preprocessor
  (`reports/m1/artifacts/{feature_set}/preprocessor.joblib`) is loaded, never
  refitted, at evaluation time (`reports/m1/test_metrics_report.json`,
  `reports/m1/explainability_report.json`).

### 3.4 Feature construction

Features come directly from the telemetry columns per the allowlist in
`configs/ml_synthetic_feature_spec.json`. `acceleration_mps2` is derived from
speed in the generator (documented in `data_dictionary.csv`); `road_grade_pct`
is a route profile field. No additional constructed features are added at model
time in the documented pipeline.

### 3.5 Target construction

The target `fuel_rate_lph` is generated by the physics formula in
`generation_config.json` §model_notes and is the supervised label. The two
target-derived fields are excluded from inputs (§2.7).

### 3.6 Leakage prevention

Leakage rules (from `configs/ml_synthetic_feature_spec.json` `leakage_rules` and
`ML_EXPERIMENT_PLAN.md` §2.1):

- No target-derived fields (`fuel_consumed_l_increment`,
  `cumulative_fuel_l_trip`) as inputs.
- No target column as an input.
- No identity fields (`truck_id`, `trip_id`, `record_id`, `timestamp_utc`) as
  inputs.
- No GPS/route coordinates (`latitude`, `longitude`) as inputs (route identity).
- No `synthetic_efficiency_factor` as input (per-truck fuel proxy).
- Only strictly-past lag features allowed; no future timestamps/telemetry.
- Hyperparameters tuned only on validation trucks; test trucks touched exactly
  once.
  The split is **truck-wise**: a truck appears in exactly one split; rows are
  never shuffled across splits (`ML_EXPERIMENT_PLAN.md` §1).

---

## 4. Feature sets

Two feature sets are defined in `configs/ml_synthetic_feature_spec.json`. Both
are subsets of the 24 columns and exclude the denylist. The `full_operational`
set has 15 named features (16 transformed columns after one-hot drop-first);
`telematics_only` has 11 named features (12 transformed columns).

### 4.1 Full-operational set (15 features)

Engine-bus + telematics + static truck attributes (primary baseline set).

### 4.2 Telematics-only set (11 features)

The four engine-internal signals (`engine_rpm`, `engine_load_pct`,
`throttle_position_pct`, `torque_nm`) are removed; the remaining fields are
external/operational telematics + static attributes.

### 4.3 Feature table

Source of units/roles: `data_dictionary.csv`; availability = in which set.

| Feature                   | Unit | Description                                    | Full | Telematics-only |
| ------------------------- | ---- | ---------------------------------------------- | ---- | --------------- |
| `vehicle_class`           | —    | Weight class 32T/40T/49T; one-hot (drop first) | ✓    | ✓               |
| `gross_vehicle_weight_kg` | kg   | Rated GVW; constant per truck                  | ✓    | ✓               |
| `payload_kg`              | kg   | Cargo mass this trip; constant within trip     | ✓    | ✓               |
| `vehicle_mass_kg`         | kg   | Curb mass + payload; constant within trip      | ✓    | ✓               |
| `speed_kph`               | km/h | Ground speed (0 when stopped)                  | ✓    | ✓               |
| `acceleration_mps2`       | m/s² | Rate of change of speed                        | ✓    | ✓               |
| `road_grade_pct`          | %    | Longitudinal road slope                        | ✓    | ✓               |
| `ignition_status`         | 0/1  | Engine on/off                                  | ✓    | ✓               |
| `engine_rpm`              | rpm  | Engine speed (0 when off; ~620 idle)           | ✓    | ✗               |
| `engine_load_pct`         | %    | Engine load relative to rated power            | ✓    | ✗               |
| `throttle_position_pct`   | %    | Throttle position                              | ✓    | ✗               |
| `torque_nm`               | N·m  | Engine torque (capped at rated)                | ✓    | ✗               |
| `coolant_temp_c`          | °C   | Coolant temperature                            | ✓    | ✓               |
| `intake_air_temp_c`       | °C   | Intake air temperature                         | ✓    | ✓               |
| `ambient_temp_c`          | °C   | Outside ambient temperature                    | ✓    | ✓               |

`engine_rpm`, `engine_load_pct`, `throttle_position_pct`, `torque_nm` are
described in `ML_EXPERIMENT_PLAN.md` §2.3 as independent physical signals (not
arithmetic transforms of fuel rate); including them and ablating them makes the
contribution of engine-internal telemetry explicit rather than hidden.

---

## 5. Data splitting and evaluation protocol

### 5.1 Truck-wise splitting

The split is performed at the **truck level**. `truckwise_split_manifest.csv`
assigns each of the 29 trucks to exactly one of three splits (verified:
`validation_summary.json` — no truck appears in more than one split). Telemetry
rows are joined to their truck's split; a truck appears in exactly one split,
and rows are never shuffled across splits (`ML_EXPERIMENT_PLAN.md` §1,
`configs/ml_synthetic_feature_spec.json` `split_contract`).

### 5.2 Train / validation / test truck counts

| Split           | Trucks | Rows    |
| --------------- | ------ | ------- |
| Train           | 20     | 657,336 |
| Validation      | 4      | 133,072 |
| Test (held-out) | 5      | 173,289 |

(Row counts from `reports/m1/metrics_report.json`; the test split has 96 trips.)

### 5.3 Held-out test identities

Test trucks: `truck_001`, `truck_004`, `truck_017`, `truck_020`,
`truck_028` (classes 32T/32T/40T/40T/49T). All three classes are represented in
the test set (`ML_EXPERIMENT_PLAN.md` §1, `reports/m1/test_metrics_report.json`).
The 49T class appears as a **single** held-out truck.

### 5.4 Why truck-wise splitting was selected

Cross-truck generalization is the central research question (RQ1, RQ3 in
`docs/M7_RESEARCH_FRAMEWORK.md`): a fleet-deployed model must predict for trucks
absent from training. A random row-level split would let the model memorize
truck identity via correlated rows. The truck-wise split enforces that every
test truck is entirely unseen. This is the documented design rationale
(`ML_EXPERIMENT_PLAN.md` §0, §4.3; `docs/M7_RESEARCH_FRAMEWORK.md` §4, §5).

### 5.5 Limitations of the five-truck test set

- Truck-level estimates, macro-averages, variance decomposition and condition
  subgroups are **descriptive and unstable**; n=5 is too small for a defensible
  between-truck random-effect estimate (ANOVA/ICC deliberately withheld;
  `docs/M7_RESEARCH_FRAMEWORK.md` §12).
- The analysis is **conditional on the five frozen test trucks** and does not
  establish population-level generalization (`reports/m3/cross_truck_generalization_report.json`
  `analysis_frame`; `reports/m3/condition_robustness_summary.md`).
- 49T is represented by one test truck; no class-level generalization is
  claimed.
- In the reference setting truck and route are confounded; a truck-wise split
  may partly measure route shift (`docs/M7_RESEARCH_FRAMEWORK.md` §12).

### 5.6 Test-set touch policy

The test set is used only once, for final evaluation of the validation-selected
model. Test data never inform fitting, tuning or selection
(`ML_EXPERIMENT_PLAN.md` §4.3; `reports/m1/test_metrics_report.json` protocol).

---

## 6. Predictive modelling

### 6.1 Baselines

Four model families are defined in `ML_EXPERIMENT_PLAN.md` §3:

- **Mean dummy** and **median dummy** (constant-prediction floors).
- **Ridge regression** (linear reference; standardized features, L2).
- **Random Forest** (non-linear reference; n_estimators 150, max_depth 20,
  min_samples_leaf 8, max_features "sqrt" — from `reports/m1/metrics_report.json`
  `config`).
- **XGBoost** (`xgboost` 3.4.1, `hist` tree method).

All models use the same feature set per run; telematics-only is run as the
ablation. Random state is fixed at `20240617`.

### 6.2 Model selection

Models are trained on the 20 train trucks and evaluated on the 4 validation
trucks. **XGBoost is selected on validation MAE** separately for each feature
set (full: 3.6211; telematics: 4.2926) — `reports/m1/test_metrics_report.json`
`selected_models_from_validation`; `reports/m1/metrics_report.json`. Validation
MAE across models: mean 38.44, median 37.17, RF 3.77 (full) / 4.77 (telematics),
Ridge 5.18 (full) / 19.22 (telematics), XGBoost 3.62 (full) / 4.29 (telematics).

### 6.3 Training procedure

- Training on 20 train trucks (963K-row suite; 657,336 train rows).
- Hyperparameter tuning on validation trucks only; test untouched
  (`ML_EXPERIMENT_PLAN.md` §4.3).
- Saved artifact per feature set: `xgboost.joblib`, `preprocessor.joblib`, plus
  baselines under `reports/m1/artifacts/`. Evaluation loads these artifacts and
  never refits/retrains (`reports/m1/test_metrics_report.json` verification;
  `reports/m3/*` protocols).

### 6.4 Hyperparameter evidence

**Design decision / limitation:** the documented artefacts record the random
forest configuration (above) and the runtime (XGBoost fit 12.3 s full /
11.9 s telematics), but **no explicit hyperparameter grid is recorded** for
XGBoost/Ridge in the reports I inspected. The exact XGBoost and Ridge
hyperparameter tuples are therefore **an evidence gap** (see §11); they are not
invented here.

### 6.5 Evaluation metrics (definitions)

For a test set with n rows, predictions ŷᵢ and targets yᵢ:

- **MAE** = (1/n) Σᵢ |yᵢ − ŷᵢ|
- **RMSE** = sqrt((1/n) Σᵢ (yᵢ − ŷᵢ)²)
- **R²** = 1 − Σᵢ(yᵢ − ŷᵢ)² / Σᵢ(yᵢ − ȳ)²
- **MAPE** = (100/n') Σᵢ∈S |yᵢ − ŷᵢ| / |yᵢ|, where S is the subset of rows
  with `fuel_rate > 1` L/h (see §6.6).

Metrics are reported **pooled** (over all test rows) and **per-truck** /
**per-class** / **per-condition**; a descriptive macro-average across the 5
test trucks is also reported and explicitly labeled not robust.

### 6.6 MAPE filtering rule

MAPE is computed on `fuel_rate > 1` L/h rows only (engine-off and near-idle rows
with tiny denominators are excluded), and the count and coverage are reported
(`ML_EXPERIMENT_PLAN.md` §4.1, §5.1; reports carry `mape_n` and
`mape_coverage`). This filter convention must be stated for readers (§11).

### 6.7 Pooled versus macro evaluation

- **Pooled** metrics aggregate all test rows (e.g. full XGBoost test MAE =
  2.5684, RMSE = 4.5275, R² = 0.9878; telematics MAE = 2.5660, RMSE =
  4.3713, R² = 0.9886 — `reports/m1/test_metrics_report.json`).
- **Per-truck** metrics are computed per truck, then a **macro-average** is
  reported but labeled descriptive (n=5).
- **Per-class** metrics are reported per 32T/40T/49T on the test set.
- Because rows within a truck/trip are autocorrelated, row-level aggregates are
  not treated as independent; per-truck and trip-level resampling is used
  (`docs/M7_RESEARCH_FRAMEWORK.md` §12; §9.2 below).

---

## 7. Explainability methodology

### 7.1 Method

Instance-level SHAP (`shap` 0.54.1) is applied to the validation-selected
XGBoost model per feature set, on a **deterministic per-truck uniform
subsample** of the five held-out test trucks (500 rows/truck, seed 20240617,
2,500 rows total). All five test trucks are represented; no train/val truck is
in the sample (`reports/m1/explainability_report.json` protocol, `sampling`).

### 7.2 Background / reference data

SHAP values are computed with the `shap.TreeExplainer` (tree-based exact
attribution) over the transformed feature space. The model and preprocessor are
loaded from disk, never refitted/retrained (`reports/m1/explainability_report.json`
verification).

### 7.3 Sampling strategy

Deterministic per-truck uniform subsample with fixed `np.random.RandomState`
seed 20240617; 500 rows per test truck (`reports/m1/explainability_report.json`).

### 7.4 Reconstruction check

For each feature set, the SHAP additive decomposition is checked against the
model output: `max_abs_deviation` (full: 1.57e-4; telematics: 1.44e-4),
`fraction_within_tolerance` = 1.0 at tolerance 0.001, and `pass: true` for
both sets (`reports/m1/explainability_report.json` `reconstruction_validation`).
This operationalizes "effective explanation" as **fidelity to the model**.

### 7.5 Reported outputs

- Global mean-absolute SHAP importance per feature and per truck
  (`reports/m1/explainability_report.json`).
- Individual explanations (waterfalls) for selected rows per truck (largest /
  median / closest-to-zero absolute residual) — `reports/m1/explainability_report.json`
  `individual_explanations`, plots under `reports/m1/plots/explainability/`.
- A readable-feature mapping (e.g. `torque_nm` → "Torque (N·m)").

### 7.6 Interpretation limitations

- SHAP attribution is **attribution for the model, not causality**, and not
  real-world fuel drivers (`reports/m1/explainability_report.json` protocol;
  `reports/m2/agent_architecture_report.md` §6).
- No human-judged interpretability study is performed; explanations are
  validated only via reconstruction (RQ3 scoping in
  `docs/M7_RESEARCH_FRAMEWORK.md` §7).
- Importance values reflect the synthetic-trained model, not observed trucks.

---

## 8. Agent methodology

### 8.1 Deterministic routing

The agent is a **stateless orchestrator over three tools**: fuel-rate
prediction, knowledge retrieval, and prediction explanation
(`src/agent/orchestrator.py`). Intent routing uses the `DeterministicIntentRouter`
(keyword/rule scoring), not an LLM — `configs/agent_config.json` sets
`routing.backend="deterministic"`, `is_llm=false` (`reports/m2/agent_architecture_report.md`
§4–5; `docs/REPRODUCIBILITY_RESEARCH.md` §1). It is **not** an LLM agent,
autonomous agent, or LangGraph system.

### 8.2 Retrieval corpus and retrieval process

- Corpus: `data/synthetic_heavy_truck_research/rag_knowledge_corpus.jsonl` —
  21 synthetic educational documents / 21 chunks, all `is_oem_manual=false`
  (`reports/m2/retrieval_evaluation_summary.md`).
- Retrieval: `sentence-transformers/all-MiniLM-L6-v2` (dim 384, local), cosine
  similarity, top-k = 5 (`reports/m2/retrieval_evaluation_summary.md`).
- An `evidence_sufficiency` flag marks low-similarity / empty retrieval
  (`reports/m2/agent_architecture_report.md` §3).
- Retrieval metrics on 55 labeled queries (M2.1) and 26 queries (M5.2) are
  reported; both are over synthetic educational text, not OEM/manufacturer
  documentation (`reports/m2/retrieval_evaluation_summary.md`,
  `reports/m5/m5_2/summary.md`).

### 8.3 Available tools

1. `FuelRatePredictionTool` — validates inputs, loads saved XGBoost +
   preprocessor, predicts L/h (`reports/m2/agent_architecture_report.md` §3).
2. `KnowledgeRetrievalTool` — vector retrieval over the synthetic corpus.
3. `PredictionExplanationTool` — instance-level SHAP + reconstruction.
   Combined intents execute tools sequentially; no cross-tool reasoning step exists
   (`reports/m2/agent_architecture_report.md` §8).

### 8.4 Response construction

`ResponseGenerator` builds a `GroundedResponse` with `driver_version` (readable)
and `technical_version` (fleet-operator). Four claim categories are kept
separate: model prediction, model explanation (non-causal), retrieved evidence
(synthetic provenance), and recommendations (evidence-backed only)
(`reports/m2/response_generation_report.md` §2–3).

### 8.5 Abstention and safety controls

A deterministic validation layer (`src/agent/response_validation.py`) enforces
10 checks: synthetic-data disclaimer, non-causal language, no OEM mislabeling,
synthetic-corpus notice, source attribution, unsupported queries not answered
with specs, insufficient evidence stated, tool failures not presented as
success, predicted value preserved, top SHAP value preserved
(`reports/m2/response_generation_report.md` §5). The agent abstains rather than
fabricating (N01–N09 controls; M5.2 7/7 unanswerable cases abstain).

### 8.6 Benchmark construction

- **M2.4:** 49 manually authored scenarios (category IDs A01–G03) with expected
  intent, tool behavior, response properties, prohibited claims, and expected
  abstention/failure. Build types: `real` (30, end-to-end orchestrator),
  `direct` (1), `mock` (18, validator-only; all 11 negative controls are mock)
  (`reports/m2/agent_evaluation_report.md` §3).
- **M5.1–M5.5:** a driver-feedback benchmark/rubric with six dimensions
  (relevance, actionability, clarity, evidence/uncertainty, safety,
  consistency); M5.4 overall 0.920 (185/201 applicable); M5.5 ablations
  A–F (`reports/m5/m5_4/summary.md`, `reports/m5/m5_5/summary.md`).

### 8.7 Manual evaluation criteria

All criteria are **manually authored, deterministic, rule-based**, evaluated on
full response text; there is no LLM judge and no subjective score
(`reports/m2/agent_evaluation_report.md` §3, §7). Safety-critical scenario count
is 0 across M5.4/M5.5. These are engineering/rubric results — not human-subject
studies and not real-world reliability.

---

## 9. Robustness and ablation methodology

All M3/M5 robustness experiments are **read-only**: they reuse the frozen
5-truck test split and saved M1.2 artifacts; no training/refitting/tuning; and
results are descriptive, conditional on the five test trucks (`reports/m3/*`
protocols).

### 9.1 Cross-truck paired comparison (M3.1)

Paired deltas between feature sets are computed per row, per truck and per trip
(delta = telematics − full; positive ⇒ full has lower error). A
**trip-clustered percentile bootstrap** (2,000 resamples of the 96 test trips,
seed 20240617) yields 95% CIs for MAE/RMSE/delta, conditional on the five test
trucks (`reports/m3/cross_truck_generalization_report.json` `bootstrap`). Pooled
delta MAE = −0.0025; delta CI [−0.237, +0.223]; per-truck deltas vary in sign.

### 9.2 Trip bootstrap

Resampling unit is `trip_id` (96 trips), sampled with replacement jointly for
both models; metrics recomputed per resample; percentile 95% interval
(`reports/m3/cross_truck_generalization_report.json`). This addresses within-
truck/trip autocorrelation (not treating rows as independent).

### 9.3 Input perturbations (M3.4)

Synthetic corruption applied to copies of the final model-input feature matrices
(after feature construction, before preprocessing): gaussian noise (0.25/0.5
× std), random missingness (5/10/20/30%), complete-sensor failure (all-NaN on
one signal), fixed bias, isolated extreme spikes (0.005 fraction, 8× std), and
physical-bound clipping. Categorical `vehicle_class` and binary `ignition_status`
are preserved. Prediction-failure rate is counted; metrics are on surviving
finite rows (`reports/m3/telemetry_robustness_summary.md`). This is corruption
**of synthetic copies**, not real-world sensor-failure validation.

### 9.4 Condition and error diagnosis (M3.2, M3.3)

Per-condition support is classified (len, trucks present); subgroups are
`descriptive_full_truck_coverage`, `descriptive_limited_truck_coverage`, or
`insufficient_rows`/`empty`. Error concentration, trip-level recurrence, and
residual direction (residual = y_pred − y_true) are reported descriptively; the
`flagged_high_error` heuristic is a descriptive deviation, not a statistical
interaction test (`reports/m3/condition_robustness_summary.md`,
`reports/m3/error_diagnosis_summary.md`).

### 9.5 Retrieval evaluation (M2.1, M5.2)

See §8.2. Grounding uses a token-overlap heuristic reported alongside manual
labels (`reports/m5/m5_2/summary.md`).

### 9.6 Agent ablation (M5.5)

Variants vs. baseline A_full_pipeline (0.920): B_without_session_context
(0.908), C_without_retrieval_grounding (0.910), D_without_shap_explanation
(0.920), E_without_response_validation (offline shadow only; unvalidated text
never delivered), F_feature_set_comparison (full 0.920 vs telematics 0.915 on
compatible inputs; no model-level accuracy recomputed; no superiority claim)
(`reports/m5/m5_5/summary.md`).

### 9.7 Limitations of current M5.5 evidence

M5.5 is an **offline engineering ablation** on a deterministic rubric, not an
independent human-subject study and not real-world validation. Score differences are
descriptive absolute percentage-point deltas; no inferential significance
testing was performed. It is reported as substantively complete as an
engineering ablation but **not as validation**, and is **not** marked
complete as validation (`reports/m5/m5_5/summary.md`;
`docs/REPRODUCIBILITY_RESEARCH.md` §12).

---

## 10. Reproducibility

Supported by `docs/REPRODUCIBILITY_RESEARCH.md`: fixed seed (20240617), fixed
environment (Python 3.13.14, numpy 2.3.2, pandas 3.1.0, scikit-learn 1.9.1,
xgboost 3.4.1, shap 0.54.1), saved model/preprocessor artifacts under
`reports/m1/artifacts/{feature_set}/`, deterministic generator, and documented
commands (`python -m src.api.server`; frontend `npm run dev`; tests
`python -m pytest tests/test_m6_2_server.py tests/test_m4_2_driver_session.py
tests/test_m4_4_driver_conversation.py -q` → 33 passed; `npm test` → 28/28;
`npm run build` → exit 0). No environment variables / secrets are required.
`scripts/verify_demo_env.py` → OVERALL: READY. The M6.5 demo video is
**unrecorded** (no screen-capture tool); the M6.5 evidence index is
`evidence/m6_5_demo/index.md`.

---

## 11. Component gaps and contradictions (recorded, not resolved)

These are **evidence gaps or contradictions** found while writing this
methodology. None are silently resolved; each is flagged for the paper.

1. **No explicit XGBoost/Ridge hyperparameter grid** is recorded in the
   artifacts I inspected (only the random-forest config and runtimes are
   logged). The exact hyperparameter tuples are an evidence gap (§6.4).
2. **`README.md` / `MILESTONES.md` are stale:** they still state "No model has
   been trained. No results exist." and mark M03+ as `todo`, yet `reports/`
   holds completed M1–M6 artifacts (recorded in `docs/M7_RESEARCH_FRAMEWORK.md`
   §15).
3. **FEAD vs. synthetic framing:** earlier framing references FEAD cross-truck
   generalization and `DATA_SOURCES.md` documents the real FEAD EMS-1 archive,
   but the modelling uses the **synthetic** suite. This document is
   unambiguous that results are over synthetic telemetry (§2.3).
4. **Validation vs. test numbers differ** for the same model (e.g. full
   XGBoost MAE 3.62 validation vs. 2.57 test) — expected (validation vs.
   held-out test) but must be labelled to avoid misreading (§6.2, §6.7).
5. **MAPE > 1 filter** convention must be stated (§6.6).
6. **M5.5 artifacts exist but the milestone is not formally closed as
   validation**; M6.5 video is incomplete (§9.7).
7. **Replay truck metadata (32T) differs from the cockpit default static inputs
   (40T)** — pre-existing, not corrected (`docs/REPRODUCIBILITY_RESEARCH.md`
   §12).
8. **RAG retrieval metrics differ across milestones** (M2.1 Recall@1 0.2455
   over 55 queries vs. M5.2 Recall@1 0.8158 over 19 answerable queries) because
   the query sets and corpora/configurations differ; the paper must cite each
   benchmark separately and not pool them.

---

_End of M7.3 methodology. No code, datasets, or evaluation outputs were
created or modified; M5.5 and M6.5 are not marked complete._
