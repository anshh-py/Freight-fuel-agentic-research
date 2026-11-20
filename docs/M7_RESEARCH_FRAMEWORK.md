# M7.1 — Research Framework and Evidence Mapping

**Project:** Cross-Truck Fuel-Rate Prediction and Explainable Agentic Driver
Feedback for Heavy-Duty Vehicles

**Stage:** Research-paper phase — framework only. This document establishes the
research framing and maps every proposed paper section to actual project
artifacts. It is **not** the full paper and contains **no new experimental
results**, no invented citations, and no claims the available evidence does not
support.

> **Scope & integrity statement.** Everything below is anchored to existing
> artifacts in `reports/`, `docs/`, `evidence/`, and `configs/`. The telemetry
> and target labels are **synthetic and formula-generated**. Neither the model
> results nor the agent evaluation establish real-world predictive validity. The
> held-out test set contains exactly five truck identities. M5.5 comparative
> evaluation is not yet formally closed, and M6.5 video evidence is not yet
> complete.

---

## 1. Proposed research title

**Working title:**
> *Synthetic Cross-Truck Fuel-Rate Prediction and Explainable Agentic Driver
> Feedback: A Pipeline-Validation Study on Formula-Generated Heavy-Duty
> Telemetry*

The title is deliberately framed as a **pipeline-validation study on synthetic
data**, mirroring the reference paper's drive-cycle study structure but never
implying real-world validation.

---

## 2. Abstract outline (not a fabricated final abstract)

The abstract will be written later from actual results. Its skeleton, as
currently supportable by evidence:

1. **Context:** instantaneous fuel-rate prediction for heavy-duty freight
   vehicles is a building block for driver feedback; predicting for *unseen*
   trucks (cross-truck generalization) is more demanding than the route-matched
   setting in the reference work.
2. **Problem:** can ML models predict `fuel_rate_lph` from telemetry for trucks
   absent from training, and can those predictions be explained and delivered to
   a driver by a grounded, retrieval-augmented agent?
3. **Method:** formula-generated synthetic telemetry from 29 truck identities
   (20 train / 4 validation / 5 held-out test); truck-wise holdout protocol;
   XGBoost variants over two feature sets (full operational vs. telematics-only);
   SHAP instance explanations; a deterministic retrieval-grounded agent with a
   rule-based response-validation guardrail battery.
4. **Results summary (facts only, no generalization claims):** held-out test
   MAE ≈ 2.57 L/h for both feature sets (R² ≈ 0.988–0.989 on synthetic data);
   errors concentrate in accelerating/highway/uphill conditions; telematics-only
   matched full-operational on these five trucks; SHAP reconstruction passes;
   agent passes its manually authored benchmark and engineering ablation.
5. **Conclusion:** the pipeline is coherent and reproducible on synthetic data;
   generalizability to real heavy-duty fleets is **not established** and is a
   central limitation.

---

## 3. Research background and motivation

- Heavy-duty vehicles are a large share of freight fuel use; near-real-time fuel
  rate is central to driver feedback and fleet operations.
- The reference paper (real-world drive-cycle development for heavy-duty freight
  vehicles from OBD/EMS data) is route-matched: the same truck and route appear
  in training and evaluation. Cross-truck generalization is not its focus.
- This project asks the harder question: does a fuel-rate model trained on some
  trucks transfer to trucks it has never seen? This is the practical situation
  for a fleet-deployed model.
- **Critical caveat:** the data used here are **synthetic**, formula-generated
  telemetry. The motivation is to validate the *research and engineering
  pipeline* (protocols, metrics, explainability, grounded agent), not to make
  claims about real fuel consumption. This is the explicit position inherited
  from `README.md`, `docs/research_brief.md`, and `ML_EXPERIMENT_PLAN.md`.

---

## 4. Precise problem statement

Given a multivariate telemetry matrix of heavy-duty truck operating and static
inputs, we study four linked problems:

1. **Cross-truck prediction:** estimate instantaneous fuel consumption rate
   (`fuel_rate_lph`, L/h) for a truck whose identity is absent from the training
   set, under a strict truck-wise holdout protocol.
2. **Feature-set decomposition:** quantify how prediction quality changes when
   engine-internal bus signals (RPM, load, throttle, torque) are removed, i.e.
   whether a telematics-only feature set is sufficient.
3. **Explainability:** produce faithful instance-level explanations of a single
   prediction, verified to reconstruct the model output, for use in driver-facing
   feedback.
4. **Grounded agentic feedback:** deliver predictions, explanations, and
   knowledge-grounded guidance through an agent that abstains rather than
   fabricating unsupported advice.

All four are studied **on synthetic data only**; the problem statement explicitly
excludes real-world generalization.

---

## 5. Research gap

- The reference paper and the broader heavy-duty fuel-modelling literature report
  high in-sample / route-matched accuracy; **unseen-truck transfer is less
  characterized**, and here it is studied only on synthetic data.
- There is a methodological gap we do **not** attempt to fill: no real fleet,
  no live OBD, no validated fuel measurement. Our contribution is a complete,
  reproducible **pipeline** for the synthetic setting, including an
  uncertainty-aware evaluation protocol and a safety-constrained agent layer.
- A substantive gap in our own evidence: the held-out test set has only **five
  truck identities**, so we cannot make population-level claims; we document the
  descriptive nature of all truck-level estimates.

---

## 6. Research objectives

1. **Objective 1 — Protocol.** Establish and document a leakage-free truck-wise
   holdout protocol with separate training/validation/test splits and explicit
   leakage rules (target-derived and identity fields excluded). *(Supported by
   `ML_EXPERIMENT_PLAN.md`, `configs/ml_synthetic_feature_spec.json`.)*
2. **Objective 2 — Cross-truck accuracy.** Train four model families on 20
   trucks, select on 4 validation trucks, and evaluate on the 5 held-out trucks;
   report MAE/RMSE/R² plus per-truck and per-condition descriptions. *(Supported
   by `reports/m1/` and `reports/m3/`.)*
3. **Objective 3 — Feature-set comparison.** Compare `full_operational` vs.
   `telematics_only`, including a trip-clustered bootstrap CI for the
   difference. *(Supported by `reports/m3/cross_truck_generalization_summary.md`.)*
4. **Objective 4 — Explainability.** Compute instance-level SHAP on held-out
   trucks with reconstruction validation, report global/per-truck importance, and
   state the non-causal caveat. *(Supported by `reports/m1/explainability_*`.)*
5. **Objective 5 — Grounded agent.** Build and evaluate a deterministic
   retrieval-grounded driver-feedback agent whose guardrails enforce
   non-fabrication, non-causal language, and abstention. *(Supported by
   `reports/m2/` and `reports/m5/`.)*

---

## 7. Research questions

The four proposed RQs are evaluated below against the evidence. **No RQ is
forced:** each is retained only because an artifact addresses it, and each is
scoped to what the evidence can answer.

| RQ | Refined wording | Status | Evidence support |
| --- | --- | --- | --- |
| RQ1 | How accurately can machine-learning models predict fuel rate on the five held-out synthetic truck identities (cross-truck generalization)? | **Retained.** | `reports/m1/test_evaluation_summary.md`, `reports/m1/test_metrics_report.json`, `reports/m3/cross_truck_generalization_summary.md` |
| RQ2 | What is the predictive-performance difference between the full-operational and telematics-only feature sets on the held-out trucks? | **Retained, reframed.** The evidence supports a *difference comparison*, not a claim that either set is superior: the pooled delta CI straddles zero. | `reports/m3/cross_truck_generalization_summary.md`, `reports/m3/condition_robustness_summary.md` |
| RQ3 | How effectively can SHAP provide faithful instance-level explanations of fuel-rate predictions that reconstruct the model output? | **Retained, reframed.** "Effectiveness" is operationalized as reconstruction fidelity + deterministic per-truck sampling; not human-judged interpretability. | `reports/m1/explainability_summary.md`, `reports/m1/explainability_report.json` |
| RQ4 | How reliably can a retrieval-grounded driver-feedback agent interpret prediction and vehicle context while avoiding unsupported advice? | **Retained, reframed.** "Reliably" is scoped to a manually authored, deterministic benchmark and an engineering ablation; not human-subject, not real-world reliability. | `reports/m2/agent_evaluation_report.md`, `reports/m2/agent_evaluation/summary.md`, `reports/m5/m5_4/summary.md`, `reports/m5/m5_5/summary.md` |

**Explicitly not forced:** no RQ claims real-world validation, statistical
significance across trucks, LLM-level reasoning, or fleet-level generalization,
because none of those are supported by the evidence.

---

## 8. Testable hypotheses (where justified)

The evidence supports *descriptive* hypotheses that are testable against the
existing artifacts (some already resolved); we do not pre-register new
experiments.

- **H1 (cross-truck transfer).** A model trained on 20 trucks achieves
  materially better-than-naive accuracy on 5 unseen trucks. *Tested:* held-out
  XGBoost MAE ≈ 2.57 L/h vs. mean-dummy MAE ≈ 37.2 L/h (`reports/m1/test_evaluation_summary.md`). Supported for the synthetic setting.
- **H2 (feature-set parity on unseen trucks).** Removing engine-internal features
  does not degrade held-out MAE by a practically meaningful margin. *Tested:*
  delta MAE = −0.0025 pooled, bootstrap CI [-0.237, +0.223] (`reports/m3/cross_truck_generalization_summary.md`). Supported descriptively for these five trucks.
- **H3 (error concentration).** Prediction error concentrates in high-dynamics
  regimes (accelerating, high-speed highway, steep uphill). *Tested:* squared-error
  shares and per-condition MAE (`reports/m3/condition_robustness_summary.md`,
  `reports/m3/error_diagnosis_summary.md`). Supported.
- **H4 (SHAP fidelity).** SHAP explanations reconstruct the model output within
  tolerance and identify a dominant feature. *Tested:* max |dev| < 1.6e-4,
  fraction within tol = 1.0 (`reports/m1/explainability_report.json`). Supported.
- **H5 (agent abstention).** The agent abstains on unsupported/insufficient
  queries rather than fabricating advice. *Tested:* 11/11 negative controls
  detected; 7/7 unanswerable retrieval cases abstained (`reports/m2/agent_evaluation_report.md`, `reports/m5/m5_2/summary.md`). Supported within the benchmark scope.

No hypothesis claims real-world generalization.

---

## 9. Main research contributions

1. **A reproducible cross-truck evaluation protocol** for synthetic heavy-duty
   telemetry (truck-wise holdout, leakage denylist, fixed seeds) and a
   trip-clustered bootstrap for uncertainty, bounded to the five held-out trucks.
2. **An empirical feature-set comparison** showing (on synthetic data, five
   held-out trucks) that telematics-only features track full-operational
   predictive performance on MAE, while engine-internal signals dominate
   full-set SHAP importance — evidence that the chosen feature ablation materially
   changes the *explanation* even when the *score* is similar.
3. **Instance-level SHAP explainability with reconstruction validation** on
   unseen trucks, with explicit non-causal framing.
4. **A deterministic, retrieval-grounded driver-feedback agent** with a
   rule-based response-validation guardrail battery, evaluated on a manually
   authored benchmark and an engineering ablation.
5. **A documented integrity boundary** distinguishing synthetic model,,
   software-integration tests, live-browser verification, and (absent) real-world
   validation.

These are pipeline/engineering contributions on synthetic data; the paper will
state this plainly.

---

## 10. Proposed paper structure

Following the reference paper's academic progression, adapted to this study:

1. **Introduction**
2. **Literature Review** (adapted; focus on heavy-duty fuel modelling and
   cross-truck transfer, explicitly not copying the reference paper's content)
3. **Study Approach** (research questions, hypotheses, scope/integrity boundary)
4. **Dataset and Data Preparation** (synthetic telemetry generation, feature
   contract, leakage rules, split protocol)
5. **Methodology** (models, evaluation, explainability, agent architecture)
6. **Results and Discussion** (cross-truck accuracy, feature comparison,
   condition robustness, SHAP, agent evaluation, integration verification)
7. **Conclusions**
8. **Acknowledgements**
9. **References** (only verifiable citations; none invented; see §13)

---

## 11. Evidence map (paper section → project artifacts)

| Proposed section | Artifact(s) | What it supports |
| --- | --- | --- |
| Dataset & Data Preparation | `configs/ml_synthetic_feature_spec.json`; `ML_EXPERIMENT_PLAN.md`; `DATA_SOURCES.md` (§7, §8) | Feature allowlist/denylist, target `fuel_rate_lph`, split contract (20/4/5), 2,697 rows, 29 trucks, synthetic provenance, EMS/IFM non-use |
| Study Approach | `docs/research_brief.md`; `README.md`; `docs/REPRODUCIBILITY_RESEARCH.md` | Protocol separation (truck-wise/chronological/random-diagnostic), baselines policy, non-claims |
| Methodology — models | `ML_EXPERIMENT_PLAN.md` §3; `reports/m1/summary.md` | Model families, hyperparameters, random state 20240617, selected XGBoost |
| Methodology — evaluation | `reports/m1/test_evaluation_summary.md`; `reports/m1/test_metrics_report.json` | MAE/RMSE/R²/MAPE(>1), per-truck, per-class, residual stats, autocorrelation caveat |
| Results — cross-truck | `reports/m3/cross_truck_generalization_summary.md` | Pooled/per-truck metrics, trip-clustered bootstrap CI, variance decomposition, tail error |
| Results — feature comparison | `reports/m3/cross_truck_generalization_summary.md`; `reports/m3/condition_robustness_summary.md`; `reports/m3/error_diagnosis_summary.md` | Delta MAE, per-condition paired comparison, error concentration |
| Results — condition robustness | `reports/m3/condition_robustness_summary.md`; `reports/m3/telemetry_robustness_summary.md`; `reports/m3/error_diagnosis_summary.md` | Route/condition/subgroup metrics, missingness/noise sensitivity, high-error cells, empty heavy-payload cells |
| Results — explainability | `reports/m1/explainability_summary.md`; `reports/m1/explainability_report.json`; `reports/m1/plots/explainability/*` | Global/per-truck SHAP importance, instance waterfalls, reconstruction pass, non-causal caveat |
| Results — agent | `reports/m2/agent_architecture_report.md`; `reports/m2/agent_evaluation_report.md`; `reports/m2/response_generation_report.md`; `reports/m2/retrieval_evaluation_summary.md`; `reports/m2/rag_corpus_validation.json` | Deterministic agent, tools, response guardrails, 49-scenario benchmark, retrieval metrics, synthetic corpus |
| Results — driver feedback | `reports/m4/driver_interaction_summary.md`; `reports/m5/m5_1`…`m5_4` summaries | M4.5 63/63 checks, M5 benchmark/rubric, M5.4 overall 0.920 |
| Results — ablation | `reports/m5/m5_5/summary.md` | M5.5 variants A–F; **report as not formally closed** |
| Results — integration verification | `reports/m1/artifacts/`; `docs/REPRODUCIBILITY_RESEARCH.md`; `evidence/m6_5_demo/index.md` | Real saved artifacts, live-browser screenshots, health checks, test counts |
| Conclusions / Limitations | `docs/REPRODUCIBILITY_RESEARCH.md` §12; `reports/m3/*` limitations; `reports/m5/*` limitations | No real-world validation, 5-truck test, synthetic data, M6.5 video incomplete |

---

## 12. Research limitations and threats to validity

1. **Synthetic data.** All telemetry and targets are formula-generated
   (`DATA_SOURCES.md` §8, `ML_EXPERIMENT_PLAN.md` §8). High R² on synthetic data
   is not evidence of real-world generalization; the target is generated, not
   measured.
2. **Five-truck test set.** Truck-level estimates, macro-averages, variance
   decomposition, and condition subgroups are descriptive and unstable; n=5 is
   too small for a defensible between-truck random-effect estimate (ANOVA/ICC
   explicitly withheld).
3. **Truck/route confound.** In the reference setting truck and route are
   confounded; a truck-wise split may partly measure route shift. This is stated,
   not hidden.
4. **49T single-truck representation.** The 49T class appears as one held-out
   truck; no class-level generalization is claimed.
5. **Empty regimes.** The heavy-payload cell (load factor > 0.66) is empty by
   construction across the whole dataset; combined heavy-condition cells cannot
   be evaluated.
6. **Autocorrelation.** Rows within a truck/trip are autocorrelated; per-truck
   and trip-level resampling is used, and row-level aggregates are not treated as
   independent.
7. **Manually authored agent benchmarks.** M2.4, M5.1–M5.5 are hand-authored,
   deterministic rubrics; they are not independent human-subject studies and
   claim no statistical significance or real-world reliability.
8. **Deterministic agent.** The agent is a keyword/rule router, not an LLM; it
   gives no free-form natural-language understanding.
9. **M5.5 not formally closed** and **M6.5 video incomplete** — these are
   documented open items, not finished results.
10. **Feature-set comparison on compatible inputs only** (M5.5 variant F) —
    no model-level accuracy recomputation and no superiority claim.

---

## 13. Claims the available evidence does NOT support (explicit exclusions)

The following are **excluded** from the paper unless future evidence (e.g. real
fleet data) changes the situation:

- **No claim of real-world predictive validation.** No real truck, live OBD,
  fleet, or validated fuel measurement exists in this work.
- **No claim that the data are observed Indian truck telemetry.** It is
  synthetic; FEAD is not an Indian dataset, and the models were not trained on
  the downloaded FEAD EMS-1 archive (the archive is documented in `DATA_SOURCES.md`
  but the modelling uses `data/synthetic_heavy_truck_research/`).
- **No claim that synthetic R² implies real-world generalization.**
- **No claim of statistical significance across trucks** (n=5 test trucks).
- **No claim that SHAP findings are causal** or reflect real-world fuel
  drivers.
- **No claim that the agent is an LLM, autonomous, or validated by human
  subjects**; benchmark results are engineering/rubric results.
- **No claim that M8.5 comparative evaluation is complete** and **no claim that
  M6.5 video evidence is complete**.
- **No invented citations, external validation, or contributions** — the
  References section will contain only verifiable, real sources; none are
  fabricated here.
- **No claim about heavy-payload behavior**, which the data cannot support.

---

## 14. Final research questions and contribution summary (for reporting)

**Final RQs:** RQ1–RQ4 as refined in §7 (all retained, none forced).

**Proposed contributions:** the five listed in §9 (protocol, feature comparison,
SHAP fidelity, grounded agent, integrity boundary), all scoped to the synthetic
pipeline.

**Evidence gaps:** (1) real-world / observed data; (2) more than five test
trucks; (3) heavy-payload regimes; (4) human-subject evaluation of agent output;
(5) completed M5.5 closure and M6.5 video.

**Unsupported claims excluded:** the full list in §13.

---

## 15. Conflicts discovered in existing documentation

While mapping evidence, the following contradictions/out-of-date statements were
noted (none are modified here; they are recorded for the research write-up):

1. **README.md / MILESTONES.md vs. actual state.** `README.md` (Project status)
   and the M03+ milestones in `MILESTONES.md` still state *"No model has been
   trained. No results exist."* and mark M03–M15 as `todo`, whereas `reports/`
   contains completed M1–M6 model, agent, and integration artifacts. These
   documents are stale relative to the evidence.
2. **FEAD vs. synthetic data framing.** The central research question and
   framing in `README.md`/`docs/research_brief.md` reference FEAD cross-truck
   generalization, and `DATA_SOURCES.md` documents downloading the real FEAD
   EMS-1 archive; however, the actual modelling and evaluation use the
   **synthetic** `data/synthetic_heavy_truck_research/` suite. The paper must
   be unambiguous that the results are over synthetic telemetry, not over the
   downloaded FEAD archive.
3. **Selection vs. test snapshots.** `reports/m1/summary.md` (validation) and
   `reports/m1/test_evaluation_summary.md` (test) report different numbers for
   the same model (e.g. full XGBoost MAE 3.62 validation vs. 2.57 test); this is
   expected (validation vs. held-out test) but must be labelled clearly to avoid
   misreading as inconsistency.
4. **MAPE filter conventions.** `ML_EXPERIMENT_PLAN.md` specifies MAPE on
   `fuel_rate > 1`; report caveats carry `MAPE (>1)` consistently, but readers
   must be told of the filter.
5. **M5.5 status text vs. artifacts.** M5.5 artifacts exist with a 0.920
   baseline score, but the milestone is intentionally **not** closed as
   validation (engineering ablation only); the paper must mirror that distinction.

---

_End of M7.1 research framework. No code, models, datasets, or evaluation outputs
were created or modified; this document is framing only._
