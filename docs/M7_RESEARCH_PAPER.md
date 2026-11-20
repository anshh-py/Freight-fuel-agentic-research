# Cross-Truck Fuel-Rate Prediction and an Explainable Deterministic Agent for Driver Feedback in Heavy-Duty Vehicles

**A synthetic-pipeline evaluation study**

---

## Abstract

Accurate instantaneous fuel-rate prediction and explainable driver feedback are central to reducing fuel expenditure and emissions in heavy-duty road freight. This study develops and evaluates a complete machine-learning and agentic pipeline for heavy-duty fuel-rate prediction on synthetic, formula-generated telemetry. The pipeline comprises (i) an XGBoost regression model trained on full-operational and telematics-only feature sets; (ii) a strict truck-wise holdout protocol in which 20 trucks are used for training, 4 for validation, and 5 previously unseen trucks for testing; (iii) instance-level SHAP explanations validated through reconstruction checks; and (iv) a deterministic, retrieval-grounded agent that provides driver feedback with explicit abstention and safety guardrails.

On the held-out synthetic test set, the full-operational XGBoost model achieved an MAE of 2.57 L/h, an RMSE of 4.53 L/h, and an R² of 0.988, with comparable macro-averaged performance across the five held-out trucks. Removing the four engine-internal signals — engine RPM, engine load, throttle position, and torque — (the telematics-only set) yielded a near-identical error (MAE 2.57 L/h) but a materially different explanation structure, indicating that the two feature sets encode very different predictive mechanisms despite similar accuracy. SHAP reconstruction checks confirmed additive consistency of the attribution values. The deterministic agent passed all 38 positive scenarios and detected all 11 negative controls on the manual functional benchmark (M2.4) and scored 92.0% (185/201) overall on the driver-feedback rubric (M5.4), including 46/46 (100%) on its safety sub-dimension with zero safety-critical violations, though both benchmarks are small and manually constructed.

All telemetry and target labels in this evaluation are synthetic and formula-generated; the results therefore do not constitute evidence of real-world generalization. The held-out test set contains only five truck identities, which is insufficient to support claims of broad fleet generalization. Within the constraints of this synthetic evaluation, the study demonstrates a reproducible protocol capable of assessing cross-truck generalization, feature-set differences, attribution consistency, and the reliability of a deterministic retrieval-grounded feedback agent.

**Keywords:** heavy-duty vehicles; fuel-rate prediction; XGBoost; SHAP; retrieval-augmented generation; driver feedback; synthetic telemetry; cross-vehicle generalization; explainable AI.

---

## 1. Introduction

Heavy-duty freight vehicles account for a substantial share of road-transport fuel consumption and associated emissions. Small reductions in per-vehicle fuel use, achieved through better prediction, diagnosis, and driver behaviour, can translate into significant operational savings at fleet scale. Two complementary enablers have attracted increasing attention: data-driven fuel-consumption models that exploit on-board telemetry, and feedback systems that translate model output into actionable guidance for drivers.

This study assembles, integrates, and evaluates a complete pipeline for heavy-duty fuel-rate prediction and driver feedback. The pipeline combines a gradient-boosted regression model trained on truck telemetry, a strict truck-wise holdout evaluation protocol, instance-level explanations derived with SHAP, and a deterministic retrieval-grounded agent that generates driver feedback with explicit abstention controls.

The evaluation in this report is conducted entirely on synthetic telemetry. Both the feature inputs and the target fuel-rate labels are generated from formulas and pre-specified parameters, not observed from operational vehicles. Consequently, the numerical results reported here describe the behaviour of the pipeline on synthetic data, they do not constitute evidence of real-world predictive performance, and they should not be read as a fleet-level operational validation.

The remainder of the manuscript is organised as follows. Section 2 reviews the related literature and motivates the research gap. Section 3 states the research objectives and questions. Section 4 describes the dataset and methodology. Section 5 reports the experimental results. Section 6 discusses the findings in relation to the research questions and the literature. Section 7 examines limitations and threats to validity, Section 8 considers implications and future work, and Section 9 concludes.

## 2. Literature Review and Research Gap

The relevant literature spans fuel-consumption prediction, cross-vehicle generalization, explainable machine learning, retrieval-augmented generation, and driver-feedback systems.

### 2.1 Fuel consumption prediction for heavy-duty vehicles

Data-driven fuel-consumption prediction has been reviewed extensively, Zhao et al. (2023) catalogue the dominant data-driven methods for vehicle-fuel-consumption prediction, ranging from statistical models to machine learning, and identify driving behaviour, vehicle condition, and operating environment as the primary determinants. Fleet-level prediction studies consistently report high within-population accuracy. Zhu et al. (2024) applied LightGBM to heavy-duty truck fuel consumption and identified key influencing factors, while Liu et al. (2025) applied CatBoost to heavy-duty diesel trucks and used SHAP to attribute the influence of engine parameters, driving behaviour, and vehicle weight.

The reference study underpinning the framing of the present work, Gavimath and Ravuri (2025), develops real-world drive cycles for heavy-duty freight vehicles from on-board diagnostics (OBD) data. That study is route-matched: it characterises drive cycles for specific routes rather than testing transfer to unseen truck identities. This distinction is central to the positioning of the present work.

### 2.2 Telematics-based and engine-feature-based prediction

Most fleet-level studies predict fuel consumption from telematics and engine-feature data. The typical feature set comprises engine speed, load, throttle position, and vehicle speed. Some studies additionally exploit fuel-related fields such as fuel rate and fuel level. The present study contrasts two feature sets: a full-operational set of 15 features and a telematics-only set of 11 features, which removes the four engine-internal signals (engine RPM, engine load, throttle position, torque). The comparison isolates the predictive contribution of the engineering features relative to the telematics-only inputs.

### 2.3 Cross-vehicle and cross-truck generalization

Predictive models are often trained on a fleet and evaluated on the same fleet, which can overstate generalisation to new vehicles. Studies that evaluate transfer to unseen trucks are comparatively rare. The reference study (Gavimath and Ravuri, 2025) does not test unseen-truck transfer, and fleet-level prediction studies surveyed in Section 2.1 report within-population accuracy rather than a strict leave-trucks-out protocol.

The present study therefore adopts a strict truck-wise holdout. 20 trucks are used for training, 4 for validation, and 5 previously unseen trucks for testing. This protocol evaluates how well a model trained on a known set of truck identities generalises to trucks it has never seen. The protocol is possible only because the synthetic dataset provides a sufficiently large number of distinct truck identities with dense trip coverage.

### 2.4 Explainable machine learning and SHAP

SHAP (SHapley Additive exPlanations; Lundberg and Lee, 2017) provides a game-theoretic attribution of model predictions to input features. In the fuel-consumption literature, SHAP has been used to attribute prediction contributions to engine, behaviour, and vehicle attributes (Liu et al., 2025). Barbado and Corcho (2022) use interpretable machine-learning models to predict and explain vehicle fuel-consumption anomalies, establishing a precedent for SHAP-based attribution in fuel analysis without claiming causal interpretation. The present study follows this precedent: SHAP values are presented as additive attribution estimates, not as causal effects.

### 2.5 Retrieval-augmented generation and evidence grounding

Retrieval-augmented generation (RAG) augments language-model output with retrieved evidence. Chen et al. (2024) benchmark large language models in RAG settings and find that models struggle with negative rejection (declining to answer when evidence is insufficient or contradictory) and with proper integration of retrieved evidence. This finding motivates a design choice in the present work: the feedback agent is deterministic and explicitly abstains rather than speculating when retrieval evidence is insufficient.

### 2.6 AI-based driver feedback and eco-driving assistance

AI-based eco-driving assistance has received growing attention. A 2026 study (Karlsson et al., 2026) presents an end-to-end conversational AI coach for heavy-duty eco-driving, combining telemetry-based prediction, SHAP, and RAG with conversational coaching. That study is evaluated on a real fleet and uses an LLM-based coach. The present work differs fundamentally in two respects: the telemetry here is synthetic, and the feedback agent here is deterministic and retrieval-grounded rather than an LLM-based conversational coach. The 2026 study is therefore not a directly comparable evaluation of the present pipeline; it is a complementary approach from which the present work draws framing but not comparable evidence.

### 2.7 Evaluation of safety, reliability, and unsupported responses

The reliability and safety of a feedback system in a safety-relevant domain depends on its ability to avoid unsupported claims and to abstain when evidence is lacking. Chen et al. (2024) highlight the difficulty of negative rejection in RAG systems. The present study operationalises this by constructing explicit safety and abstention benchmarks and by measuring the agent's behaviour on them. The benchmarks are manually authored and small, and therefore provide preliminary rather than conclusive evidence of safe behaviour.

### 2.8 Research gap and positioning

No study identified in this review reports a strict leave-trucks-out holdout for instantaneous heavy-duty fuel prediction, combines it with reconstruction-validated SHAP attribution, and pairs it with a deterministic abstention-oriented feedback agent. The present study does not claim to be the first to combine prediction, SHAP, RAG, and driver feedback — the 2026 conversational coach study already does so. Instead, the contribution is narrower and evidence-supported: **(i)** a reproducible truck-wise holdout protocol for synthetic heavy-duty fuel-rate prediction; **(ii)** an empirical comparison of full-operational and telematics-only feature sets with distinct explanation structures; **(iii)** instance-level SHAP attribution validated for additive consistency; and **(iv)** a deterministic retrieval-grounded agent with an explicit abstention guardrail evaluated against manually constructed benchmarks.

Because the evaluation is synthetic, the contribution is methodological and protocol-level: it demonstrates a reproducible framework for assessing cross-truck generalisation, feature-set differences, attribution consistency, and agent reliability before any real-world data are available, and it does not claim real-world predictive validation.

## 3. Research Objectives and Questions

### 3.1 Objectives

The study pursues four objectives:

1. **O1 — Cross-truck prediction.** Develop and evaluate a machine-learning model that predicts instantaneous fuel rate on truck identities held out during training.
2. **O2 — Feature-set comparison.** Compare the predictive performance of a full-operational feature set against a telematics-only feature set.
3. **O3 — Explainability.** Produce instance-level SHAP explanations of fuel-rate predictions and validate their additive consistency.
4. **O4 — Agent reliability.** Build and evaluate a deterministic retrieval-grounded driver-feedback agent that interprets prediction and vehicle context while abstaining rather than offering unsupported advice.

### 3.2 Research questions

The research questions map directly onto the objectives:

- **RQ1.** How accurately can machine-learning models predict fuel rate across held-out synthetic heavy-duty truck identities?
- **RQ2.** What is the predictive-performance difference between full-operational and telematics-only feature sets?
- **RQ3.** How effectively can SHAP provide instance-level explanations of fuel-rate predictions?
- **RQ4.** How reliably can a retrieval-grounded driver-feedback agent interpret prediction and vehicle context while avoiding unsupported advice?

Each question is answered only to the extent the existing experiments support it. In particular, RQ1 is answered within the five held-out synthetic trucks only; RQ3 is answered in terms of attribution consistency, not human-judged interpretability; and RQ4 is answered against manually authored benchmarks, not a human-subject evaluation.

## 4. Dataset and Methodology

### 4.1 Study design

The study is an evidence-synthesis of completed experiments; it introduces no new experiments. The overall design comprises seven distinguishable components:

1. **Synthetic telemetry generation.** Telemetry and fuel-rate labels are generated from formulas and pre-specified parameters.
2. **Predictive modelling.** An XGBoost regression model is trained on feature sets derived from the telemetry.
3. **Cross-truck evaluation.** The model is evaluated on truck identities held out during training.
4. **Feature-set comparison.** Full-operational and telematics-only feature sets are compared.
5. **SHAP explanation.** Instance-level attributions are produced and validated for additive consistency.
6. **Agent retrieval and response evaluation.** A deterministic retrieval-grounded agent is evaluated against manually constructed benchmarks.
7. **System integration.** The components are integrated, and the integration is verified at the API, component, and live-browser levels.

The synthetic and evaluation components are deliberately kept distinct: model evaluation, agent evaluation, and software integration are reported as separate evidence streams.

### 4.2 Dataset description and provenance

The modelling dataset is synthetic. It is stored under `data/synthetic_heavy_truck_research/` and is accompanied by a dataset card, a generation configuration, a data dictionary, a validation summary, truck metadata, and a data manifest. The dataset comprises 29 synthetic truck identities, with 20 trucks used for training, 4 for validation, and 5 for the held-out test set. Each truck is described by metadata including a truck ID, a truck class, a load-factor parameter, and a drive-cycle signature.

The synthetic telemetry is sampled at 1 Hz along generated drive cycles. Variables include vehicle speed, engine speed (RPM), throttle position, engine load, grade, wind speed, fuel rate, and fuel-level-related fields. The target variable is instantaneous fuel rate (L/h). The fuel-rate label is generated from a formula of the input variables, so the relationship between inputs and target is deterministic in the generative sense.

The dataset explicitly distinguishes the synthetic modelling data from the FEAD dataset. FEAD is not an Indian dataset, and the synthetic modelling suite is not the FEAD dataset. The synthetic generator and the FEAD reference motivate parts of the data design, but the actual modelling is performed entirely on the synthetic suite. This distinction is important for integrity and is preserved throughout this manuscript.

**Dataset limitations.** (i) The data are synthetic and formula-generated, so no field observation of truck telemetry or fuel use is represented. (ii) The load-factor parameter in the synthetic generator never exceeds 0.3886, so the heavy-payload regime is absent from the evaluated data. (iii) The five-truck test set is small and cannot support broad fleet-generalisation claims. (iv) The data dictionary documents the generated fields but does not claim real-world provenance for any of them.

### 4.3 Data preprocessing

Preprocessing follows the documented pipeline:

- **Data validation.** Generated trucks are validated against expected trip counts and vehicle tri, with issues reported in a validation summary.
- **Missing-value handling.** Missing telemetry values are handled according to the documented pipeline; the specific imputation rule is recorded in the data documentation.
- **Feature transformations and construction.** Derived features are computed from raw telemetry. Target-derived fields (e.g., fuel increment and cumulative fuel) are excluded from both feature sets under the leakage rules.
- **Target construction.** The target is instantaneous fuel rate in litres per hour; it is formula-generated together with the input features.
- **Leakage prevention.** Splits are performed at the truck level so that no truck identity appears in more than one partition. Features that are formulaically derived from the target are excluded from both feature sets, so they cannot leak the target into either model input.

### 4.4 Feature sets

Two feature sets are defined.

- **Full-operational feature set (15 features).** Includes the telematics variables plus the four engine-internal signals (engine RPM, engine load, throttle position, torque).
- **Telematics-only feature set (11 features).** Removes the four engine-internal signals, retaining only the telematics variables. Target-derived fields and fuel-level-derived fields are excluded from both feature sets.

The counts 15 and 11 refer to the _logical_ predictor specification. One of those predictors, `vehicle_class` (a categorical variable), is encoded into the model input matrices using one-hot encoding with `drop='first'`, which expands it from a single column into two dummy columns. Consequently, the full-operational and telematics-only model input matrices contain 16 and 12 post-encoding columns respectively, even though the logical predictor counts are 15 and 11.

The feature table below lists the included predictors and the excluded fields for each set.

| Feature                                                 | Unit        | Full-operational (15) | Telematics-only (11) |
| ------------------------------------------------------- | ----------- | --------------------- | -------------------- |
| Vehicle class (one-hot encoded)                         | categorical | Yes                   | Yes                  |
| Gross vehicle weight                                    | kg          | Yes                   | Yes                  |
| Payload                                                 | kg          | Yes                   | Yes                  |
| Vehicle mass                                            | kg          | Yes                   | Yes                  |
| Vehicle speed                                           | km/h        | Yes                   | Yes                  |
| Acceleration                                            | m/s²        | Yes                   | Yes                  |
| Road grade                                              | %           | Yes                   | Yes                  |
| Ignition status                                         | binary      | Yes                   | Yes                  |
| Coolant temperature                                     | °C          | Yes                   | Yes                  |
| Intake air temperature                                  | °C          | Yes                   | Yes                  |
| Ambient temperature                                     | °C          | Yes                   | Yes                  |
| Engine RPM                                              | rpm         | Yes                   | No                   |
| Engine load                                             | %           | Yes                   | No                   |
| Throttle position                                       | %           | Yes                   | No                   |
| Torque                                                  | N·m         | Yes                   | No                   |
| Target-derived fields (fuel increment, cumulative fuel) | —           | Excluded              | Excluded             |
| Fuel-level-derived fields                               | —           | Excluded              | Excluded             |

The exact set of 15 and 11 features, with units and descriptions, is recorded in the feature specification (`configs/ml_synthetic_feature_spec.json`). The two sets are deliberately designed so that the difference isolates the engine-internal signals.

### 4.5 Data splitting and evaluation protocol

The train/validation/test split is truck-wise, not row-wise, with counts:

- **Training:** 20 truck identities.
- **Validation:** 4 truck identities.
- **Test:** 5 previously unseen truck identities.

Truck-wise splitting is selected because it evaluates generalisation to unseen truck identities rather than to unseen trips of familiar trucks. This matches the scientific question of whether a model trained on a known fleet transfers to new vehicles. The alternative, a random row-wise split, would overstate performance by allowing the model to see trips from every truck during training.

The held-out test identities are the five trucks reserved from training; their identities are recorded in the split manifest. The limitations of the five-truck test set are (i) its small size, which precludes broad fleet-generalisation claims, and (ii) the descriptive, not inferential, nature of any per-truck comparison.

### 4.6 Predictive modelling

The primary model is an XGBoost regressor. A random-forest model and a linear baseline (Ridge regression) are also evaluated as baselines. Models are trained on the training partition, selected on the validation partition, and evaluated on the held-out test partition.

The training procedure follows the ML experiment plan. The final model configurations are recoverable from the source code and serialized model artifacts and are recorded here. XGBoost uses n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, tree_method='hist', and random_state=20240617; Ridge uses alpha=1.0. All remaining parameters use the relevant library defaults. No grid, randomized, or other hyperparameter-search procedure is claimed, because none was recorded in the inspected artifacts; the absence of a recorded search procedure is a separate limitation from the reproducibility of the final configuration, which is fully specified above. The random-forest configuration and runtimes are logged in the experiment plan.

**Evaluation metrics.** Let the predictions be $\hat{y}_i$ and the observed values $y_i$ for $i = 1, \dots, n$.

- **MAE** (L/h): $\mathrm{MAE} = \frac{1}{n}\sum_i |y_i - \hat{y}_i|$.
- **RMSE** (L/h): $\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_i (y_i - \hat{y}_i)^2}$.
- **R²**: $R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$.
- **MAPE** (%): $\mathrm{MAPE} = \frac{100}{n}\sum_i \left|\frac{y_i - \hat{y}_i}{y_i}\right|$.

**MAPE filtering rule.** MAPE is computed only on records whose observed fuel-rate target exceeds 1.0 L/h (the lower bound defined per the experiment plan, §4.1), because division by near-zero targets would otherwise make MAPE undefined or wildly inflated. Applied to the held-out test partition, this filter retains $n = 172{,}057$ of $173{,}289$ test rows, giving a coverage of $0.99289$ of the partition. The rows excluded are at the low-target end of the fuel-rate distribution; the reported MAPE values are computed on this retained subset and are therefore not directly comparable to a metric computed over the unfiltered set.

**Pooled versus macro evaluation.** Pooled metrics are computed over all samples in the relevant partition regardless of truck identity. Macro-averaged metrics are computed by averaging the per-truck metric values across trucks. The two are reported separately and are not interchangeable.

### 4.7 Explainability methodology

SHAP (Lundberg and Lee, 2017) is used to produce instance-level attributions of the fuel-rate prediction to input features. The SHAP explainer is fitted on a background/reference data sample drawn from the training partition. Attributions are computed for test instances.

**Reconstruction check.** Because SHAP values are additive, the sum of the attribution values plus the expected value should reconstruct the model prediction. The reconstruction check quantifies how closely this holds. Reconstruction consistency validates the additive structure of the attributions; it does not establish human interpretability or causal validity.

**Interpretation limitations.** SHAP values are presented as additive attribution estimates, not causal effects. The results do not claim that a feature causally drives fuel use, only that the model's prediction is attributed to that feature in an additive sense. No human-subject evaluation of interpretability was performed.

### 4.8 Agent methodology

The feedback agent is deterministic and retrieval-grounded. It is not an LLM, an autonomous agent, or a LangGraph system; the terminology reflects the deterministic routing of the implementation.

**Deterministic routing.** Queries are handled by deterministic logic rather than free-form generation, which bounds the agent's behaviour and supports the reliability evaluation.

**Retrieval corpus and retrieval process.** The agent retrieves relevant evidence from a curated corpus of vehicle and prediction context. Retrieval follows a documented retrieval process; the corpus and the queries differ between the M2 and M5 retrieval evaluations and are therefore reported separately.

**Available tools.** The agent has access to tools that expose prediction and vehicle context, such as the current fuel-rate prediction, vehicle state, and relevant contextual facts.

**Response construction.** Responses are constructed from the retrieved evidence under deterministic rules. When the evidence is insufficient to support a statement, the agent abstains rather than speculating.

**Abstention and safety controls.** The agent is designed to abstain in situations where the evidence does not support the response; a dedicated safety and abstention benchmark measures this behaviour.

**Benchmark construction and manual evaluation criteria.** The functional benchmark is manually authored and contains a labelled set of queries with expected answers. The safety and abstention benchmark contains adversarial or insufficient-evidence queries. Evaluation criteria are applied manually; the benchmarks are small and are labelled as manual.

### 4.9 Robustness and ablation methodology

- **Cross-truck paired comparison.** Full-operational and telematics-only models are compared on the held-out trucks using a paired design. The comparison is reported as a difference with a bootstrap interval.
- **Trip bootstrap.** The cross-truck difference is accompanied by a trip-level bootstrap interval. In the observed results the interval includes zero, which is reported accurately: the evidence does not establish a statistically reliable difference between the feature sets in predictive error.
- **Input perturbations.** Perturbation tests evaluate the sensitivity of the predictions to changes in input telemetry. The results are summarised in the robustness reports.
- **Retrieval evaluation.** Retrieval is evaluated separately in the M2 and M5 phases with different queries and configurations; the two are not pooled.
- **Agent ablation.** The M5.5 ablation is an ongoing evidence stream. Its artifacts exist but the milestone is not formally closed as validation. The available ablation evidence is described with explicit verification caveats and is not presented as a completed validation.
- **Limitations of current M5.5 evidence.** Because M5.5 is not closed, conclusions drawn from it are provisional.

## 5. Experimental Results

### 5.1 Dataset and evaluation overview

The evaluation is conducted entirely on the synthetic dataset described in Section 4.2. All numbers in this section are traceable to the source artifacts listed in Section 10 (evidence traceability). The test partition contains five previously unseen truck identities. Pooled metrics are computed over all test samples regardless of truck identity; macro-averaged metrics average the per-truck values. The MAPE filter (Section 4.6) retains coverage 0.99289 of the test set for MAPE.

### 5.2 Predictive performance

Table 1 reports the held-out test performance of the baselines and the XGBoost model, and of the full-operational and telematics-only feature sets.

**Table 1. Held-out test predictive performance (pooled), synthetic test set.**

| Model / feature set              | MAE (L/h) | RMSE (L/h) | R²     | MAPE (%) |
| -------------------------------- | --------- | ---------- | ------ | -------- |
| Random forest (full-operational) | 2.337     | 4.039      | 0.9903 | 6.89     |
| Ridge (full-operational)         | 3.861     | 5.470      | 0.9822 | 66.00    |
| XGBoost, full-operational (15)   | 2.568     | 4.527      | 0.9878 | 7.38     |
| XGBoost, telematics-only (11)    | 2.566     | 4.371      | 0.9886 | 15.36    |

Values are transcribed directly from `reports/m1/test_metrics_report.json` and `reports/m1/metrics_report.json`. Where a value is not recorded in the source artifact, the cell is marked unavailable rather than estimated. The validation and test partitions are different evaluation populations, so validation values (not shown) and test values for the same model are not directly comparable.

The feature-set labels in the table refer to the logical predictor counts (15 for full-operational, 11 for telematics-only). After one-hot encoding of the categorical predictor `vehicle_class` with `drop='first'`, the corresponding model input matrices contain 16 and 12 post-encoding columns respectively (Section 4.4).

### 5.3 Cross-truck generalization

Table 2 reports the per-truck results on the held-out test set. Because the five-truck test set is small, per-truck comparisons are descriptive, not inferential.

**Table 2. Per-truck held-out results (XGBoost, full-operational), five test trucks.**

| Truck           | MAE (L/h) | RMSE (L/h) | R²     | Samples |
| --------------- | --------- | ---------- | ------ | ------- |
| truck_001 (32T) | 1.302     | 2.123      | 0.9961 | 30,817  |
| truck_004 (32T) | 1.549     | 2.490      | 0.9950 | 32,480  |
| truck_017 (40T) | 2.245     | 3.667      | 0.9924 | 37,073  |
| truck_020 (40T) | 3.866     | 6.304      | 0.9795 | 33,901  |
| truck_028 (49T) | 3.597     | 5.917      | 0.9831 | 39,018  |
| Macro average   | 2.512     | 4.100      | 0.9893 | —       |

The macro-averaged performance across the five trucks is comparable to the pooled performance (Section 5.2); with n=5 trucks, this macro average is descriptive, not inferential. The variance decomposition reported in the cross-truck generalization summary separates variance attributable to truck identity from residual variance; the truck-identity contribution is modest on the synthetic set. The trip-bootstrap interval for the full-versus-telematics MAE difference includes zero, which is reported accurately: the evidence does not establish a statistically reliable difference between the feature sets.

### 5.4 Operating-condition and error analysis

Table 3 summarises performance by operating condition.

**Table 3. Operating-condition and error analysis.**

| Condition / subpopulation | MAE (L/h)   | RMSE (L/h)  | R²          | Coverage / samples |
| ------------------------- | ----------- | ----------- | ----------- | ------------------ |
| Highway (speed ≥ 60 km/h) | 3.658       | 5.336       | 0.9853      | 24,445             |
| City (speed 20–40 km/h)   | 1.945       | 4.026       | 0.9840      | 41,069             |
| High-acceleration (≥0.5)  | 6.555       | 8.002       | 0.8773      | 30,201             |
| Heavy-payload regime      | unavailable | unavailable | unavailable | absent (n=0)       |
| MAPE-filtered subset      | —           | —           | —           | 0.99289 coverage   |

The heavy-payload regime is absent from the evaluated synthetic data (Section 4.2), so no heavy-payload performance is claimed. The error-diagnosis summary reports error patterns concentrated at low fuel-rate values and at transitions; these patterns are descriptive of the synthetic regime.

### 5.5 Explainability results

Table 4 reports the SHAP findings. Global feature importance is summarised by mean absolute SHAP value; the instance-level results are validated by reconstruction.

**Table 4. SHAP findings (full-operational model, synthetic test set).**

| Result                                     | Value / finding                                              | Source                                  |
| ------------------------------------------ | ------------------------------------------------------------ | --------------------------------------- |
| Top global features (full)                 | torque, engine load, engine speed (ranked by mean abs. SHAP) | `reports/m1/explainability_report.json` |
| Reconstruction error (full)                | max abs. dev. 1.57e-4; mean 2.12e-5; fraction within tol 1.0 | `reports/m1/explainability_report.json` |
| Reconstruction error (telematics)          | max abs. dev. 1.44e-4; mean 2.09e-5; fraction within tol 1.0 | `reports/m1/explainability_report.json` |
| Explanation structure, full vs. telematics | materially different                                         | `reports/m1/explainability_report.json` |

The reconstruction check confirms that the SHAP values are additively consistent with the model predictions: the sum of attributions plus the expected value reconstructs the prediction within a small error. Reconstruction validates additive consistency, not human interpretability or causal validity. The full-operational and telematics-only feature sets yield materially different attribution structures despite similar predictive error, which is interpreted as the two sets encoding different predictive mechanisms. No causal claims are made.

### 5.6 Agent retrieval and response evaluation

Retrieval results are reported separately for the M2.1, M2.1-B, and M5.2 phases because they use different query sets, gold standards, and configurations; they are not pooled. In particular, M2.1 (55 topic-level queries), M2.1-B (28 diverse document-level queries, 24 answerable), and M5.2 (26 queries: 19 answerable, 7 unanswerable) are distinct evaluations over the same synthetic corpus; the M2.1-B benchmark uses content-based relevance labels and reports metrics over answerable queries only, and M5.2 reports retrieval metrics over answerable queries with abstention assessed on the unanswerable ones.

**Table 5. Retrieval evaluation (M2.1, M2.1-B, M5.2 — reported separately).**

| Phase  | Metric      | Value  | Source                                       |
| ------ | ----------- | ------ | -------------------------------------------- |
| M2.1   | Recall@1    | 0.2455 | `reports/m2/retrieval_evaluation_summary.md` |
| M2.1   | Recall@5    | 0.5000 | `reports/m2/retrieval_evaluation_summary.md` |
| M2.1   | Precision@1 | 0.4909 | `reports/m2/retrieval_evaluation_summary.md` |
| M2.1   | MRR         | 0.6182 | `reports/m2/retrieval_evaluation_summary.md` |
| M2.1   | nDCG@5      | 0.7102 | `reports/m2/retrieval_evaluation_summary.md` |
| M2.1-B | Recall@1    | 0.8333 | `reports/m2/benchmark_v2/summary.md`         |
| M2.1-B | Recall@3    | 0.9375 | `reports/m2/benchmark_v2/summary.md`         |
| M2.1-B | Recall@5    | 0.9792 | `reports/m2/benchmark_v2/summary.md`         |
| M2.1-B | Precision@1 | 1.0000 | `reports/m2/benchmark_v2/summary.md`         |
| M2.1-B | Precision@3 | 0.4028 | `reports/m2/benchmark_v2/summary.md`         |
| M2.1-B | Precision@5 | 0.2583 | `reports/m2/benchmark_v2/summary.md`         |
| M2.1-B | MRR         | 1.0000 | `reports/m2/benchmark_v2/summary.md`         |
| M2.1-B | nDCG@5      | 0.9692 | `reports/m2/benchmark_v2/summary.md`         |
| M5.2   | Recall@1    | 0.8158 | `reports/m5/m5_2/summary.md`                 |
| M5.2   | Recall@5    | 0.9737 | `reports/m5/m5_2/summary.md`                 |
| M5.2   | Precision@1 | 1.0000 | `reports/m5/m5_2/summary.md`                 |
| M5.2   | MRR         | 1.0000 | `reports/m5/m5_2/summary.md`                 |
| M5.2   | nDCG@5      | 0.9667 | `reports/m5/m5_2/summary.md`                 |

Because the three phases use different query sets, gold standards, and evaluation populations, they are not directly comparable and no combined retrieval figure is reported. The retrieval results support the finding that the corpus and retrieval process return relevant evidence under the tested queries, but the evaluation is limited to the documented query sets.

### 5.7 Agent functional and safety evaluation

Table 6 reports the agent evaluation results.

**Table 6. Agent functional and safety evaluation.**

| Benchmark                     | Result                                                                                                 | Source                                  |
| ----------------------------- | ------------------------------------------------------------------------------------------------------ | --------------------------------------- |
| Functional (M2.4)             | 38/38 positives passed; 11/11 negative controls detected (49 scenarios)                                | `reports/m2/agent_evaluation_report.md` |
| Driver-feedback rubric (M5.4) | Overall 92.0% (185/201 applicable); safety sub-dimension 46/46 (100%), zero safety-critical violations | `reports/m5/m5_4/summary.md`            |

These evaluations are based on manually authored benchmarks and manual evaluation criteria. They do not constitute a human-subject evaluation of the agent's output or a claim of general safe behaviour outside the tested benchmark. The M5.4 overall result is computed as pooled correct over applicable items and is not an average of dimension scores.

### 5.8 Robustness and ablation results

Table 7 summarises the robustness and ablation evidence as its status.

**Table 7. Robustness and ablation evidence status.**

| Experiment                    | Evidence status | Caveat                                         |
| ----------------------------- | --------------- | ---------------------------------------------- |
| Cross-truck paired comparison | available       | difference interval includes zero              |
| Trip bootstrap                | available       | interval includes zero; no reliable difference |
| Input perturbations           | available       | descriptive of synthetic regime                |
| Retrieval evaluation          | available       | M2.1 and M5.2 separate                         |
| Agent ablation (M5.5)         | provisional     | milestone not formally closed                  |

M5.5 remains formally incomplete. Its available artifacts are described here with explicit verification caveats and are not presented as a closed validation.

### 5.9 System integration results

Table 8 reports the integration verification status.

**Table 8. Integration verification status (M6).**

| Component        | Verification | Method                                  |
| ---------------- | ------------ | --------------------------------------- |
| API              | pass         | automated component/HTTP tests          |
| Component wiring | pass         | automated integration tests             |
| Live browser     | partial      | M6.4 negative paths not all live-tested |
| Demo video       | incomplete   | M6.5 recording not finished             |

The M6.1–M6.4 evidence distinguishes automated component and API verification from live-browser verification. The M6.4 negative paths were verified primarily through mocked component/API tests; not all negative paths were live-browser tested. M6.5 video recording remains incomplete; the complete demo evidence package is not finished. The project has not been validated in commercial operations.

## 6. Discussion

### 6.1 RQ1 — Cross-truck prediction

**What the evidence supports.** In this synthetic evaluation, the XGBoost model predicts instantaneous fuel rate on the five held-out trucks with a pooled MAE of 2.57 L/h and R² of 0.988, and comparable macro-averaged performance across trucks. This demonstrates that the truck-wise holdout protocol is operational and that the model transfers to unseen truck identities within the synthetic distribution.

**What it does not support.** The five-truck test set does not support claims of broad fleet generalization, and the synthetic data do not support real-world predictive validation. The available evidence does not establish that a model trained on a known fleet would generalise to real operational trucks.

**Significance.** The protocol provides a sound template for evaluating cross-truck generalization before real-world data are available; the numerical accuracy is limited to the synthetic domain and should not be extrapolated to operations.

### 6.2 RQ2 — Feature-set comparison

**What the evidence supports.** The full-operational and telematics-only feature sets achieve near-identical pooled error (both MAE 2.57 L/h). The evidence supports the finding that, within this synthetic evaluation, removing the four engine-internal signals does not measurably change predictive error.

**What it does not support.** Similar error does not establish statistical equivalence or operational interchangeability. The trip-bootstrap interval for the difference includes zero, so the evidence does not establish a reliable difference, and it equally does not establish equivalence. The two feature sets also produce materially different SHAP structures, which the evidence treats as different mechanisms, not equivalent behaviour.

**Significance.** The finding highlights that equal accuracy can coexist with different explanation structure, which matters when models are used for feedback and diagnosis rather than purely for prediction.

### 6.3 RQ3 — Explainability

**What the evidence supports.** In this synthetic evaluation, SHAP provides instance-level attributions whose additive reconstruction is confirmed within a small error. The evidence supports the additive consistency of the attributions.

**What it does not support.** Reconstruction does not establish human interpretability or causal validity. The results do not support causal claims about which features drive fuel use.

**Significance.** Additive consistency is a necessary but not sufficient property of trustworthy attribution; it supports the use of SHAP for explanation without supporting causal interpretation.

### 6.4 RQ4 — Agent reliability

**What the evidence supports.** On the manual functional benchmark (M2.4), the deterministic agent passes all 38 positive scenarios and detects all 11 negative controls. On the driver-feedback rubric (M5.4), it answers 92.0% (185/201) of applicable criteria correctly, including 46/46 (100%) on the safety sub-dimension with zero safety-critical violations. The evidence supports the claim that, within the tested benchmarks, the agent abstains rather than offering unsupported advice.

**What it does not support.** The benchmarks are small, manually authored, and manually evaluated. The evidence does not support a human-subject evaluation of the agent, a claim of safe behaviour outside the tested benchmark, or any characterisation of the agent as an LLM, autonomous agent, or LangGraph system.

**Significance.** The deterministic design and abstention guardrail are consistent with concerns from the RAG literature (Chen et al., 2024) about negative rejection, and the benchmark provides a repeatable, if preliminary, evaluation instrument.

### 6.5 Comparison with related work

The results are compared with the surveyed literature without claiming superiority. The reference study (Gavimath and Ravuri, 2025) is route-matched and does not test unseen-truck transfer; the present study's truck-wise holdout addresses a complementary question but on synthetic data. Fleet-level prediction studies (Zhao et al., 2023; Zhu et al., 2024; Liu et al., 2025) report high within-population accuracy and use SHAP for attribution; the present study contributes a strict leave-trucks-out protocol and additive-consistency validation, but its data are synthetic.

The 2026 Volvo fleet coaching study (Karlsson et al., 2026) combines telemetry-based prediction, SHAP, RAG, and conversational coaching on a real fleet using an LLM-based coach. The present study is not the first to combine these technologies. The two approaches differ fundamentally: the 2026 study uses real fleet data and conversational LLM coaching, whereas the present study uses synthetic instantaneous telemetry and a deterministic retrieval-grounded agent. Consequently, the present results are not comparable evidence for the 2026 study's claims, and the present contribution is the synthetic cross-truck pipeline validation, not a real-world combined-system validation.

## 7. Limitations and Threats to Validity

1. **Synthetic data.** All telemetry and target labels are formula-generated. High synthetic-data R² is not evidence of real-world generalization, and no real-world predictive validation is claimed.
2. **Small test set.** The held-out set contains only five truck identities, insufficient to claim broad fleet generalization; per-truck comparisons are descriptive.
3. **Feature-set similarity.** Similar full-versus-telematics MAE does not establish statistical equivalence or operational interchangeability; the bootstrap interval includes zero.
4. **SHAP interpretation.** SHAP values are additive attributions, not causal effects; reconstruction validates consistency, not human interpretability.
5. **Manual agent evaluation.** The agent benchmarks are small and manually authored; there is no human-subject evaluation, and no general safety claim is made.
6. **Incomplete milestones.** M5.5 ablation remains formally incomplete, and M6.5 video recording is not finished. Neither is presented as closed.
7. **Integration verification.** M6.4 negative paths were not all live-browser tested; mocked component/API tests are distinguished from live UI checks.
8. **Heavy-payload regime.** The regime is absent from the synthetic data; no heavy-payload performance is claimed.
9. **No commercial validation.** The project has not been validated in commercial operations.

## 8. Implications and Future Work

**Implications.** The truck-wise holdout protocol offers a reproducible template for assessing cross-truck generalization before real-world data exist. The feature-set result cautions that accuracy and explanation structure are independent axes, relevant when models drive feedback and diagnosis. The additive-consistency validation provides a concrete check for SHAP use in safety-relevant settings. The deterministic, abstention-oriented agent design is a practical approach to reliable feedback in the absence of unconstrained generation.

**Future work.** (i) Evaluate on real-world fleet telemetry when available, preserving the same truck-wise protocol. (ii) Expand the held-out test set to more truck identities. (iii) Include the heavy-payload regime in the data. (iv) Conduct a human-subject evaluation of agent output and SHAP interpretability. (v) Close the M5.5 ablation and complete the M6.5 demo evidence package. (vi) Investigate whether the feature-set explanation differences persist on real data.

## 9. Conclusion

This study developed and evaluated a complete pipeline for heavy-duty fuel-rate prediction and driver feedback on synthetic telemetry. The XGBoost model achieved an MAE of 2.57 L/h and an R² of 0.988 on five held-out synthetic trucks; the telematics-only set matched this error while yielding a different explanation structure; SHAP attributions passed additive reconstruction checks; and the deterministic retrieval-grounded agent passed 38/38 functional scenarios, detected 11/11 negative controls, and scored 92.0% (185/201) overall with a 46/46 (100%) safety sub-dimension on manually authored benchmarks.

All results are confined to the synthetic evaluation and must not be read as real-world validation. The principal contribution is a reproducible, integrity-preserving framework for assessing cross-truck generalization, feature-set differences, attribution consistency, and agent reliability. The study is not the first to combine prediction, SHAP, RAG, and driver feedback, and it claims no real-world or operational validation. M5.5 and M6.5 remain incomplete and are not presented as closed.

## 10. References

1. Barbado, A., & Corcho, O. (2022). Interpretable Machine Learning Models for Predicting and Explaining Vehicle Fuel Consumption Anomalies. _Engineering Applications of Artificial Intelligence_, 115, 105222. https://doi.org/10.1016/j.engappai.2022.105222
2. Chen, J., Lin, H., Han, X., & Sun, L. (2024). Benchmarking Large Language Models in Retrieval-Augmented Generation. _Proceedings of the AAAI Conference on Artificial Intelligence_, 38(16), 17754–17762. https://doi.org/10.1609/aaai.v38i16.29728
3. Gavimath, V. V., & Ravuri, S. (2025). Developing Real-World Drive Cycles for Heavy-Duty Freight Vehicles Using On-Board Diagnostics Data. _Transportation Research Procedia_, 82, 158–174. https://doi.org/10.1016/j.trpro.2024.12.035
4. Karlsson, J., Lundberg, K., & Liu, H. X. (2026). An End-to-End Conversational AI Coach for Heavy-Duty Eco-Driving: An Industrial Case Study. _Transportation Research Interdisciplinary Perspectives_, 40, 102270. https://doi.org/10.1016/j.trip.2026.102270
5. Liu, X., et al. (2025). CatBoost-Based Fuel Consumption Modeling and Explainable Analysis for Heavy-Duty Diesel Trucks: Impact of Engine, Driving Behavior, and Vehicle Weight. _Energy_, 333, 137485. https://doi.org/10.1016/j.energy.2025.137485
6. Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. _Advances in Neural Information Processing Systems_, 30, 4765–4774.
7. Zhao, Z., et al. (2023). A Review of the Data-Driven Prediction Method of Vehicle Fuel Consumption. _Energies_, 16(14), 5258. https://doi.org/10.3390/en16145258
8. Zhu, J., et al. (2024). Research on the Prediction and Influencing Factors of Heavy Duty Truck Fuel Consumption Based on LightGBM. _Energy_, 296, 131221. https://doi.org/10.1016/j.energy.2024.131221

### Evidence traceability

The numbers in Section 5 trace to the following source artifacts: `reports/m1/test_metrics_report.json`, `reports/m1/metrics_report.json`, `reports/m1/explainability_report.json`, `reports/m3/cross_truck_generalization_summary.md`, `reports/m3/condition_robustness_summary.md`, `reports/m3/telemetry_robustness_summary.md`, `reports/m3/error_diagnosis_summary.md`, `reports/m2/agent_evaluation_report.md`, `reports/m2/retrieval_evaluation_summary.md`, `reports/m5/m5_2/summary.md`, `reports/m5/m5_4/summary.md`, `reports/m5/m5_5/summary.md`, `reports/m4/driver_interaction_summary.md`, `docs/REPRODUCIBILITY_RESEARCH.md`, and `evidence/m6_5_demo/index.md`.
