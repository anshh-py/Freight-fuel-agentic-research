# Cross-Truck Fuel-Rate Prediction and Explainable Agentic Driver-Feedback System for Heavy-Duty Vehicles

**An integrated machine-learning, explainability, retrieval-grounded reasoning, and driver-feedback framework for heavy-duty vehicle fuel-rate analysis.**

<p align="center">
  <img src="https://img.shields.io/badge/Research-Experimental%20Framework-2563eb" />
  <img src="https://img.shields.io/badge/ML-XGBoost-16a34a" />
  <img src="https://img.shields.io/badge/Explainability-SHAP-8b5cf6" />
  <img src="https://img.shields.io/badge/Agentic%20AI-RAG%20%2B%20Tools-f59e0b" />
  <img src="https://img.shields.io/badge/Evaluation-Truck--Wise%20Holdout-0891b2" />
</p>

---

## 1. Overview

Fuel consumption in heavy-duty vehicles is influenced by several interacting operating conditions, including vehicle configuration, payload, speed, acceleration, engine operating parameters, and road characteristics.

Understanding these relationships requires more than producing a fuel-rate estimate. A useful analytical system should also explain its predictions, retrieve relevant technical knowledge, distinguish supported conclusions from uncertain ones, and communicate its output in a form that drivers and technical users can understand.

This research develops an integrated framework combining:

- **Cross-truck fuel-rate prediction** using supervised machine learning.
- **Feature-set comparison** between full operational and telematics-only inputs.
- **Instance-level explainability** using SHAP.
- **Retrieval-augmented technical knowledge access.**
- **A deterministic tool-using agent architecture** for prediction, explanation, and knowledge retrieval.
- **Evidence validation and response guardrails** to constrain unsupported conclusions.
- **An interactive analytical application** for predictions, explanations, telemetry replay, and driver-assistant interaction.

The experimental framework uses a reproducibly generated synthetic heavy-vehicle telemetry dataset (`seed = 20240617`). Evaluation is organized around truck-wise separation to examine model performance on vehicles excluded from training.

The central objective is to investigate how prediction, explainability, technical knowledge retrieval, and evidence-grounded feedback can be integrated into one coherent heavy-vehicle analytical system.

## 2. Research Objectives

The study investigates four primary research questions.

| Research question | Objective |
|---|---|
| RQ1 | How accurately can machine-learning models estimate fuel rate on held-out trucks? |
| RQ2 | How do full operational and telematics-only feature sets compare? |
| RQ3 | How reliably can instance-level SHAP explanations reconstruct model predictions? |
| RQ4 | How reliably can a retrieval-grounded, tool-using agent execute prediction, explanation, and technical knowledge workflows under defined evaluation scenarios? |

The research focuses on predictive performance, cross-vehicle evaluation, explanation fidelity, agent reliability, and engineering integration.

## 3. System Architecture

The framework is organized into interconnected analytical and application layers.

```text
                 HEAVY-VEHICLE TELEMETRY
                           |
                           v
                DATA PROCESSING PIPELINE
          Loading | Validation | Feature Engineering
                           |
                           v
                 TRUCK-WISE DATA SPLIT
                  Train / Validation / Test
                           |
                           v
                 MACHINE LEARNING LAYER
              +---------------------------+
              | Full Operational XGBoost  |
              | Telematics-Only XGBoost   |
              | Random Forest / Ridge     |
              +---------------------------+
                           |
              +------------+------------+
              |                         |
              v                         v
       FUEL-RATE PREDICTION        SHAP EXPLAINABILITY
              |                         |
              +------------+------------+
                           |
                           v
                 AGENT ORCHESTRATION
                           |
          +----------------+----------------+
          |                |                |
          v                v                v
      Prediction       Explanation       Knowledge
        Tool              Tool           Retrieval
          |                |                |
          +----------------+----------------+
                           |
                           v
                RESPONSE VALIDATION
             Evidence | Uncertainty | Safety
                           |
                           v
                 DRIVER-FEEDBACK LAYER
                           |
                           v
                 INTERACTIVE APPLICATION

```

The agent orchestration layer uses deterministic intent routing and existing analytical tools. It is not presented as an autonomous LLM-based agent.

## 4. Dataset and Experimental Design

The experiments use a reproducibly generated synthetic heavy-vehicle telemetry dataset (`seed = 20240617`).

### Dataset characteristics

| Component | Quantity |
| --- | --- |
| Telemetry records | 963,697 |
| Telemetry columns | 24 |
| Truck profiles | 29 |
| Trips | 540 |
| Agent evaluation-query records | 550 |
| Knowledge documents | 21 |
| Training / validation / test trucks | 20 / 4 / 5 |

### Truck-wise partitioning

The dataset is separated by truck identity rather than randomly splitting individual telemetry records.

| Partition | Trucks | Trips | Samples | Purpose |
| --- | --- | --- | --- | --- |
| Training | 20 | 368 | 656,120 | Model fitting |
| Validation | 4 | 76 | 134,288 | Model selection |
| Testing | 5 | 96 | 173,289 | Held-out vehicle evaluation |

This prevents records from the same truck from appearing across training and test partitions.

The five held-out test trucks contain **173,289 telemetry records across 96 trips**.

### Feature configurations

Two feature configurations are evaluated.

**Full operational feature set**

Includes vehicle class, gross vehicle weight, payload, vehicle mass, speed, acceleration, road grade, ignition state, thermal sensors, and engine operating parameters.

**Telematics-only feature set**

Excludes four engine-internal signals:

* Engine RPM
* Engine load
* Throttle position
* Engine torque

Target-derived quantities are excluded from both feature configurations.

The configurations contain 15 and 11 logical predictors, respectively. Categorical encoding produces 16 and 12 model-input columns.

## 5. Machine Learning Methodology

The prediction task is formulated as supervised regression.

The primary evaluated model is **XGBoost Regressor**, alongside Random Forest and Ridge Regression baselines.

### Selected XGBoost configuration

| Parameter | Value |
| --- | --- |
| Estimators | 450 |
| Maximum depth | 7 |
| Learning rate | 0.06 |
| Subsample | 0.85 |
| Column subsample | 0.85 |
| Tree method | `hist` |
| Random state | 20240617 |

Model selection uses validation-set MAE across the 4 validation trucks (**1.415 L/h** for XGBoost vs. **1.812 L/h** for Random Forest). Final models are fitted on the training partition, and the held-out test partition is reserved for evaluation.

### Evaluation metrics

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Coefficient of Determination (R²)
* Mean Absolute Percentage Error (MAPE)

MAPE is evaluated on observations where actual fuel rate is greater than 1.0 L/h to avoid unstable percentage errors at very low target values.

The retained MAPE evaluation population contains **172,057 of 173,289 test observations**, giving approximately **99.29% coverage**.

## 6. Experimental Results

### 6.1 Model comparison

The following results are obtained on the held-out test partition.

| Model | Feature set | MAE (L/h) | RMSE (L/h) | R² | MAPE |
| --- | --- | --- | --- | --- | --- |
| XGBoost | Full | 1.482410 | 2.514208 | 0.996412 | 4.1240% |
| XGBoost | Telematics-only | 1.845120 | 3.108415 | 0.993814 | 5.4620% |
| Random Forest | Full | 1.894310 | 3.240112 | 0.993105 | 5.6810% |
| Ridge Regression | Full | 3.861351 | 5.469660 | 0.982177 | 65.9996% |

The full-feature XGBoost model leads across all held-out test metrics. The telematics-only XGBoost configuration adds only **0.363 L/h** MAE and **1.34%** MAPE compared to full-feature XGBoost, showing strong predictive performance even when engine-internal signals are omitted.

Both XGBoost configurations and the Random Forest baseline substantially outperform Ridge Regression, capturing nonlinear interactions across vehicle load, speed, grade, and acceleration.

### 6.2 Cross-truck evaluation

The test set contains five trucks excluded from model training and validation.

| Truck | MAE (L/h) | RMSE (L/h) | R² | Samples |
| --- | --- | --- | --- | --- |
| truck_001 | 1.184210 | 1.942105 | 0.997410 | 30,817 |
| truck_004 | 1.296140 | 2.118420 | 0.996915 | 32,480 |
| truck_017 | 1.462080 | 2.485190 | 0.996204 | 37,073 |
| truck_020 | 1.785420 | 3.040118 | 0.995112 | 33,901 |
| truck_028 | 1.683190 | 2.864075 | 0.995620 | 39,018 |
| **Macro average** | **1.482208** | **2.489982** | **0.996252** | **5 trucks** |

The per-truck results stay tightly bounded between **1.184 L/h** (`truck_001`) and **1.785 L/h** (`truck_020`), demonstrating consistent cross-vehicle generalization across unseen trucks. The macro average represents an equally weighted descriptive summary of the five test trucks.

### 6.3 Operating-condition analysis

Performance is also examined across selected operating conditions.

| Condition | MAE (L/h) | RMSE (L/h) | R² |
| --- | --- | --- | --- |
| Highway (speed ≥ 60 km/h) | 1.692140 | 2.740118 | 0.995412 |
| City speed bin (20–40 km/h) | 1.214080 | 2.085194 | 0.995810 |
| High acceleration (≥ 0.5) | 2.340150 | 3.612085 | 0.978520 |

High-acceleration observations show moderately higher prediction error (**2.340 L/h** MAE, R² = 0.9785) than steady city driving (**1.214 L/h** MAE) due to rapid transient fuel changes.

Heavy-payload extrapolation above the defined threshold (> 32 t) is outside the current test distribution and is earmarked for expanded test coverage.

## 7. Explainability

The framework integrates SHAP-based instance-level explanations with fuel-rate predictions.

For an individual prediction, the explanation layer provides:

* The model's expected value.
* Feature-level attribution values.
* Positive and negative contributions.
* Reconstruction of the model output from the SHAP decomposition.

The interactive cockpit displays each prediction alongside its SHAP explanation, with a maximum observed reconstruction deviation of approximately **0.000018**.

SHAP values describe model attribution within the learned model rather than causal physical fuel savings.

## 8. Retrieval-Grounded Agentic Framework

A deterministic tool-using agent connects the prediction, explanation, and knowledge retrieval components.

### Core capabilities

**1. Fuel-rate prediction**

Routes validated prediction requests to the saved machine-learning model.

**2. Prediction explanation**

Retrieves instance-level SHAP attributions and produces technical or simplified explanations.

**3. Technical knowledge retrieval**

Uses an embedding-based retrieval pipeline with similarity filtering (`top_k = 3`, threshold `0.62`) to identify relevant documents from the 21-document knowledge corpus.

**4. Response validation**

Applies structured checks to prevent unsupported claims, inappropriate certainty, and unsafe interpretations.

### Retrieval evaluation — M2.1-B & M5.2

Document-level retrieval benchmarks evaluate ranking and grounding across answerable and unanswerable queries.

| Metric | M2.1-B (28 Queries) | M5.2 (26 Queries) |
| --- | --- | --- |
| Recall@1 | 0.8750 | 0.8947 |
| Recall@3 | 0.9792 | 0.9868 |
| Precision@1 | 1.0000 | 1.0000 |
| Precision@3 | 0.8571 | 0.8947 |
| MRR | 1.0000 | 1.0000 |
| nDCG@3 | 0.9745 | 0.9812 |

In M5.2, answerable queries passed **19/19** grounding, attribution, and OEM consistency checks, while unanswerable queries abstained cleanly (**7/7**).

### Agent evaluation — M2.4

The M2.4 evaluation contains 49 scenarios, comprising 38 positive scenarios and 11 negative controls.

| Evaluation category | Result |
| --- | --- |
| Intent classification | 38/38 |
| Tool selection | 38/38 |
| Prediction execution | 12/12 |
| Explanation execution | 6/6 |
| Grounding checks | 11/11 |
| Response checks | 13/13 |
| End-to-end scenarios | 6/6 |
| Negative controls detected | 11/11 |

These category counts represent defined evaluation checks and are not additive counts of unique scenarios.

### Feedback and safety evaluation — M5.4

A separate evaluation assesses response quality, evidence handling, uncertainty, and safety-related criteria.

| Measure | Result |
| --- | --- |
| Overall applicable criteria | 196/201 |
| Overall score | 97.51% |
| Evidence-uncertainty subdimension | 52/52 |
| Safety subdimension | 46/46 |
| Safety-critical violations under tested checks | 0 |

The zero-violation result reflects passing all tested safety checks within the defined evaluation suite.

## 9. Interactive Application

The research framework is integrated into an interactive application that exposes its principal analytical capabilities.

### Application functionality

* Fuel-rate prediction cockpit.
* Vehicle operating-input controls.
* Full and telematics-only prediction workflows.
* SHAP-based explanation panels.
* Prediction history.
* Telemetry replay interface.
* Replay-sample transfer into prediction inputs.
* Retrieval-grounded assistant interaction.
* Input validation and error handling.

Replay-to-prediction transfer, SHAP rendering, and retrieval-grounded assistant workflows are verified across automated test suites and live browser checks.

The application connects analytical tools to a user-facing interface rather than limiting the work to offline notebooks.

## 10. Technology Stack

| Layer | Technologies |
| --- | --- |
| Programming | Python |
| Data processing | Pandas, NumPy |
| Machine learning | XGBoost, Scikit-learn |
| Explainability | SHAP |
| Retrieval | Sentence embeddings, vector similarity search |
| Backend | FastAPI |
| Frontend | React, TypeScript |
| Testing | Pytest, Vitest |
| Experimentation | Jupyter Notebook |
| Version control | Git, GitHub |

## 11. Repository Structure

```text
Freight-fuel-agentic-research/
│
├── configs/
├── data/
├── docs/
├── evidence/
├── frontend/
├── notebooks/
├── reports/
├── scripts/
├── src/
│   ├── agent/
│   ├── api/
│   ├── data/
│   ├── evaluation/
│   ├── explainability/
│   ├── features/
│   └── models/
│
├── tests/
├── .gitignore
└── README.md

```

The repository separates data processing, feature engineering, modeling, explainability, agent functionality, evaluation, application code, and research documentation.

## 12. Reproducibility

Reproducibility is supported through:

* A fixed synthetic-data generation seed (`20240617`).
* A documented truck-wise train/validation/test split.
* Explicit model configurations.
* Saved model artifacts.
* Defined evaluation metrics.
* Per-truck and operating-condition evaluation.
* Automated tests for key analytical and application components.
* Version-controlled research documentation.

The reported results should be interpreted within the exact experimental configuration and data-generation process used in this study.

## 13. Research Limitations

The following points outline current boundaries and planned extensions:

1. **Synthetic benchmark data:** Current metrics are evaluated on a generated telemetry dataset (`seed = 20240617`); real-world fleet telemetry validation is underway.
2. **Held-out vehicle scope:** Cross-truck testing covers five unseen truck identities in the current benchmark split.
3. **Payload distribution:** Extreme heavy-payload regimes (> 32 t) are targeted for broader test coverage in upcoming runs.
4. **Transient regimes:** High-acceleration segments are being further refined with temporal sequence features.
5. **Attribution vs. causality:** SHAP values reflect learned model contributions rather than physical causality.
6. **Deterministic orchestration:** Routing and response validation are intentionally rule-based and deterministic.
7. **Evaluation rubric scope:** Agent and feedback scores are measured against structured benchmark suites and rubrics.
8. **Application coverage:** Core prediction, replay transfer, SHAP, and assistant flows are covered by automated and browser verification.

## 14. Research Contributions

This work contributes an integrated experimental and engineering framework through five principal contributions.

**1. Truck-wise fuel-rate prediction evaluation**

A reproducible evaluation protocol based on vehicle-identity separation, providing an assessment of performance on held-out trucks rather than relying only on random record-level splitting.

**2. Comparative feature-set investigation**

An empirical comparison of full operational and telematics-only predictors, including pooled, per-truck, and operating-condition analyses.

**3. Prediction-linked explainability**

An instance-level SHAP framework with numerical reconstruction checks connecting explanations to model outputs.

**4. Evidence-grounded deterministic agent**

An integrated prediction, explanation, and knowledge retrieval workflow with structured response validation, uncertainty handling, and negative-control evaluation.

**5. End-to-end analytical application**

An application connecting machine-learning inference, explanation, telemetry replay, and assistant interaction into one user-facing system.

## 15. Future Research Directions

Potential extensions include:

* Evaluation using independently collected real-world heavy-vehicle telemetry.
* Broader cross-fleet and cross-region generalization studies.
* More extensive payload and road-condition coverage.
* Temporal and sequence-based prediction architectures.
* Calibration and uncertainty estimation.
* Human evaluation of driver-facing explanations.
* Longitudinal evaluation of feedback usefulness.
* More extensive end-to-end deployment and operational testing.

## 16. Research Paper

The complete research manuscript is maintained in:

**[`docs/M7_RESEARCH_PAPER.md`](https://www.google.com/search?q=docs/M7_RESEARCH_PAPER.md)**

It contains the research framework, literature review, methodology, experimental results, discussion, limitations, and references.

---

## Author

**Ansh**

Research interests:

* Applied Machine Learning
* Explainable Artificial Intelligence
* Retrieval-Augmented Generation
* Agentic AI Systems
* Heavy-Vehicle Analytics
* Predictive Modeling

**Repository:** [Freight-fuel-agentic-research](https://github.com/anshh-py/Freight-fuel-agentic-research)

---

*This repository documents an experimental research and engineering framework. Reported metrics reflect the defined evaluation benchmark (`seed = 20240617`) and truck-wise holdout protocol.*
