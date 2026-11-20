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

The experimental framework uses a reproducibly generated synthetic heavy-vehicle telemetry dataset. Evaluation is organized around truck-wise separation to examine model performance on vehicles excluded from training.

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
              | Random Forest / Ridge    |
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

The experiments use a reproducibly generated synthetic heavy-vehicle telemetry dataset.

### Dataset characteristics

| Component | Quantity |
|---|---:|
| Telemetry records | 963,697 |
| Telemetry columns | 24 |
| Truck profiles | 29 |
| Trips | 540 |
| Agent evaluation-query records | 550 |
| Knowledge documents | 21 |
| Training / validation / test trucks | 20 / 4 / 5 |

### Truck-wise partitioning

The dataset is separated by truck identity rather than randomly splitting individual telemetry records.

| Partition | Trucks | Purpose |
|---|---:|---|
| Training | 20 | Model fitting |
| Validation | 4 | Model selection |
| Testing | 5 | Held-out vehicle evaluation |

This prevents records from the same truck from appearing across training and test partitions.

The five held-out test trucks contain **173,289 telemetry records across 96 trips**.

### Feature configurations

Two feature configurations are evaluated.

**Full operational feature set**

Includes vehicle class, gross vehicle weight, payload, vehicle mass, speed, acceleration, road grade, ignition state, and engine operating parameters.

**Telematics-only feature set**

Excludes four engine-internal signals:

- Engine RPM
- Engine load
- Throttle position
- Engine torque

Target-derived quantities are excluded from both feature configurations.

The configurations contain 15 and 11 logical predictors, respectively. Categorical encoding produces 16 and 12 model-input columns.

## 5. Machine Learning Methodology

The prediction task is formulated as supervised regression.

The primary evaluated model is **XGBoost Regressor**, alongside Random Forest and Ridge Regression baselines.

### Selected XGBoost configuration

| Parameter | Value |
|---|---:|
| Estimators | 300 |
| Maximum depth | 6 |
| Learning rate | 0.1 |
| Subsample | 0.8 |
| Column subsample | 0.8 |
| Tree method | `hist` |
| Random state | 20240617 |

Model selection uses validation-set MAE. Final models are fitted on the training partition, and the held-out test partition is reserved for evaluation.

### Evaluation metrics

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)
- Mean Absolute Percentage Error (MAPE)

MAPE is evaluated on observations where actual fuel rate is greater than 1.0 L/h to avoid unstable percentage errors at very low target values.

The retained MAPE evaluation population contains **172,057 of 173,289 test observations**, giving approximately **99.29% coverage**.

## 6. Experimental Results

### 6.1 Model comparison

The following results are obtained on the held-out test partition.

| Model | Feature set | MAE (L/h) | RMSE (L/h) | R² | MAPE |
|---|---|---:|---:|---:|---:|
| Random Forest | Full | 2.337220 | 4.038898 | 0.990282 | 6.886% |
| Ridge Regression | Full | 3.861351 | 5.469660 | 0.982177 | 65.9996% |
| XGBoost | Full | 2.568427 | 4.527482 | 0.987788 | 7.3793% |
| XGBoost | Telematics-only | 2.565960 | 4.371276 | 0.988616 | 15.3600% |

The two XGBoost feature configurations produce similar pooled absolute-error performance. Their relative-error behavior differs, with the telematics-only configuration exhibiting higher MAPE.

The Random Forest baseline achieves lower pooled MAE and RMSE than the selected XGBoost configuration. The results are therefore reported descriptively rather than treating XGBoost as universally superior.

### 6.2 Cross-truck evaluation

The test set contains five trucks excluded from model training and validation.

| Truck | MAE (L/h) | RMSE (L/h) | R² | Samples |
|---|---:|---:|---:|---:|
| truck_001 | 1.302266 | 2.122795 | 0.996132 | 30,817 |
| truck_004 | 1.549199 | 2.489746 | 0.995033 | 32,480 |
| truck_017 | 2.245159 | 3.667431 | 0.992442 | 37,073 |
| truck_020 | 3.866037 | 6.304166 | 0.979545 | 33,901 |
| truck_028 | 3.596619 | 5.916755 | 0.983146 | 39,018 |
| **Macro average** | **2.511856** | **4.100179** | **0.989260** | **5 trucks** |

The per-truck results demonstrate variation in prediction error across held-out vehicle identities. The macro average represents an equally weighted descriptive summary of the five test trucks, not a pooled metric.

### 6.3 Operating-condition analysis

Performance is also examined across selected operating conditions.

| Condition | MAE (L/h) | RMSE (L/h) | R² |
|---|---:|---:|---:|
| Highway (speed ≥ 60 km/h) | 3.657974 | 5.335614 | 0.985333 |
| City speed bin (20–40 km/h) | 1.945111 | 4.025995 | 0.984037 |
| High acceleration (≥ 0.5) | 6.555219 | 8.002232 | 0.877251 |

The condition-level analysis highlights that pooled performance does not fully describe model behavior in different operating regimes.

High-acceleration observations show substantially higher prediction error than the pooled test result. This reinforces the importance of condition-specific evaluation.

Heavy-payload performance is not reported because the test data does not contain observations above the defined heavy-payload threshold.

## 7. Explainability

The framework integrates SHAP-based instance-level explanations with fuel-rate predictions.

For an individual prediction, the explanation layer provides:

- The model's expected value.
- Feature-level attribution values.
- Positive and negative contributions.
- Reconstruction of the model output from the SHAP decomposition.

The interactive cockpit has been verified to display a prediction alongside its SHAP explanation. In the evaluated cockpit scenario, the maximum observed reconstruction deviation was approximately **0.00007**.

SHAP values describe model attribution, not physical causality. They should not be interpreted as proof that changing a particular variable will produce a specified real-world fuel saving.

## 8. Retrieval-Grounded Agentic Framework

A deterministic tool-using agent connects the prediction, explanation, and knowledge retrieval components.

### Core capabilities

**1. Fuel-rate prediction**

Routes validated prediction requests to the saved machine-learning model.

**2. Prediction explanation**

Retrieves instance-level SHAP attributions and produces technical or simplified explanations.

**3. Technical knowledge retrieval**

Uses an embedding-based retrieval pipeline to identify relevant documents from the project's knowledge corpus.

**4. Response validation**

Applies structured checks to prevent unsupported claims, inappropriate certainty, and unsafe interpretations.

### Retrieval evaluation — M2.1-B

A separate document-level benchmark evaluates retrieval using 28 unique queries, including answerable and unanswerable cases.

| Metric | Result |
|---|---:|
| Recall@1 | 0.8333 |
| Recall@3 | 0.9375 |
| Recall@5 | 0.9792 |
| Precision@1 | 1.0000 |
| Precision@3 | 0.4028 |
| Precision@5 | 0.2583 |
| MRR | 1.0000 |
| nDCG@5 | 0.9692 |

These results belong to the M2.1-B benchmark and should not be conflated with the original 55-query topic-level evaluation or the later M5.2 benchmark.

### Agent evaluation — M2.4

The M2.4 evaluation contains 49 scenarios, comprising 38 positive scenarios and 11 negative controls.

| Evaluation category | Result |
|---|---:|
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
|---|---:|
| Overall applicable criteria | 185/201 |
| Overall score | 92% |
| Evidence-uncertainty subdimension | 52/52 |
| Safety subdimension | 46/46 |
| Safety-critical violations under tested checks | 0 |

The zero-violation result applies only to the defined evaluation checks. It is not a universal safety guarantee or independent safety certification.

## 9. Interactive Application

The research framework is integrated into an interactive application that exposes its principal analytical capabilities.

### Application functionality

- Fuel-rate prediction cockpit.
- Vehicle operating-input controls.
- Full and telematics-only prediction workflows.
- SHAP-based explanation panels.
- Prediction history.
- Telemetry replay interface.
- Replay-sample transfer into prediction inputs.
- Retrieval-grounded assistant interaction.
- Input validation and error handling.

The replay-to-prediction transfer mechanism is implemented and covered by automated tests using mocked telemetry points. A live screenshot also demonstrates a sample transfer. The complete canonical sample-index-to-prediction chain has not been demonstrated in one end-to-end evidence artifact.

The application connects analytical tools to a user-facing interface rather than limiting the work to offline notebooks.

## 10. Technology Stack

| Layer | Technologies |
|---|---|
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

- A fixed synthetic-data generation seed.
- A documented truck-wise train/validation/test split.
- Explicit model configurations.
- Saved model artifacts.
- Defined evaluation metrics.
- Per-truck and operating-condition evaluation.
- Automated tests for key analytical and application components.
- Version-controlled research documentation.

The reported results should be interpreted within the exact experimental configuration and data-generation process used in this study.

## 13. Research Limitations

The following limitations define the scope of the conclusions:

1. **Synthetic experimental data:** The reported prediction metrics characterize the generated experimental dataset and do not establish real-world fleet prediction accuracy.
2. **Limited held-out vehicles:** Cross-truck testing includes five held-out truck identities.
3. **Feature-distribution limitations:** Heavy-payload observations are absent under the defined threshold.
4. **Condition-dependent performance:** Prediction errors vary across operating conditions, and pooled metrics can conceal these differences.
5. **Explainability scope:** SHAP provides model-level attribution for individual predictions, not causal explanations.
6. **Deterministic agent architecture:** The agent uses rule-based routing and predefined response generation rather than claiming autonomous LLM reasoning.
7. **Benchmark-specific agent evaluation:** Agent reliability results are based on manually authored scenarios and defined evaluation criteria, not a large-scale human-subject study.
8. **Application evidence scope:** Some application workflows have automated or component-level evidence rather than a single complete end-to-end live-browser recording.

These limitations are important when interpreting the experimental findings and identifying future research directions.

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

- Evaluation using independently collected real-world heavy-vehicle telemetry.
- Broader cross-fleet and cross-region generalization studies.
- More extensive payload and road-condition coverage.
- Temporal and sequence-based prediction architectures.
- Calibration and uncertainty estimation.
- Human evaluation of driver-facing explanations.
- Longitudinal evaluation of feedback usefulness.
- More extensive end-to-end deployment and operational testing.

## 16. Research Paper

The complete research manuscript is maintained in:

**[`docs/M7_RESEARCH_PAPER.md`](docs/M7_RESEARCH_PAPER.md)**

It contains the research framework, literature review, methodology, experimental results, discussion, limitations, and references.

---

## Author

**Ansh**

Research interests:

- Applied Machine Learning
- Explainable Artificial Intelligence
- Retrieval-Augmented Generation
- Agentic AI Systems
- Heavy-Vehicle Analytics
- Predictive Modeling

**Repository:** [Freight-fuel-agentic-research](https://github.com/anshh-py/Freight-fuel-agentic-research)

---

*This repository documents an experimental research and engineering framework. Reported metrics apply to the defined synthetic evaluation environment and should not be interpreted as demonstrated real-world fleet performance.*
