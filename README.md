# FEAD Cross-Truck Generalization Study

Research code and documentation for a study of **cross-truck generalization** on
FEAD engine management system (EMS) telemetry.

## Central research question

> How well does a fuel-rate prediction model trained on
> telemetry from a subset of trucks generalize to trucks it has never seen?

## Important framing (read before interpreting any result)

- The original FEAD paper describes **route-matched** training and evaluation:
  same truck, same route.
- **Unseen-truck evaluation is an extension of that work by us.** It is our own
  investigation, not an established or reproduced finding from FEAD.
- Any result in this repository about unseen trucks must be read as a provisional
  experiment, not as a validated claim.

## Evaluation protocols (kept strictly separate)

| Protocol                              | Role                 | Notes                                         |
| ------------------------------------- | -------------------- | --------------------------------------------- |
| Truck-wise holdout (leave-trucks-out) | **Primary evidence** | Test trucks never appear in training          |
| Chronological split                   | **Primary evidence** | Train on earlier data, test on later data     |
| Random row split                      | **Diagnostic only**  | Leakage-prone; never used as primary evidence |

Per-truck mean predictors are **not** valid unseen-truck baselines. They may appear
only inside a clearly labelled _seen-truck_ experiment, and never in the main
unseen-truck results table.

## Metrics

Primary regression metrics: **MAE, RMSE, R²**.
MAPE is optional and only reported where it is valid for the target distribution
(e.g. targets bounded away from zero); it must not be the headline metric.

## Explainability

No feature is assumed to be important in advance. SHAP results are only reported
from actual model runs, and are stored alongside the run configuration.

## Project status

- **No dataset has been downloaded.**
- **No model has been trained. No results exist.**
- No ML dependencies are installed yet.
- Dataset file sizes and redistribution/licensing permissions are **not yet verified**.
- This project does not claim to use Indian truck data, and does not claim to
  perform live OBD-II data collection. It works retrospectively from FEAD.

## Repository layout

```
docs/            research brief and notes
src/data/        loading, validation, splitting
src/features/    feature construction
src/models/      model definitions and training
src/evaluation/  metrics, protocols, cross-truck evaluation
src/explainability/ SHAP and other attribution
src/api/         local serving layer (later milestone)
notebooks/       exploratory work
scripts/         reproducible entry points
configs/         experiment configuration
tests/           unit and protocol tests
```

## Documents

- [DATA_SOURCES.md](DATA_SOURCES.md) — provenance, sampling, accuracy, EMS vs IFM
- [MILESTONES.md](MILESTONES.md) — roadmap and current state
- [docs/research_brief.md](docs/research_brief.md) — full research design
