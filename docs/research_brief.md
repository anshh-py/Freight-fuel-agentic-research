# Research Brief

## 1. Central question

How well does a regression model trained on FEAD EMS telemetry from a subset of
trucks generalize to trucks it has never seen?

## 2. Position relative to prior work

The FEAD paper reports **route-matched** training and evaluation — same truck,
same route. This repository does not reproduce that setting as its main result.

The unseen-truck evaluation here is **our extension**. It is an investigation,
not an established finding, and must be described that way in every write-up,
figure caption and README line.

## 3. Evaluation protocols

Three protocols are kept permanently separate in code, configs and reports.

1. **Truck-wise holdout (primary).** Entire trucks are held out. A truck present
   in training must never appear in test, in any form, including in fitted
   scalers or target statistics.
2. **Chronological (primary).** Train on earlier observations, test on later ones,
   within a documented time boundary per truck.
3. **Random row split (diagnostic only).** Reported only to demonstrate leakage
   behaviour and protocol sensitivity. It is never primary evidence and never
   appears in headline results.

Every results table carries an explicit protocol column.

## 4. Baselines

- Per-truck mean is **not** an unseen-truck baseline. It is undefined for a truck
  with no training data, and using it as such would misrepresent the task.
- Per-truck mean may appear only inside a clearly labelled **seen-truck**
  experiment, in its own table.
- Naive or persistence baselines are used only where the target and time indexing
  make them well defined.

## 5. Metrics

Primary: **MAE, RMSE, R²**.

MAPE is optional and reported only when the target distribution makes it valid
(values bounded away from zero, no zero/near-zero targets). When reported, the
validity argument must be stated next to the number.

Report dispersion across truck folds, not only the pooled score.

## 6. Explainability

No feature is assumed important before measurement. SHAP values are produced only
from actual trained models, stored with the run ID and config, and interpreted per
protocol. Attribution computed on a random-split model is not evidence about
cross-truck behaviour.

## 7. Data documentation requirements

Before modelling, the following must be documented in DATA_SOURCES.md:

- provenance and access path (FEAD repository links via Google Drive)
- sampling frequency per source
- measurement accuracy / resolution where stated
- differences between EMS and IFM
- file sizes and license/redistribution status

Current state: none verified, nothing downloaded.

## 8. Non-claims

- Not Indian truck data.
- Not live OBD-II collection.
- No model results exist yet.
- No dataset has been downloaded.

## 9. Threats to validity

- Truck and route are confounded in FEAD; a truck-wise split may partly measure
  route shift rather than truck shift. This must be stated, not hidden.
- Small truck counts limit the resolution of unseen-truck estimates.
- Leakage risk from any per-truck statistics computed before splitting.
