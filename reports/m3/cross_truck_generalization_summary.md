# M3.1 Cross-Truck Generalization Analysis

**Milestone:** M3.1

**Analysis frame:** conditional on the five frozen test trucks (truck_001, truck_004, truck_017, truck_020, truck_028); does NOT establish population-level generalization across trucks

**Protocol:** read-only: reuse frozen 5-truck test split and saved M1.2 artifacts; no training/refitting/tuning; test used only for evaluation; trip-level resampling (not telemetry rows)

**Synthetic-data caveat:** results are pipeline validation; they do not represent observed real-world Indian truck telemetry.

## Selected models (from M1.2 validation, not test)

| Feature set | Selected model | Validation MAE |
| --- | --- | --- |
| full_operational | xgboost | 3.621111 |
| telematics_only | xgboost | 4.292624 |

## Verification / reconciliation

- Test trucks: truck_001, truck_004, truck_017, truck_020, truck_028

- Test rows: 173289
- Test trips: 96 (expected 96)
- All expected test trucks present: True
- Full vs telematics same rows: True
- Per-feature-set row counts match: True
- Bootstrap resamples: 2000 (seed 20240617)

## Pooled paired comparison (full-operational vs telematics-only)

Sign convention: delta = MAE_telematics_only - MAE_full_operational; positive => full-operational has lower error (better)

| Metric | Full-operational | Telematics-only | Delta |
| --- | --- | --- | --- |
| MAE | 2.5684 | 2.5660 | -0.0025 |
| RMSE | 4.5275 | 4.3713 | -0.1562 |

## Per-truck paired MAE delta

| Truck | n | MAE full | MAE telematics | Δ MAE |
| --- | --- | --- | --- | --- |
| truck_001 | 30817 | 1.3023 | 1.8832 | 0.5809 |
| truck_004 | 32480 | 1.5492 | 1.5358 | -0.0134 |
| truck_017 | 37073 | 2.2452 | 2.0527 | -0.1925 |
| truck_020 | 33901 | 3.8660 | 2.9305 | -0.9355 |
| truck_028 | 39018 | 3.5966 | 4.1337 | 0.5371 |

## Trip-clustered bootstrap (95% percentile intervals)

Method: trip-clustered percentile bootstrap; resample the 96 test trips with replacement, jointly for both models; 95% percentile interval

Resampling unit: trip_id; n trips = 96; resamples = 2000; seed = 20240617

**estimates uncertainty conditional on these five test trucks; does NOT establish population-level generalization across trucks**

| Quantity | 95% CI |
| --- | --- |
| mae_full | [2.2218, 2.9182] |
| rmse_full | [3.9563, 5.0939] |
| mae_tel | [2.2054, 2.9298] |
| rmse_tel | [3.6759, 4.9965] |
| delta_mae | [-0.2374, 0.2227] |

## Feature set: `full_operational`

Pooled MAE = 2.5684; rows = 173289; trips = 96

Pooled tail error (L/h):
| Level | p90 | p95 | p99 | max |
| --- | --- | --- | --- | --- |
| pooled | 7.663 | 11.150 | 16.328 | 25.982 |

Per-truck tail error (L/h):
| Truck | n | p90 | p95 | p99 | max |
| --- | --- | --- | --- | --- | --- |
| truck_001 | 30817 | 4.247 | 4.724 | 5.325 | 9.118 |
| truck_004 | 32480 | 4.854 | 5.149 | 5.958 | 10.204 |
| truck_017 | 37073 | 7.671 | 8.285 | 9.010 | 12.596 |
| truck_020 | 33901 | 13.099 | 14.045 | 15.625 | 22.489 |
| truck_028 | 39018 | 9.655 | 14.359 | 19.794 | 25.982 |

Variance decomposition of squared error (descriptive):
- Between-truck share: 0.343136
- Between-trip (within truck) share: 0.09636
- Within-trip share: 0.560504
- Note: 5-truck sample: truck-level variance component is descriptive and unstable; a formal ANOVA ICC is NOT reported because n=5 trucks is too small for a defensible between-truck random-effect estimate. No causal interpretation.

Worst 5 trips by MAE:
| Trip | Truck | Route | Class | n | MAE | RMSE | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| trip_000384 | truck_020 | highway | 40T | 945 | 9.106 | 11.365 | 16.586 |
| trip_000378 | truck_020 | mixed | 40T | 699 | 7.229 | 9.798 | 16.834 |
| trip_000517 | truck_028 | highway | 49T | 979 | 7.010 | 8.792 | 15.391 |
| trip_000368 | truck_020 | highway | 40T | 1778 | 6.937 | 9.368 | 14.540 |
| trip_000520 | truck_028 | urban | 49T | 1105 | 6.897 | 9.933 | 19.180 |

## Feature set: `telematics_only`

Pooled MAE = 2.5660; rows = 173289; trips = 96

Pooled tail error (L/h):
| Level | p90 | p95 | p99 | max |
| --- | --- | --- | --- | --- |
| pooled | 7.076 | 9.713 | 16.920 | 46.665 |

Per-truck tail error (L/h):
| Truck | n | p90 | p95 | p99 | max |
| --- | --- | --- | --- | --- | --- |
| truck_001 | 30817 | 5.007 | 6.883 | 10.397 | 12.412 |
| truck_004 | 32480 | 4.155 | 5.253 | 7.439 | 16.365 |
| truck_017 | 37073 | 5.955 | 8.125 | 11.105 | 18.343 |
| truck_020 | 33901 | 8.352 | 10.301 | 14.169 | 21.340 |
| truck_028 | 39018 | 11.315 | 15.988 | 24.243 | 46.665 |

Variance decomposition of squared error (descriptive):
- Between-truck share: 0.277189
- Between-trip (within truck) share: 0.127782
- Within-trip share: 0.595029
- Note: 5-truck sample: truck-level variance component is descriptive and unstable; a formal ANOVA ICC is NOT reported because n=5 trucks is too small for a defensible between-truck random-effect estimate. No causal interpretation.

Worst 5 trips by MAE:
| Trip | Truck | Route | Class | n | MAE | RMSE | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| trip_000517 | truck_028 | highway | 49T | 979 | 8.865 | 11.431 | 18.896 |
| trip_000520 | truck_028 | urban | 49T | 1105 | 8.835 | 13.257 | 26.921 |
| trip_000513 | truck_028 | mountainous | 49T | 1661 | 7.665 | 12.007 | 26.419 |
| trip_000368 | truck_020 | highway | 40T | 1778 | 6.969 | 8.824 | 14.739 |
| trip_000378 | truck_020 | mixed | 40T | 699 | 6.178 | 8.600 | 16.236 |

## Limitations

- Synthetic telemetry: results are pipeline validation, not observed real-world Indian truck telemetry.
- Only five test trucks: truck-level estimates (tail, variance decomposition, macro) are descriptive and unstable; not population estimates.
- Trip-clustered bootstrap resamples the 96 test trips and therefore reflects uncertainty conditional on these five trucks only.
- Repeated trips of the same truck are not fully independent; we mitigate this by resampling at the trip level rather than rows.
- 49T class is represented by a single truck; no class-level generalization claims are made.
