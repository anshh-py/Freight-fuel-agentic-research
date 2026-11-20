# M3.3 Cross-Truck and Cross-Condition Error Diagnosis

**Milestone:** M3.3

**Analysis frame:** conditional on the five frozen test trucks; descriptive only; no hypothesis tests, no causal claims

**Protocol:** read-only: reuse frozen 5-truck test split and saved M1.2 artifacts; no training/refitting/tuning; no test-set-driven model selection

**Residual convention:** residual = y_pred - y_true; positive => overprediction; negative => underprediction. Overprediction, underprediction and approximately-unbiased percentages overlap by definition (an approximately-unbiased row is also counted in one of the other two bins), so they do not sum to 100%.

**Synthetic-data caveat:** results are pipeline validation; they do not represent observed real-world Indian truck telemetry.

**Diagnostic scope:** descriptive only. The `flagged_high_error` heuristic is a deviation from truck-level and pooled-condition baselines, NOT a statistical interaction test (only five trucks). It is an investigative indicator only and may appear on cells flagged insufficient_support, where it is indicative rather than conclusive. `recurring` vs `dominated_by_worst_trip` is a descriptive categorization, not a mechanistic finding.

## Selected models (from M1.2 validation, not test)

| Feature set | Selected model |
| --- | --- |
| full_operational | xgboost |
| telematics_only | xgboost |

## Verification / reconciliation

- Test trucks: truck_001, truck_004, truck_017, truck_020, truck_028

- Test rows: 173289
- Test trips: 96 (expected 96)
- All expected test trucks present: True
- Full vs telematics same rows: True
- Per-feature-set row counts: {'full_operational': 173289, 'telematics_only': 173289}
- Cell support statuses used: ['descriptive', 'empty', 'insufficient_rows']
- All requested conditions present per feature set: {'full_operational': True, 'telematics_only': True}

## Feature set: `full_operational`

Rows = 173289; trips = 96; trucks = 5

### Experiment 1 — Truck × condition error table

| Truck | Condition | n rows | n trips | MAE | RMSE | median | P90 | P95 | truck ratio | cond ratio | flagged | support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| truck_020 | grade_ge_1.5_steep_uphill | 255 | 3 | 12.2679 | 13.1974 | 14.8302 | 15.6791 | 16.6591 | 3.1732 | 2.3753 | True | insufficient_rows |
| truck_028 | combined_uphill_accelerating | 134 | 5 | 12.2560 | 13.5606 | 11.4128 | 19.1806 | 19.6138 | 3.4076 | 1.5666 | True | insufficient_rows |
| truck_020 | combined_uphill_accelerating | 89 | 5 | 11.7291 | 11.9368 | 12.0072 | 13.9462 | 14.1655 | 3.0339 | 1.4993 | True | insufficient_rows |
| truck_028 | accel_ge_0.5_accelerating | 6331 | 17 | 9.9853 | 11.7922 | 8.6119 | 18.8087 | 20.2007 | 2.7763 | 1.5233 | True | descriptive |
| truck_020 | accel_ge_0.5_accelerating | 5725 | 23 | 9.0823 | 9.9822 | 10.1544 | 13.6955 | 14.1370 | 2.3493 | 1.3855 | True | descriptive |
| truck_020 | speed_ge_60_highway | 3730 | 3 | 7.0591 | 9.3539 | 5.9118 | 14.9155 | 15.5931 | 1.8259 | 1.9298 | True | descriptive |
| truck_017 | accel_ge_0.5_accelerating | 7203 | 23 | 5.8804 | 6.5119 | 6.8067 | 8.6748 | 8.9286 | 2.6191 | 0.8971 | False | descriptive |
| truck_017 | combined_uphill_accelerating | 137 | 8 | 5.5215 | 6.2769 | 6.8931 | 8.8213 | 8.9989 | 2.4593 | 0.7058 | False | insufficient_rows |
| truck_028 | grade_ge_1.5_steep_uphill | 672 | 3 | 5.4663 | 6.0094 | 5.1731 | 9.3239 | 10.3030 | 1.5198 | 1.0584 | False | descriptive |
| truck_017 | grade_ge_1.5_steep_uphill | 635 | 4 | 4.9597 | 5.4072 | 5.1725 | 7.6218 | 8.3195 | 2.2091 | 0.9603 | False | descriptive |
| truck_028 | speed_ge_60_highway | 4774 | 3 | 4.4204 | 5.7175 | 4.6148 | 9.5631 | 11.3232 | 1.2290 | 1.2084 | False | descriptive |
| truck_004 | combined_uphill_accelerating | 157 | 6 | 4.3536 | 4.6353 | 4.8511 | 6.0310 | 6.3201 | 2.8102 | 0.5565 | False | insufficient_rows |
| truck_004 | accel_ge_0.5_accelerating | 5799 | 13 | 4.3024 | 4.5105 | 4.6254 | 5.4274 | 5.9037 | 2.7772 | 0.6563 | False | descriptive |
| truck_001 | grade_ge_1.5_steep_uphill | 820 | 2 | 4.2056 | 4.3745 | 4.3902 | 5.0584 | 5.2930 | 3.2295 | 0.8143 | False | descriptive |
| truck_001 | combined_uphill_accelerating | 22 | 3 | 4.1126 | 4.2355 | 4.4774 | 4.7484 | 4.7640 | 3.1580 | 0.5257 | False | insufficient_rows |
| truck_004 | grade_ge_1.5_steep_uphill | 646 | 2 | 3.4664 | 4.0775 | 4.5184 | 5.6092 | 5.6374 | 2.2375 | 0.6712 | False | descriptive |
| truck_017 | speed_ge_60_highway | 5533 | 5 | 3.4340 | 4.5749 | 2.8246 | 7.9377 | 8.3764 | 1.5295 | 0.9388 | False | descriptive |
| truck_001 | accel_ge_0.5_accelerating | 5143 | 19 | 3.0050 | 3.4071 | 3.7158 | 4.8562 | 5.1079 | 2.3075 | 0.4584 | False | descriptive |
| truck_001 | speed_ge_60_highway | 4014 | 2 | 2.5796 | 3.3671 | 3.7060 | 5.0057 | 5.3251 | 1.9808 | 0.7052 | False | descriptive |
| truck_004 | speed_ge_60_highway | 6394 | 3 | 1.9754 | 2.8568 | 0.7907 | 4.9637 | 5.4222 | 1.2751 | 0.5400 | False | descriptive |
| truck_028 | combined_downhill_braking | 275 | 6 | 0.3581 | 0.4026 | 0.3178 | 0.6120 | 0.6381 | 0.0996 | 1.6655 | False | insufficient_rows |
| truck_028 | accel_le_-0.5_braking | 6232 | 17 | 0.3432 | 0.4205 | 0.2948 | 0.7066 | 0.7918 | 0.0954 | 1.7003 | False | descriptive |
| truck_020 | combined_downhill_braking | 187 | 8 | 0.2866 | 0.3901 | 0.2013 | 0.8647 | 0.8744 | 0.0741 | 1.3327 | False | insufficient_rows |
| truck_028 | idle_engine_on_stopped | 2361 | 17 | 0.2627 | 0.3005 | 0.2413 | 0.4926 | 0.4931 | 0.0730 | 1.6847 | False | descriptive |
| truck_017 | combined_downhill_braking | 246 | 11 | 0.2284 | 0.2691 | 0.2164 | 0.4355 | 0.4492 | 0.1017 | 1.0620 | False | insufficient_rows |
| truck_020 | accel_le_-0.5_braking | 5252 | 24 | 0.2256 | 0.2851 | 0.1991 | 0.4483 | 0.4932 | 0.0583 | 1.1177 | False | descriptive |
| truck_028 | engine_off_stopped | 213 | 16 | 0.1900 | 0.2312 | 0.1686 | 0.4057 | 0.4223 | 0.0528 | 1.5291 | False | insufficient_rows |
| truck_017 | accel_le_-0.5_braking | 5910 | 23 | 0.1692 | 0.1983 | 0.1458 | 0.2996 | 0.3226 | 0.0754 | 0.8384 | False | descriptive |
| truck_004 | idle_engine_on_stopped | 805 | 13 | 0.1394 | 0.1789 | 0.1296 | 0.2207 | 0.4263 | 0.0900 | 0.8940 | False | descriptive |
| truck_020 | idle_engine_on_stopped | 1863 | 24 | 0.1358 | 0.1590 | 0.1289 | 0.2432 | 0.2963 | 0.0351 | 0.8707 | False | descriptive |
| truck_017 | engine_off_stopped | 314 | 23 | 0.1347 | 0.1824 | 0.0962 | 0.3460 | 0.4206 | 0.0600 | 1.0846 | False | insufficient_rows |
| truck_004 | accel_le_-0.5_braking | 4947 | 13 | 0.1282 | 0.1614 | 0.1184 | 0.2372 | 0.3812 | 0.0827 | 0.6351 | False | descriptive |
| truck_004 | combined_downhill_braking | 337 | 5 | 0.1253 | 0.1592 | 0.1072 | 0.2055 | 0.3769 | 0.0809 | 0.5827 | False | insufficient_rows |
| truck_020 | engine_off_stopped | 330 | 24 | 0.1205 | 0.1547 | 0.1084 | 0.2596 | 0.2976 | 0.0312 | 0.9698 | False | insufficient_rows |
| truck_001 | accel_le_-0.5_braking | 4776 | 19 | 0.1079 | 0.1320 | 0.0942 | 0.2154 | 0.2603 | 0.0829 | 0.5347 | False | descriptive |
| truck_004 | engine_off_stopped | 153 | 13 | 0.1016 | 0.1426 | 0.0653 | 0.2421 | 0.3798 | 0.0656 | 0.8175 | False | insufficient_rows |
| truck_017 | idle_engine_on_stopped | 1513 | 23 | 0.0904 | 0.1212 | 0.0392 | 0.2528 | 0.2724 | 0.0402 | 0.5796 | False | descriptive |
| truck_001 | idle_engine_on_stopped | 1502 | 19 | 0.0880 | 0.0985 | 0.0955 | 0.1478 | 0.1620 | 0.0676 | 0.5645 | False | descriptive |
| truck_001 | combined_downhill_braking | 201 | 7 | 0.0868 | 0.1122 | 0.0643 | 0.1709 | 0.2515 | 0.0667 | 0.4037 | False | insufficient_rows |
| truck_001 | engine_off_stopped | 222 | 19 | 0.0675 | 0.0987 | 0.0394 | 0.1314 | 0.1882 | 0.0518 | 0.5435 | False | insufficient_rows |
| truck_001 | payload_heavy_gt0.66 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | - | - | False | empty |
| truck_004 | payload_heavy_gt0.66 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | - | - | False | empty |
| truck_017 | payload_heavy_gt0.66 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | - | - | False | empty |
| truck_020 | payload_heavy_gt0.66 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | - | - | False | empty |
| truck_028 | payload_heavy_gt0.66 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | - | - | False | empty |
| truck_001 | combined_highway_heavy_payload | 0 | 0 | n/a | n/a | n/a | n/a | n/a | - | - | False | empty |
| truck_004 | combined_highway_heavy_payload | 0 | 0 | n/a | n/a | n/a | n/a | n/a | - | - | False | empty |
| truck_017 | combined_highway_heavy_payload | 0 | 0 | n/a | n/a | n/a | n/a | n/a | - | - | False | empty |
| truck_020 | combined_highway_heavy_payload | 0 | 0 | n/a | n/a | n/a | n/a | n/a | - | - | False | empty |
| truck_028 | combined_highway_heavy_payload | 0 | 0 | n/a | n/a | n/a | n/a | n/a | - | - | False | empty |
| truck_001 | combined_steep_uphill_heavy_payload | 0 | 0 | n/a | n/a | n/a | n/a | n/a | - | - | False | empty |
| truck_004 | combined_steep_uphill_heavy_payload | 0 | 0 | n/a | n/a | n/a | n/a | n/a | - | - | False | empty |
| truck_017 | combined_steep_uphill_heavy_payload | 0 | 0 | n/a | n/a | n/a | n/a | n/a | - | - | False | empty |
| truck_020 | combined_steep_uphill_heavy_payload | 0 | 0 | n/a | n/a | n/a | n/a | n/a | - | - | False | empty |
| truck_028 | combined_steep_uphill_heavy_payload | 0 | 0 | n/a | n/a | n/a | n/a | n/a | - | - | False | empty |

`truck ratio` = cell MAE / truck-level baseline MAE; `cond ratio` = cell MAE / pooled-condition baseline MAE. A cell is `flagged_high_error=True` only when BOTH ratios >= 1.25 (descriptive heuristic, not a statistical interaction test).

### Experiment 2 — Trip-level recurrence

| Truck | Condition | n rows | n trips | median trip MAE | std trip MAE | worst-trip share | pattern | support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| truck_001 | speed_ge_60_highway | 4014 | 2 | 2.4532 | 0.4262 | 0.7213 | dominated_by_worst_trip | descriptive |
| truck_004 | speed_ge_60_highway | 6394 | 3 | 1.9361 | 0.0811 | 0.5468 | dominated_by_worst_trip | descriptive |
| truck_017 | speed_ge_60_highway | 5533 | 5 | 3.7840 | 0.6542 | 0.3961 | recurring_pattern | descriptive |
| truck_020 | speed_ge_60_highway | 3730 | 3 | 7.0713 | 1.7322 | 0.4426 | recurring_pattern | descriptive |
| truck_028 | speed_ge_60_highway | 4774 | 3 | 4.8657 | 1.7590 | 0.4742 | recurring_pattern | descriptive |
| truck_001 | idle_engine_on_stopped | 1502 | 19 | 0.0930 | 0.0489 | 0.2903 | recurring_pattern | descriptive |
| truck_004 | idle_engine_on_stopped | 805 | 13 | 0.1235 | 0.1066 | 0.5964 | dominated_by_worst_trip | descriptive |
| truck_017 | idle_engine_on_stopped | 1513 | 23 | 0.1226 | 0.0793 | 0.3833 | recurring_pattern | descriptive |
| truck_020 | idle_engine_on_stopped | 1863 | 24 | 0.1617 | 0.1147 | 0.1286 | recurring_pattern | descriptive |
| truck_028 | idle_engine_on_stopped | 2361 | 17 | 0.2050 | 0.1688 | 0.3171 | recurring_pattern | descriptive |
| truck_001 | engine_off_stopped | 222 | 19 | 0.0836 | 0.0610 | 0.5574 | insufficient | insufficient_rows |
| truck_004 | engine_off_stopped | 153 | 13 | 0.0556 | 0.0915 | 0.6066 | insufficient | insufficient_rows |
| truck_017 | engine_off_stopped | 314 | 23 | 0.0870 | 0.1126 | 0.3663 | insufficient | insufficient_rows |
| truck_020 | engine_off_stopped | 330 | 24 | 0.1069 | 0.1008 | 0.2765 | insufficient | insufficient_rows |
| truck_028 | engine_off_stopped | 213 | 16 | 0.1685 | 0.1240 | 0.2445 | insufficient | insufficient_rows |
| truck_001 | accel_ge_0.5_accelerating | 5143 | 19 | 3.0923 | 1.2327 | 0.1851 | recurring_pattern | descriptive |
| truck_004 | accel_ge_0.5_accelerating | 5799 | 13 | 4.2485 | 0.6296 | 0.2252 | recurring_pattern | descriptive |
| truck_017 | accel_ge_0.5_accelerating | 7203 | 23 | 6.2278 | 2.3564 | 0.1536 | recurring_pattern | descriptive |
| truck_020 | accel_ge_0.5_accelerating | 5725 | 23 | 9.9094 | 3.2444 | 0.1738 | recurring_pattern | descriptive |
| truck_028 | accel_ge_0.5_accelerating | 6331 | 17 | 7.3233 | 5.3592 | 0.2279 | recurring_pattern | descriptive |
| truck_001 | accel_le_-0.5_braking | 4776 | 19 | 0.0976 | 0.0513 | 0.2988 | recurring_pattern | descriptive |
| truck_004 | accel_le_-0.5_braking | 4947 | 13 | 0.1064 | 0.0891 | 0.4660 | recurring_pattern | descriptive |
| truck_017 | accel_le_-0.5_braking | 5910 | 23 | 0.1737 | 0.0968 | 0.1486 | recurring_pattern | descriptive |
| truck_020 | accel_le_-0.5_braking | 5252 | 24 | 0.1940 | 0.2144 | 0.1232 | recurring_pattern | descriptive |
| truck_028 | accel_le_-0.5_braking | 6232 | 17 | 0.3876 | 0.2155 | 0.2344 | recurring_pattern | descriptive |
| truck_001 | grade_ge_1.5_steep_uphill | 820 | 2 | 4.1197 | 0.1925 | 0.7248 | dominated_by_worst_trip | descriptive |
| truck_004 | grade_ge_1.5_steep_uphill | 646 | 2 | 3.3874 | 0.2008 | 0.7358 | dominated_by_worst_trip | descriptive |
| truck_017 | grade_ge_1.5_steep_uphill | 635 | 4 | 5.3282 | 0.6738 | 0.3987 | recurring_pattern | descriptive |
| truck_020 | grade_ge_1.5_steep_uphill | 255 | 3 | 11.3253 | 3.0460 | 0.7071 | insufficient | insufficient_rows |
| truck_028 | grade_ge_1.5_steep_uphill | 672 | 3 | 5.4085 | 2.2862 | 0.4082 | recurring_pattern | descriptive |
| truck_001 | combined_uphill_accelerating | 22 | 3 | 4.3817 | 0.3255 | 0.5893 | insufficient | insufficient_rows |
| truck_004 | combined_uphill_accelerating | 157 | 6 | 4.4915 | 0.8272 | 0.3734 | insufficient | insufficient_rows |
| truck_017 | combined_uphill_accelerating | 137 | 8 | 5.7025 | 2.5104 | 0.4711 | insufficient | insufficient_rows |
| truck_020 | combined_uphill_accelerating | 89 | 5 | 11.7781 | 0.9462 | 0.3974 | insufficient | insufficient_rows |
| truck_028 | combined_uphill_accelerating | 134 | 5 | 9.5590 | 5.2778 | 0.6501 | insufficient | insufficient_rows |
| truck_001 | combined_downhill_braking | 201 | 7 | 0.0939 | 0.0588 | 0.4756 | insufficient | insufficient_rows |
| truck_004 | combined_downhill_braking | 337 | 5 | 0.1420 | 0.1191 | 0.3860 | insufficient | insufficient_rows |
| truck_017 | combined_downhill_braking | 246 | 11 | 0.1589 | 0.1201 | 0.5325 | insufficient | insufficient_rows |
| truck_020 | combined_downhill_braking | 187 | 8 | 0.3121 | 0.2498 | 0.6624 | insufficient | insufficient_rows |
| truck_028 | combined_downhill_braking | 275 | 6 | 0.4057 | 0.1455 | 0.2726 | insufficient | insufficient_rows |

### Experiment 3 — Error distribution (pooled conditions)

| Condition | n rows | n trips | n trucks | MAE | RMSE | median | P90 | P95 | max | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| speed_ge_60_highway | 24445 | 16 | 5 | 3.6580 | 5.3356 | 2.9660 | 8.4471 | 12.4891 | 18.6469 | 4.2108 |
| idle_engine_on_stopped | 8044 | 96 | 5 | 0.1559 | 0.2003 | 0.1252 | 0.3336 | 0.4573 | 0.7581 | 3.6523 |
| engine_off_stopped | 1232 | 95 | 5 | 0.1242 | 0.1686 | 0.1011 | 0.2976 | 0.3952 | 0.4552 | 3.9075 |
| accel_ge_0.5_accelerating | 30201 | 95 | 5 | 6.5552 | 8.0022 | 5.2395 | 13.1063 | 16.4366 | 25.9823 | 3.1371 |
| accel_le_-0.5_braking | 27117 | 96 | 5 | 0.2018 | 0.2698 | 0.1465 | 0.4419 | 0.5593 | 2.2366 | 3.8179 |
| grade_ge_1.5_steep_uphill | 3028 | 14 | 5 | 5.1648 | 6.1272 | 4.7831 | 8.7812 | 13.0281 | 16.9691 | 2.7238 |
| payload_heavy_gt0.66 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | zero-median |
| combined_highway_heavy_payload | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | zero-median |
| combined_uphill_accelerating | 539 | 27 | 5 | 7.8231 | 9.2871 | 6.6014 | 16.8544 | 18.2859 | 20.9651 | 2.7700 |
| combined_downhill_braking | 1246 | 37 | 5 | 0.2150 | 0.2860 | 0.1826 | 0.4510 | 0.5209 | 1.4657 | 2.8530 |
| combined_steep_uphill_heavy_payload | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | zero-median |

### Experiment 4 — Residual direction (per truck)

| Truck | n rows | mean signed | median signed | over% | under% | ~unbiased% |
| --- | --- | --- | --- | --- | --- | --- |
| truck_001 | 30817 | -1.2202 | -0.2464 | 23.75 | 76.25 | 13.30 |
| truck_004 | 32480 | 1.4257 | 0.3533 | 85.75 | 14.25 | 9.81 |
| truck_017 | 37073 | 1.9949 | 0.2834 | 76.21 | 23.79 | 8.03 |
| truck_020 | 33901 | -3.8383 | -0.9603 | 11.27 | 88.73 | 7.81 |
| truck_028 | 39018 | 3.5487 | 1.0463 | 88.84 | 11.16 | 2.88 |

### Experiment 4 — Residual direction (per condition)

| Condition | n rows | mean signed | median signed | over% | under% | ~unbiased% |
| --- | --- | --- | --- | --- | --- | --- |
| speed_ge_60_highway | 24445 | 0.4295 | 0.0744 | 54.63 | 45.37 | 7.83 |
| idle_engine_on_stopped | 8044 | 0.0320 | 0.0392 | 58.83 | 41.17 | 19.92 |
| engine_off_stopped | 1232 | 0.0162 | 0.0104 | 55.60 | 44.40 | 33.69 |
| accel_ge_0.5_accelerating | 30201 | 2.0003 | 3.7315 | 60.78 | 39.22 | 0.33 |
| accel_le_-0.5_braking | 27117 | 0.0548 | 0.0505 | 59.18 | 40.82 | 16.51 |
| grade_ge_1.5_steep_uphill | 3028 | 0.7273 | 3.9489 | 59.02 | 40.98 | 1.16 |
| payload_heavy_gt0.66 | 0 | n/a | n/a | n/a | n/a | n/a |
| combined_highway_heavy_payload | 0 | n/a | n/a | n/a | n/a | n/a |
| combined_uphill_accelerating | 539 | 3.4025 | 4.9145 | 74.77 | 25.23 | 0.00 |
| combined_downhill_braking | 1246 | 0.0217 | -0.0008 | 49.84 | 50.16 | 21.75 |
| combined_steep_uphill_heavy_payload | 0 | n/a | n/a | n/a | n/a | n/a |

### Experiment 5 — Paired model comparison (per condition)

| Condition | n rows | MAE full | MAE tel | delta MAE | RMSE full | RMSE tel | delta RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| speed_ge_60_highway | 24445 | 3.6580 | 3.7330 | 0.0750 | 5.3356 | 5.3835 | 0.0479 |
| idle_engine_on_stopped | 8044 | 0.1559 | 0.4687 | 0.3127 | 0.2003 | 1.1748 | 0.9745 |
| engine_off_stopped | 1232 | 0.1242 | 0.3341 | 0.2099 | 0.1686 | 0.4390 | 0.2704 |
| accel_ge_0.5_accelerating | 30201 | 6.5552 | 5.5051 | -1.0501 | 8.0022 | 7.5521 | -0.4501 |
| accel_le_-0.5_braking | 27117 | 0.2018 | 0.5328 | 0.3310 | 0.2698 | 0.7391 | 0.4693 |
| grade_ge_1.5_steep_uphill | 3028 | 5.1648 | 4.5764 | -0.5884 | 6.1272 | 6.2163 | 0.0891 |
| payload_heavy_gt0.66 | 0 | nan | nan | nan | nan | nan | nan |
| combined_highway_heavy_payload | 0 | nan | nan | nan | nan | nan | nan |
| combined_uphill_accelerating | 539 | 7.8231 | 6.7569 | -1.0661 | 9.2871 | 8.9240 | -0.3630 |
| combined_downhill_braking | 1246 | 0.2150 | 0.6997 | 0.4846 | 0.2860 | 0.9524 | 0.6664 |
| combined_steep_uphill_heavy_payload | 0 | nan | nan | nan | nan | nan | nan |

Paired delta convention: delta = metric_telematics_only - metric_full_operational; positive => full-operational has lower error (better)

Small deltas are descriptive; they are not evidence of model superiority. No hypothesis tests were run.

## Feature set: `telematics_only`

Rows = 173289; trips = 96; trucks = 5

### Experiment 1 — Truck × condition error table

| Truck | Condition | n rows | n trips | MAE | RMSE | median | P90 | P95 | truck ratio | cond ratio | flagged | support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| truck_028 | combined_uphill_accelerating | 134 | 5 | 14.2349 | 15.3455 | 15.4330 | 20.9307 | 22.1371 | 3.4436 | 2.1067 | True | insufficient_rows |
| truck_028 | accel_ge_0.5_accelerating | 6331 | 17 | 11.2753 | 13.3744 | 10.7424 | 21.8019 | 24.9664 | 2.7277 | 2.0482 | True | descriptive |
| truck_028 | grade_ge_1.5_steep_uphill | 672 | 3 | 8.4860 | 9.8227 | 8.6030 | 14.0710 | 16.6823 | 2.0529 | 1.8543 | True | descriptive |
| truck_020 | grade_ge_1.5_steep_uphill | 255 | 3 | 8.1712 | 8.8940 | 7.8276 | 13.3153 | 14.2632 | 2.7883 | 1.7855 | True | insufficient_rows |
| truck_020 | combined_uphill_accelerating | 89 | 5 | 7.7648 | 8.1027 | 7.6286 | 10.8916 | 11.4629 | 2.6496 | 1.1492 | False | insufficient_rows |
| truck_020 | speed_ge_60_highway | 3730 | 3 | 6.5169 | 8.2260 | 6.0788 | 13.3786 | 14.3632 | 2.2238 | 1.7458 | True | descriptive |
| truck_028 | speed_ge_60_highway | 4774 | 3 | 5.8526 | 7.6530 | 4.8569 | 12.8957 | 16.1137 | 1.4158 | 1.5678 | True | descriptive |
| truck_020 | accel_ge_0.5_accelerating | 5725 | 23 | 4.7811 | 5.5229 | 4.4374 | 8.5597 | 9.4565 | 1.6315 | 0.8685 | False | descriptive |
| truck_017 | accel_ge_0.5_accelerating | 7203 | 23 | 4.5890 | 5.7360 | 3.7953 | 9.3362 | 10.5209 | 2.2356 | 0.8336 | False | descriptive |
| truck_001 | combined_uphill_accelerating | 22 | 3 | 4.3656 | 4.7970 | 4.4392 | 7.2227 | 7.7298 | 2.3182 | 0.6461 | False | insufficient_rows |
| truck_001 | accel_ge_0.5_accelerating | 5143 | 19 | 4.0491 | 4.8637 | 3.5973 | 8.4445 | 9.6256 | 2.1501 | 0.7355 | False | descriptive |
| truck_017 | grade_ge_1.5_steep_uphill | 635 | 4 | 3.9225 | 5.2964 | 3.0705 | 9.8874 | 13.1498 | 1.9110 | 0.8571 | False | descriptive |
| truck_017 | combined_uphill_accelerating | 137 | 8 | 3.7925 | 4.6920 | 3.3423 | 7.3381 | 9.6024 | 1.8476 | 0.5613 | False | insufficient_rows |
| truck_004 | combined_uphill_accelerating | 157 | 6 | 2.7251 | 3.5734 | 2.1569 | 5.2611 | 7.2878 | 1.7743 | 0.4033 | False | insufficient_rows |
| truck_004 | grade_ge_1.5_steep_uphill | 646 | 2 | 2.5427 | 3.2630 | 2.0405 | 5.1850 | 6.1973 | 1.6556 | 0.5556 | False | descriptive |
| truck_017 | speed_ge_60_highway | 5533 | 5 | 2.4807 | 3.5945 | 1.6173 | 5.8643 | 8.1207 | 1.2085 | 0.6645 | False | descriptive |
| truck_004 | speed_ge_60_highway | 6394 | 3 | 2.4395 | 3.2159 | 1.8749 | 5.4527 | 6.5183 | 1.5884 | 0.6535 | False | descriptive |
| truck_001 | speed_ge_60_highway | 4014 | 2 | 2.4115 | 3.1110 | 1.8615 | 5.1882 | 6.0950 | 1.2805 | 0.6460 | False | descriptive |
| truck_001 | grade_ge_1.5_steep_uphill | 820 | 2 | 2.3629 | 2.9859 | 1.6709 | 4.7597 | 5.7865 | 1.2547 | 0.5163 | False | descriptive |
| truck_004 | accel_ge_0.5_accelerating | 5799 | 13 | 2.3495 | 3.1287 | 1.5975 | 5.3514 | 6.4893 | 1.5298 | 0.4268 | False | descriptive |
| truck_017 | combined_downhill_braking | 246 | 11 | 0.9902 | 1.2541 | 0.8984 | 2.0837 | 2.4803 | 0.4824 | 1.4152 | False | insufficient_rows |
| truck_028 | accel_le_-0.5_braking | 6232 | 17 | 0.8673 | 1.1004 | 0.7458 | 1.6761 | 2.0221 | 0.2098 | 1.6277 | False | descriptive |
| truck_020 | combined_downhill_braking | 187 | 8 | 0.8509 | 1.0163 | 0.8108 | 1.5021 | 1.6615 | 0.2904 | 1.2162 | False | insufficient_rows |
| truck_028 | idle_engine_on_stopped | 2361 | 17 | 0.8488 | 1.9488 | 0.7220 | 1.4327 | 1.7493 | 0.2053 | 1.8111 | False | descriptive |
| truck_028 | combined_downhill_braking | 275 | 6 | 0.8415 | 1.1449 | 0.6090 | 1.7036 | 2.2583 | 0.2036 | 1.2027 | False | insufficient_rows |
| truck_028 | engine_off_stopped | 213 | 16 | 0.6145 | 0.7394 | 0.5681 | 1.1529 | 1.2894 | 0.1487 | 1.8392 | False | insufficient_rows |
| truck_001 | combined_downhill_braking | 201 | 7 | 0.4753 | 0.6617 | 0.4005 | 0.9059 | 1.5337 | 0.2524 | 0.6793 | False | insufficient_rows |
| truck_020 | accel_le_-0.5_braking | 5252 | 24 | 0.4734 | 0.6267 | 0.3634 | 1.0841 | 1.3045 | 0.1615 | 0.8884 | False | descriptive |
| truck_017 | accel_le_-0.5_braking | 5910 | 23 | 0.4730 | 0.6734 | 0.3438 | 1.0036 | 1.2154 | 0.2304 | 0.8878 | False | descriptive |
| truck_004 | accel_le_-0.5_braking | 4947 | 13 | 0.4328 | 0.5461 | 0.3676 | 0.8569 | 0.9910 | 0.2818 | 0.8122 | False | descriptive |
| truck_004 | combined_downhill_braking | 337 | 5 | 0.4218 | 0.5491 | 0.3675 | 0.9333 | 1.1367 | 0.2746 | 0.6028 | False | insufficient_rows |
| truck_017 | idle_engine_on_stopped | 1513 | 23 | 0.3731 | 0.9827 | 0.2438 | 0.6367 | 0.9649 | 0.1818 | 0.7960 | False | descriptive |
| truck_017 | engine_off_stopped | 314 | 23 | 0.3429 | 0.4165 | 0.3099 | 0.6887 | 0.7272 | 0.1670 | 1.0263 | False | insufficient_rows |
| truck_001 | accel_le_-0.5_braking | 4776 | 19 | 0.3395 | 0.4681 | 0.2410 | 0.7639 | 0.9445 | 0.1803 | 0.6372 | False | descriptive |
| truck_020 | idle_engine_on_stopped | 1863 | 24 | 0.3329 | 0.4362 | 0.2746 | 0.6521 | 0.8549 | 0.1136 | 0.7102 | False | descriptive |
| truck_004 | idle_engine_on_stopped | 805 | 13 | 0.2987 | 0.3862 | 0.2573 | 0.5883 | 0.7109 | 0.1945 | 0.6373 | False | descriptive |
| truck_020 | engine_off_stopped | 330 | 24 | 0.2902 | 0.3618 | 0.2397 | 0.6071 | 0.7046 | 0.0990 | 0.8686 | False | insufficient_rows |
| truck_004 | engine_off_stopped | 153 | 13 | 0.2437 | 0.2841 | 0.1866 | 0.5101 | 0.5101 | 0.1587 | 0.7296 | False | insufficient_rows |
| truck_001 | idle_engine_on_stopped | 1502 | 19 | 0.2270 | 0.3650 | 0.1222 | 0.5223 | 0.6764 | 0.1205 | 0.4843 | False | descriptive |
| truck_001 | engine_off_stopped | 222 | 19 | 0.1802 | 0.2222 | 0.1546 | 0.4004 | 0.4371 | 0.0957 | 0.5394 | False | insufficient_rows |
| truck_001 | payload_heavy_gt0.66 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | - | - | False | empty |
| truck_004 | payload_heavy_gt0.66 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | - | - | False | empty |
| truck_017 | payload_heavy_gt0.66 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | - | - | False | empty |
| truck_020 | payload_heavy_gt0.66 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | - | - | False | empty |
| truck_028 | payload_heavy_gt0.66 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | - | - | False | empty |
| truck_001 | combined_highway_heavy_payload | 0 | 0 | n/a | n/a | n/a | n/a | n/a | - | - | False | empty |
| truck_004 | combined_highway_heavy_payload | 0 | 0 | n/a | n/a | n/a | n/a | n/a | - | - | False | empty |
| truck_017 | combined_highway_heavy_payload | 0 | 0 | n/a | n/a | n/a | n/a | n/a | - | - | False | empty |
| truck_020 | combined_highway_heavy_payload | 0 | 0 | n/a | n/a | n/a | n/a | n/a | - | - | False | empty |
| truck_028 | combined_highway_heavy_payload | 0 | 0 | n/a | n/a | n/a | n/a | n/a | - | - | False | empty |
| truck_001 | combined_steep_uphill_heavy_payload | 0 | 0 | n/a | n/a | n/a | n/a | n/a | - | - | False | empty |
| truck_004 | combined_steep_uphill_heavy_payload | 0 | 0 | n/a | n/a | n/a | n/a | n/a | - | - | False | empty |
| truck_017 | combined_steep_uphill_heavy_payload | 0 | 0 | n/a | n/a | n/a | n/a | n/a | - | - | False | empty |
| truck_020 | combined_steep_uphill_heavy_payload | 0 | 0 | n/a | n/a | n/a | n/a | n/a | - | - | False | empty |
| truck_028 | combined_steep_uphill_heavy_payload | 0 | 0 | n/a | n/a | n/a | n/a | n/a | - | - | False | empty |

`truck ratio` = cell MAE / truck-level baseline MAE; `cond ratio` = cell MAE / pooled-condition baseline MAE. A cell is `flagged_high_error=True` only when BOTH ratios >= 1.25 (descriptive heuristic, not a statistical interaction test).

### Experiment 2 — Trip-level recurrence

| Truck | Condition | n rows | n trips | median trip MAE | std trip MAE | worst-trip share | pattern | support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| truck_001 | speed_ge_60_highway | 4014 | 2 | 2.4966 | 0.2869 | 0.5428 | dominated_by_worst_trip | descriptive |
| truck_004 | speed_ge_60_highway | 6394 | 3 | 2.6748 | 0.9548 | 0.6860 | dominated_by_worst_trip | descriptive |
| truck_017 | speed_ge_60_highway | 5533 | 5 | 1.8286 | 0.6713 | 0.5345 | dominated_by_worst_trip | descriptive |
| truck_020 | speed_ge_60_highway | 3730 | 3 | 6.2685 | 0.6609 | 0.5350 | dominated_by_worst_trip | descriptive |
| truck_028 | speed_ge_60_highway | 4774 | 3 | 5.1961 | 1.9595 | 0.4489 | recurring_pattern | descriptive |
| truck_001 | idle_engine_on_stopped | 1502 | 19 | 0.2513 | 0.3251 | 0.3879 | recurring_pattern | descriptive |
| truck_004 | idle_engine_on_stopped | 805 | 13 | 0.2604 | 0.1842 | 0.2297 | recurring_pattern | descriptive |
| truck_017 | idle_engine_on_stopped | 1513 | 23 | 0.3677 | 1.3069 | 0.8160 | dominated_by_worst_trip | descriptive |
| truck_020 | idle_engine_on_stopped | 1863 | 24 | 0.3290 | 0.2176 | 0.2246 | recurring_pattern | descriptive |
| truck_028 | idle_engine_on_stopped | 2361 | 17 | 0.6132 | 6.0041 | 0.6034 | dominated_by_worst_trip | descriptive |
| truck_001 | engine_off_stopped | 222 | 19 | 0.1679 | 0.0835 | 0.2191 | insufficient | insufficient_rows |
| truck_004 | engine_off_stopped | 153 | 13 | 0.2077 | 0.1234 | 0.3553 | insufficient | insufficient_rows |
| truck_017 | engine_off_stopped | 314 | 23 | 0.3714 | 0.1813 | 0.1271 | insufficient | insufficient_rows |
| truck_020 | engine_off_stopped | 330 | 24 | 0.2399 | 0.1459 | 0.1988 | insufficient | insufficient_rows |
| truck_028 | engine_off_stopped | 213 | 16 | 0.5391 | 0.3125 | 0.2265 | insufficient | insufficient_rows |
| truck_001 | accel_ge_0.5_accelerating | 5143 | 19 | 2.8004 | 1.7693 | 0.5078 | dominated_by_worst_trip | descriptive |
| truck_004 | accel_ge_0.5_accelerating | 5799 | 13 | 1.9857 | 1.8219 | 0.3614 | recurring_pattern | descriptive |
| truck_017 | accel_ge_0.5_accelerating | 7203 | 23 | 3.9670 | 2.7206 | 0.2097 | recurring_pattern | descriptive |
| truck_020 | accel_ge_0.5_accelerating | 5725 | 23 | 5.1757 | 2.8158 | 0.1353 | recurring_pattern | descriptive |
| truck_028 | accel_ge_0.5_accelerating | 6331 | 17 | 7.7871 | 6.4660 | 0.1894 | recurring_pattern | descriptive |
| truck_001 | accel_le_-0.5_braking | 4776 | 19 | 0.3510 | 0.1396 | 0.0935 | recurring_pattern | descriptive |
| truck_004 | accel_le_-0.5_braking | 4947 | 13 | 0.3791 | 0.1657 | 0.2824 | recurring_pattern | descriptive |
| truck_017 | accel_le_-0.5_braking | 5910 | 23 | 0.4395 | 0.3253 | 0.2008 | recurring_pattern | descriptive |
| truck_020 | accel_le_-0.5_braking | 5252 | 24 | 0.3929 | 0.2581 | 0.1615 | recurring_pattern | descriptive |
| truck_028 | accel_le_-0.5_braking | 6232 | 17 | 0.8416 | 0.2867 | 0.1711 | recurring_pattern | descriptive |
| truck_001 | grade_ge_1.5_steep_uphill | 820 | 2 | 2.6684 | 0.6845 | 0.5414 | dominated_by_worst_trip | descriptive |
| truck_004 | grade_ge_1.5_steep_uphill | 646 | 2 | 2.6714 | 0.3274 | 0.6491 | dominated_by_worst_trip | descriptive |
| truck_017 | grade_ge_1.5_steep_uphill | 635 | 4 | 2.3544 | 1.4951 | 0.8759 | dominated_by_worst_trip | descriptive |
| truck_020 | grade_ge_1.5_steep_uphill | 255 | 3 | 8.8919 | 0.8401 | 0.5398 | insufficient | insufficient_rows |
| truck_028 | grade_ge_1.5_steep_uphill | 672 | 3 | 8.8572 | 3.2864 | 0.4036 | recurring_pattern | descriptive |
| truck_001 | combined_uphill_accelerating | 22 | 3 | 4.8899 | 0.4698 | 0.5006 | insufficient | insufficient_rows |
| truck_004 | combined_uphill_accelerating | 157 | 6 | 3.5753 | 2.0427 | 0.4678 | insufficient | insufficient_rows |
| truck_017 | combined_uphill_accelerating | 137 | 8 | 2.8468 | 3.3450 | 0.4081 | insufficient | insufficient_rows |
| truck_020 | combined_uphill_accelerating | 89 | 5 | 6.8606 | 1.4499 | 0.3639 | insufficient | insufficient_rows |
| truck_028 | combined_uphill_accelerating | 134 | 5 | 14.3815 | 6.1831 | 0.4570 | insufficient | insufficient_rows |
| truck_001 | combined_downhill_braking | 201 | 7 | 0.3376 | 0.2221 | 0.8183 | insufficient | insufficient_rows |
| truck_004 | combined_downhill_braking | 337 | 5 | 0.3300 | 0.1285 | 0.5670 | insufficient | insufficient_rows |
| truck_017 | combined_downhill_braking | 246 | 11 | 0.6269 | 0.4628 | 0.4326 | insufficient | insufficient_rows |
| truck_020 | combined_downhill_braking | 187 | 8 | 0.6918 | 0.2597 | 0.5672 | insufficient | insufficient_rows |
| truck_028 | combined_downhill_braking | 275 | 6 | 0.4909 | 0.4199 | 0.7676 | insufficient | insufficient_rows |

### Experiment 3 — Error distribution (pooled conditions)

| Condition | n rows | n trips | n trucks | MAE | RMSE | median | P90 | P95 | max | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| speed_ge_60_highway | 24445 | 16 | 5 | 3.7330 | 5.3835 | 2.3065 | 9.6855 | 12.3770 | 27.4365 | 5.3660 |
| idle_engine_on_stopped | 8044 | 96 | 5 | 0.4687 | 1.1748 | 0.2813 | 1.0612 | 1.3218 | 46.6647 | 4.6982 |
| engine_off_stopped | 1232 | 95 | 5 | 0.3341 | 0.4390 | 0.2450 | 0.7159 | 0.8895 | 1.6912 | 3.6302 |
| accel_ge_0.5_accelerating | 30201 | 95 | 5 | 5.5051 | 7.5521 | 4.0736 | 11.7500 | 16.6880 | 30.9036 | 4.0966 |
| accel_le_-0.5_braking | 27117 | 96 | 5 | 0.5328 | 0.7391 | 0.3937 | 1.1715 | 1.4723 | 7.4878 | 3.7392 |
| grade_ge_1.5_steep_uphill | 3028 | 14 | 5 | 4.5764 | 6.2163 | 3.2886 | 10.3574 | 13.0389 | 46.6647 | 3.9648 |
| payload_heavy_gt0.66 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | zero-median |
| combined_highway_heavy_payload | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | zero-median |
| combined_uphill_accelerating | 539 | 27 | 5 | 6.7569 | 8.9240 | 5.1179 | 16.4357 | 19.1583 | 25.9083 | 3.7434 |
| combined_downhill_braking | 1246 | 37 | 5 | 0.6997 | 0.9524 | 0.5090 | 1.5129 | 1.9220 | 4.3261 | 3.7761 |
| combined_steep_uphill_heavy_payload | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | zero-median |

### Experiment 4 — Residual direction (per truck)

| Truck | n rows | mean signed | median signed | over% | under% | ~unbiased% |
| --- | --- | --- | --- | --- | --- | --- |
| truck_001 | 30817 | -1.6242 | -0.7484 | 25.52 | 74.48 | 5.60 |
| truck_004 | 32480 | -0.7010 | -0.1186 | 45.75 | 54.25 | 3.81 |
| truck_017 | 37073 | 1.3081 | 0.3115 | 61.58 | 38.42 | 4.37 |
| truck_020 | 33901 | -2.6476 | -1.1226 | 24.28 | 75.72 | 3.75 |
| truck_028 | 39018 | 3.6610 | 1.4549 | 76.13 | 23.87 | 1.96 |

### Experiment 4 — Residual direction (per condition)

| Condition | n rows | mean signed | median signed | over% | under% | ~unbiased% |
| --- | --- | --- | --- | --- | --- | --- |
| speed_ge_60_highway | 24445 | -0.0709 | -0.0930 | 48.68 | 51.32 | 1.47 |
| idle_engine_on_stopped | 8044 | 0.0969 | 0.0755 | 58.60 | 41.40 | 10.79 |
| engine_off_stopped | 1232 | 0.2002 | 0.1605 | 67.94 | 32.06 | 9.25 |
| accel_ge_0.5_accelerating | 30201 | 1.3266 | -0.5395 | 45.91 | 54.09 | 0.67 |
| accel_le_-0.5_braking | 27117 | 0.0293 | 0.0295 | 52.41 | 47.59 | 8.22 |
| grade_ge_1.5_steep_uphill | 3028 | 1.5224 | 1.1145 | 58.59 | 41.41 | 0.83 |
| payload_heavy_gt0.66 | 0 | n/a | n/a | n/a | n/a | n/a |
| combined_highway_heavy_payload | 0 | n/a | n/a | n/a | n/a | n/a |
| combined_uphill_accelerating | 539 | 1.9906 | -0.2912 | 46.75 | 53.25 | 0.37 |
| combined_downhill_braking | 1246 | -0.1762 | -0.0741 | 44.86 | 55.14 | 6.90 |
| combined_steep_uphill_heavy_payload | 0 | n/a | n/a | n/a | n/a | n/a |

### Experiment 5 — Paired model comparison (per condition)

| Condition | n rows | MAE full | MAE tel | delta MAE | RMSE full | RMSE tel | delta RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| speed_ge_60_highway | 24445 | 3.6580 | 3.7330 | 0.0750 | 5.3356 | 5.3835 | 0.0479 |
| idle_engine_on_stopped | 8044 | 0.1559 | 0.4687 | 0.3127 | 0.2003 | 1.1748 | 0.9745 |
| engine_off_stopped | 1232 | 0.1242 | 0.3341 | 0.2099 | 0.1686 | 0.4390 | 0.2704 |
| accel_ge_0.5_accelerating | 30201 | 6.5552 | 5.5051 | -1.0501 | 8.0022 | 7.5521 | -0.4501 |
| accel_le_-0.5_braking | 27117 | 0.2018 | 0.5328 | 0.3310 | 0.2698 | 0.7391 | 0.4693 |
| grade_ge_1.5_steep_uphill | 3028 | 5.1648 | 4.5764 | -0.5884 | 6.1272 | 6.2163 | 0.0891 |
| payload_heavy_gt0.66 | 0 | nan | nan | nan | nan | nan | nan |
| combined_highway_heavy_payload | 0 | nan | nan | nan | nan | nan | nan |
| combined_uphill_accelerating | 539 | 7.8231 | 6.7569 | -1.0661 | 9.2871 | 8.9240 | -0.3630 |
| combined_downhill_braking | 1246 | 0.2150 | 0.6997 | 0.4846 | 0.2860 | 0.9524 | 0.6664 |
| combined_steep_uphill_heavy_payload | 0 | nan | nan | nan | nan | nan | nan |

Paired delta convention: delta = metric_telematics_only - metric_full_operational; positive => full-operational has lower error (better)

Small deltas are descriptive; they are not evidence of model superiority. No hypothesis tests were run.

## Limitations

- Synthetic telemetry: results are pipeline validation, not observed real-world Indian truck telemetry.
- Only five test trucks: all truck-condition cells and residual tables are descriptive and unstable; not population estimates.
- 49T is represented by a single held-out truck; any 49T truck-condition result is descriptive and is NOT generalized to the 49T population.
- Truck-condition cells span a single truck by construction (n_trucks=1). Support is judged on rows and trips, not truck count; cells with < MIN_SUPPORT_ROWS rows or a single trip are flagged insufficient.
- The flagged_high_error heuristic is a descriptive deviation from truck-level and pooled-condition baselines; it is NOT a statistical interaction test (only five trucks). It is an investigative indicator only and may appear on cells flagged insufficient_support, where it is indicative rather than conclusive.
- Recurring-vs-isolated is a descriptive categorization based on the worst-trip share of squared error; it is not a mechanistic finding.
- Residual direction uses the explicit convention residual = y_pred - y_true (positive => overprediction); the approximately-unbiased figure uses a documented |residual| <= 0.05 L/h tolerance.
- Paired model deltas are descriptive; small differences are not evidence of model superiority. No hypothesis tests were run.
