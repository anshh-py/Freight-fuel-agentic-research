# M3.2 Operating-Condition Robustness Analysis

**Milestone:** M3.2

**Analysis frame:** conditional on the five frozen test trucks; descriptive only; no hypothesis tests or significance claims

**Protocol:** read-only: reuse frozen 5-truck test split and saved M1.2 artifacts; no training/refitting/tuning; no test-set-driven model selection

**Synthetic-data caveat:** results are pipeline validation; they do not represent observed real-world Indian truck telemetry.

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

## Insufficient-support note

Support is classified, not a single boolean. A subgroup is 'insufficient' (insufficient_support=True) when it is empty, has fewer than 500 rows, or spans fewer than 2 distinct trucks. A subgroup with enough rows and at least 2 trucks but fewer than all 5 test trucks is reported DESCRIPTIVELY with limited_truck_coverage=True and an explicit warning: it does not support population-level generalization. Row count alone never proves reliability.

## Feature set: `full_operational`

Rows = 173289; trips = 96; trucks = 5

### Route-type metrics (trip level)

| Route type | n rows | n trips | n trucks | MAE | RMSE |
| --- | --- | --- | --- | --- | --- |
| highway | 25919 | 14 | 5 | 3.5797 | 5.2955 |
| urban | 30815 | 16 | 5 | 2.7510 | 4.6470 |
| mountainous | 33550 | 19 | 5 | 2.7199 | 4.8159 |
| mixed | 42286 | 26 | 5 | 2.4994 | 4.3467 |
| city | 40719 | 21 | 4 | 1.7334 | 3.7831 |

### Condition table (coverage + support status + MAE/RMSE)

| Condition | n rows | n trips | n trucks | MAE | RMSE | support status |
| --- | --- | --- | --- | --- | --- | --- |
| all | 173289 | 96 | 5 | 2.5684 | 4.5275 | descriptive_full_truck_coverage |
| engine_on | 172057 | 96 | 5 | 2.5859 | 4.5436 | descriptive_full_truck_coverage |
| engine_off | 1232 | 95 | 5 | 0.1242 | 0.1686 | descriptive_full_truck_coverage |
| speed_ge_60_highway | 24445 | 16 | 5 | 3.6580 | 5.3356 | descriptive_full_truck_coverage |
| idle_engine_on_stopped | 8044 | 96 | 5 | 0.1559 | 0.2003 | descriptive_full_truck_coverage |
| engine_off_stopped | 1232 | 95 | 5 | 0.1242 | 0.1686 | descriptive_full_truck_coverage |
| accel_ge_0.5_accelerating | 30201 | 95 | 5 | 6.5552 | 8.0022 | descriptive_full_truck_coverage |
| accel_le_-0.5_braking | 27117 | 96 | 5 | 0.2018 | 0.2698 | descriptive_full_truck_coverage |
| grade_ge_1.5_steep_uphill | 3028 | 14 | 5 | 5.1648 | 6.1272 | descriptive_full_truck_coverage |
| payload_light_le_0.33 | 158889 | 88 | 5 | 2.6421 | 4.6449 | descriptive_full_truck_coverage |
| payload_medium_gt0.33_le0.66 | 14400 | 8 | 3 | 1.7555 | 2.9349 | descriptive_limited_truck_coverage |
| payload_heavy_gt0.66 | 0 | 0 | 0 | nan | nan | empty |
| payload_invalid_gvw | 0 | 0 | 0 | nan | nan | empty |
| combined_highway_heavy_payload | 0 | 0 | 0 | nan | nan | empty |
| combined_uphill_accelerating | 539 | 27 | 5 | 7.8231 | 9.2871 | descriptive_full_truck_coverage |
| combined_downhill_braking | 1246 | 37 | 5 | 0.2150 | 0.2860 | descriptive_full_truck_coverage |
| combined_steep_uphill_heavy_payload | 0 | 0 | 0 | nan | nan | empty |

*Support status legend:* `descriptive_full_truck_coverage` covers all 5 frozen test trucks; `descriptive_limited_truck_coverage` has enough rows and at least 2 trucks but fewer than the full test-truck set — reported descriptively with an explicit limited-coverage warning, NOT as unusable; `insufficient_rows` / `insufficient_truck_coverage` / `empty` are genuinely under-sampled (insufficient_support=True).

> **Medium-payload subgroup:** descriptive result with incomplete truck coverage (n_trucks=3 of 5). It is NOT flagged unusable, but it does not support population-level generalization.

### Error concentration (worst first, by squared-error share)

| Condition | sq-err share | row share | MAE | RMSE |
| --- | --- | --- | --- | --- |
| accel_ge_0.5_accelerating | 0.5445 | 0.1743 | 6.5552 | 8.0022 |
| speed_ge_60_highway | 0.1959 | 0.1411 | 3.6580 | 5.3356 |
| grade_ge_1.5_steep_uphill | 0.0320 | 0.0175 | 5.1648 | 6.1272 |
| combined_uphill_accelerating | 0.0131 | 0.0031 | 7.8231 | 9.2871 |
| accel_le_-0.5_braking | 0.0006 | 0.1565 | 0.2018 | 0.2698 |
| idle_engine_on_stopped | 0.0001 | 0.0464 | 0.1559 | 0.2003 |
| combined_downhill_braking | 0.0000 | 0.0072 | 0.2150 | 0.2860 |
| engine_off_stopped | 0.0000 | 0.0071 | 0.1242 | 0.1686 |
| payload_heavy_gt0.66 | 0.0000 | 0.0000 | n/a | n/a |
| combined_highway_heavy_payload | 0.0000 | 0.0000 | n/a | n/a |
| combined_steep_uphill_heavy_payload | 0.0000 | 0.0000 | n/a | n/a |

### Per-truck × condition (informative conditions)

**speed_ge_60_highway** (n=24445)
| Truck | n rows | MAE | RMSE |
| --- | --- | --- | --- |
| truck_001 | 4014 | 2.5796 | 3.3671 |
| truck_004 | 6394 | 1.9754 | 2.8568 |
| truck_017 | 5533 | 3.4340 | 4.5749 |
| truck_020 | 3730 | 7.0591 | 9.3539 |
| truck_028 | 4774 | 4.4204 | 5.7175 |

**idle_engine_on_stopped** (n=8044)
| Truck | n rows | MAE | RMSE |
| --- | --- | --- | --- |
| truck_001 | 1502 | 0.0880 | 0.0985 |
| truck_004 | 805 | 0.1394 | 0.1789 |
| truck_017 | 1513 | 0.0904 | 0.1212 |
| truck_020 | 1863 | 0.1358 | 0.1590 |
| truck_028 | 2361 | 0.2627 | 0.3005 |

**engine_off_stopped** (n=1232)
| Truck | n rows | MAE | RMSE |
| --- | --- | --- | --- |
| truck_001 | 222 | 0.0675 | 0.0987 |
| truck_004 | 153 | 0.1016 | 0.1426 |
| truck_017 | 314 | 0.1347 | 0.1824 |
| truck_020 | 330 | 0.1205 | 0.1547 |
| truck_028 | 213 | 0.1900 | 0.2312 |

**accel_ge_0.5_accelerating** (n=30201)
| Truck | n rows | MAE | RMSE |
| --- | --- | --- | --- |
| truck_001 | 5143 | 3.0050 | 3.4071 |
| truck_004 | 5799 | 4.3024 | 4.5105 |
| truck_017 | 7203 | 5.8804 | 6.5119 |
| truck_020 | 5725 | 9.0823 | 9.9822 |
| truck_028 | 6331 | 9.9853 | 11.7922 |

**accel_le_-0.5_braking** (n=27117)
| Truck | n rows | MAE | RMSE |
| --- | --- | --- | --- |
| truck_001 | 4776 | 0.1079 | 0.1320 |
| truck_004 | 4947 | 0.1282 | 0.1614 |
| truck_017 | 5910 | 0.1692 | 0.1983 |
| truck_020 | 5252 | 0.2256 | 0.2851 |
| truck_028 | 6232 | 0.3432 | 0.4205 |

**grade_ge_1.5_steep_uphill** (n=3028)
| Truck | n rows | MAE | RMSE |
| --- | --- | --- | --- |
| truck_001 | 820 | 4.2056 | 4.3745 |
| truck_004 | 646 | 3.4664 | 4.0775 |
| truck_017 | 635 | 4.9597 | 5.4072 |
| truck_020 | 255 | 12.2679 | 13.1974 |
| truck_028 | 672 | 5.4663 | 6.0094 |

**payload_heavy_gt0.66** (n=0)
| Truck | n rows | MAE | RMSE |
| --- | --- | --- | --- |

**combined_highway_heavy_payload** (n=0)
| Truck | n rows | MAE | RMSE |
| --- | --- | --- | --- |

**combined_uphill_accelerating** (n=539)
| Truck | n rows | MAE | RMSE |
| --- | --- | --- | --- |
| truck_001 | 22 | 4.1126 | 4.2355 |
| truck_004 | 157 | 4.3536 | 4.6353 |
| truck_017 | 137 | 5.5215 | 6.2769 |
| truck_020 | 89 | 11.7291 | 11.9368 |
| truck_028 | 134 | 12.2560 | 13.5606 |

**combined_downhill_braking** (n=1246)
| Truck | n rows | MAE | RMSE |
| --- | --- | --- | --- |
| truck_001 | 201 | 0.0868 | 0.1122 |
| truck_004 | 337 | 0.1253 | 0.1592 |
| truck_017 | 246 | 0.2284 | 0.2691 |
| truck_020 | 187 | 0.2866 | 0.3901 |
| truck_028 | 275 | 0.3581 | 0.4026 |

**combined_steep_uphill_heavy_payload** (n=0)
| Truck | n rows | MAE | RMSE |
| --- | --- | --- | --- |

Payload bin edges (load factor): [0.33, 0.66]

## Feature set: `telematics_only`

Rows = 173289; trips = 96; trucks = 5

### Route-type metrics (trip level)

| Route type | n rows | n trips | n trucks | MAE | RMSE |
| --- | --- | --- | --- | --- | --- |
| highway | 25919 | 14 | 5 | 3.6832 | 5.3701 |
| mountainous | 33550 | 19 | 5 | 2.7814 | 5.2869 |
| urban | 30815 | 16 | 5 | 2.6131 | 4.6157 |
| mixed | 42286 | 26 | 5 | 2.2201 | 3.4556 |
| city | 40719 | 21 | 4 | 2.0008 | 3.3777 |

### Condition table (coverage + support status + MAE/RMSE)

| Condition | n rows | n trips | n trucks | MAE | RMSE | support status |
| --- | --- | --- | --- | --- | --- | --- |
| all | 173289 | 96 | 5 | 2.5660 | 4.3713 | descriptive_full_truck_coverage |
| engine_on | 172057 | 96 | 5 | 2.5819 | 4.3867 | descriptive_full_truck_coverage |
| engine_off | 1232 | 95 | 5 | 0.3341 | 0.4390 | descriptive_full_truck_coverage |
| speed_ge_60_highway | 24445 | 16 | 5 | 3.7330 | 5.3835 | descriptive_full_truck_coverage |
| idle_engine_on_stopped | 8044 | 96 | 5 | 0.4687 | 1.1748 | descriptive_full_truck_coverage |
| engine_off_stopped | 1232 | 95 | 5 | 0.3341 | 0.4390 | descriptive_full_truck_coverage |
| accel_ge_0.5_accelerating | 30201 | 95 | 5 | 5.5051 | 7.5521 | descriptive_full_truck_coverage |
| accel_le_-0.5_braking | 27117 | 96 | 5 | 0.5328 | 0.7391 | descriptive_full_truck_coverage |
| grade_ge_1.5_steep_uphill | 3028 | 14 | 5 | 4.5764 | 6.2163 | descriptive_full_truck_coverage |
| payload_light_le_0.33 | 158889 | 88 | 5 | 2.6205 | 4.4715 | descriptive_full_truck_coverage |
| payload_medium_gt0.33_le0.66 | 14400 | 8 | 3 | 1.9645 | 3.0549 | descriptive_limited_truck_coverage |
| payload_heavy_gt0.66 | 0 | 0 | 0 | nan | nan | empty |
| payload_invalid_gvw | 0 | 0 | 0 | nan | nan | empty |
| combined_highway_heavy_payload | 0 | 0 | 0 | nan | nan | empty |
| combined_uphill_accelerating | 539 | 27 | 5 | 6.7569 | 8.9240 | descriptive_full_truck_coverage |
| combined_downhill_braking | 1246 | 37 | 5 | 0.6997 | 0.9524 | descriptive_full_truck_coverage |
| combined_steep_uphill_heavy_payload | 0 | 0 | 0 | nan | nan | empty |

*Support status legend:* `descriptive_full_truck_coverage` covers all 5 frozen test trucks; `descriptive_limited_truck_coverage` has enough rows and at least 2 trucks but fewer than the full test-truck set — reported descriptively with an explicit limited-coverage warning, NOT as unusable; `insufficient_rows` / `insufficient_truck_coverage` / `empty` are genuinely under-sampled (insufficient_support=True).

> **Medium-payload subgroup:** descriptive result with incomplete truck coverage (n_trucks=3 of 5). It is NOT flagged unusable, but it does not support population-level generalization.

### Error concentration (worst first, by squared-error share)

| Condition | sq-err share | row share | MAE | RMSE |
| --- | --- | --- | --- | --- |
| accel_ge_0.5_accelerating | 0.5202 | 0.1743 | 5.5051 | 7.5521 |
| speed_ge_60_highway | 0.2140 | 0.1411 | 3.7330 | 5.3835 |
| grade_ge_1.5_steep_uphill | 0.0353 | 0.0175 | 4.5764 | 6.2163 |
| combined_uphill_accelerating | 0.0130 | 0.0031 | 6.7569 | 8.9240 |
| accel_le_-0.5_braking | 0.0045 | 0.1565 | 0.5328 | 0.7391 |
| idle_engine_on_stopped | 0.0034 | 0.0464 | 0.4687 | 1.1748 |
| combined_downhill_braking | 0.0003 | 0.0072 | 0.6997 | 0.9524 |
| engine_off_stopped | 0.0001 | 0.0071 | 0.3341 | 0.4390 |
| payload_heavy_gt0.66 | 0.0000 | 0.0000 | n/a | n/a |
| combined_highway_heavy_payload | 0.0000 | 0.0000 | n/a | n/a |
| combined_steep_uphill_heavy_payload | 0.0000 | 0.0000 | n/a | n/a |

### Per-truck × condition (informative conditions)

**speed_ge_60_highway** (n=24445)
| Truck | n rows | MAE | RMSE |
| --- | --- | --- | --- |
| truck_001 | 4014 | 2.4115 | 3.1110 |
| truck_004 | 6394 | 2.4395 | 3.2159 |
| truck_017 | 5533 | 2.4807 | 3.5945 |
| truck_020 | 3730 | 6.5169 | 8.2260 |
| truck_028 | 4774 | 5.8526 | 7.6530 |

**idle_engine_on_stopped** (n=8044)
| Truck | n rows | MAE | RMSE |
| --- | --- | --- | --- |
| truck_001 | 1502 | 0.2270 | 0.3650 |
| truck_004 | 805 | 0.2987 | 0.3862 |
| truck_017 | 1513 | 0.3731 | 0.9827 |
| truck_020 | 1863 | 0.3329 | 0.4362 |
| truck_028 | 2361 | 0.8488 | 1.9488 |

**engine_off_stopped** (n=1232)
| Truck | n rows | MAE | RMSE |
| --- | --- | --- | --- |
| truck_001 | 222 | 0.1802 | 0.2222 |
| truck_004 | 153 | 0.2437 | 0.2841 |
| truck_017 | 314 | 0.3429 | 0.4165 |
| truck_020 | 330 | 0.2902 | 0.3618 |
| truck_028 | 213 | 0.6145 | 0.7394 |

**accel_ge_0.5_accelerating** (n=30201)
| Truck | n rows | MAE | RMSE |
| --- | --- | --- | --- |
| truck_001 | 5143 | 4.0491 | 4.8637 |
| truck_004 | 5799 | 2.3495 | 3.1287 |
| truck_017 | 7203 | 4.5890 | 5.7360 |
| truck_020 | 5725 | 4.7811 | 5.5229 |
| truck_028 | 6331 | 11.2753 | 13.3744 |

**accel_le_-0.5_braking** (n=27117)
| Truck | n rows | MAE | RMSE |
| --- | --- | --- | --- |
| truck_001 | 4776 | 0.3395 | 0.4681 |
| truck_004 | 4947 | 0.4328 | 0.5461 |
| truck_017 | 5910 | 0.4730 | 0.6734 |
| truck_020 | 5252 | 0.4734 | 0.6267 |
| truck_028 | 6232 | 0.8673 | 1.1004 |

**grade_ge_1.5_steep_uphill** (n=3028)
| Truck | n rows | MAE | RMSE |
| --- | --- | --- | --- |
| truck_001 | 820 | 2.3629 | 2.9859 |
| truck_004 | 646 | 2.5427 | 3.2630 |
| truck_017 | 635 | 3.9225 | 5.2964 |
| truck_020 | 255 | 8.1712 | 8.8940 |
| truck_028 | 672 | 8.4860 | 9.8227 |

**payload_heavy_gt0.66** (n=0)
| Truck | n rows | MAE | RMSE |
| --- | --- | --- | --- |

**combined_highway_heavy_payload** (n=0)
| Truck | n rows | MAE | RMSE |
| --- | --- | --- | --- |

**combined_uphill_accelerating** (n=539)
| Truck | n rows | MAE | RMSE |
| --- | --- | --- | --- |
| truck_001 | 22 | 4.3656 | 4.7970 |
| truck_004 | 157 | 2.7251 | 3.5734 |
| truck_017 | 137 | 3.7925 | 4.6920 |
| truck_020 | 89 | 7.7648 | 8.1027 |
| truck_028 | 134 | 14.2349 | 15.3455 |

**combined_downhill_braking** (n=1246)
| Truck | n rows | MAE | RMSE |
| --- | --- | --- | --- |
| truck_001 | 201 | 0.4753 | 0.6617 |
| truck_004 | 337 | 0.4218 | 0.5491 |
| truck_017 | 246 | 0.9902 | 1.2541 |
| truck_020 | 187 | 0.8509 | 1.0163 |
| truck_028 | 275 | 0.8415 | 1.1449 |

**combined_steep_uphill_heavy_payload** (n=0)
| Truck | n rows | MAE | RMSE |
| --- | --- | --- | --- |

Payload bin edges (load factor): [0.33, 0.66]

## Limitations

- Synthetic telemetry: results are pipeline validation, not observed real-world Indian truck telemetry.
- Payload load-factor ceiling: across the entire synthetic dataset (all 963,697 rows) the load factor payload_kg / gross_vehicle_weight_kg never exceeds 0.3886. There is therefore NO heavy-payload regime (LF > 0.66) in the data; the heavy and combined-heavy cells are empty by construction, not by an error, and cannot be evaluated. Only the light (<= 0.33) and medium (0.33-0.66) regimes are populated.
- Only five test trucks: any truck-level or per-condition breakdown is descriptive and unstable; not a population estimate.
- 49T is represented by a single held-out truck; its per-condition results are descriptive only and are NOT generalized to the 49T population.
- Combined-condition subgroups are small; they are reported with explicit insufficient-support flags and must not be over-interpreted.
- Payload load factor uses payload_kg / gross_vehicle_weight_kg (intra-trip constant); invalid denominators are bucketed separately.
- Insufficient-support flag is a combined caution (rows AND trucks), not a proof of reliability.
