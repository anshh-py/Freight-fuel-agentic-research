# M3.4 Missing / Noisy / Corrupted Telemetry Robustness

**Milestone:** M3.4

**Analysis frame:** synthetic corruption applied to copies of the model-input feature matrices; results are descriptive pipeline validation, NOT observed real-world sensor-failure data

**Protocol:** read-only: reuse frozen 5-truck test split and saved M1.2 artifacts; corruption applied to copies of the final model-input feature matrices after feature construction and before preprocessing; no training/refitting/tuning

**Synthetic-data caveat:** corruption is applied to copies of the model-input feature matrices and is explicitly NOT real-world sensor-failure validation.

## Selected models (from M1.2 validation, not test)

| Feature set | Selected model |
| --- | --- |
| full_operational | xgboost |
| telematics_only | xgboost |

## Verification / reconciliation

- Test trucks: truck_001, truck_004, truck_017, truck_020, truck_028

- Test rows: 173289; test trips: 96
- Categorical/binary columns preserved: True
- Prediction failures counted: True
- Synthetic corruption distinct from real-world validation: True

### Clean baseline reproduces M1.3 (within tolerance)

| Feature set | ref model | model match | |MAE diff| | |RMSE diff| | |R² diff| | within tol |
| --- | --- | --- | --- | --- | --- | --- |
| full_operational | xgboost | True | 0.00e+00 | 0.00e+00 | 0.00e+00 | True |
| telematics_only | xgboost | True | 0.00e+00 | 0.00e+00 | 0.00e+00 | True |

### Per-truck reconciliation (weighted per-truck MAE == pooled MAE)

| Feature set | pooled MAE | weighted per-truck MAE | max abs diff | reconciles |
| --- | --- | --- | --- | --- |
| full_operational | 2.568427 | 2.568427 | 2.97e-07 | True |
| telematics_only | 2.56596 | 2.56596 | 2.98e-07 | True |

## Corruption conventions

- Seed base: 20240617
- Preserved categorical: ['vehicle_class']; binary: ['ignition_status']
- Gaussian-noise severities (fraction of feature std): [0.25, 0.5]
- Random-missingness fractions: [0.05, 0.1, 0.2, 0.3]
- Spike fraction: 0.005; multiplier: 8.0 x std
- Clipping bounds: {"gross_vehicle_weight_kg": [15000.0, 60000.0], "payload_kg": [0.0, 40000.0], "vehicle_mass_kg": [15000.0, 60000.0], "speed_kph": [0.0, 90.0], "acceleration_mps2": [-5.0, 5.0], "road_grade_pct": [-10.0, 10.0], "engine_rpm": [0.0, 5000.0], "engine_load_pct": [0.0, 100.0], "throttle_position_pct": [0.0, 100.0], "torque_nm": [0.0, 4000.0], "coolant_temp_c": [0.0, 120.0], "intake_air_temp_c": [-40.0, 100.0], "ambient_temp_c": [-40.0, 60.0]}
- Gaussian-noise sigma & spike references are training-derived only (training_split_only).
- Random streams per-column deterministic: shared columns equivalent across feature sets.
- Combined scenario: The combined missingness+noise scenario uses an independent realization from the standalone 20% missingness and 0.25-noise scenarios (separate seeds), not a superset of them.

## Feature set: `full_operational`

Clean baseline (xgboost): MAE=2.568427, RMSE=4.527482, R²=0.987788, rows=173289

Numeric features corrupted: ['gross_vehicle_weight_kg', 'payload_kg', 'vehicle_mass_kg', 'speed_kph', 'acceleration_mps2', 'road_grade_pct', 'engine_rpm', 'engine_load_pct', 'throttle_position_pct', 'torque_nm', 'coolant_temp_c', 'intake_air_temp_c', 'ambient_temp_c']

### Scenario results

| Scenario | applied | affected cols | cells changed | fail rate | MAE | RMSE | R² | ΔMAE | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_missingness_05pct | yes | gross_vehicle_weight_kg, payload_kg, vehicle_mass_kg, speed_kph, acceleration_mps2, road_grade_pct, engine_rpm, engine_load_pct, throttle_position_pct, torque_nm, coolant_temp_c, intake_air_temp_c, ambient_temp_c | 112632 | 0.0000 | 5.7634 | 14.4458 | 0.8757 | 3.1949 | 2.2439 |
| random_missingness_10pct | yes | gross_vehicle_weight_kg, payload_kg, vehicle_mass_kg, speed_kph, acceleration_mps2, road_grade_pct, engine_rpm, engine_load_pct, throttle_position_pct, torque_nm, coolant_temp_c, intake_air_temp_c, ambient_temp_c | 225277 | 0.0000 | 9.2112 | 21.0459 | 0.7361 | 6.6428 | 3.5863 |
| random_missingness_20pct | yes | gross_vehicle_weight_kg, payload_kg, vehicle_mass_kg, speed_kph, acceleration_mps2, road_grade_pct, engine_rpm, engine_load_pct, throttle_position_pct, torque_nm, coolant_temp_c, intake_air_temp_c, ambient_temp_c | 450554 | 0.0000 | 16.8216 | 32.7424 | 0.3613 | 14.2532 | 6.5494 |
| random_missingness_30pct | yes | gross_vehicle_weight_kg, payload_kg, vehicle_mass_kg, speed_kph, acceleration_mps2, road_grade_pct, engine_rpm, engine_load_pct, throttle_position_pct, torque_nm, coolant_temp_c, intake_air_temp_c, ambient_temp_c | 675831 | 0.0000 | 25.2789 | 43.3512 | -0.1196 | 22.7105 | 9.8422 |
| complete_sensor_failure_engine_rpm | yes | engine_rpm | 173289 | 0.0000 | 5.5840 | 7.7863 | 0.9639 | 3.0156 | 2.1741 |
| complete_sensor_failure_throttle_position_pct | yes | throttle_position_pct | 173289 | 0.0000 | 2.7223 | 4.5531 | 0.9877 | 0.1539 | 1.0599 |
| complete_sensor_failure_speed_kph | yes | speed_kph | 173289 | 0.0000 | 2.7228 | 4.7280 | 0.9867 | 0.1543 | 1.0601 |
| complete_sensor_failure_coolant_temp_c | yes | coolant_temp_c | 173289 | 0.0000 | 2.5666 | 4.5270 | 0.9878 | -0.0018 | 0.9993 |
| gaussian_noise_severity_25pct | yes | gross_vehicle_weight_kg, payload_kg, vehicle_mass_kg, speed_kph, acceleration_mps2, road_grade_pct, engine_rpm, engine_load_pct, throttle_position_pct, torque_nm, coolant_temp_c, intake_air_temp_c, ambient_temp_c | 2252757 | 0.0000 | 6.5705 | 9.4748 | 0.9465 | 4.0021 | 2.5582 |
| gaussian_noise_severity_50pct | yes | gross_vehicle_weight_kg, payload_kg, vehicle_mass_kg, speed_kph, acceleration_mps2, road_grade_pct, engine_rpm, engine_load_pct, throttle_position_pct, torque_nm, coolant_temp_c, intake_air_temp_c, ambient_temp_c | 2252757 | 0.0000 | 11.8161 | 16.7499 | 0.8329 | 9.2477 | 4.6005 |
| fixed_bias_engine_rpm_plus200 | yes | engine_rpm | 173289 | 0.0000 | 4.0146 | 6.2414 | 0.9768 | 1.4462 | 1.5631 |
| fixed_bias_ambient_temp_c_plus5 | yes | ambient_temp_c | 173289 | 0.0000 | 2.5331 | 4.4983 | 0.9879 | -0.0353 | 0.9863 |
| fixed_bias_speed_kph_plus10 | yes | speed_kph | 173289 | 0.0000 | 2.5847 | 4.5399 | 0.9877 | 0.0163 | 1.0063 |
| isolated_extreme_spikes_engine_load_pct | yes | engine_load_pct | 866 | 0.0000 | 2.5979 | 4.5675 | 0.9876 | 0.0295 | 1.0115 |
| isolated_extreme_spikes_speed_kph | yes | speed_kph | 866 | 0.0000 | 2.5692 | 4.5282 | 0.9878 | 0.0007 | 1.0003 |
| isolated_extreme_spikes_torque_nm | yes | torque_nm | 866 | 0.0000 | 2.8301 | 6.0667 | 0.9781 | 0.2616 | 1.1019 |
| clipping_physical_bounds | yes | acceleration_mps2, ambient_temp_c, coolant_temp_c, engine_load_pct, engine_rpm, gross_vehicle_weight_kg, intake_air_temp_c, payload_kg, road_grade_pct, speed_kph, throttle_position_pct, torque_nm, vehicle_mass_kg | 11 | 0.0000 | 2.5684 | 4.5275 | 0.9878 | 0.0000 | 1.0000 |
| combined_missingness20pct_noise0.25 | yes | gross_vehicle_weight_kg, payload_kg, vehicle_mass_kg, speed_kph, acceleration_mps2, road_grade_pct, engine_rpm, engine_load_pct, throttle_position_pct, torque_nm, coolant_temp_c, intake_air_temp_c, ambient_temp_c | 2703311 | 0.0000 | 19.3110 | 33.3944 | 0.3356 | 16.7426 | 7.5186 |

`ΔMAE` = corrupted MAE − clean MAE (L/h); `ratio` = corrupted MAE / clean MAE. Prediction-failure rate = fraction of test rows with aborted or NaN predictions (failures are counted, never silently dropped).

### Per-truck MAE degradation (most damaging scenario)

Scenario: `random_missingness_30pct` (ratio 9.84219)

| Truck | n rows | MAE | clean MAE | ΔMAE |
| --- | --- | --- | --- | --- |
| truck_001 | 30817 | 25.6551 | 1.3023 | 24.3529 |
| truck_004 | 32480 | 25.5914 | 1.5492 | 24.0422 |
| truck_017 | 37073 | 24.6845 | 2.2452 | 22.4393 |
| truck_020 | 33901 | 24.4084 | 3.8660 | 20.5424 |
| truck_028 | 39018 | 26.0429 | 3.5966 | 22.4463 |

## Feature set: `telematics_only`

Clean baseline (xgboost): MAE=2.56596, RMSE=4.371276, R²=0.988616, rows=173289

Numeric features corrupted: ['gross_vehicle_weight_kg', 'payload_kg', 'vehicle_mass_kg', 'speed_kph', 'acceleration_mps2', 'road_grade_pct', 'coolant_temp_c', 'intake_air_temp_c', 'ambient_temp_c']

### Scenario results

| Scenario | applied | affected cols | cells changed | fail rate | MAE | RMSE | R² | ΔMAE | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_missingness_05pct | yes | gross_vehicle_weight_kg, payload_kg, vehicle_mass_kg, speed_kph, acceleration_mps2, road_grade_pct, coolant_temp_c, intake_air_temp_c, ambient_temp_c | 77976 | 0.0000 | 8.1684 | 21.1761 | 0.7329 | 5.6025 | 3.1834 |
| random_missingness_10pct | yes | gross_vehicle_weight_kg, payload_kg, vehicle_mass_kg, speed_kph, acceleration_mps2, road_grade_pct, coolant_temp_c, intake_air_temp_c, ambient_temp_c | 155961 | 0.0000 | 13.9841 | 30.6576 | 0.4401 | 11.4182 | 5.4499 |
| random_missingness_20pct | yes | gross_vehicle_weight_kg, payload_kg, vehicle_mass_kg, speed_kph, acceleration_mps2, road_grade_pct, coolant_temp_c, intake_air_temp_c, ambient_temp_c | 311922 | 0.0000 | 26.3157 | 46.2645 | -0.2751 | 23.7497 | 10.2557 |
| random_missingness_30pct | yes | gross_vehicle_weight_kg, payload_kg, vehicle_mass_kg, speed_kph, acceleration_mps2, road_grade_pct, coolant_temp_c, intake_air_temp_c, ambient_temp_c | 467883 | 0.0000 | 39.5802 | 60.4839 | -1.1794 | 37.0142 | 15.4251 |
| complete_sensor_failure_speed_kph | yes | speed_kph | 173289 | 0.0000 | 25.1372 | 37.2140 | 0.1750 | 22.5713 | 9.7964 |
| complete_sensor_failure_coolant_temp_c | yes | coolant_temp_c | 173289 | 0.0000 | 2.6038 | 4.3686 | 0.9886 | 0.0379 | 1.0148 |
| gaussian_noise_severity_25pct | yes | gross_vehicle_weight_kg, payload_kg, vehicle_mass_kg, speed_kph, acceleration_mps2, road_grade_pct, coolant_temp_c, intake_air_temp_c, ambient_temp_c | 1559601 | 0.0000 | 9.4652 | 15.5158 | 0.8566 | 6.8993 | 3.6888 |
| gaussian_noise_severity_50pct | yes | gross_vehicle_weight_kg, payload_kg, vehicle_mass_kg, speed_kph, acceleration_mps2, road_grade_pct, coolant_temp_c, intake_air_temp_c, ambient_temp_c | 1559601 | 0.0000 | 16.3648 | 26.5139 | 0.5812 | 13.7988 | 6.3776 |
| fixed_bias_ambient_temp_c_plus5 | yes | ambient_temp_c | 173289 | 0.0000 | 2.5732 | 4.3871 | 0.9885 | 0.0073 | 1.0028 |
| fixed_bias_speed_kph_plus10 | yes | speed_kph | 173289 | 0.0000 | 7.2105 | 11.8751 | 0.9160 | 4.6445 | 2.8101 |
| isolated_extreme_spikes_speed_kph | yes | speed_kph | 866 | 0.0000 | 2.6786 | 5.0662 | 0.9847 | 0.1126 | 1.0439 |
| clipping_physical_bounds | yes | acceleration_mps2, ambient_temp_c, coolant_temp_c, gross_vehicle_weight_kg, intake_air_temp_c, payload_kg, road_grade_pct, speed_kph, vehicle_mass_kg | 11 | 0.0000 | 2.5660 | 4.3713 | 0.9886 | 0.0000 | 1.0000 |
| combined_missingness20pct_noise0.25 | yes | gross_vehicle_weight_kg, payload_kg, vehicle_mass_kg, speed_kph, acceleration_mps2, road_grade_pct, coolant_temp_c, intake_air_temp_c, ambient_temp_c | 1871523 | 0.0000 | 28.6033 | 46.2124 | -0.2723 | 26.0373 | 11.1472 |

`ΔMAE` = corrupted MAE − clean MAE (L/h); `ratio` = corrupted MAE / clean MAE. Prediction-failure rate = fraction of test rows with aborted or NaN predictions (failures are counted, never silently dropped).

### Per-truck MAE degradation (most damaging scenario)

Scenario: `random_missingness_30pct` (ratio 15.425096)

| Truck | n rows | MAE | clean MAE | ΔMAE |
| --- | --- | --- | --- | --- |
| truck_001 | 30817 | 34.9076 | 1.8832 | 33.0244 |
| truck_004 | 32480 | 35.1750 | 1.5358 | 33.6392 |
| truck_017 | 37073 | 39.9496 | 2.0527 | 37.8970 |
| truck_020 | 33901 | 37.6559 | 2.9305 | 34.7254 |
| truck_028 | 39018 | 48.2586 | 4.1337 | 44.1249 |

## Limitations

- Synthetic telemetry: these are pipeline-corruption experiments on synthetic data, not observed real-world sensor-failure validation. They characterize model sensitivity to the injected corruptions only.
- Only five test trucks: all per-truck breakdowns are descriptive and unstable; not population estimates.
- Corruption is restricted to numeric model-input features; categorical vehicle_class and binary ignition_status are kept unchanged. Real-world sensor failures may also affect other signal classes.
- No feature columns are removed; complete sensor failure is simulated with all-NaN values while preserving the model's expected schema.
- Prediction failures (aborted, NaN, +inf or -inf predictions) are counted and reported as a failure rate; metrics are computed on the surviving finite rows and should always be read alongside the failure rate.
- The clipping bounds are documented physical limits; they are already satisfied by most data, so the clipping scenario primarily measures the effect of the bound enforcement itself.
- MAE degradation is reported against each feature set's own clean baseline; cross-feature-set ratios are not directly comparable because baselines differ.
