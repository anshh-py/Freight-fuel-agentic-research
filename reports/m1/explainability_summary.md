# M1.4 Explainability & Feature Analysis

Protocol: explain validation-selected XGBoost models on a deterministic sample of the 5 held-out test trucks; preprocessors loaded from disk, never refitted; no retraining; results are model explanations on synthetic data, not causal/real-world claims

**Scope:** validation-selected XGBoost models, both feature sets, explained on a deterministic sample of the five held-out test trucks.

## Sampling (deterministic per-truck uniform subsample with a fixed np.random.RandomState seed)

- Sample size per truck: 500
- Seed: 20240617
- All five test trucks represented: True

## Verification

- All five test trucks represented: True
- No train/validation truck in sample: True
- Preprocessors loaded, never refitted: True
- Models loaded, not retrained: True
- Source of truth for truck split: split manifest

## Important caveat

These explanations describe models trained on the **synthetic** telemetry suite. They are model-behavior explanations, not causal claims and not evidence of real-world fuel-consumption drivers.

## Feature set: `full_operational`

- Sample rows: 2500 (per truck: {'truck_001': 500, 'truck_004': 500, 'truck_017': 500, 'truck_020': 500, 'truck_028': 500})
- SHAP reconstruction: max|dev|=1.57e-04, fraction within tol=1.0000, pass=True
- Model artifact: 469059 bytes; preprocessor: 865 bytes

### Global feature importance (mean |SHAP|)

| # | Feature | Mean |SHAP| (L/h) |
| --- | --- | --- |
| 1 | Torque (N·m) | 31.7242 |
| 2 | Engine load (%) | 4.9365 |
| 3 | Engine speed (rpm) | 1.2966 |
| 4 | Vehicle mass (kg) | 0.5798 |
| 5 | Gross vehicle weight (kg) | 0.4867 |
| 6 | Payload (kg) | 0.3697 |
| 7 | Speed (kph) | 0.2704 |
| 8 | Throttle position (%) | 0.2492 |
| 9 | Intake air temp (°C) | 0.1729 |
| 10 | Ambient temp (°C) | 0.1116 |
| 11 | Acceleration (m/s²) | 0.0708 |
| 12 | Coolant temp (°C) | 0.0215 |
| 13 | Vehicle class (40T) | 0.0139 |
| 14 | Vehicle class (49T) | 0.0123 |
| 15 | Ignition status | 0.0062 |
| 16 | Road grade (%) | 0.0054 |

### Per-truck mean |SHAP| (top 5 per truck)

| Truck | Top features (mean |SHAP|) |
| --- | --- |
| truck_001 | torque_nm=25.852; engine_load_pct=4.649; engine_rpm=1.418; gross_vehicle_weight_kg=0.832; vehicle_mass_kg=0.324 |
| truck_004 | torque_nm=28.496; engine_load_pct=4.930; engine_rpm=0.933; gross_vehicle_weight_kg=0.787; payload_kg=0.346 |
| truck_017 | torque_nm=34.026; engine_load_pct=4.959; engine_rpm=1.322; vehicle_mass_kg=0.494; payload_kg=0.349 |
| truck_020 | torque_nm=31.164; engine_load_pct=5.104; engine_rpm=1.311; vehicle_mass_kg=0.421; payload_kg=0.313 |
| truck_028 | torque_nm=39.084; engine_load_pct=5.041; engine_rpm=1.498; vehicle_mass_kg=1.364; gross_vehicle_weight_kg=0.681 |

### Individual explanation records (deterministic rules)

| Truck | Rule | Actual L/h | Pred L/h | Residual |
| --- | --- | --- | --- | --- |
| truck_001 | largest_abs_residual | 60.828 | 51.711 | +9.118 |
| truck_001 | median_abs_residual | 2.087 | 1.816 | +0.271 |
| truck_001 | closest_to_zero_residual | 1.090 | 1.094 | -0.004 |
| truck_004 | largest_abs_residual | 82.150 | 89.127 | -6.977 |
| truck_004 | median_abs_residual | 13.958 | 14.335 | -0.377 |
| truck_004 | closest_to_zero_residual | 2.054 | 2.054 | -0.000 |
| truck_017 | largest_abs_residual | 101.681 | 112.957 | -11.277 |
| truck_017 | median_abs_residual | 10.478 | 11.013 | -0.535 |
| truck_017 | closest_to_zero_residual | 1.371 | 1.371 | +0.000 |
| truck_020 | largest_abs_residual | 94.318 | 77.015 | +17.303 |
| truck_020 | median_abs_residual | 2.690 | 1.830 | +0.860 |
| truck_020 | closest_to_zero_residual | 2.648 | 2.650 | -0.002 |
| truck_028 | largest_abs_residual | 107.317 | 131.215 | -23.898 |
| truck_028 | median_abs_residual | 6.903 | 8.443 | -1.540 |
| truck_028 | closest_to_zero_residual | 12.045 | 12.055 | -0.009 |

### Plots

- plots/explainability/global_importance_full_operational.png
- plots/explainability/summary_beeswarm_full_operational.png
- plots/explainability/dependence_full_operational_torque_nm.png
- plots/explainability/dependence_full_operational_engine_load_pct.png
- plots/explainability/dependence_full_operational_engine_rpm.png
- plots/explainability/per_truck_importance_full_operational.png
- plots/explainability/waterfall_full_operational_truck_001_largest_abs_residual.png
- plots/explainability/waterfall_full_operational_truck_001_median_abs_residual.png

---

## Feature set: `telematics_only`

- Sample rows: 2500 (per truck: {'truck_001': 500, 'truck_004': 500, 'truck_017': 500, 'truck_020': 500, 'truck_028': 500})
- SHAP reconstruction: max|dev|=1.44e-04, fraction within tol=1.0000, pass=True
- Model artifact: 468595 bytes; preprocessor: 813 bytes

### Global feature importance (mean |SHAP|)

| # | Feature | Mean |SHAP| (L/h) |
| --- | --- | --- |
| 1 | Acceleration (m/s²) | 32.2338 |
| 2 | Speed (kph) | 10.2649 |
| 3 | Vehicle mass (kg) | 3.5022 |
| 4 | Road grade (%) | 2.9530 |
| 5 | Gross vehicle weight (kg) | 1.9570 |
| 6 | Payload (kg) | 0.9817 |
| 7 | Vehicle class (49T) | 0.8791 |
| 8 | Intake air temp (°C) | 0.2742 |
| 9 | Vehicle class (40T) | 0.2586 |
| 10 | Ambient temp (°C) | 0.1614 |
| 11 | Coolant temp (°C) | 0.1094 |
| 12 | Ignition status | 0.0053 |

### Per-truck mean |SHAP| (top 5 per truck)

| Truck | Top features (mean |SHAP|) |
| --- | --- |
| truck_001 | acceleration_mps2=28.557; speed_kph=9.599; vehicle_mass_kg=4.414; gross_vehicle_weight_kg=3.317; road_grade_pct=3.298 |
| truck_004 | acceleration_mps2=30.137; speed_kph=7.637; vehicle_mass_kg=4.407; gross_vehicle_weight_kg=3.548; road_grade_pct=3.030 |
| truck_017 | acceleration_mps2=33.787; speed_kph=10.143; road_grade_pct=2.773; vehicle_mass_kg=2.108; payload_kg=0.968 |
| truck_020 | acceleration_mps2=32.947; speed_kph=10.985; road_grade_pct=2.439; vehicle_mass_kg=2.038; payload_kg=0.732 |
| truck_028 | acceleration_mps2=35.741; speed_kph=12.961; vehicle_mass_kg=4.544; road_grade_pct=3.226; gross_vehicle_weight_kg=2.448 |

### Individual explanation records (deterministic rules)

| Truck | Rule | Actual L/h | Pred L/h | Residual |
| --- | --- | --- | --- | --- |
| truck_001 | largest_abs_residual | 85.479 | 74.130 | +11.348 |
| truck_001 | median_abs_residual | 2.181 | 1.296 | +0.885 |
| truck_001 | closest_to_zero_residual | 1.090 | 1.090 | +0.000 |
| truck_004 | largest_abs_residual | 85.196 | 69.243 | +15.953 |
| truck_004 | median_abs_residual | 1.995 | 2.800 | -0.806 |
| truck_004 | closest_to_zero_residual | 2.045 | 2.042 | +0.003 |
| truck_017 | largest_abs_residual | 95.820 | 111.487 | -15.667 |
| truck_017 | median_abs_residual | 2.651 | 3.523 | -0.872 |
| truck_017 | closest_to_zero_residual | 71.126 | 71.127 | -0.000 |
| truck_020 | largest_abs_residual | 112.809 | 93.281 | +19.529 |
| truck_020 | median_abs_residual | 56.889 | 55.549 | +1.340 |
| truck_020 | closest_to_zero_residual | 2.609 | 2.608 | +0.001 |
| truck_028 | largest_abs_residual | 90.134 | 119.786 | -29.652 |
| truck_028 | median_abs_residual | 118.936 | 117.156 | +1.781 |
| truck_028 | closest_to_zero_residual | 1.337 | 1.335 | +0.002 |

### Plots

- plots/explainability/global_importance_telematics_only.png
- plots/explainability/summary_beeswarm_telematics_only.png
- plots/explainability/dependence_telematics_only_acceleration_mps2.png
- plots/explainability/dependence_telematics_only_speed_kph.png
- plots/explainability/dependence_telematics_only_vehicle_mass_kg.png
- plots/explainability/per_truck_importance_telematics_only.png
- plots/explainability/waterfall_telematics_only_truck_001_largest_abs_residual.png
- plots/explainability/waterfall_telematics_only_truck_001_median_abs_residual.png

---

## Caveats

- **Synthetic data**: SHAP explains models trained on the formula-generated suite; do not interpret as real-world fuel-consumption drivers.
- **Per-truck importance** is over a small deterministic sample per truck and reflects that truck's own operating profile, not a population estimate.
- SHAP is an additive model-explanations tool; feature attributions are not causal effects.