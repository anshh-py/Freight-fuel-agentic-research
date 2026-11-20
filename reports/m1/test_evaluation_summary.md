# M1.3 Unseen-Truck Final Evaluation

Protocol: evaluate saved M1.2 models on the 5 held-out test trucks only; test data never used for fitting/tuning/selection

**Selected models (from M1.2 validation, not test):**

| Feature set | Selected model | Validation MAE |
| --- | --- | --- |
| `full_operational` | `xgboost` | 3.6211 |
| `telematics_only` | `xgboost` | 4.2926 |

Model selection used **validation** evidence only; the test set was not used for tuning, feature choice, or retrospective selection.

## Verification

- truck_001, truck_004, truck_017, truck_020, truck_028
- All expected test trucks present: True
- Train/validation truck in test evaluation: False
- Test rows: 173289
- Preprocessors loaded from saved M1.2 artifacts, never refitted: True

## Feature set: `full_operational`

- Test rows: 173289, features: 16
- Plots: plots/residuals_hist_full_operational.png, plots/residual_vs_actual_full_operational.png, plots/error_vs_speed_full_operational.png, plots/error_vs_engine_load_full_operational.png, plots/per_truck_mean_residual_full_operational.png

### `mean_dummy`

> feature-independent prediction (constant); metrics are identical across feature sets by design

| metric | MAE | RMSE | R² | MAPE (>1) | rows |
| --- | --- | --- | --- | --- | --- |
| pooled | 37.1862 | 41.0708 | -0.0049 | 825.0981 | 173289 |
| macro-avg | 36.9470 | 40.5316 | -0.0159 | 829.2472 | — |
| **Per-truck** | | | | | |
| truck | MAE | RMSE | R² | MAPE (>1) | rows |
| --- | --- | --- | --- | --- | --- |
| truck_001 | 32.7948 | 35.1534 | -0.0608 | 932.4911 | 30817 |
| truck_004 | 33.2833 | 35.6369 | -0.0175 | 866.7734 | 32480 |
| truck_017 | 38.3948 | 42.1977 | -0.0006 | 746.9833 | 37073 |
| truck_020 | 39.6309 | 44.0936 | -0.0007 | 789.4045 | 33901 |
| truck_028 | 40.6311 | 45.5765 | -0.0000 | 810.5837 | 39018 |

| **Per-class** | | | | | |
| class | MAE | RMSE | R² | MAPE (>1) | rows |
| --- | --- | --- | --- | --- | --- |
| 32T | 33.0455 | 35.4023 | -0.0349 | 898.7278 | 63297 |
| 40T | 38.9852 | 43.1137 | -0.0006 | 767.2325 | 70974 |
| 49T | 40.6311 | 45.5765 | -0.0000 | 810.5837 | 39018 |

| **Residual bias** | mean=-2.872, median=-21.933, std=40.970, underpredicted=39.5%, overpredicted=60.5% |

### `median_dummy`

> feature-independent prediction (constant); metrics are identical across feature sets by design

| metric | MAE | RMSE | R² | MAPE (>1) | rows |
| --- | --- | --- | --- | --- | --- |
| pooled | 34.8370 | 44.7137 | -0.1911 | 408.1880 | 173289 |
| macro-avg | 34.5965 | 43.9905 | -0.1920 | 410.2647 | — |
| **Per-truck** | | | | | |
| truck | MAE | RMSE | R² | MAPE (>1) | rows |
| --- | --- | --- | --- | --- | --- |
| truck_001 | 29.6683 | 36.3026 | -0.1313 | 460.1339 | 30817 |
| truck_004 | 32.0255 | 38.8252 | -0.2078 | 431.6224 | 32480 |
| truck_017 | 36.4297 | 46.5865 | -0.2195 | 370.0913 | 37073 |
| truck_020 | 36.9383 | 48.2447 | -0.1980 | 389.5781 | 33901 |
| truck_028 | 37.9209 | 49.9937 | -0.2033 | 399.8979 | 39018 |

| **Per-class** | | | | | |
| class | MAE | RMSE | R² | MAPE (>1) | rows |
| --- | --- | --- | --- | --- | --- |
| 32T | 30.8779 | 37.6182 | -0.1685 | 445.4857 | 63297 |
| 40T | 36.6726 | 47.3858 | -0.2088 | 379.3930 | 70974 |
| 49T | 37.9209 | 49.9937 | -0.2033 | 399.8979 | 39018 |

| **Residual bias** | mean=17.910, median=-1.151, std=40.970, underpredicted=49.4%, overpredicted=50.6% |

### `ridge`

| metric | MAE | RMSE | R² | MAPE (>1) | rows |
| --- | --- | --- | --- | --- | --- |
| pooled | 3.8614 | 5.4697 | 0.9822 | 65.9996 | 173289 |
| macro-avg | 3.8232 | 5.2824 | 0.9829 | 67.6894 | — |
| **Per-truck** | | | | | |
| truck | MAE | RMSE | R² | MAPE (>1) | rows |
| --- | --- | --- | --- | --- | --- |
| truck_001 | 3.2083 | 4.0860 | 0.9857 | 97.6334 | 30817 |
| truck_004 | 3.3944 | 4.3371 | 0.9849 | 94.0005 | 32480 |
| truck_017 | 3.2136 | 4.7618 | 0.9873 | 45.9052 | 37073 |
| truck_020 | 4.2952 | 6.2951 | 0.9796 | 52.2169 | 33901 |
| truck_028 | 5.0044 | 6.9318 | 0.9769 | 48.6909 | 39018 |

| **Per-class** | | | | | |
| class | MAE | RMSE | R² | MAPE (>1) | rows |
| --- | --- | --- | --- | --- | --- |
| 32T | 3.3038 | 4.2167 | 0.9853 | 95.7669 | 63297 |
| 40T | 3.7302 | 5.5473 | 0.9834 | 48.9180 | 70974 |
| 49T | 5.0044 | 6.9318 | 0.9769 | 48.6909 | 39018 |

| **Residual bias** | mean=-0.343, median=0.267, std=5.459, underpredicted=52.9%, overpredicted=47.1% |

### `random_forest`

| metric | MAE | RMSE | R² | MAPE (>1) | rows |
| --- | --- | --- | --- | --- | --- |
| pooled | 2.3372 | 4.0389 | 0.9903 | 6.8862 | 173289 |
| macro-avg | 2.2671 | 3.4965 | 0.9917 | 6.6394 | — |
| **Per-truck** | | | | | |
| truck | MAE | RMSE | R² | MAPE (>1) | rows |
| --- | --- | --- | --- | --- | --- |
| truck_001 | 1.3946 | 2.2320 | 0.9957 | 4.0232 | 30817 |
| truck_004 | 1.0678 | 1.7147 | 0.9976 | 2.7192 | 32480 |
| truck_017 | 1.4019 | 2.2661 | 0.9971 | 3.1821 | 37073 |
| truck_020 | 3.1524 | 4.9598 | 0.9873 | 8.8311 | 33901 |
| truck_028 | 4.3189 | 6.3098 | 0.9808 | 14.4413 | 39018 |

| **Per-class** | | | | | |
| class | MAE | RMSE | R² | MAPE (>1) | rows |
| --- | --- | --- | --- | --- | --- |
| 32T | 1.2269 | 1.9835 | 0.9968 | 3.3532 | 63297 |
| 40T | 2.2380 | 3.7990 | 0.9922 | 5.8785 | 70974 |
| 49T | 4.3189 | 6.3098 | 0.9808 | 14.4413 | 39018 |

| **Residual bias** | mean=-0.502, median=-0.009, std=4.008, underpredicted=48.8%, overpredicted=50.6% |

### `xgboost` ✅ selected

| metric | MAE | RMSE | R² | MAPE (>1) | rows |
| --- | --- | --- | --- | --- | --- |
| pooled | 2.5684 | 4.5275 | 0.9878 | 7.3793 | 173289 |
| macro-avg | 2.5119 | 4.1002 | 0.9893 | 7.1981 | — |
| **Per-truck** | | | | | |
| truck | MAE | RMSE | R² | MAPE (>1) | rows |
| --- | --- | --- | --- | --- | --- |
| truck_001 | 1.3023 | 2.1228 | 0.9961 | 4.3467 | 30817 |
| truck_004 | 1.5492 | 2.4897 | 0.9950 | 5.1222 | 32480 |
| truck_017 | 2.2452 | 3.6674 | 0.9924 | 5.8003 | 37073 |
| truck_020 | 3.8660 | 6.3042 | 0.9795 | 8.8291 | 33901 |
| truck_028 | 3.5966 | 5.9168 | 0.9831 | 11.8922 | 39018 |

| **Per-class** | | | | | |
| class | MAE | RMSE | R² | MAPE (>1) | rows |
| --- | --- | --- | --- | --- | --- |
| 32T | 1.4290 | 2.3184 | 0.9956 | 4.7451 | 63297 |
| 40T | 3.0194 | 5.0999 | 0.9860 | 7.2460 | 70974 |
| 49T | 3.5966 | 5.9168 | 0.9831 | 11.8922 | 39018 |

| **Residual bias** | mean=-0.525, median=-0.102, std=4.497, underpredicted=41.2%, overpredicted=58.8% |

---

## Feature set: `telematics_only`

- Test rows: 173289, features: 12
- Plots: plots/residuals_hist_telematics_only.png, plots/residual_vs_actual_telematics_only.png, plots/error_vs_speed_telematics_only.png, plots/error_vs_engine_load_telematics_only.png, plots/per_truck_mean_residual_telematics_only.png

### `mean_dummy`

> feature-independent prediction (constant); metrics are identical across feature sets by design

| metric | MAE | RMSE | R² | MAPE (>1) | rows |
| --- | --- | --- | --- | --- | --- |
| pooled | 37.1862 | 41.0708 | -0.0049 | 825.0981 | 173289 |
| macro-avg | 36.9470 | 40.5316 | -0.0159 | 829.2472 | — |
| **Per-truck** | | | | | |
| truck | MAE | RMSE | R² | MAPE (>1) | rows |
| --- | --- | --- | --- | --- | --- |
| truck_001 | 32.7948 | 35.1534 | -0.0608 | 932.4911 | 30817 |
| truck_004 | 33.2833 | 35.6369 | -0.0175 | 866.7734 | 32480 |
| truck_017 | 38.3948 | 42.1977 | -0.0006 | 746.9833 | 37073 |
| truck_020 | 39.6309 | 44.0936 | -0.0007 | 789.4045 | 33901 |
| truck_028 | 40.6311 | 45.5765 | -0.0000 | 810.5837 | 39018 |

| **Per-class** | | | | | |
| class | MAE | RMSE | R² | MAPE (>1) | rows |
| --- | --- | --- | --- | --- | --- |
| 32T | 33.0455 | 35.4023 | -0.0349 | 898.7278 | 63297 |
| 40T | 38.9852 | 43.1137 | -0.0006 | 767.2325 | 70974 |
| 49T | 40.6311 | 45.5765 | -0.0000 | 810.5837 | 39018 |

| **Residual bias** | mean=-2.872, median=-21.933, std=40.970, underpredicted=39.5%, overpredicted=60.5% |

### `median_dummy`

> feature-independent prediction (constant); metrics are identical across feature sets by design

| metric | MAE | RMSE | R² | MAPE (>1) | rows |
| --- | --- | --- | --- | --- | --- |
| pooled | 34.8370 | 44.7137 | -0.1911 | 408.1880 | 173289 |
| macro-avg | 34.5965 | 43.9905 | -0.1920 | 410.2647 | — |
| **Per-truck** | | | | | |
| truck | MAE | RMSE | R² | MAPE (>1) | rows |
| --- | --- | --- | --- | --- | --- |
| truck_001 | 29.6683 | 36.3026 | -0.1313 | 460.1339 | 30817 |
| truck_004 | 32.0255 | 38.8252 | -0.2078 | 431.6224 | 32480 |
| truck_017 | 36.4297 | 46.5865 | -0.2195 | 370.0913 | 37073 |
| truck_020 | 36.9383 | 48.2447 | -0.1980 | 389.5781 | 33901 |
| truck_028 | 37.9209 | 49.9937 | -0.2033 | 399.8979 | 39018 |

| **Per-class** | | | | | |
| class | MAE | RMSE | R² | MAPE (>1) | rows |
| --- | --- | --- | --- | --- | --- |
| 32T | 30.8779 | 37.6182 | -0.1685 | 445.4857 | 63297 |
| 40T | 36.6726 | 47.3858 | -0.2088 | 379.3930 | 70974 |
| 49T | 37.9209 | 49.9937 | -0.2033 | 399.8979 | 39018 |

| **Residual bias** | mean=17.910, median=-1.151, std=40.970, underpredicted=49.4%, overpredicted=50.6% |

### `ridge`

| metric | MAE | RMSE | R² | MAPE (>1) | rows |
| --- | --- | --- | --- | --- | --- |
| pooled | 18.4213 | 22.9559 | 0.6861 | 414.6266 | 173289 |
| macro-avg | 18.2255 | 22.6548 | 0.6819 | 414.6955 | — |
| **Per-truck** | | | | | |
| truck | MAE | RMSE | R² | MAPE (>1) | rows |
| --- | --- | --- | --- | --- | --- |
| truck_001 | 15.4045 | 20.2720 | 0.6472 | 452.7765 | 30817 |
| truck_004 | 15.3801 | 19.7213 | 0.6884 | 429.4539 | 32480 |
| truck_017 | 18.6156 | 22.7263 | 0.7098 | 368.7814 | 37073 |
| truck_020 | 19.6045 | 24.3998 | 0.6936 | 356.9089 | 33901 |
| truck_028 | 22.1231 | 26.1547 | 0.6707 | 465.5569 | 39018 |

| **Per-class** | | | | | |
| class | MAE | RMSE | R² | MAPE (>1) | rows |
| --- | --- | --- | --- | --- | --- |
| 32T | 15.3920 | 19.9913 | 0.6700 | 440.7942 | 63297 |
| 40T | 19.0880 | 23.5405 | 0.7017 | 363.1143 | 70974 |
| 49T | 22.1231 | 26.1547 | 0.6707 | 465.5569 | 39018 |

| **Residual bias** | mean=-0.499, median=-2.902, std=22.950, underpredicted=45.6%, overpredicted=54.4% |

### `random_forest`

| metric | MAE | RMSE | R² | MAPE (>1) | rows |
| --- | --- | --- | --- | --- | --- |
| pooled | 3.4973 | 5.5790 | 0.9815 | 20.7523 | 173289 |
| macro-avg | 3.4636 | 5.4932 | 0.9807 | 20.0952 | — |
| **Per-truck** | | | | | |
| truck | MAE | RMSE | R² | MAPE (>1) | rows |
| --- | --- | --- | --- | --- | --- |
| truck_001 | 2.7329 | 4.5087 | 0.9826 | 11.5687 | 30817 |
| truck_004 | 3.8533 | 6.0648 | 0.9705 | 20.6814 | 32480 |
| truck_017 | 2.7001 | 4.6394 | 0.9879 | 14.9872 | 37073 |
| truck_020 | 3.2877 | 5.7194 | 0.9832 | 13.1394 | 33901 |
| truck_028 | 4.7442 | 6.5334 | 0.9795 | 40.0994 | 39018 |

| **Per-class** | | | | | |
| class | MAE | RMSE | R² | MAPE (>1) | rows |
| --- | --- | --- | --- | --- | --- |
| 32T | 3.3078 | 5.3639 | 0.9762 | 16.2504 | 63297 |
| 40T | 2.9807 | 5.1835 | 0.9855 | 14.1052 | 70974 |
| 49T | 4.7442 | 6.5334 | 0.9795 | 40.0994 | 39018 |

| **Residual bias** | mean=0.253, median=-0.275, std=5.573, underpredicted=38.5%, overpredicted=61.4% |

### `xgboost` ✅ selected

| metric | MAE | RMSE | R² | MAPE (>1) | rows |
| --- | --- | --- | --- | --- | --- |
| pooled | 2.5660 | 4.3713 | 0.9886 | 15.3600 | 173289 |
| macro-avg | 2.5072 | 3.9741 | 0.9899 | 15.0522 | — |
| **Per-truck** | | | | | |
| truck | MAE | RMSE | R² | MAPE (>1) | rows |
| --- | --- | --- | --- | --- | --- |
| truck_001 | 1.8832 | 2.9720 | 0.9924 | 12.1451 | 30817 |
| truck_004 | 1.5358 | 2.3242 | 0.9957 | 12.2213 | 32480 |
| truck_017 | 2.0527 | 3.3264 | 0.9938 | 12.6128 | 37073 |
| truck_020 | 2.9305 | 4.5446 | 0.9894 | 13.9538 | 33901 |
| truck_028 | 4.1337 | 6.7031 | 0.9784 | 24.3281 | 39018 |

| **Per-class** | | | | | |
| class | MAE | RMSE | R² | MAPE (>1) | rows |
| --- | --- | --- | --- | --- | --- |
| 32T | 1.7049 | 2.6594 | 0.9942 | 12.1843 | 63297 |
| 40T | 2.4720 | 3.9554 | 0.9916 | 13.2529 | 70974 |
| 49T | 4.1337 | 6.7031 | 0.9784 | 24.3281 | 39018 |

| **Residual bias** | mean=-0.166, median=0.049, std=4.368, underpredicted=51.8%, overpredicted=48.2% |

---

## Caveats

- **Synthetic data**: metrics reflect the formula-generated suite; they do **not** establish real-world predictive performance.
- **Macro-average** is a descriptive average over the five test trucks only (n=5); it is not a statistically robust population estimate.
- **Small class samples**: per-class test estimates come from 2×32T, 2×40T and 1×49T trucks and should be read accordingly.
- **Rows within a truck/trip are autocorrelated**; per-truck metrics are the more meaningful unit than row-level aggregates.