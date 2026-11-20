# M1.2 Baseline Results

Protocol: train on 20 trucks, validate on 4 trucks; test untouched
Random state: 20240617

Random Forest configuration: n_estimators=150, max_depth=20, min_samples_leaf=8, max_features="sqrt", n_jobs=-1, random_state=20240617.

Split: train=657336 rows (20 trucks), val=133072 rows (4 trucks), test (unused)=173289 rows.

## Feature set: `full_operational`

| Model                   | MAE     | RMSE    | R²      | MAPE (>1) | fit (s) |
| ----------------------- | ------- | ------- | ------- | --------- | ------- |
| Mean Dummy Regressor    | 38.4434 | 42.4356 | -0.0000 | 777.617%  | 0.0     |
| Median Dummy Regressor  | 37.1713 | 47.3628 | -0.2457 | 387.086%  | 0.0     |
| Random Forest Regressor | 3.7714  | 5.8096  | 0.9813  | 9.317%    | 83.9    |
| Ridge Regression        | 5.1810  | 6.9422  | 0.9732  | 68.102%   | 0.3     |
| XGBoost Regressor       | 3.6211  | 5.8850  | 0.9808  | 9.854%    | 12.3    |

## Feature set: `telematics_only`

| Model                   | MAE     | RMSE    | R²      | MAPE (>1) | fit (s) |
| ----------------------- | ------- | ------- | ------- | --------- | ------- |
| Mean Dummy Regressor    | 38.4434 | 42.4356 | -0.0000 | 777.617%  | 0.0     |
| Median Dummy Regressor  | 37.1713 | 47.3628 | -0.2457 | 387.086%  | 0.0     |
| Random Forest Regressor | 4.7736  | 7.5077  | 0.9687  | 21.255%   | 49.4    |
| Ridge Regression        | 19.2174 | 23.6918 | 0.6883  | 406.995%  | 0.1     |
| XGBoost Regressor       | 4.2926  | 6.6228  | 0.9756  | 17.602%   | 11.9    |
