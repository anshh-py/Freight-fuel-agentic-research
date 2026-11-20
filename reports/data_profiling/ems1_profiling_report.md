# FEAD EMS-1 Data Profiling Report

Generated: `2026-11-19T09:31:09Z` (UTC)
Profiler: `src/data/profile_ems1.py`

> **Metric scope.** Exact full-dataset metrics are labelled `(exact)`;
> reservoir-sample estimates are labelled `(sample)`. Nothing is silently
> downgraded. See `metrics_scope` in the JSON summary for the split list.

## 1. Scan overview

- Input: `data\interim\dwa_inceptio_ems_detail_location_manual.csv`
- Total records (physical lines, **exact**): **100,001**
- Bytes scanned (exact): 57,284,858 (0.06 GB)
- Scan wall time: 0.97 s (streaming, single lean pass)
- Zero-length / malformed lines (exact): 0

## 2. Trucks

- Distinct trucks (device-id based, **exact**): **2**
- Records per truck (exact, anonymized):

| truck | records (exact) |
|---|---|
| truck_001 | 61,808 |
| truck_002 | 38,192 |

## 3. Chronology & sampling (exact)

- Rows missing a `date` field (exact): 0
- Date minimum (exact): 1575130244377 ms (epoch)
- Date maximum (exact): 1575216013248 ms (epoch)
- Global date-nondecrease rows (exact): 974
- Per-truck date-nondecrease rows (exact): 973
- Chronological order is assessed from the exact full-scan date scan;
  per-truck sampling-interval statistics would require a full JSON parse
  of every row and are deliberately not computed in this pass.

## 4. JSON validity (exact structural + sample semantic)

- Structurally invalid `ems_value` (non-`{...}` / empty, **exact**):
  **0**
- Full JSON parse failures (sample): 0 (0.0%)
- JSON keys seen in sample (16): `date, ems_type, imei, x000B, x006C, x7000, x7001, x7002, x7003, x7004, x7005, x7006, x7007, x7035, x7091, x7095`

## 5. Outer columns (21, tab-separated, header-less)

| idx | provisional name | blank rate (sample) | numeric min (sample) | numeric max (sample) |
|---|---|---|---|---|
| 0 | record_id | 0.0% | 1.6591809551070116e+27 | 2.6591809551070116e+27 |
| 1 | ems_value | 0.0% | — | — |
| 2 | event_time_str | 0.0% | — | — |
| 3 | imei | 0.0% | — | — |
| 4 | imei_suffix | 0.0% | — | — |
| 5 | device_hex_id | 0.0% | — | — |
| 6 | road_route_id | 0.0% | 977.0 | 977.0 |
| 7 | device_type | 0.0% | — | — |
| 8 | license_plate | 0.0% | — | — |
| 9 | plate_other_id | 0.0% | — | — |
| 10 | gps_latitude | 0.0% | 22.945775 | 30.10617 |
| 11 | gps_longitude | 0.0% | 104.59639 | 114.19131 |
| 12 | speed_kmh | 0.0% | 0.0 | 359.0 |
| 13 | event_time2 | 0.0% | — | — |
| 14 | province | 0.0% | — | — |
| 15 | city | 0.0% | — | — |
| 16 | unknown_outer_16 | 9.25% | — | — |
| 17 | address | 0.0% | — | — |
| 18 | road_name | 0.0% | — | — |
| 19 | road_type | 0.0% | — | — |
| 20 | file_date | 0.0% | — | — |

Column 1 (`ems_value`) is the JSON blob (see §4). Names are *provisional*;
unverified outer columns are marked `unknown_outer_*`. License plates are
never written here.

## 6. Key JSON sub-fields (sample)

| field | present (sample) | n | min | median | mean | max | p95 |
|---|---|---|---|---|---|---|---|
| x7000 | 100.0% | 100000 | 0.0 | 1085.0 | 1066.541 | 1952.0 | 1321.0 |
| x7001 | 100.0% | 100000 | 0.0 | 20.95 | 23.123 | 88.4 | 59.8 |
| x7002 | 98.16% | 98156 | 33479.5 | 33773.5 | 38588.544 | 46936.0 | 46870.0 |
| x7003 | 100.0% | 100000 | 0.0 | 0.0 | 0.049 | 1.0 | 0.0 |
| x7004 | 100.0% | 100000 | 0.0 | 70.0 | 61.798 | 103.0 | 88.0 |
| x7005 | 100.0% | 100000 | 92641.3 | 93488.7 | 104640.055 | 123937.1 | 123759.80500000001 |
| x7006 | 100.0% | 100000 | 0.0 | 38.0 | 33.378 | 100.0 | 72.0 |
| x7007 | 99.95% | 99949 | 0.0 | 79.0 | 78.663 | 99.0 | 85.0 |
| x006C | 88.68% | 88682 | 0.0 | 68.8 | 60.277 | 124.1 | 86.7 |
| x7091 | 100.0% | 100000 | 0.0 | 5.2 | 5.229 | 10.0 | 9.5 |
| x7035 | 100.0% | 100000 | 0.0 | 0.0 | 0.047 | 1.0 | 0.0 |
| x7095 | 100.0% | 100000 | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 |
| ems_type | 100.0% | 100000 | 6.0 | 6.0 | 6.0 | 6.0 | 6.0 |
| date | 100.0% | 100000 | 1575130244377.0 | 1575180223641.0 | 1575178932301.059 | 1575216013248.0 | 1575212732250.0 |
| imei | 100.0% | 0 | — | — | — | — | — |

### x7001 (fuel-rate target) — suspicious values (sample)

- negative: 0
- zero: 23,728
- > 500: 0
- > 1000: 0

## 7. Duplicates (sample)

- Exact duplicate line-hashes within 100,000 sampled rows: **0**
- Full-file duplicate scan was not performed (memory-bound); see limitations.

## 8. x7002 (cumulative fuel counter) monotonicity (sample)

- Sample-observed decreases (counter going backwards within a truck): **905**
- Sample rows carrying an x7002 value: 2

## 9. PII / identifier consistency (sample)

- Rows where an outer license-plate mapped to a different truck key: 0
- Distinct (outer-device, JSON-imei) pairs in sample: 2
- No raw identifier (plate, IMEI, device id) is printed in this report.

## 10. Limitations

- JSON deep stats, duplicate counts, x7002 monotonicity and PII consistency
  are based on a **reservoir sample** of 100,000 rows, not all 10.27M.
- Exact per-truck sampling intervals require a full JSON parse of every row;
  not performed in this profiling pass.
- Duplicate detection is sample-based; a global duplicate count is unknown.
- Column 0 and several outer columns are provisional names (`unknown_outer_*`)
  until matched to the FEAD message spec.

## 11. Notable observations (not yet conclusions)

- `x7002` is a cumulative counter; its raw value should **not** be used as a
  model feature. `x7001` (target) appears inside the feature range; it must
  be excluded from any feature allowlist.
- `x7091` (fuel level) is present but its exact semantics need confirmation.
- The dataset is a single merged CSV, not per-day files as the repo loaders
  expect — ingestion design must account for that.

