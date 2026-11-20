# Data Sources

Status scheme used below:

- **Verified** — confirmed directly by us against a live source (repo, paper, loaders).
- **Published, not independently verified** — stated in the FEAD paper/repo but not confirmed against the actual downloaded files.
- **Unverified** — not yet checked.
- **Blocked** — cannot be checked under the current no-download rule.

---

## M02.1 — Source and Dataset Preflight Findings

The findings below are the result of the M02.1 preflight investigation (no download
performed). They record what is Verified versus Published-not-independently-verified
versus Unverified/Blocked.

## 1. Official sources (Verified)

| Source                       | URL                                                                    | Status                                                                                                                                           |
| ---------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| FEAD GitHub repository       | https://github.com/Torreskai0722/FEAD                                  | Verified: public repo exists, default branch `master`, no LICENSE detected by GitHub API (`license: null`), no LICENSE file anywhere in the tree |
| Original paper               | https://weisongshi.org/papers/liu23-fuel.pdf                           | Verified: reachable (HTTP 200), ~5.98 MB PDF, text extracted                                                                                     |
| EMS dataset-1 (Google Drive) | https://drive.google.com/file/d/1lLeACfOWyvT_LtFnrN9viqU3SNu_eTBo/view | Verified: link resolves (HTTP 200)                                                                                                               |
| EMS dataset-2 (Google Drive) | https://drive.google.com/file/d/15Awwj26PE0MIGRUW1CiCO2xkP3j8-z2o/view | Verified: link resolves (HTTP 200)                                                                                                               |
| IFM dataset (Google Drive)   | https://drive.google.com/file/d/1PVlbxufMIaEIHC9TfAOzaEGvMZQ8OF4p/view | Verified: link resolves (HTTP 200)                                                                                                               |

All three dataset links are explicitly referenced by the repository README. Only HTTP
reachability was checked; no file was downloaded and no transfer was initiated.

---

## 2. Licensing / redistribution (Unverified → Blocked without download)

- The GitHub API reports `license: null` for the FEAD repository; no `LICENSE` file
  (or per-dataset license) exists anywhere in the repo tree.
- The repo README and the loader code comments contain no dataset-specific terms.
- The download endpoints are raw Google Drive file pages; Google Drive does not expose
  a redistribution grant, and public access does not imply a license.

**Conclusion:** access/download is implicitly permitted by the public links, but
**redistribution is not established**. Treat the raw data as **non-redistributable**
until a license is confirmed. Permission to _download/use_ is separate from permission
to _redistribute_; the latter remains unresolved.

---

## 3. Published dataset metadata (Published, not independently verified)

From the FEAD paper, Table III (also echoed in the repo README):

| Dataset       | Time (paper) | Trucks (paper) | Rows (paper) | Source          |
| ------------- | ------------ | -------------- | ------------ | --------------- |
| EMS dataset-1 | 12/2019      | 9              | 10,273,969   | paper Table III |
| EMS dataset-2 | 04/2020      | 29             | 26,145,539   | paper Table III |
| IFM dataset   | 06/2020      | 1              | 872,844      | paper Table III |

These are **published counts**, not verified against the actual files. File names,
formats, compression and exact downloadable sizes are **Unverified / Blocked** until
a download is permitted (public Drive pages do not expose them without initiating a
transfer).

---

## 4. Channels, measurement and parsing (Verified from repo loaders + paper)

### EMS (primary feature source)

- **Target / label = `x7001` = instantaneous fuel rate, unit L/h** (paper Table II; repo
  spec workbook `秒级数据报文说明.xlsx` lists `x7001` 瞬时油耗 升/小时).
- Channel definitions (from the repo spec workbook):
  - `x7000` engine speed · `x7002` **total fuel consumption (cumulative, unit 0.5 L)**
    · `x7003` brake on/off · `x7004` EMS mileage (0.1 km) · `x7005` cumulative odometer
    · `x7006` throttle % · `x7007` coolant temperature · `x006C` GPS speed · `x7035`
    handbrake · `x7091` main-tank remaining fuel % · `x70EB` unknown · `x000B` time;
    `ems_type` subfields `0x716D` brake-pedal position, `0x704F` torque/load %,
    `0x705F` engine request (gear info).
- **Parsing logic (repo)** — raw files are daily tab-separated `.txt` (e.g.
  `ems_yingche_20200406.txt`) loaded by `csv_load_ems_2.sql` into a SQL table
  (`emsdata_3`) with an `ems_value` JSON-string column plus `truckid`, `triggertime`,
  `province`/`city`/`county`, `tdate`, etc. `read_ems.py` parses `ems_value` JSON and
  forms features `x7000,x7002..x7007,x006C,x7035,x7091` + lat/lng; label `x7001`.
- Measurement (paper): fuel consumption read from the truck's EMS via an off-the-shelf
  CAN-bus parser with **10 mL measurement resolution**; the paper's percent-error
  figure did not extract cleanly from the PDF text (raw token `6:3 0:2%`) so it is
  recorded as _published, exact value could not be cleanly read_.

### IFM (auxiliary)

- The paper describes an Onosokki FP-2140H fuel meter with **0.1 mL resolution** and
  error **< 0.2%** per 50 ms; captures **57 features total**, reported at **20 Hz**,
  collected on **one truck on the same path for several days**.
- **IFM contains two fuel-rate labels**: `x8001` (instant flow rate) and `x8002`
  (engine fuel rate read from the CAN bus — the same quantity as EMS `x7001`).
- **Parsing logic (repo)** — loader scripts combine three CSVs
  (`localization_result.csv`, `vehicle_report.csv`, and one of
  `0x721_Ins_flow_rate.csv` OR `0x18fef2fe_EngFuelRate.csv`) aligned by timestamp;
  features come from `vehicle_report` columns (brake, retarder torque, engine torque,
  engine torque loss, engine speed, gear, clutch slip, vehicle weight, speed). The two
  loader variants use **different** fuel-rate fields as the label:
  `dataset_clean_csv.py` uses `0x721_Ins_flow_rate`; `dataset_ems_read.py` uses
  `0x18fef2fe_EngFuelRate` × 10.

### EMS vs IFM — do not mix

EMS and IFM are different instruments; their fuel-rate fields are **not
interchangeable**. IFM fuel-rate columns (`x8001`/`x8002`) and EMS `x7001` must never
be mixed as one label, and an IFM-derived rate must not be used as a feature when the
other is the label.

---

## 5. Target-leakage note (Verified from code/spec/schema)

- `x7001` (label) sits **inside** the `x7000`–`x7007` feature range. It must be
  **excluded from every model input**.
- `x7002` (**total fuel consumption**, cumulative, unit 0.5 L) is effectively an
  integral/re-count of fuel rate; if present as a feature it dominates or leaks the
  target. The paper's own correlation table lists `x7002` among features, but we must
  **not** use it as a model input.
- `x7091` (remaining-fuel %) is correlated with cumulative consumption; review before
  inclusion.
- IFM: if `x8001` or `x8002` is the label, the other must not be a feature.

---

## 6. Storage & environment (partially Verified)

- Local env measured: **RAM ~7.3 GB total** (CPU-first), **disk free ~266 GB**.
- Required download space, extracted size, and temporary extraction space are
  **Unverified / Blocked** until download (exact file sizes unknown).
- Recommended processing approach for large files (do not load all into pandas):
  - Process per-day `.txt` files line-by-line; parse each row's `ems_value` JSON and
    append to chunked Parquet/partitioned outputs.
  - Use the SQL bulk-loader pattern from the repo for a DB route.
  - Never build the full frame in memory; stream in chunks.
- To check disk in Windows PowerShell yourself: `Get-PSDrive C | Select-Object Used,Free`.

---

## 7. Recommended first dataset

**EMS-1** (9 trucks, ~10.3 M rows) is the best first dataset: it is the smaller EMS
set, single instrument, and enough to validate the ingestion pipeline before tackling
EMS-2 (29 trucks, ~26.1 M rows). Keep EMS-1 and EMS-2 separate; do not mix in IFM.

---

## 8. What this project is not

- Not Indian truck data.
- Not live OBD-II collection. All analysis is retrospective over FEAD.
- Not a reproduction of the FEAD route-matched results; unseen-truck evaluation is our
  own extension.

## 9. Status

- **EMS dataset-1 downloaded** (see M02.2 below). EMS-2 and IFM are **not** downloaded.
- **No model results exist.**
- No ML dependencies installed; no extraction performed; no Git commit made.

---

### M02.2 — EMS-1 acquisition & integrity (Verified)

Dataset: **EMS dataset-1** (the smaller EMS set, 9 trucks).

| Item                  | Value                                                                                                                         |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Source URL            | https://drive.google.com/file/d/1lLeACfOWyvT_LtFnrN9viqU3SNu_eTBo/view                                                        |
| Destination           | `data/raw/`                                                                                                                   |
| Filename (preserved)  | `dwa_inceptio_ems_detail_location_manual2020-3-18.tar.gz`                                                                     |
| Format                | gzip-compressed tar archive (`tar.gz`) — signature `1F 8B` confirmed from first bytes; not HTML                               |
| Download size         | 822,374,003 bytes (~784.2 MiB)                                                                                                |
| SHA-256               | `2CC30ECB239857CB8157CF10C611D8FAD3701485C68B639982015777FD190731`                                                            |
| Retrieval timestamp   | recorded at download time from local machine clock                                                                            |
| Permission provenance | Written email permission received from FEAD author for all three datasets; private correspondence not copied into public docs |

Google Drive required the virus-scan confirmation flow (`confirm=t` + uuid) because the
file exceeds the scan threshold; the public Drive page reported size "784M" before
download. Filename, size and format were verified from the public Drive page first, and
the signature check confirmed a real gzip stream (not an HTML/error page). Free disk
after download measured ≈265 GB.

---

### M02.3 — EMS-1 archive integrity & content inspection (Verified)

Inspection performed **without** extracting the archive (streamed reads only; archive
preserved byte-for-byte unchanged).

| Check                                       | Result (Verified)                                                                                                                                                                                |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Re-computed SHA-256                         | `2CC30ECB239857CB8157CF10C611D8FAD3701485C68B639982015777FD190731` — **matches** M02.2 record                                                                                                    |
| gzip integrity                              | Full-stream read of the decompressed archive completed cleanly: `tar -tvzf` (members listed, exit 0) and `tar -xOf … \| Out-Null` (complete member read, exit 0). No gzip CRC/truncation errors. |
| Member count                                | **1** archive member                                                                                                                                                                             |
| Member path                                 | `dwa_inceptio_ems_detail_location_manual.csv` (relative, no leading `/`, no `..` components)                                                                                                     |
| Member type                                 | Regular file (`-rw-rw-r--`, owner `hive`); **not** a symlink, directory or special entry                                                                                                         |
| Member uncompressed size                    | **5,848,864,872 bytes (~5.85 GB)**                                                                                                                                                               |
| Total declared uncompressed size            | 5,848,864,872 bytes (only member)                                                                                                                                                                |
| Nested archives                             | **None** — the single member is a `.csv`; no tar/zip/gz inside                                                                                                                                   |
| Extension/format (from listing + signature) | gzip-compressed tar containing one plain CSV file                                                                                                                                                |

**Structure vs repo documentation (observation, not resolved):** the FEAD repo loaders
(`csv_load_ems_2.sql`, `read_ems.py`) expect **daily tab-separated `.txt` files** (e.g.
`ems_yingche_20200406.txt`) with an `ems_value` JSON column. The actual archive instead
contains a **single consolidated CSV** (`dwa_inceptio_ems_detail_location_manual.csv`).
This implies the repo loader may need adaptation to the delivered layout, or that this
archive was packaged as one merged file. **Resolved only at extraction time** — no
extraction has been performed, so row/column layout is not yet verified.

**Row count:** still **not verified** at M02.3 (blocked until extraction); the
published EMS-1 count of 10,273,969 rows was _published, not independently verified_.
**Resolved in M02.4** — see below.

---

### M02.4 — EMS-1 controlled extraction & schema inspection (Verified)

Extracted the single verified member into `data/interim/dwa_inceptio_ems_detail_location_manual.csv`
(only that relative member path was extracted; the raw archive in `data/raw/` was left
unchanged and never overwritten).

| Check                          | Result (Verified)                                                                                                                        |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Extracted file size            | **5,848,864,872 bytes** — exactly matches the archive's declared member size                                                             |
| Extracted SHA-256              | `8A5CF353BFDA52A237FFF3353F7EEDF41C141A0509E96902EDADD9E54404784D` (hash of the extracted CSV; expected to differ from the archive hash) |
| Delimiter                      | **TAB** (`0x09`)-separated — not comma                                                                                                   |
| Header row                     | **None** — first physical line is a data row                                                                                             |
| Encoding                       | UTF-8 (first-line decode OK)                                                                                                             |
| Columns per row                | **21**, consistent across all rows                                                                                                       |
| Row count                      | **10,273,969 data rows = 10,273,969 physical newlines** (streamed parse) — independently verifies the published count                    |
| Column-width consistency       | All 10,273,969 rows have exactly 21 columns                                                                                              |
| `ems_value` JSON presence      | All 10,273,969 rows contain a JSON blob in column index 1                                                                                |
| Malformed/non-JSON `ems_value` | 0 in the 50,000-row sample; 0 non-JSON across the full pass                                                                              |

**Schema (50,000-row bounded sample; 21 outer columns):**

- **Outer metadata columns (non-JSON):** large numeric record id, datetime strings, device
  `imei`, **license plate** (e.g. `闽K59938`), GPS latitude/longitude, altitude, province/city,
  road class, and a date key. Identifiers and geo context live **outside** the JSON blob.
- **`ems_value` JSON keys observed (frequency over the 50,000-row sample):**
  - Always present (100%): `x7000`, `x7001`, `x7003`, `x7004`, `x7005`, `x7006`,
    `x7035`, `date`, `ems_type`, `imei`, `x7095`, `x000B`, `x7091`.
  - Near-universal: `x7007` (99.9%), `x7002` (98.6%), `x006C` (93%).
- **`x7001` = fuel-rate target (instantaneous, L/h)** — present in 100% of sample rows.
- **`x7002` = cumulative fuel counter (0.5 L units)** — present in ~98.6%; confirmed as a
  **potential leakage field** (cumulative total; must be excluded from model inputs).
- Timestamps: JSON `date` (epoch ms) and `x000B` (human-readable string) both present; outer
  datetime columns also present.

**Leakage status (flagged; allowlist not finalized):**

- `x7001` lies inside the proposed `x7000–x7007` feature range → must be **excluded** from
  all model inputs (it is the target).
- `x7002` cumulative fuel counter → potential leakage; exclusion recommended pending
  feature-engineering review.
- `x7091` (fuel level %) → retained for review.
- **Feature allowlist intentionally not finalized** at M02.4 (M03 feature engineering owns this).

**Row-count verdict:** the published EMS-1 count of **10,273,969 rows is now independently
verified** (streamed count equals the published value exactly). The earlier _published, not
independently verified_ status is resolved for the row count.
