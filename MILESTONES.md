# Milestones

Status legend: `todo` / `in progress` / `done` / `blocked`

## M01 — Repository scaffolding · done

- [x] Directory structure (src/data, src/features, src/models, src/evaluation,
      src/explainability, src/api, notebooks, scripts, configs, tests, docs)
- [x] README, DATA_SOURCES, MILESTONES, research brief
- [x] .gitignore
- [x] Status statement: no data, no results
- [x] Confirm workspace has no duplicate nested project directory

## M02 — Data acquisition and provenance · in progress

### M02.1 — FEAD source, licensing & dataset preflight · done

- [x] Verified repo, paper and all three Google Drive dataset links
- [x] No repository LICENSE found; redistribution unresolved (blocked on download)
- [x] Published row/truck counts recorded from paper Table III (not independently verified)
- [x] EMS/IFM distinctions and channel definitions documented; target = x7001
- [x] Leakage flags recorded (x7001 in feature range; x7002 cumulative counter)
- [x] Storage measured (RAM ~7.3 GB, disk free ~266 GB); chunked processing recommended
- [x] Recommended first dataset: EMS-1

### M02.2 — EMS-1 acquisition · done (integrity passed)

- [x] Downloaded EMS dataset-1 only → `data/raw/dwa_inceptio_ems_detail_location_manual2020-3-18.tar.gz`
- [x] Preserved original filename and bytes; no extraction/transformation
- [x] Format confirmed by signature (gzip `1F 8B`, tar.gz), not by extension
- [x] Size 822,374,003 bytes recorded; SHA-256 `2CC30ECB…190731` computed
- [x] Source URL, retrieval details and permission provenance recorded (DATA_SOURCES.md)

### M02.3 — EMS-1 archive integrity & content inspection · done

- [x] Re-computed SHA-256 matches M02.2 record (`2CC30ECB…190731`)
- [x] gzip integrity passed (full-stream read clean; `tar -tvzf` and `tar -xOf` exit 0)
- [x] Listed archive members without extracting: exactly 1 member, `dwa_inceptio_ems_detail_location_manual.csv`
- [x] No path traversal, absolute paths, symlinks, unusual entries, or nested archives
- [x] Total declared uncompressed size 5,848,864,872 bytes recorded
- [x] Archive structure (single merged CSV) recorded as differing from repo's daily-.txt loader expectation; resolution deferred to extraction

### M02.4 — EMS-1 controlled extraction & schema inspection · done

- [x] Extracted the single verified member to `data/interim/dwa_inceptio_ems_detail_location_manual.csv`; raw archive untouched
- [x] Extracted size (5,848,864,872 B) matches declared member size; SHA-256 `8A5CF353…04784D` computed
- [x] Schema: TAB-delimited, no header, UTF-8, 21 consistent columns; `ems_value` JSON in column index 1
- [x] Row count independently verified: **10,273,969 data rows = 10,273,969 physical newlines** (matches published count)
- [x] `x7001` confirmed as fuel-rate target; `x7002` cumulative-fuel counter flagged as potential leakage (allowlist not finalized)
- [x] Timestamp, identifier (imei, license plate) and geo fields documented; no embedded-newline/quoted-field edge cases observed

Remaining (M02, still in progress):

- [ ] Download EMS-2 and IFM (permission received; not yet acquired)
- [ ] Confirm license and redistribution terms (still blocked until a definitive license is found/documented)
- [ ] Write loader in `src/data/`; no modelling yet

## M03 — Data profiling and validation

- [ ] Per-truck, per-route inventory
- [ ] Sampling frequency and gap analysis
- [ ] EMS vs IFM comparison documented
- [ ] Define target variable(s) precisely

## M04 — Splitting protocols

- [ ] Truck-wise holdout splitter
- [ ] Chronological splitter
- [ ] Random row splitter, marked diagnostic-only in code and outputs
- [ ] Tests asserting no truck leakage across splits

## M05 — Baselines

- [ ] Naive / persistence baselines where defined
- [ ] Per-truck mean restricted to a labelled seen-truck experiment only
- [ ] Baseline results table with protocol column

## M06 — Modelling

- [ ] Linear / regularized baselines
- [ ] Gradient-boosted trees
- [ ] Fixed configs under `configs/`
- [ ] Seed control and reproducible runs

## M07 — Evaluation

- [ ] MAE, RMSE, R² as primary metrics
- [ ] MAPE only where valid, with justification
- [ ] Separate result tables per protocol; unseen-truck results never mixed with
      seen-truck or random-split results
- [ ] Uncertainty / variance across truck folds

## M08 — Research validation

- [ ] Pre-registered analysis plan before final runs
- [ ] Ablation of features and protocols
- [ ] Leakage audit
- [ ] Honest limitations section; explicitly label unseen-truck work as our extension
- [ ] Reproducibility check from clean environment

## M09 — Explainability

- [ ] SHAP computed only from trained runs
- [ ] No pre-assumed important features
- [ ] Per-truck attribution stability analysis
- [ ] Artefacts saved with run ID

## M10 — Agent evaluation

- [ ] Define agent tasks over the trained pipeline
- [ ] Task success criteria and rubric
- [ ] Evaluation harness in `src/evaluation/`
- [ ] Report agent failure modes

## M11 — API and serving

- [ ] Minimal local API in `src/api/`
- [ ] Input validation and schema
- [ ] No external data egress

## M12 — Performance

- [ ] Training and inference timing
- [ ] Memory profile
- [ ] Dataset-size scaling behaviour

## M13 — Security

- [ ] Dependency audit
- [ ] Secrets handling; no credentials in repo
- [ ] Data handling policy for non-redistributable data
- [ ] Input validation at API boundary

## M14 — Documentation

- [ ] Reproducible setup instructions
- [ ] Results documented with protocol labels
- [ ] Limitations and non-claims stated
- [ ] Data provenance section finalised

## M15 — Recruiter release

- [ ] Clean public branch
- [ ] No raw data, no restricted artefacts
- [ ] Results summary with correct framing
- [ ] Final review of claims against evidence
