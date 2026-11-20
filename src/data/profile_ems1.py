"""Streaming profiler for the FEAD EMS-1 dataset (performance-optimized).

Reads the ~5.85 GB tab-separated, header-less CSV without ever loading it
into memory. Emits:

  * reports/data_profiling/ems1_profiling_summary.json   (machine-readable)
  * reports/data_profiling/ems1_profiling_report.md      (human-readable)

Performance notes (revision)
----------------------------
The previous implementation did heavy per-row work for every one of the
10.27M rows (SHA-256 per row, a full 21-part split per row, per-row float
parsing of 11 columns, and regex date scanning). That exceeded the 300 s
execution window.

This revision splits the work into two clearly labelled passes:

1. FULL SCAN (exact, lean): streams every row but only extracts the fields
   it must have for the required exact metrics -- the ``ems_value`` JSON
   column (col 1) and the device/truck column (col 2) -- using manual
   ``bytes.find`` tab-finding instead of splitting into all 21 fields. No
   per-row SHA-256 (raw device id bytes are used directly as dict keys and
   only the anonymized ``truck_NNN`` name is ever written). Dates are pulled
   with a fast manual digit scan, not a regex.

   Exact metrics produced here: total records, unique trucks, records per
   truck, timestamp min/max, global & per-truck chronological ordering, JSON
   *structural* failures (col 1 empty / not a ``{ ... }`` object), total
   bytes, zero-length rows.

2. RESERVOIR SAMPLE (estimates): a seeded random sample (default 300k rows)
   receives all the expensive per-column / per-JSON analysis: outer-column
   numeric min/max, per-column blank rates, full JSON key discovery, JSON
   sub-field distributions, suspicious x7001 values, x7002 monotonicity,
   exact duplicate line-hashes, and PII identifier-consistency checks.

Every metric in the outputs is tagged exact vs sample so nothing is silently
downgraded.

PII policy: license plates, IMEIs and raw device ids are never written to
outputs. Trucks appear only as stable pseudonyms ``truck_001`` ...
``truck_0NN`` assigned in first-seen order. No raw identifier is printed.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

DATASET_PATH = Path("data/interim/dwa_inceptio_ems_detail_location_manual.csv")
SUMMARY_PATH = Path("reports/data_profiling/ems1_profiling_summary.json")
REPORT_PATH = Path("reports/data_profiling/ems1_profiling_report.md")

# Tab-separated, header-less, 21 columns. Only the index-based column
# meaning is used; names are descriptive and unknown fields are marked.
KNOWN_OUTER_COLUMNS = {
    0: "record_id",              # per-row id (unverified semantic)
    1: "ems_value",             # JSON blob (documented)
    2: "event_time_str",        # human-readable event time (local)
    3: "imei",                  # truck/device identifier (NEVER output raw)
    4: "imei_suffix",           # short imei suffix (derived)
    5: "device_hex_id",         # hex device id (NEVER output raw)
    6: "road_route_id",         # numeric id (unverified semantic)
    7: "device_type",           # e.g. E-Router-4G
    8: "license_plate",         # NEVER written to outputs
    9: "plate_other_id",        # secondary plate/vehicle id
    10: "gps_latitude",
    11: "gps_longitude",
    12: "speed_kmh",            # numeric (unverified semantic)
    13: "event_time2",          # second time field
    14: "province",
    15: "city",
    16: "unknown_outer_16",
    17: "address",
    18: "road_name",
    19: "road_type",
    20: "file_date",            # YYYYMMDD (date only)
}

# Outer columns that are numeric (profiled in the SAMPLE phase).
NUMERIC_OUTER_COLUMNS = [0, 6, 10, 11, 12]

# JSON sub-fields of interest.
JSON_SUBFIELDS = [
    "x7000",  # engine speed
    "x7001",  # instantaneous fuel rate (target)
    "x7002",  # cumulative fuel counter
    "x7003",  # brake
    "x7004",  # distance / mileage
    "x7005",  # cumulative total mileage
    "x7006",  # throttle
    "x7007",  # coolant temperature
    "x006C",  # GPS speed
    "x7091",  # fuel level
    "x7035",  # handbrake
    "x7095",  # unknown
    "ems_type",
    "date",    # epoch ms
    "imei",
]

# JSON sub-fields whose values are numeric (may arrive as JSON strings)
JSON_NUMERIC_SUBFIELDS = {
    "x7000", "x7001", "x7002", "x7003", "x7004", "x7005", "x7006",
    "x7007", "x006C", "x7091", "x7035", "x7095", "ems_type", "date",
}

# sample size for JSON sub-field deep stats / duplicate / PII analysis
JSON_SAMPLE_SIZE = 300_000

RNG_SEED = 20240501

_READ_CHUNK = 1 << 24          # 16 MiB read chunks
_PROGRESS_EVERY = 1_000_000    # print progress every N rows


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def _fmt_int(n):
    return f"{n:,}"


def _num(s: bytes):
    """Parse a numeric byte-string; None on empty/non-numeric."""
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _coerce_num(v):
    """Coerce a decoded JSON value to float, or None.

    Handles int/float and numeric strings; rejects None, bool, empty
    strings, malformed strings, NaN and Infinity.
    """
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        f = float(v)
    elif isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            f = float(s)
        except ValueError:
            return None
    else:
        return None
    if f != f or f == float("inf") or f == float("-inf"):
        return None
    return f


def _percentile(sorted_vals, p):
    if not sorted_vals:
        return None
    idx = (len(sorted_vals) - 1) * p / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


class Reservoir:
    """Algorithm-R reservoir of raw line bytes (seeded, deterministic)."""

    __slots__ = ("k", "seen", "items", "rng")

    def __init__(self, k, rng: random.Random):
        self.k = k
        self.seen = 0
        self.items = []
        self.rng = rng

    def offer(self, line: bytes):
        self.seen += 1
        if len(self.items) < self.k:
            self.items.append(line)
        else:
            j = self.rng.randrange(self.seen)
            if j < self.k:
                self.items[j] = line


def _iter_raw_lines(path: Path):
    """Yield raw line bytes without loading the file. 16 MiB chunks."""
    with open(path, "rb") as fh:
        carry = b""
        while True:
            buf = fh.read(_READ_CHUNK)
            if not buf:
                break
            buf = carry + buf
            lines = buf.split(b"\n")
            carry = lines.pop()
            for ln in lines:
                yield ln
        if carry:
            yield carry


def _extract_date_fast(jb: bytes):
    """Manually scan ``"date":<digits>`` inside the ems_value JSON bytes.

    Fast (no regex, no full JSON parse); returns epoch-ms int or None.
    """
    i = jb.find(b'"date"')
    if i < 0:
        return None
    j = jb.find(b":", i)
    if j < 0:
        return None
    j += 1
    n = len(jb)
    while j < n and jb[j] in (32, 9):   # skip space/tab
        j += 1
    s = j
    while j < n and 48 <= jb[j] <= 57:  # digits
        j += 1
    if j == s:
        return None
    return int(jb[s:j])


def _is_json_structurally_valid(jb: bytes) -> bool:
    """Cheap exact structural check: a non-empty ``{ ... }`` object."""
    return len(jb) >= 2 and jb[0] == 123 and jb[-1] == 125   # { }


def _parse_json_dict(raw: bytes):
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


# --------------------------------------------------------------------------
# Full scan (exact, lean) + reservoir sample analysis
# --------------------------------------------------------------------------

def _scan_line(line: bytes):
    """Manually deconstruct one raw line.

    Returns (json_bytes, truck_key). json_bytes is the ``ems_value``
    column (col 1). truck_key is the raw imei bytes of col 3 (the stable
    truck/device identifier), falling back to col 5 (hex device id) or
    b"" when malformed. No per-row SHA-256; the raw bytes are used directly
    as the dict key and only their anonymized name is output.
    """
    i1 = line.find(b"\t")
    if i1 < 0:
        return None, b""
    i2 = line.find(b"\t", i1 + 1)
    if i2 < 0:
        return None, b""
    i3 = line.find(b"\t", i2 + 1)
    if i3 < 0:
        return None, b""
    i4 = line.find(b"\t", i3 + 1)
    if i4 < 0:
        i4 = len(line)
    truck = line[i3 + 1:i4]
    if not truck:
        i5 = line.find(b"\t", i4 + 1)
        if i5 < 0:
            i5 = len(line)
        truck = line[i4 + 1:i5]
    return line[i1 + 1:i2], truck


def run_profile(path: Path, sample_size: int = JSON_SAMPLE_SIZE,
                max_rows: int | None = None) -> tuple[dict, dict]:
    rng = random.Random(RNG_SEED)
    reservoir = Reservoir(sample_size, rng)

    # ---- exact full-scan accumulators ----
    line_count = 0
    bytes_read = 0
    zero_len_lines = 0
    json_struct_fail = 0
    json_empty = 0
    date_missing = 0
    date_min = None
    date_max = None
    global_prev = None
    n_decrease_global = 0
    n_decrease_per_truck = 0
    truck_prev_date = {}

    truck_count = {}
    truck_first = {}

    t0 = time.time()
    for line in _iter_raw_lines(path):
        line_count += 1
        bytes_read += len(line)
        if max_rows is not None and line_count > max_rows:
            break
        if not line:
            zero_len_lines += 1
            continue
        if line_count % _PROGRESS_EVERY == 0:
            print(f"  ... scanned {_fmt_int(line_count)} rows "
                  f"({time.time() - t0:.1f}s)", flush=True)

        jb, tk = _scan_line(line)
        if jb is None:
            zero_len_lines += 1
            continue

        # exact JSON structural failure count
        if not jb:
            json_empty += 1
            json_struct_fail += 1
        elif not _is_json_structurally_valid(jb):
            json_struct_fail += 1

        # exact truck identity (raw bytes key, no SHA)
        if tk not in truck_count:
            truck_count[tk] = 0
            truck_first[tk] = len(truck_count) - 1
        truck_count[tk] += 1

        # exact date min/max + chronology
        if jb:
            d = _extract_date_fast(jb)
        else:
            d = None
        if d is None:
            date_missing += 1
        else:
            if date_min is None or d < date_min:
                date_min = d
            if date_max is None or d > date_max:
                date_max = d
            if global_prev is not None and d < global_prev:
                n_decrease_global += 1
            global_prev = d
            tp = truck_prev_date.get(tk)
            if tp is not None and d < tp:
                n_decrease_per_truck += 1
            truck_prev_date[tk] = d

        reservoir.offer(line)

    scan_seconds = time.time() - t0

    # ---- anonymize trucks ----
    trucks = sorted(truck_count.items(), key=lambda kv: truck_first[kv[0]])
    truck_names = {k: f"truck_{i + 1:03d}" for i, (k, _) in enumerate(trucks)}
    records_per_truck = [
        {"truck": truck_names[k], "records": int(v)} for k, v in trucks
    ]

    # ---- sample phase (estimates) ----
    o = _analyze_sample(reservoir.items, sample_size)

    # ---- assemble exact scan block ----
    scan = {
        "lines_total_exact": int(line_count),
        "bytes_read_exact": int(bytes_read),
        "scan_seconds": round(scan_seconds, 2),
        "zero_len_lines_exact": int(zero_len_lines),
        "json_structural_fail_exact": int(json_struct_fail),
        "json_empty_exact": int(json_empty),
        "date_missing_rows_exact": int(date_missing),
        "date_min_ms_exact": int(date_min) if date_min is not None else None,
        "date_max_ms_exact": int(date_max) if date_max is not None else None,
        "date_nondecrease_global_exact": int(n_decrease_global),
        "date_nondecrease_per_truck_exact": int(n_decrease_per_truck),
    }

    summary = {
        "input_path": str(path),
        "scan": scan,
        "trucks": {
            "distinct_exact": int(len(truck_count)),
            "records_exact": records_per_truck,
        },
        "metrics_scope": {
            "exact": [
                "lines_total", "bytes_read", "zero_len_lines",
                "unique_trucks", "records_per_truck",
                "date_min", "date_max", "date_missing",
                "chronology_global", "chronology_per_truck",
                "json_structural_fail",
            ],
            "sample": [
                "outer_column_numeric_minmax", "outer_column_blank_rates",
                "json_keys", "json_subfield_stats", "x7001_suspicious",
                "x7002_monotonicity", "duplicate_line_hashes",
                "pii_identifier_consistency",
            ],
        },
        "columns": o["columns"],
        "json_sample": o["json_sample"],
        "json_fields": o["json_fields"],
        "duplicates": o["duplicates"],
        "x7002": o["x7002"],
        "pii": o["pii"],
        "row_col_count_neq_21_sample": int(o["row_col_count_neq_21_sample"]),
        "meta": {
            "profiler": "src/data/profile_ems1.py",
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seed": RNG_SEED,
            "json_sample_size": int(sample_size),
        },
    }
    return summary, truck_names


# --------------------------------------------------------------------------
# Reservoir sample analysis (estimates only)
# --------------------------------------------------------------------------

def _analyze_sample(items: list[bytes], sample_size: int) -> dict:
    json_total = 0
    json_fail = 0
    json_keys = collections.Counter()
    sub = {f: [] for f in JSON_SUBFIELDS}
    sub_missing = {f: 0 for f in JSON_SUBFIELDS}
    sub_numeric_ok = {f: 0 for f in JSON_SUBFIELDS}
    dup_hashes = collections.Counter()
    x7002_decreases = 0
    x7002_seq = {}
    x7001_suspicious = {"negative": 0, "zero": 0, "gt_500": 0, "gt_1000": 0}
    x7001_vals = []

    outer_blank = [0] * 21
    outer_numeric = {c: {"n": 0, "min": None, "max": None}
                     for c in NUMERIC_OUTER_COLUMNS}

    plate_truck = {}
    multi_plate = 0
    outer_imei_pairs = set()
    outer_json_imei_mismatch = 0
    unexpected_col_count = 0

    for line in items:
        h = hashlib.sha256(line).digest()[:8]
        dup_hashes[h] += 1

        parts = line.split(b"\t")
        np = len(parts)
        tk = parts[3] if np > 3 else b""

        if np != 21:
            unexpected_col_count += 1

        for c in range(21):
            if c >= np or parts[c] == b"":
                outer_blank[c] += 1
        for c in NUMERIC_OUTER_COLUMNS:
            if c < np:
                v = _num(parts[c])
                if v is not None:
                    o = outer_numeric[c]
                    o["n"] += 1
                    if o["min"] is None or v < o["min"]:
                        o["min"] = v
                    if o["max"] is None or v > o["max"]:
                        o["max"] = v

        if np > 8 and parts[8]:
            tk = parts[3] if np > 3 else b""
            if parts[8] not in plate_truck:
                plate_truck[parts[8]] = tk
            elif plate_truck[parts[8]] != tk:
                multi_plate += 1

        json_total += 1
        if np < 2:
            json_fail += 1
            for f in sub_missing:
                sub_missing[f] += 1
            continue
        d = _parse_json_dict(parts[1])
        if d is None:
            json_fail += 1
            for f in sub_missing:
                sub_missing[f] += 1
            continue
        for k in d:
            json_keys[k] += 1
        for f in JSON_SUBFIELDS:
            if f in d:
                nv = _coerce_num(d[f]) if f in JSON_NUMERIC_SUBFIELDS else None
                if nv is not None:
                    sub[f].append(nv)
                    sub_numeric_ok[f] += 1
                else:
                    sub[f].append(d[f])
            else:
                sub_missing[f] += 1

        x1 = _coerce_num(d.get("x7001"))
        if x1 is not None:
            x7001_vals.append(x1)
            if x1 < 0:
                x7001_suspicious["negative"] += 1
            elif x1 == 0:
                x7001_suspicious["zero"] += 1
            if x1 > 500:
                x7001_suspicious["gt_500"] += 1
            if x1 > 1000:
                x7001_suspicious["gt_1000"] += 1

        imei = d.get("imei")
        if imei is not None:
            ikey = _bytes_key(str(imei))
            if np > 3 and parts[3]:
                outer_imei_pairs.add((parts[3], ikey))
                if parts[3] != ikey:
                    outer_json_imei_mismatch += 1
        x2 = _coerce_num(d.get("x7002"))
        if x2 is not None:
            prev = x7002_seq.get(tk)
            if prev is not None and x2 < prev:
                x7002_decreases += 1
            x7002_seq[tk] = x2

    # columns profile (sample based)
    col_profile = {}
    for c in range(21):
        name = KNOWN_OUTER_COLUMNS.get(c, f"unknown_outer_{c}")
        entry = {
            "index": c,
            "name": name,
            "blank_count_sample": int(outer_blank[c]),
            "blank_rate_sample": round(outer_blank[c] / max(1, json_total), 6),
        }
        if c in NUMERIC_OUTER_COLUMNS:
            o = outer_numeric[c]
            entry["numeric_min_sample"] = o["min"]
            entry["numeric_max_sample"] = o["max"]
            entry["numeric_seen_sample"] = int(o["n"])
        col_profile[str(c)] = entry

    json_field_stats = {}
    for f in JSON_SUBFIELDS:
        vals = sub[f]
        numvals = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
        present = len(vals)
        entry = {
            "present_in_sample": int(present),
            "present_rate_sample": round(present / max(1, json_total), 6),
            "missing_in_sample": int(sub_missing[f]),
            "sample_n": int(len(numvals)),
        }
        if numvals:
            sv = sorted(numvals)
            entry["min"] = sv[0]
            entry["max"] = sv[-1]
            entry["median"] = _percentile(sv, 50)
            entry["mean"] = sum(sv) / len(sv)
            entry["p05"] = _percentile(sv, 5)
            entry["p95"] = _percentile(sv, 95)
        json_field_stats[f] = entry

    sv1 = sorted(x7001_vals)
    x1 = json_field_stats.get("x7001", {})
    x1["sample_n"] = len(sv1)
    if sv1:
        x1["min"] = sv1[0]
        x1["max"] = sv1[-1]
        x1["median"] = _percentile(sv1, 50)
        x1["mean"] = sum(sv1) / len(sv1)
        x1["p05"] = _percentile(sv1, 5)
        x1["p95"] = _percentile(sv1, 95)
    x1["suspicious"] = x7001_suspicious
    json_field_stats["x7001"] = x1

    duplicates = sum(1 for _, cnt in dup_hashes.items() if cnt > 1)

    return {
        "columns": col_profile,
        "json_sample": {
            "rows_examined_sample": int(json_total),
            "parse_failures_sample": int(json_fail),
            "failure_rate_sample": round(json_fail / max(1, json_total), 6),
            "keys": dict(json_keys.most_common()),
        },
        "json_fields": json_field_stats,
        "duplicates": {
            "exact_line_hashes_in_sample": int(duplicates),
            "sample_rows": int(json_total),
        },
        "row_col_count_neq_21_sample": int(unexpected_col_count),
        "x7002": {
            "sample_decreases": int(x7002_decreases),
            "sample_rows_with_counter": int(len(x7002_seq)),
        },
        "pii": {
            "multi_outer_plate_to_truck_sample": int(multi_plate),
            "sample_outer_imei_pairs": int(len(outer_imei_pairs)),
            "outer_json_imei_mismatch_sample": int(outer_json_imei_mismatch),
        },
    }


def _bytes_key(s) -> bytes:
    """Stable byte key for an int/str identifier (never printed raw)."""
    if isinstance(s, int):
        return s.to_bytes((s.bit_length() + 7) // 8 or 1, "little")
    return s.encode("utf-8", "replace")


# --------------------------------------------------------------------------
# Report writer
# --------------------------------------------------------------------------

def _write_report(summary: dict, report_path: Path = REPORT_PATH) -> None:
    lines = []
    a = lines.append
    a("# FEAD EMS-1 Data Profiling Report")
    a("")
    a(f"Generated: `{summary['meta']['generated_utc']}` (UTC)")
    a(f"Profiler: `{summary['meta']['profiler']}`")
    a("")
    a("> **Metric scope.** Exact full-dataset metrics are labelled `(exact)`;")
    a("> reservoir-sample estimates are labelled `(sample)`. Nothing is silently")
    a("> downgraded. See `metrics_scope` in the JSON summary for the split list.")
    a("")
    a("## 1. Scan overview")
    a("")
    s = summary["scan"]
    a(f"- Input: `{summary['input_path']}`")
    a(f"- Total records (physical lines, **exact**): **{_fmt_int(s['lines_total_exact'])}**")
    a(f"- Bytes scanned (exact): {_fmt_int(s['bytes_read_exact'])} ({round(s['bytes_read_exact'] / 1e9, 2)} GB)")
    a(f"- Scan wall time: {s['scan_seconds']} s (streaming, single lean pass)")
    a(f"- Zero-length / malformed lines (exact): {_fmt_int(s['zero_len_lines_exact'])}")
    a("")
    a("## 2. Trucks")
    a("")
    tr = summary["trucks"]
    a(f"- Distinct trucks (device-id based, **exact**): **{tr['distinct_exact']}**")
    a(f"- Records per truck (exact, anonymized):")
    a("")
    a("| truck | records (exact) |")
    a("|---|---|")
    for r in tr["records_exact"]:
        a(f"| {r['truck']} | {_fmt_int(r['records'])} |")
    a("")
    a("## 3. Chronology & sampling (exact)")
    a("")
    a(f"- Rows missing a `date` field (exact): {_fmt_int(s['date_missing_rows_exact'])}")
    a(f"- Date minimum (exact): {s['date_min_ms_exact']} ms (epoch)")
    a(f"- Date maximum (exact): {s['date_max_ms_exact']} ms (epoch)")
    a(f"- Global date-nondecrease rows (exact): {_fmt_int(s['date_nondecrease_global_exact'])}")
    a(f"- Per-truck date-nondecrease rows (exact): {_fmt_int(s['date_nondecrease_per_truck_exact'])}")
    a("- Chronological order is assessed from the exact full-scan date scan;")
    a("  per-truck sampling-interval statistics would require a full JSON parse")
    a("  of every row and are deliberately not computed in this pass.")
    a("")
    a("## 4. JSON validity (exact structural + sample semantic)")
    a("")
    a(f"- Structurally invalid `ems_value` (non-`{{...}}` / empty, **exact**):")
    a(f"  **{_fmt_int(s['json_structural_fail_exact'])}**")
    j = summary["json_sample"]
    a(f"- Full JSON parse failures (sample): {_fmt_int(j['parse_failures_sample'])} "
      f"({round(j['failure_rate_sample'] * 100, 4)}%)")
    a(f"- JSON keys seen in sample ({len(j['keys'])}): `{', '.join(sorted(j['keys']))}`")
    a("")
    a("## 5. Outer columns (21, tab-separated, header-less)")
    a("")
    a("| idx | provisional name | blank rate (sample) | numeric min (sample) | numeric max (sample) |")
    a("|---|---|---|---|---|")
    for idx in sorted(summary["columns"], key=lambda x: int(x)):
        c = summary["columns"][idx]
        a(
            f"| {c['index']} | {c['name']} | "
            f"{round(c['blank_rate_sample'] * 100, 4)}% | "
            f"{c.get('numeric_min_sample', '—')} | {c.get('numeric_max_sample', '—')} |"
        )
    a("")
    a("Column 1 (`ems_value`) is the JSON blob (see §4). Names are *provisional*;")
    a("unverified outer columns are marked `unknown_outer_*`. License plates are")
    a("never written here.")
    a("")
    a("## 6. Key JSON sub-fields (sample)")
    a("")
    a("| field | present (sample) | n | min | median | mean | max | p95 |")
    a("|---|---|---|---|---|---|---|---|")
    for f in JSON_SUBFIELDS:
        st = summary["json_fields"].get(f, {})
        if not st:
            continue
        a(
            f"| {f} | {round(st.get('present_rate_sample', 0) * 100, 2)}% | "
            f"{st.get('sample_n', 0)} | {st.get('min', '—')} | "
            f"{st.get('median', '—')} | {round(st.get('mean', 0), 3) if st.get('mean') is not None else '—'} | "
            f"{st.get('max', '—')} | {st.get('p95', '—')} |"
        )
    a("")
    a("### x7001 (fuel-rate target) — suspicious values (sample)")
    a("")
    sus = summary["json_fields"].get("x7001", {}).get("suspicious", {})
    a(f"- negative: {_fmt_int(sus.get('negative', 0))}")
    a(f"- zero: {_fmt_int(sus.get('zero', 0))}")
    a(f"- > 500: {_fmt_int(sus.get('gt_500', 0))}")
    a(f"- > 1000: {_fmt_int(sus.get('gt_1000', 0))}")
    a("")
    a("## 7. Duplicates (sample)")
    a("")
    dup = summary["duplicates"]
    a(f"- Exact duplicate line-hashes within {_fmt_int(dup['sample_rows'])} sampled rows: **{dup['exact_line_hashes_in_sample']}**")
    a("- Full-file duplicate scan was not performed (memory-bound); see limitations.")
    a("")
    a("## 8. x7002 (cumulative fuel counter) monotonicity (sample)")
    a("")
    a(f"- Sample-observed decreases (counter going backwards within a truck): **{summary['x7002']['sample_decreases']}**")
    a(f"- Sample rows carrying an x7002 value: {_fmt_int(summary['x7002']['sample_rows_with_counter'])}")
    a("")
    a("## 9. PII / identifier consistency (sample)")
    a("")
    p = summary["pii"]
    a(f"- Rows where an outer license-plate mapped to a different truck key: {p['multi_outer_plate_to_truck_sample']}")
    a(f"- Distinct (outer-device, JSON-imei) pairs in sample: {p['sample_outer_imei_pairs']}")
    a("- No raw identifier (plate, IMEI, device id) is printed in this report.")
    a("")
    a("## 10. Limitations")
    a("")
    a("- JSON deep stats, duplicate counts, x7002 monotonicity and PII consistency")
    a("  are based on a **reservoir sample** of "
      f"{_fmt_int(summary['json_sample']['rows_examined_sample'])} rows, not all 10.27M.")
    a("- Exact per-truck sampling intervals require a full JSON parse of every row;")
    a("  not performed in this profiling pass.")
    a("- Duplicate detection is sample-based; a global duplicate count is unknown.")
    a("- Column 0 and several outer columns are provisional names (`unknown_outer_*`)")
    a("  until matched to the FEAD message spec.")
    a("")
    a("## 11. Notable observations (not yet conclusions)")
    a("")
    a("- `x7002` is a cumulative counter; its raw value should **not** be used as a")
    a("  model feature. `x7001` (target) appears inside the feature range; it must")
    a("  be excluded from any feature allowlist.")
    a("- `x7091` (fuel level) is present but its exact semantics need confirmation.")
    a("- The dataset is a single merged CSV, not per-day files as the repo loaders")
    a("  expect — ingestion design must account for that.")
    a("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Streaming profiler for FEAD EMS-1")
    ap.add_argument("--input", type=Path, default=DATASET_PATH)
    ap.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    ap.add_argument("--report", type=Path, default=REPORT_PATH)
    ap.add_argument("--sample-size", type=int, default=JSON_SAMPLE_SIZE)
    ap.add_argument("--max-rows", type=int, default=None,
                    help="Stop after N rows (for benchmarking only)")
    args = ap.parse_args()

    summary, _ = run_profile(args.input, args.sample_size, args.max_rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(summary, args.report)
    print(f"done -> {args.summary} | {args.report}")
    print(f"lines={summary['scan']['lines_total_exact']} "
          f"trucks={summary['trucks']['distinct_exact']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
