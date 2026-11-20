import json, re, io, sys

d = json.load(open("reports/data_profiling/_bench_summary.json", encoding="utf-8"))
print("TOPKEYS", sorted(d.keys()))
print("ROWCOL_NEQ21", d.get("row_col_count_neq_21_sample"))
print("PII", d.get("pii"))
print("DUP", d.get("duplicates"))
x1 = d["json_fields"].get("x7001", {})
x2 = d["json_fields"].get("x7002", {})
x0 = d["json_fields"].get("x7000", {})
x6 = d["json_fields"].get("x7006", {})
xc = d["json_fields"].get("x006C", {})
x9 = d["json_fields"].get("x7091", {})
for k, lbl in ((x1, "X7001"), (x2, "X7002"), (x0, "X7000"), (x6, "X7006"), (xc, "X006C"), (x9, "X7091")):
    print(lbl, "sample_n=", k.get("sample_n"), "present_rate=", k.get("present_rate_sample"),
          "min=", k.get("min"), "max=", k.get("max"), "susp=", k.get("suspicious"))

s = io.open("reports/data_profiling/_bench_report.md", encoding="utf-8").read()
print("REPORT_EXISTS len=", len(s))
print("raw_plate", bool(re.search(r"[\u4e00-\u9fff][A-Z][A-Z0-9]{5}", s)))
print("raw_imei15", bool(re.search(r"\b[0-9]{15}\b", s)))
print("raw_hex32", bool(re.search(r"\b[0-9A-F]{32}\b", s)))
print("truck_anon", sorted(set(re.findall(r"truck_\d{3}", s))))
print("exact_labelled", ("exact" in s))
print("sample_labelled", ("sample" in s))
