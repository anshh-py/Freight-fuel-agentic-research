"""Temporary read-only diagnostic: print failing check details for M2.3 eval."""
import json
from pathlib import Path

d = json.loads(Path("reports/m2/response_evaluation/evaluation.json").read_text())
for r in d["results"]:
    if r["status"] == "error":
        print("=== ERROR", r["id"], r.get("error"))
        continue
    if r["all_passed"]:
        continue
    print("===", r["id"], r["category"], "expected_pass=", r["expected_pass"],
          "expected_violations=", r["expected_violations"])
    for c in r["checks"]:
        if not c["passed"]:
            print("   FAILED", c["check_id"], "::", c["detail"])
