"""Temp read-only diag: recompute verdicts and show driver previews."""
import json
from pathlib import Path

d = json.loads(Path("reports/m2/response_evaluation/evaluation.json").read_text())
for r in d["results"]:
    if r["status"] != "ok":
        continue
    # recompute from stored checks
    by = {c["check_id"]: c for c in r["checks"]}
    pass_ok = all(by.get(cid) is not None and by[cid]["passed"] for cid in r["expected_pass"])
    viol_ok = all(by.get(cid) is not None and not by[cid]["passed"] for cid in r["expected_violations"])
    rec = pass_ok and viol_ok
    marker = "MATCH" if rec == r["all_passed"] else "<<< MISMATCH"
    print(f"[{marker}] {r['id']} reported_all_passed={r['all_passed']} recomputed={rec} "
          f"pass_ok={pass_ok} viol_ok={viol_ok}")
    print("   preview:", (r.get("driver_postview_preview") or "")[:200].replace("\n", " | "))
