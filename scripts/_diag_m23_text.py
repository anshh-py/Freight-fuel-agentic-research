"""Temp read-only diag: print full driver_version for real scenarios."""
import json
from pathlib import Path

d = json.loads(Path("reports/m2/response_evaluation/evaluation.json").read_text())
for want in ("K01", "E01", "C01", "I01"):
    for r in d["results"]:
        if r["id"] == want:
            print("=" * 30, want, "=" * 30)
            print(r.get("driver_postview_preview"))
            print()
