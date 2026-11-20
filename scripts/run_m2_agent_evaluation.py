#!/usr/bin/env python
"""M2.4 — agent evaluation runner.

Runs the M2.2 orchestrator + M2.3 response generator + M2.3 validation
battery over a manually authored 49-scenario benchmark, and reports seven
dimensions (intent classification, tool execution, prediction integrity,
explanation integrity, knowledge grounding, response reliability,
end-to-end reliability) with per-dimension pass rates, failed scenario ids,
and failure categories.

Scoring is fully deterministic. No LLM judge, no paid API, no subjective
quality score. Real scenarios run against the actual saved models and corpus.
"direct" scenarios call a tool directly (used for telematics-only prediction,
which the orchestrator does not expose). Mock scenarios inject controlled
failures to prove the validator reacts correctly.

Negative scenarios (those asserting an expected violation) are guardrail
controls: a negative passes iff its expected violation actually fired. They
are excluded from per-dimension pass-rate denominators, which measure normal
positive behavior.

Outputs (reports/m2/agent_evaluation/):
    benchmark.json        — the scenario dataset
    scenario_results.json — per-scenario detailed outcomes
    evaluation.json       — per-dimension aggregates + per-scenario verdicts
    summary.md            — human-readable summary
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.agent.intents import Intent
from src.agent.orchestrator import AgentOrchestrator, AgentResponse
from src.agent.response_generator import GroundedResponse, ResponseGenerator
from src.agent.response_validation import validate_grounded_response
from src.tools.fuel_rate_prediction import FuelRatePredictionTool
from src.tools.knowledge_retrieval import KnowledgeRetrievalTool
from src.tools.prediction_explanation import PredictionExplanationTool
from tests.test_tools import FULL_INPUTS, TELE_INPUTS

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "reports" / "m2" / "agent_evaluation"
BENCHMARK_PATH = OUT_DIR / "benchmark.json"

# A manufactured technical specification is a concrete number attached to an
# engineering-spec unit. Fuel-rate / SHAP values are expressed in L/h and are
# NOT specs; feature-name mentions like "Torque (N·m)" carry no numeric value
# immediately before the unit, so they are not matched either.
SPEC_FABRICATED_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:N·m|Nm|bar|psi|°C|°F|kg|t(?:ons)?)\b",
    re.I,
)

# prohibited-claim -> validation check that must PASS for the claim to be absent
# NOTE: "fabricated_spec" is intentionally NOT in this map: whose validator only
# runs for unsupported/ambiguous intents, so it cannot serve knowledge-intent
# scenarios (e.g. I01). It is evaluated content-based in `_prohibit` instead.
PROHIBIT_CHECK = {
    "causal_claim": "causal_language",
    "oem_claim": "oem_mislabeling",
    "fabricated_source": "source_attribution",
    "fabricated_prediction": "failure_not_presented_as_success",
}


def _build_inputs(sc: dict) -> dict | None:
    ref = sc.get("required_input")
    if ref is None:
        return None
    if ref == "FULL":
        return dict(FULL_INPUTS)
    if ref == "TELE":
        return dict(TELE_INPUTS)
    if isinstance(ref, dict) and "missing" in ref:
        base = dict(FULL_INPUTS)
        for k in ref["missing"]:
            base.pop(k, None)
        return base
    if isinstance(ref, dict) and "override" in ref:
        base = dict(FULL_INPUTS)
        base.update(ref["override"])
        return base
    return dict(FULL_INPUTS)


def _run_direct(sc: dict) -> AgentResponse:
    """Direct tool call (telematics-only prediction)."""
    tool = FuelRatePredictionTool()
    if sc["required_input"] == "TELE":
        res = tool.predict(dict(TELE_INPUTS), "telematics_only")
    else:
        res = tool.predict(dict(FULL_INPUTS), "full_operational")
    resp = AgentResponse()
    resp.intent = Intent.PREDICT_FUEL
    resp.selected_tools = ["fuel_rate_prediction"]
    resp.tool_results = [{"tool": "fuel_rate_prediction", "status": "ok",
                          "result": res}]
    return resp


def _source_results(resp) -> tuple:
    """Extract raw tool result dicts from an AgentResponse (matching the raw
    tool output shape the validation checks expect)."""
    src_pred = None
    src_expl = None
    for r in (getattr(resp, "tool_results", None) or []):
        if r.get("status") != "ok":
            continue
        if r["tool"] == "fuel_rate_prediction":
            src_pred = r["result"]
        elif r["tool"] == "prediction_explanation":
            src_expl = r["result"]
    return src_pred, src_expl


def run_scenario(orch, gen, sc: dict) -> dict:
    build = sc["build"]
    if build == "mock":
        m = sc.get("mock", {})
        grounded = GroundedResponse(
            intent=sc["expected_intent"],
            detected_intent=sc["expected_intent"],
            router_backend="deterministic",
            selected_tools=list(sc["expected_tools"]),
            prediction=m.get("prediction"),
            explanation=m.get("explanation"),
            evidence=m.get("evidence", []),
            evidence_insufficient=m.get("evidence_insufficient", False),
            tool_failures=m.get("tool_failures", []),
            limitations=[],
            technical_version=m.get("technical_version", {}),
            driver_version=m.get("driver_version", ""),
        )
        resp = None
        src_pred, src_expl = m.get("prediction"), m.get("explanation")
        detected_intent = sc["expected_intent"]
    else:
        resp = _run_direct(sc) if build == "direct" else orch.handle(
            sc["query"], _build_inputs(sc))
        grounded = gen.generate(resp)
        src_pred, src_expl = _source_results(resp)
        detected_intent = resp.intent.value if resp.intent else None

    known = [e["document_id"] for e in (grounded.evidence or [])]
    issues = validate_grounded_response(
        grounded,
        source_prediction=src_pred,
        source_explanation=src_expl,
        known_doc_ids=known,
    )
    by_id = {i.check_id: i for i in issues}
    passed_ids = {cid for cid, i in by_id.items() if i.passed}

    text = grounded.driver_version or ""
    tl = text.lower()

    exp_pass_ok = all(by_id.get(cid) is not None and by_id[cid].passed
                      for cid in sc["expect_pass"])
    exp_viol_ok = all(by_id.get(cid) is not None and not by_id[cid].passed
                      for cid in sc["expect_violations"])

    def _require(req):
        if req == "prediction_present":
            return "L/h" in text
        if req == "shap_present":
            return "SHAP" in text
        if req == "non_causal_stated":
            return "not causality" in tl or "not a causal" in tl
        if req == "retrieval_present":
            return "synthetic" in tl and ("passage" in tl or "knowledge" in tl)
        if req == "feature_set_identified":
            return "feature set:" in tl
        if req == "disclaimer_present":
            return "synthetic" in tl
        if req == "refused":
            return "can't answer" in tl or "scope" in tl
        if req == "clarification_requested":
            return "couldn't pin down" in tl or "clarification" in tl
        if req == "failure_disclosed":
            return "failed" in tl or "failure" in tl
        if req == "technical_format_present":
            return bool(grounded.technical_version)
        if req == "synthetic_corpus_noted":
            return "synthetic educational" in tl
        if req == "insufficiency_stated":
            return "no sufficiently supportive passage" in tl
        if req == "synthetic_model_stated":
            return "synthetic" in tl and "model" in tl
        return True

    require_ok = all(_require(r) for r in sc.get("require", []))

    def _prohibit(req):
        if req == "fabricated_spec":
            # Content-based, intent-independent: flag a concrete engineering
            # specification (number + spec unit) that is not a legitimate
            # fuel-rate/SHAP L/h value.
            return SPEC_FABRICATED_RE.search(text) is not None
        if req in PROHIBIT_CHECK:
            return by_id.get(PROHIBIT_CHECK[req]) is None or not by_id[PROHIBIT_CHECK[req]].passed
        if req == "invented_driver_history":
            return any(w in tl for w in ("driver", "route", "vin"))
        if req == "real_world_claim":
            return any(s in tl for s in ("actual fuel consumption", "field-verified", "measured in real trucks"))
        if req == "global_importance_substitution":
            return "feature importance" in tl and "SHAP" not in text
        return False

    prohib_ok = all(not _prohibit(r) for r in sc.get("prohibited", []))

    is_negative = bool(sc["expect_violations"])
    scenario_pass = exp_viol_ok if is_negative else (exp_pass_ok and require_ok and prohib_ok)

    return {
        "id": sc["id"],
        "category": sc["category"],
        "description": sc["description"],
        "build": build,
        "status": "ok",
        "pass": scenario_pass,
        "is_negative": is_negative,
        "detected_intent": detected_intent,
        "expected_intent": sc["expected_intent"],
        "selected_tools": list(getattr(resp, "selected_tools", []) if resp else sc["expected_tools"]),
        "expected_tools": sc["expected_tools"],
        "tool_exec_status": {
            t.get("tool"): t.get("status") for t in (getattr(resp, "tool_results", None) or [])
        },
        "expected_pass": sc["expect_pass"],
        "expected_pass_ok": exp_pass_ok,
        "expected_violations_ok": exp_viol_ok,
        "requirements_ok": require_ok,
        "prohibited_ok": prohib_ok,
        "check_summary": {
            "total": len(issues),
            "passed": len(passed_ids),
            "failed": len(issues) - len(passed_ids),
            "passed_checks": sorted(passed_ids),
            "failed_checks": sorted({cid for cid, i in by_id.items() if not i.passed}),
        },
        "validation_issues": [
            {"check_id": i.check_id, "passed": i.passed, "detail": i.detail, "severity": i.severity}
            for i in issues
        ],
        "text_full": text,
        "text_preview": text[:240],
        "failure_categories": sorted({cid for cid, i in by_id.items() if not i.passed}),
    }


def dimensions_for_scenario(r: dict) -> dict:
    exp_tools = r["expected_tools"]
    intent = r["expected_intent"]
    cat = r["category"]
    checks = set(r["check_summary"]["passed_checks"])
    txt = r.get("text_full") or r["text_preview"]
    txt = txt.lower()
    out = {}

    ia_ok = r["detected_intent"] == intent
    amb_ok = intent != "ambiguous" or "couldn't pin down" in txt
    uns_ok = intent != "unsupported" or "can't answer" in txt or "scope" in txt
    out["A_intent_classification"] = {"applicable": True,
                                       "passed": ia_ok and amb_ok and uns_ok}

    tool_ok = r["selected_tools"] == exp_tools
    fail_check_ok = "failure_not_presented_as_success" in checks
    out["B_tool_execution"] = {"applicable": True,
                                "passed": tool_ok and fail_check_ok}

    # Applicability is tied to the scenario's DECLARED expectations, not to
    # mere tool presence: a scenario that expects a successful prediction/top
    # (predict expectation in expect_pass) counts toward that dimension; a
    # scenario that expects a failure does not get penalized there.
    pred_app = "prediction_value_preserved" in r.get("expected_pass", [])
    out["C_prediction_integrity"] = {
        "applicable": pred_app,
        "passed": (not pred_app) or (
            "prediction_value_preserved" in checks
            and "synthetic_disclaimer" in checks
            and "synthetic" in txt and "model" in txt),
    }

    expl_app = "shap_value_preserved" in r.get("expected_pass", [])
    non_causal = "not causality" in txt or "not a causal" in txt
    out["D_explanation_integrity"] = {
        "applicable": expl_app,
        "passed": (not expl_app) or (
            "shap_value_preserved" in checks
            and "causal_language" in checks
            and non_causal),
    }

    kn_app = "source_attribution" in r.get("expected_pass", [])
    kn_checks = {"source_attribution", "synthetic_corpus_notice",
                 "oem_mislabeling", "insufficient_evidence_stated"}
    out["E_knowledge_grounding"] = {
        "applicable": kn_app,
        "passed": (not kn_app) or kn_checks.issubset(checks),
    }

    f_app = intent in ("unsupported", "ambiguous") or cat in (
        "missing_input", "invalid_input", "tool_failure", "tool_exception",
        "insufficient_evidence")
    f_passed = True
    if intent == "unsupported":
        f_passed = f_passed and ("can't answer" in txt or "scope" in txt)
    if intent == "ambiguous":
        f_passed = f_passed and "couldn't pin down" in txt
    if cat in ("missing_input", "invalid_input", "tool_failure", "tool_exception"):
        f_passed = f_passed and ("failed" in txt or "failure" in txt)
    if cat == "insufficient_evidence":
        f_passed = f_passed and "no sufficiently supportive passage" in txt
    out["F_response_reliability"] = {"applicable": f_app, "passed": (not f_app) or f_passed}

    g_app = cat in ("combined", "end_to_end")
    exp_set = set(exp_tools)
    # Tool selection is order-independent.
    tools_ok = set(r["selected_tools"]) == exp_set
    # Each expected tool must have executed successfully (status == "ok").
    # For mock builds the response is constructed directly (no orchestrator
    # trace exists), so execution is confirmed by output presence below rather
    # than by a runtime status map.
    exec_status = r.get("tool_exec_status", {})
    exec_ok = all(exec_status.get(t) == "ok" for t in exp_set) if exec_status else True
    # Required usable output depends on which tools THIS scenario expects.
    out_ok = True
    if "fuel_rate_prediction" in exp_set:
        out_ok = out_ok and "prediction_value_preserved" in checks and "l/h" in txt
    if "prediction_explanation" in exp_set:
        out_ok = out_ok and "shap_value_preserved" in checks and "shap" in txt
    if "knowledge_retrieval" in exp_set:
        out_ok = out_ok and "source_attribution" in checks and "synthetic" in txt
    # This scenario's own expected response requirements and prohibited claims.
    own_ok = r["expected_pass_ok"] and r["requirements_ok"] and r["prohibited_ok"]
    g_passed = tools_ok and exec_ok and out_ok and own_ok
    out["G_end_to_end_reliability"] = {"applicable": g_app, "passed": (not g_app) or g_passed}
    return out


def aggregate_dimensions(results: list[dict]) -> dict:
    dim_keys = ["A_intent_classification", "B_tool_execution",
                "C_prediction_integrity", "D_explanation_integrity",
                "E_knowledge_grounding", "F_response_reliability",
                "G_end_to_end_reliability"]
    dim_out = {}
    for key in dim_keys:
        total = 0
        passed = 0
        failed_ids = []
        cats = set()
        for r in results:
            if r["status"] != "ok" or r["is_negative"]:
                continue
            di = r.get("dimensions", {}).get(key, {})
            if not di.get("applicable"):
                continue
            total += 1
            if di["passed"]:
                passed += 1
            else:
                failed_ids.append(r["id"])
                cats.update(r["failure_categories"])
        dim_out[key] = {
            "label": key.split("_", 1)[1].replace("_", " ").title(),
            "total_applicable": total,
            "passed": passed,
            "pass_rate": round(passed / total, 4) if total else 1.0,
            "failed_ids": sorted(failed_ids),
            "failure_categories": sorted(cats),
        }
    return dim_out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    scenarios = benchmark["scenarios"]

    tools = {
        "fuel_rate_prediction": FuelRatePredictionTool(),
        "knowledge_retrieval": KnowledgeRetrievalTool(),
        "prediction_explanation": PredictionExplanationTool(),
    }
    orch = AgentOrchestrator(tools=tools)
    gen = ResponseGenerator(inputs=FULL_INPUTS)

    results = []
    for sc in scenarios:
        try:
            r = run_scenario(orch, gen, sc)
            r["dimensions"] = dimensions_for_scenario(r)
            results.append(r)
        except Exception as exc:  # noqa: BLE001 - surface scenario errors
            results.append({
                "id": sc["id"], "category": sc["category"],
                "description": sc["description"], "build": sc["build"],
                "status": "error", "pass": False, "is_negative": bool(sc["expect_violations"]),
                "error": f"{type(exc).__name__}: {exc}",
                "failure_categories": ["runner_error"], "dimensions": {},
                "check_summary": {"total": 0, "passed": 0, "failed": 0,
                                  "passed_checks": [], "failed_checks": []},
                "validation_issues": [], "text_preview": "",
            })

    n_pass = sum(1 for r in results if r["status"] == "ok" and r["pass"])
    n_neg = sum(1 for r in results if r["is_negative"])
    n_neg_pass = sum(1 for r in results if r["is_negative"] and r["pass"])
    n_pos_pass = n_pass - n_neg_pass
    n_fail = len(results) - n_pass
    dims = aggregate_dimensions(results)

    (OUT_DIR / "scenario_results.json").write_text(
        json.dumps({"dataset": "M2.4 agent evaluation", "n_scenarios": len(results),
                    "n_negative_controls": n_neg, "results": results}, indent=2),
        encoding="utf-8")

    eval_payload = {
        "dataset": "M2.4 agent evaluation",
        "runner": "scripts/run_m2_agent_evaluation.py",
        "routing_backend": "deterministic",
        "is_llm": False,
        "note": "Deterministic scoring; no LLM judge, no paid API, no subjective quality score. Small manually authored benchmark: no claim of statistical significance or broad real-world reliability.",
        "n_scenarios": len(results),
        "n_positive": len(results) - n_neg,
        "n_negative_controls": n_neg,
        "n_passed_positive": n_pos_pass,
        "n_negative_controls_passed": n_neg_pass,
        "n_failed": n_fail,
        "dimensions": dims,
        "results": results,
        "limitations": [
            "Small manually authored benchmark (49 scenarios) - not statistically significant.",
            "Dimension pass-rates exclude negative guardrail controls (which intentionally should fail validation).",
            "Negative controls prove the validator rejects injected prohibited claims; they are not a measure of normal agent behavior.",
            "Deterministic checks only; subjective quality is not scored.",
        ],
    }
    (OUT_DIR / "evaluation.json").write_text(
        json.dumps(eval_payload, indent=2), encoding="utf-8")

    lines = [
        "# M2.4 — Agent Evaluation Summary",
        "",
        f"- Dataset: `reports/m2/agent_evaluation/benchmark.json` ({len(scenarios)} scenarios)",
        f"- Scenarios: **{len(results)}** | positive **{len(results) - n_neg}** | negative controls **{n_neg}**",
        f"- Positive passed: **{n_pos_pass}** | negative controls behaving as expected: **{n_neg_pass}**",
        f"- Overall failed scenarios: **{n_fail}**",
        f"- Scoring: fully deterministic; no LLM judge, no paid API, no subjective quality score.",
        "",
        "## Per-dimension results (positive scenarios only)",
        "",
        "| Dimension | Passed | Applicable | Pass rate | Failed IDs |",
        "| --- | --- | --- | --- | --- |",
    ]
    for k, d in sorted(dims.items()):
        lines.append(
            f"| {d['label']} ({k}) | {d['passed']} | {d['total_applicable']} | "
            f"{d['pass_rate']:.2%} | {', '.join(d['failed_ids']) or '-'} |"
        )

    lines += ["", "## Failed scenarios (positive)", ""]
    for r in results:
        if r["status"] == "ok" and not r["pass"] and not r["is_negative"]:
            lines.append(
                f"- `{r['id']}` ({r['category']}, {r['build']}): "
                f"failed checks {r['check_summary']['failed_checks']}; "
                f"expected-pass-ok={r['expected_pass_ok']} "
                f"requirements-ok={r['requirements_ok']} "
                f"prohibited-ok={r['prohibited_ok']}"
            )
    lines += ["", "## Negative controls (must trigger their expected violation)", ""]
    for r in results:
        if r["is_negative"] and r["status"] == "ok":
            flag = "OK" if r["pass"] else "FAIL"
            lines.append(
                f"- `{r['id']}` ({r['category']}) [{flag}] "
                f"violations-ok={r['expected_violations_ok']}"
            )
    lines += ["", "## Category coverage", ""]
    seen = {}
    for r in results:
        seen.setdefault(r["category"], []).append(r["id"])
    for cat, ids in sorted(seen.items()):
        lines.append(f"- {cat}: {', '.join(ids)}")
    lines += ["", "## Caveats", "",
              "- Small manually authored benchmark: do not interpret as statistical significance or broad real-world reliability.",
              "- Deterministic checks only; subjective quality is not scored.",
              "- Negative controls exercise the validator; they are excluded from dimension pass-rates."]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {OUT_DIR}")
    print(f"Scenarios: {len(results)} | positive {len(results) - n_neg} | "
          f"negative controls {n_neg} | failed {n_fail}")
    for k, d in sorted(dims.items()):
        print(f"  {d['label']}: {d['passed']}/{d['total_applicable']} "
              f"({d['pass_rate']:.2%}) failed={d['failed_ids']}")
    for r in results:
        if r["status"] == "error":
            flag = "ERR"
        elif r["is_negative"]:
            flag = "N-OK" if r["pass"] else "N-FAIL"
        else:
            flag = "PASS" if r["pass"] else "FAIL"
        print(f"  [{flag}] {r['id']} ({r['category']})")


if __name__ == "__main__":
    main()
