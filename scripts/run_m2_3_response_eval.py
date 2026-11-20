#!/usr/bin/env python
"""M2.3 — response-evaluation benchmark runner.

Builds a deterministic response-evaluation dataset (>=20 scenarios), runs the
M2.3 ``ResponseGenerator`` over inputs ``AgentResponse`` objects (real tool
executions plus synthetic/injected cases for the failure modes real tools
cannot produce), applies the validation battery, and writes:

    reports/m2/response_evaluation/benchmark.json
    reports/m2/response_evaluation/evaluation.json
    reports/m2/response_evaluation/summary.md

No LLM judge and no paid API is used: scoring is fully deterministic
(``src.agent.response_validation``). Subjective quality is only reported
through documented ``manual_review`` notes, never as an objective numeric
score.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from src.agent.intents import Intent
from src.agent.orchestrator import AgentOrchestrator, AgentResponse
from src.agent.response_generator import GroundedResponse, ResponseGenerator
from src.agent.response_validation import validate_grounded_response
from src.tools.fuel_rate_prediction import FuelRatePredictionTool
from src.tools.knowledge_retrieval import KnowledgeRetrievalTool
from src.tools.prediction_explanation import PredictionExplanationTool
from tests.test_tools import FULL_INPUTS, TELE_INPUTS

OUT_DIR = Path(__file__).resolve().parent.parent / "reports" / "m2" / "response_evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def synthetic_response(intent, tool_results, selected_tools, router=None, limitations=None):
    r = AgentResponse()
    r.intent = intent
    r.selected_tools = list(selected_tools)
    r.tool_results = tool_results
    r.router = router
    r.limitations = list(limitations or [])
    return r


def grounded_response(**kw):
    g = GroundedResponse(
        intent=kw.get("intent", "prediction"),
        detected_intent=kw.get("intent", "prediction"),
        router_backend=kw.get("router_backend", "deterministic"),
        selected_tools=kw.get("selected_tools", []),
        prediction=kw.get("prediction"),
        explanation=kw.get("explanation"),
        evidence=kw.get("evidence", []),
        evidence_insufficient=kw.get("evidence_insufficient", False),
        tool_failures=kw.get("tool_failures", []),
        limitations=kw.get("limitations", []),
        driver_version=kw.get("driver_version", ""),
    )
    return g


def round_agent_response(orch, query, inputs=None, min_score=None):
    """Run a real query through the orchestrator and return the response."""
    return orch.handle(query, inputs)


# ---------------------------------------------------------------------------
# Scenario table
# ---------------------------------------------------------------------------
def build_scenarios(orch, store, mod_dir):
    scenarios = []

    tool_pred = FuelRatePredictionTool(model_dir=mod_dir)
    tool_expl = PredictionExplanationTool(model_dir=mod_dir)
    tool_retr = KnowledgeRetrievalTool(store=store)

    # ---- real prediction ----
    resp = round_agent_response(orch, "predict fuel consumption", FULL_INPUTS)
    scenarios.append({
        "id": "P01",
        "category": "prediction_only",
        "description": "Real prediction query with full_operational inputs.",
        "build": "real",
        "agent_response": resp,
        "source_prediction": resp.tool_results[0]["result"] if resp.tool_results else None,
        "source_explanation": None,
        "known_doc_ids": [],
        "expect_pass": ["synthetic_disclaimer", "prediction_value_preserved",
                        "causal_language", "oem_mislabeling",
                        "failure_not_presented_as_success"],
        "expect_violations": [],
        "manual_review": "Check readability of the no-jargon driver summary.",
    })

    # ---- telematics-only prediction (real tool output, assembled response) ----
    res = tool_pred.predict(TELE_INPUTS, "telematics_only")
    sres = synthetic_response(
        Intent.PREDICT_FUEL,
        [{"tool": "fuel_rate_prediction", "status": "ok", "result": res}],
        ["fuel_rate_prediction"])
    scenarios.append({
        "id": "P02", "category": "prediction_only",
        "description": "Telematics-only feature set prediction.",
        "build": "synthetic", "agent_response": sres,
        "source_prediction": res, "source_explanation": None,
        "known_doc_ids": [],
        "expect_pass": ["synthetic_disclaimer", "prediction_value_preserved",
                        "causal_language", "oem_mislabeling"],
        "expect_violations": [],
        "manual_review": "Confirm the telematics feature set is clearly labelled.",
    })

    # ---- knowledge-only ----
    resp = round_agent_response(orch, "why does road grade affect fuel consumption on trucks")
    rv = resp.tool_results[0]["result"] if resp.tool_results else None
    docs = [p["document_id"] for p in (rv or {}).get("passages", [])]
    scenarios.append({
        "id": "K01", "category": "knowledge_only",
        "description": "Real knowledge query.",
        "build": "real", "agent_response": resp,
        "source_prediction": None, "source_explanation": None,
        "known_doc_ids": docs,
        "expect_pass": ["source_attribution", "synthetic_corpus_notice",
                        "oem_mislabeling", "insufficient_evidence_stated",
                        "causal_language"],
        "expect_violations": [],
        "manual_review": "Passages are clearly cited with ids and synthetic provenance.",
    })

    # ---- prediction + explanation (real outputs) ----
    res_p = tool_pred.predict(FULL_INPUTS, "full_operational")
    res_e = tool_expl.explain(FULL_INPUTS, "full_operational")
    sres = synthetic_response(
        Intent.COMBINED_PREDICT_EXPLAIN,
        [{"tool": "fuel_rate_prediction", "status": "ok", "result": res_p},
         {"tool": "prediction_explanation", "status": "ok", "result": res_e}],
        ["fuel_rate_prediction", "prediction_explanation"])
    scenarios.append({
        "id": "E01", "category": "prediction_plus_explanation",
        "description": "Real prediction + instance SHAP explanation.",
        "build": "synthetic", "agent_response": sres,
        "source_prediction": res_p, "source_explanation": res_e,
        "known_doc_ids": [],
        "expect_pass": ["prediction_value_preserved", "shap_value_preserved",
                        "synthetic_disclaimer", "causal_language",
                        "oem_mislabeling"],
        "expect_violations": [],
        "manual_review": "Driver version should show estimated rate, top drivers, and limitations.",
    })

    # ---- combined knowledge + prediction (real orchestrator) ----
    resp = round_agent_response(
        orch, "predict fuel consumption and explain why engine load affects it", FULL_INPUTS)
    rv = next((r for r in resp.tool_results if r["tool"] == "knowledge_retrieval"), None)
    docs = [p["document_id"] for p in (rv or {}).get("result", {}).get("passages", [])] if rv else []
    pred_res = next((r for r in resp.tool_results if r["tool"] == "fuel_rate_prediction"), None)
    scenarios.append({
        "id": "C01", "category": "combined_knowledge_prediction",
        "description": "Real combined knowledge + prediction.",
        "build": "real", "agent_response": resp,
        "source_prediction": pred_res["result"] if pred_res else None,
        "source_explanation": None,
        "known_doc_ids": docs,
        "expect_pass": ["source_attribution", "synthetic_corpus_notice",
                        "prediction_value_preserved", "causal_language",
                        "oem_mislabeling"],
        "expect_violations": [],
        "manual_review": "Combined response must keep prediction and evidence clearly separated.",
    })

    # ---- ambiguous ----
    resp = round_agent_response(orch, "banana split")
    scenarios.append({
        "id": "U01", "category": "ambiguous",
        "description": "Ambiguous query (no decisive intent).",
        "build": "real", "agent_response": resp,
        "source_prediction": None, "source_explanation": None,
        "known_doc_ids": [],
        "expect_pass": ["unsupported_not_answered", "causal_language",
                        "oem_mislabeling"],
        "expect_violations": [],
        "manual_review": "Response should ask for clarification, not guess.",
    })

    # ---- unsupported OEM query (real router) ----
    resp = round_agent_response(orch, "what is the official manufacturer torque limit for this truck")
    scenarios.append({
        "id": "U02", "category": "unsupported_oem_spec",
        "description": "Out-of-scope OEM specification request.",
        "build": "real", "agent_response": resp,
        "source_prediction": None, "source_explanation": None,
        "known_doc_ids": [],
        "expect_pass": ["unsupported_not_answered", "causal_language", "oem_mislabeling"],
        "expect_violations": [],
        "manual_review": "Must decline without inventing a torque value.",
    })

    # ---- missing input -> tool failure (real orchestrator, bad inputs) ----
    bad = copy.deepcopy(FULL_INPUTS)
    bad["speed_kph"] = -40.0
    resp = round_agent_response(orch, "predict fuel consumption", bad)
    scenarios.append({
        "id": "F01", "category": "tool_failure",
        "description": "Invalid input causes prediction-tool failure; must be disclosed.",
        "build": "real", "agent_response": resp,
        "source_prediction": None, "source_explanation": None,
        "known_doc_ids": [],
        "expect_pass": ["failure_not_presented_as_success", "causal_language"],
        "expect_violations": [],
        "manual_review": "Failure must be explicit; no predicted value fabricated.",
    })

    # ---- insufficient retrieval evidence (real store, high floor) ----
    tools = {"fuel_rate_prediction": FuelRatePredictionTool(model_dir=mod_dir),
             "knowledge_retrieval": KnowledgeRetrievalTool(store=store, min_score=0.99),
             "prediction_explanation": PredictionExplanationTool(model_dir=mod_dir)}
    orch_hi = AgentOrchestrator(tools=tools)
    resp = round_agent_response(orch_hi, "why does road grade affect fuel consumption")
    scenarios.append({
        "id": "I01", "category": "insufficient_evidence",
        "description": "Real retrieval with a high similarity floor yields insufficient evidence.",
        "build": "real", "agent_response": resp,
        "source_prediction": None, "source_explanation": None,
        "known_doc_ids": [],
        "expect_pass": ["insufficient_evidence_stated", "causal_language", "oem_mislabeling"],
        "expect_violations": [],
        "manual_review": "Response must explicitly say evidence is insufficient.",
    })

    # ---- negative: causal claim (validator must reject) ----
    g = grounded_response(intent="prediction", selected_tools=["prediction_explanation"],
                          explanation={"positive_contributions": [{"feature": "Torque (N·m)",
                                                                  "shap": 20.1, "actual_input_value": 1200.0}]},
                          driver_version="The torque caused the higher fuel rate; reducing torque "
                                        "will reduce fuel consumption by 20 L/h.")
    scenarios.append({
        "id": "N01", "category": "contradictory_or_prohibited",
        "description": "Hypothetical bad response claiming a causal effect.",
        "build": "synthetic", "agent_response": g,
        "source_prediction": None,
        "source_explanation": {"most_influential_features": {},
                               "predicted_fuel_rate_lph": 52.5773},
        "known_doc_ids": [],
        "expect_pass": [],
        "expect_violations": ["causal_language"],
        "manual_review": "Validation must flag the causal claim.",
    })

    # ---- negative: OEM mislabeling ----
    g = grounded_response(intent="knowledge", evidence=[{"document_id": "doc_001", "title": "T",
                                                         "score": 0.72}],
                          driver_version="Per the manufacturer specification in doc_001, grade raises fuel use.")
    scenarios.append({
        "id": "N02", "category": "contradictory_or_prohibited",
        "description": "Hypothetical response presenting corpus as OEM documentation.",
        "build": "synthetic", "agent_response": g,
        "source_prediction": None, "source_explanation": None,
        "known_doc_ids": ["doc_001"],
        "expect_pass": [],
        "expect_violations": ["oem_mislabeling"],
        "manual_review": "Validation must flag OEM framing of synthetic corpus.",
    })

    # ---- negative: missing synthetic disclaimer ----
    g = grounded_response(intent="prediction", selected_tools=["fuel_rate_prediction"],
                          prediction={"predicted_fuel_rate_lph": 52.5773, "feature_set": "full_operational"},
                          driver_version="Estimated fuel rate: 52.5773 L/h.")
    scenarios.append({
        "id": "N03", "category": "contradictory_or_prohibited",
        "description": "Prediction without the synthetic-trained-model disclaimer.",
        "build": "synthetic", "agent_response": g,
        "source_prediction": {"predicted_fuel_rate_lph": 52.5773},
        "source_explanation": None, "known_doc_ids": [],
        "expect_pass": [],
        "expect_violations": ["synthetic_disclaimer"],
        "manual_review": "Validation must detect the missing disclaimer.",
    })

    # ---- negative: fabricated source ----
    g = grounded_response(intent="knowledge", evidence=[{"document_id": "doc_001", "title": "T",
                                                         "score": 0.72}],
                          driver_version="Per [doc_999], grades raise fuel use.")
    scenarios.append({
        "id": "N04", "category": "contradictory_or_prohibited",
        "description": "Response cites a source id not present in retrieved evidence.",
        "build": "synthetic", "agent_response": g,
        "source_prediction": None, "source_explanation": None,
        "known_doc_ids": ["doc_001"],
        "expect_pass": [],
        "expect_violations": ["source_attribution"],
        "manual_review": "Validation must flag the fabricated citation.",
    })

    # ---- negative: unsupported fabricates spec ----
    g = grounded_response(intent="unsupported", driver_version="The official torque limit is 89 L/h.")
    scenarios.append({
        "id": "N05", "category": "contradictory_or_prohibited",
        "description": "Unsupported query answered with a fabricated specification.",
        "build": "synthetic", "agent_response": g,
        "source_prediction": None, "source_explanation": None,
        "known_doc_ids": [],
        "expect_pass": [],
        "expect_violations": ["unsupported_not_answered"],
        "manual_review": "Validation must flag fabricated numbers under an unsupported query.",
    })

    # ---- negative: insufficient evidence not stated ----
    g = grounded_response(intent="knowledge", evidence=[], evidence_insufficient=True,
                          driver_version="Here is the answer: grades always double fuel use.")
    scenarios.append({
        "id": "N06", "category": "contradictory_or_prohibited",
        "description": "Insufficient evidence but the response answers anyway.",
        "build": "synthetic", "agent_response": g,
        "source_prediction": None, "source_explanation": None,
        "known_doc_ids": [],
        "expect_pass": [],
        "expect_violations": ["insufficient_evidence_stated"],
        "manual_review": "Validation must require explicit insufficiency disclosure.",
    })

    # ---- negative: tool failure presented as success ----
    g = grounded_response(intent="prediction", selected_tools=["fuel_rate_prediction"],
                          tool_failures=["fuel_rate_prediction"],
                          prediction={"predicted_fuel_rate_lph": 52.5773},
                          driver_version="Estimated fuel rate: 52.5773 L/h.")
    scenarios.append({
        "id": "N07", "category": "contradictory_or_prohibited",
        "description": "A failed tool's value is presented as a successful prediction.",
        "build": "synthetic", "agent_response": g,
        "source_prediction": None, "source_explanation": None,
        "known_doc_ids": [],
        "expect_pass": [],
        "expect_violations": ["failure_not_presented_as_success"],
        "manual_review": "Validation must flag failure presented as success.",
    })

    # ---- negative: prediction value dropped ----
    g = grounded_response(intent="prediction", selected_tools=["fuel_rate_prediction"],
                          prediction={"predicted_fuel_rate_lph": 52.5773},
                          driver_version="We estimated fuel consumption but hid the number.")
    scenarios.append({
        "id": "N08", "category": "contradictory_or_prohibited",
        "description": "Rendered response omits the predicted value.",
        "build": "synthetic", "agent_response": g,
        "source_prediction": {"predicted_fuel_rate_lph": 52.5773},
        "source_explanation": None, "known_doc_ids": [],
        "expect_pass": [],
        "expect_violations": ["prediction_value_preserved"],
        "manual_review": "Validation must flag the missing value.",
    })

    # ---- negative: shap value dropped ----
    g = grounded_response(intent="prediction", selected_tools=["prediction_explanation"],
                          explanation={"positive_contributions": [{"feature": "Torque (N·m)",
                                                                  "shap": 20.1063, "actual_input_value": 1200.0}]},
                          driver_version="The explanation omitted the top SHAP magnitude.")
    scenarios.append({
        "id": "N09", "category": "contradictory_or_prohibited",
        "description": "Explanation rendered without the top SHAP contribution.",
        "build": "synthetic", "agent_response": g,
        "source_prediction": None,
        "source_explanation": {"most_influential_features":
                               {"top_positive_shap": [{"shap": 20.1063}]}},
        "known_doc_ids": [],
        "expect_pass": [],
        "expect_violations": ["shap_value_preserved"],
        "manual_review": "Validation must flag the dropped SHAP value.",
    })

    # ---- positive: contradictory retrieved passages are still handled with provenance ----
    g = grounded_response(intent="knowledge",
                          evidence=[
                              {"document_id": "doc_001", "title": "A", "topic": "x",
                               "score": 0.72, "source_reference": "synthetic", "content_excerpt": "Grades raise fuel."},
                              {"document_id": "doc_002", "title": "B", "topic": "y",
                               "score": 0.60, "source_reference": "synthetic", "content_excerpt": "Grades lower fuel."},
                          ],
                          driver_version="Relevant knowledge (retrieved passages):\n"
                                        "  - [doc_001] A (similarity 0.720).\n"
                                        "  - [doc_002] B (similarity 0.600).\n"
                                        "Retrieved passages are SYNTHETIC educational material (not OEM or "
                                        "manufacturer documentation). Retrieval similarity is not proof of factual correctness.")
    scenarios.append({
        "id": "C02", "category": "contradictory_or_prohibited",
        "description": "Contradictory retrieved passages — presented with provenance, not resolved by invention.",
        "build": "synthetic", "agent_response": g,
        "source_prediction": None, "source_explanation": None,
        "known_doc_ids": ["doc_001", "doc_002"],
        "expect_pass": ["source_attribution", "synthetic_corpus_notice", "oem_mislabeling"],
        "expect_violations": [],
        "manual_review": "Both passages cited with provenance; no fabricated resolution.",
    })

    # ---- missing required input -> tool failure (real orchestrator) ----
    bad_missing = copy.deepcopy(FULL_INPUTS)
    del bad_missing["speed_kph"]
    resp = round_agent_response(orch, "predict fuel consumption", bad_missing)
    scenarios.append({
        "id": "M01", "category": "tool_failure",
        "description": "Missing required input field causes prediction-tool failure; must be disclosed.",
        "build": "real", "agent_response": resp,
        "source_prediction": None, "source_explanation": None,
        "known_doc_ids": [],
        "expect_pass": ["failure_not_presented_as_success", "causal_language",
                        "oem_mislabeling"],
        "expect_violations": [],
        "manual_review": "Failure must be explicit; no predicted value fabricated from missing data.",
    })

    # ---- telematics-only prediction + explanation (real tool outputs) ----
    res_p2 = tool_pred.predict(TELE_INPUTS, "telematics_only")
    res_e2 = tool_expl.explain(TELE_INPUTS, "telematics_only")
    sres2 = synthetic_response(
        Intent.COMBINED_PREDICT_EXPLAIN,
        [{"tool": "fuel_rate_prediction", "status": "ok", "result": res_p2},
         {"tool": "prediction_explanation", "status": "ok", "result": res_e2}],
        ["fuel_rate_prediction", "prediction_explanation"])
    scenarios.append({
        "id": "E02", "category": "prediction_plus_explanation",
        "description": "Telematics-only prediction + instance SHAP explanation (real tool outputs).",
        "build": "synthetic", "agent_response": sres2,
        "source_prediction": res_p2, "source_explanation": res_e2,
        "known_doc_ids": [],
        "expect_pass": ["prediction_value_preserved", "shap_value_preserved",
                        "synthetic_disclaimer", "causal_language", "oem_mislabeling"],
        "expect_violations": [],
        "manual_review": "Telematics feature set clearly labelled; SHAP caveat present.",
    })

    return scenarios


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def run_evaluation(scenarios):
    gen = ResponseGenerator(inputs=FULL_INPUTS)
    results = []
    for sc in scenarios:
        resp = sc["agent_response"]
        try:
            if isinstance(resp, GroundedResponse):
                # Already-rendered response (negative / C scenarios): validate
                # it directly, so the injected text is what actually gets
                # checked. Running the generator here would regenerate a clean
                # answer and defeat the purpose of the scenario.
                grounded = resp
            else:
                grounded = gen.generate(resp)
        except Exception as exc:  # surfaces generator failure
            results.append({
                "id": sc["id"], "category": sc["category"],
                "status": "error", "error": f"{type(exc).__name__}: {exc}",
                "checks": [], "all_passed": False, "expected_pass": sc["expect_pass"],
                "expected_violations": sc["expect_violations"],
                "manual_review": sc["manual_review"],
            })
            continue

        issues = validate_grounded_response(
            grounded,
            source_prediction=sc["source_prediction"],
            source_explanation=sc["source_explanation"],
            known_doc_ids=sc["known_doc_ids"],
        )
        by_id = {i.check_id: i for i in issues}

        pass_issues = {}
        for cid in sc["expect_pass"]:
            pass_issues[cid] = by_id.get(cid)
        viol_note = {}
        for cid in sc["expect_violations"]:
            viol_note[cid] = by_id.get(cid)

        # verdict on expected-pass checks
        pass_ok = all(p is not None and p.passed for p in pass_issues.values())
        viol_ok = all(v is not None and not v.passed for v in viol_note.values())
        all_passed = pass_ok and viol_ok

        results.append({
            "id": sc["id"], "category": sc["category"],
            "description": sc["description"], "build": sc["build"],
            "status": "ok",
            "all_passed": all_passed,
            "expected_pass": {cid: bool(p and p.passed) for cid, p in pass_issues.items()},
            "expected_violations": {cid: bool(v and not v.passed) for cid, v in viol_note.items()},
            "check_summary": {
                "total": len(issues),
                "passed": sum(1 for i in issues if i.passed),
                "failed": sum(1 for i in issues if not i.passed),
                "failed_checks": sorted({i.check_id for i in issues if not i.passed}),
            },
            "driver_postview_preview": grounded.driver_version[:240],
            "manual_review": sc["manual_review"],
            "checks": [{"check_id": i.check_id, "passed": i.passed, "detail": i.detail}
                       for i in issues],
        })
    return results


def main():
    tools = {"fuel_rate_prediction": FuelRatePredictionTool(),
             "knowledge_retrieval": KnowledgeRetrievalTool(),
             "prediction_explanation": PredictionExplanationTool()}
    orch = AgentOrchestrator(tools=tools)
    store = tools["knowledge_retrieval"]._ensure_store()
    mod_dir = FuelRatePredictionTool().model_dir

    scenarios = build_scenarios(orch, store, mod_dir)
    results = run_evaluation(scenarios)

    n_ok = sum(1 for r in results if r["status"] == "ok" and r["all_passed"])
    n_fail = len(results) - n_ok

    benchmark_payload = {
        "dataset": "M2.3 response-evaluation",
        "generated_by": "scripts/run_m2_3_response_eval.py",
        "routing_backend": "deterministic",
        "is_llm": False,
        "note": "Deterministic checks only; subjective quality is recorded as document review notes, not numeric scores.",
        "n_scenarios": len(scenarios),
        "scenarios": [
            {"id": s["id"], "category": s["category"], "description": s["description"],
             "build": s["build"], "expect_pass": s["expect_pass"],
             "expect_violations": s["expect_violations"],
             "manual_review": s["manual_review"]}
            for s in scenarios
        ],
    }
    (OUT_DIR / "benchmark.json").write_text(
        json.dumps(benchmark_payload, indent=2), encoding="utf-8")

    evaluation_payload = {
        "dataset": "M2.3 response-evaluation",
        "n_scenarios": len(results),
        "n_passed": n_ok,
        "n_failed": n_fail,
        "all_passed": n_fail == 0,
        "results": results,
        "limitations": [
            "These checks are deterministic heuristics, not an LLM judge and not proof of human-level quality.",
            "Subjective readability/completeness is only noted manually, never scored numerically.",
            "Negative scenarios (N01-N09) deliberately inject bad responses to prove the validator rejects them.",
        ],
    }
    (OUT_DIR / "evaluation.json").write_text(
        json.dumps(evaluation_payload, indent=2), encoding="utf-8")

    # summary.md
    lines = [
        "# M2.3 — Response Evaluation Summary",
        "",
        f"- Dataset: `reports/m2/response_evaluation/benchmark.json`",
        f"- Scenarios: **{len(results)}**",
        f"- Passed: **{n_ok}**, Failed: **{n_fail}**",
        f"- Scoring: fully deterministic (`src.agent.response_validation`); no LLM judge, no paid API.",
        "",
        "## Categories covered",
        "",
    ]
    seen_cat = {}
    for r in results:
        seen_cat.setdefault(r["category"], []).append(r["id"])
    for cat, ids in sorted(seen_cat.items()):
        lines.append(f"- {cat}: {', '.join(ids)}")
    lines += ["", "## Positive (expected-pass) check coverage", ""]
    for r in results:
        if r["status"] == "ok" and r["expected_pass"]:
            lines.append(f"- `{r['id']}` ({r['category']}): " + ", ".join(r["expected_pass"]))
    lines += ["", "## Negative (expected-violation) coverage", ""]
    for r in results:
        if r["status"] == "ok" and r["expected_violations"]:
            lines.append(f"- `{r['id']}` ({r['category']}): " + ", ".join(r["expected_violations"]))
    lines += ["", "## Manual review notes (documented, not scored)", ""]
    for r in results:
        lines.append(f"- `{r['id']}`: {r['manual_review']}")
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {OUT_DIR}")
    print(f"Scenarios: {len(results)} | passed {n_ok} | failed {n_fail}")
    for r in results:
        flag = "PASS" if r["status"] == "ok" and r["all_passed"] else "FAIL"
        print(f"  [{flag}] {r['id']} ({r['category']})")


if __name__ == "__main__":
    main()
