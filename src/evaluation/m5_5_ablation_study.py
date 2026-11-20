"""M5.5 — comparative experiments & ablation study (deterministic).

This milestone runs CONTROLLED component ablations of the existing agentic
driver-feedback system by removing one component at a time from the real
``DriverConversationService`` and scoring the resulting driver-facing feedback
with the frozen M5.4 rubric (unchanged).

Variant A — full pipeline (baseline, == M5.4).
Variant B — without session context (each turn processed on a fresh session).
Variant C — without retrieval grounding (real service, empty-retrieval stub).
Variant D — without SHAP explanation (real service, no-op explanation stub).
Variant E — without response validation (OFFLINE shadow analysis only; the
            unvalidated text is never delivered; we measure what the validator
            detects on the raw generated response).
Variant F — feature-set comparison (full_operational vs telematics_only on
            compatible inputs; M5.3 model/tool metrics are reused, not
            recomputed).

Non-duplication / honesty guarantees
  * Baseline execution is real (the actual service + saved M1.2 tools).
  * Adaptors are evaluation-only stubs that REMOVE a component; they never
    add, fabricate or improve content, and never weaken the rubric.
  * Scores are computed with the frozen M5.4 ``evaluate_turn``; no rubric
    criterion is modified.
  * All numbers are absolute percentage-point differences. No claim of
    statistical significance is made (no justified test is available).
  * Safety-critical failures are reported separately and never hidden by an
    average.
  * This is an engineering study; results do not transfer to real-world
    trucks (all telemetry/targets are synthetic).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.agent.driver_conversation import (
    DriverConversationResponse,
    DriverConversationService,
)
from src.agent.driver_session import DriverSession
from src.evaluation.m5_4_driver_feedback import (
    CriterionResult,
    DIMENSIONS,
    FULL_INPUTS,
    PARTIAL_INPUTS,
    _SCENARIOS,
    evaluate_turn,
)
from src.tools.fuel_rate_prediction import FuelRatePredictionTool
from src.tools.knowledge_retrieval import KnowledgeRetrievalTool
from src.tools.prediction_explanation import PredictionExplanationTool

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "m5" / "m5_5"
DEFAULT_FEATURE_SET = "full_operational"
TELEMATICS_FEATURE_SET = "telematics_only"
SEED = 20250801

# Feature names used to build compatible inputs for the feature-set variant.
_TELEMATICS_FEATURES = [
    "vehicle_class",
    "gross_vehicle_weight_kg",
    "payload_kg",
    "vehicle_mass_kg",
    "speed_kph",
    "acceleration_mps2",
    "road_grade_pct",
    "ignition_status",
    "coolant_temp_c",
    "intake_air_temp_c",
    "ambient_temp_c",
]


# ---------------------------------------------------------------------------
# Evaluation-only adapter tools (remove a component; never add/fabricate)
# ---------------------------------------------------------------------------

class _NoOpExplanationTool:
    """Evaluation-only stub: removes the SHAP explanation component.

    ``explain`` returns ``None``, so the generator sees no explanation result
    and renders prediction-only feedback (no explanation block, no tool
    failure). Used ONLY for Variant D.
    """

    def explain(self, inputs, feature_set=None, top_features=None):  # noqa: ARG002
        return None


class _EmptyRetrievalTool:
    """Evaluation-only stub: removes retrieval grounding.

    ``retrieve`` returns zero passages with evidence flagged insufficient, so
    the generator emits the honest "no sufficiently supportive passage" path
    instead of grounding a response in retrieved context. Used ONLY for
    Variant C. No replacement evidence is fabricated.
    """

    def retrieve(self, query, k=None, min_score=None):  # noqa: ARG002
        return {
            "status": "ok",
            "query": query,
            "n_results": 0,
            "passages": [],
            "evidence_sufficiency": {
                "insufficient": True,
                "reason": "no passage retrieved",
                "min_score_floor": 0.45,
            },
            "provenance": {
                "corpus_synthetic": True,
                "note": "all retrieved passages are synthetic educational material",
            },
        }


# ---------------------------------------------------------------------------
# Shared scenario source (frozen M5.4 scenario set, unchanged)
# ---------------------------------------------------------------------------

def shared_scenarios() -> list[dict]:
    return [dict(sc) for sc in _SCENARIOS]


def telematics_compatible_inputs() -> dict:
    """FULL_INPUTS restricted to the telematics_only feature set (fair inputs)."""
    return {k: v for k, v in FULL_INPUTS.items() if k in _TELEMATICS_FEATURES}


# ---------------------------------------------------------------------------
# Per-turn runner + aggregation (reuses frozen M5.4 rubric)
# ---------------------------------------------------------------------------

@dataclass
class TurnRecord:
    scenario: str
    turn: int
    message: str
    expected_intent: str
    intent: str | None = None
    delivered: bool = False
    feedback: str = ""
    criteria: list[CriterionResult] = field(default_factory=list)
    note: str = ""

    @property
    def applicable_criteria(self) -> list[CriterionResult]:
        return [c for c in self.criteria if c.applicable]

    @property
    def passed_criteria(self) -> list[CriterionResult]:
        return [c for c in self.applicable_criteria if c.passed]

    @property
    def safety_critical(self) -> list[CriterionResult]:
        return [c for c in self.applicable_criteria if not c.passed and c.severity == "safety"]


def run_scenario_variant(
    service: DriverConversationService,
    sc: dict,
    *,
    carry_session: bool = True,
) -> dict[str, Any]:
    """Run one scenario through a variant of the service.

    ``carry_session=False`` processes every turn on a fresh ``DriverSession``
    with no history (Variant B). Otherwise the session carries context as in
    the real pipeline.
    """
    session = DriverSession(feature_set=service.default_feature_set)
    turns: list[TurnRecord] = []
    history: list[dict] = []
    negative_control = sc.get("negative_control", False)

    for i, turn_spec in enumerate(sc["turns"], start=1):
        if carry_session:
            resp = service.handle(session, turn_spec["message"], turn_spec.get("inputs"))
            spec = dict(turn_spec)
            spec["negative_control"] = negative_control
            criteria = evaluate_turn(resp, spec, history)
            rec = TurnRecord(
                scenario=sc["name"],
                turn=i,
                message=turn_spec["message"],
                expected_intent=turn_spec["expected_intent"],
                intent=resp.driver_intent,
                delivered=resp.delivered,
                feedback=resp.driver_feedback,
                criteria=criteria,
                note=turn_spec.get("note", ""),
            )
        else:
            fresh = DriverSession(feature_set=service.default_feature_set)
            resp = service.handle(fresh, turn_spec["message"], turn_spec.get("inputs"))
            spec = dict(turn_spec)
            spec["negative_control"] = negative_control
            criteria = evaluate_turn(resp, spec, [])
            rec = TurnRecord(
                scenario=sc["name"],
                turn=i,
                message=turn_spec["message"],
                expected_intent=turn_spec["expected_intent"],
                intent=resp.driver_intent,
                delivered=resp.delivered,
                feedback=resp.driver_feedback,
                criteria=criteria,
                note=turn_spec.get("note", ""),
            )
        turns.append(rec)
        history.append({"feedback": resp.driver_feedback, "intent": resp.driver_intent})

    total_app = sum(len(t.applicable_criteria) for t in turns)
    total_pass = sum(len(t.passed_criteria) for t in turns)
    safety_crit = [c for t in turns for c in t.safety_critical]

    dim_scores: dict[str, dict] = {}
    for dim in DIMENSIONS:
        applicable = [c for t in turns for c in t.applicable_criteria if c.dimension == dim]
        passed = [c for c in applicable if c.passed]
        dim_scores[dim] = {
            "applicable": len(applicable),
            "passed": len(passed),
            "score": (len(passed) / len(applicable)) if applicable else None,
        }

    return {
        "name": sc["name"],
        "type": sc.get("type", "functional"),
        "negative_control": sc.get("negative_control", False),
        "turns": [
            {
                "turn": t.turn,
                "message": t.message,
                "expected_intent": t.expected_intent,
                "intent": t.intent,
                "delivered": t.delivered,
                "feedback": t.feedback,
                "note": t.note,
                "criteria": [
                    {
                        "criterion_id": c.criterion_id,
                        "criterion_name": c.criterion_name,
                        "dimension": c.dimension,
                        "passed": c.passed,
                        "applicable": c.applicable,
                        "severity": c.severity,
                        "evidence": c.evidence,
                    }
                    for c in t.criteria
                ],
            }
            for t in turns
        ],
        "aggregate": {
            "applicable_criteria": total_app,
            "passed_criteria": total_pass,
            "overall_score": (total_pass / total_app) if total_app else None,
        },
        "dimension_scores": dim_scores,
        "safety_critical_failures": [
            {
                "criterion_id": c.criterion_id,
                "criterion_name": c.criterion_name,
                "turn": next(t.turn for t in turns if c in t.criteria),
                "evidence": c.evidence,
            }
            for c in safety_crit
        ],
    }


def aggregate_variant(scenarios: list[dict]) -> dict[str, Any]:
    """Aggregate scenario results into dimension scores + overall."""
    total_app = sum(s["aggregate"]["applicable_criteria"] for s in scenarios)
    total_pass = sum(s["aggregate"]["passed_criteria"] for s in scenarios)
    overall = (total_pass / total_app) if total_app else None

    dim_overall: dict[str, dict] = {}
    for dim in DIMENSIONS:
        applicable = sum(s["dimension_scores"][dim]["applicable"] for s in scenarios)
        passed = sum(s["dimension_scores"][dim]["passed"] for s in scenarios)
        dim_overall[dim] = {
            "applicable": applicable,
            "passed": passed,
            "score": (passed / applicable) if applicable else None,
        }

    safety_critical = [s for s in scenarios if s["safety_critical_failures"]]

    return {
        "evaluation": {
            "dimensions": dim_overall,
            "overall_score": overall,
            "overall_calculation": (
                "pooled across all applicable criteria (passed / applicable), "
                "not an unweighted average of dimension scores; dimensions with "
                "zero applicable criteria are excluded"
            ),
            "applicable_criteria": total_app,
            "passed_criteria": total_pass,
            "scenarios_total": len(scenarios),
            "scenarios_with_safety_critical_failures": len(safety_critical),
        },
        "safety_critical_violations": [
            {"scenario": s["name"], "failures": s["safety_critical_failures"]}
            for s in safety_critical
        ],
    }


# ---------------------------------------------------------------------------
# Variant service factories
# ---------------------------------------------------------------------------

def _base_service() -> DriverConversationService:
    """Full pipeline service (Variant A / B): all real tools + validation."""
    return DriverConversationService(default_feature_set=DEFAULT_FEATURE_SET)


def _no_retrieval_service() -> DriverConversationService:
    """Variant C: real service with retrieval grounding removed (empty stub)."""
    return DriverConversationService(
        default_feature_set=DEFAULT_FEATURE_SET,
        tools={
            "fuel_rate_prediction": FuelRatePredictionTool(),
            "knowledge_retrieval": _EmptyRetrievalTool(),
            "prediction_explanation": PredictionExplanationTool(),
        },
    )


def _no_explanation_service() -> DriverConversationService:
    """Variant D: real service with SHAP explanation removed (no-op stub)."""
    return DriverConversationService(
        default_feature_set=DEFAULT_FEATURE_SET,
        tools={
            "fuel_rate_prediction": FuelRatePredictionTool(),
            "knowledge_retrieval": KnowledgeRetrievalTool(),
            "prediction_explanation": _NoOpExplanationTool(),
        },
    )


def _telematics_service() -> DriverConversationService:
    """Variant F telematics_only side: real tools on the telematics feature set."""
    return DriverConversationService(default_feature_set=TELEMATICS_FEATURE_SET)


_VARIANT_SERVICES: dict[str, Callable[[], DriverConversationService]] = {
    "A_full_pipeline": _base_service,
    "C_without_retrieval": _no_retrieval_service,
    "D_without_explanation": _no_explanation_service,
    "F_telematics_only": _telematics_service,
}


# ---------------------------------------------------------------------------
# Scenario-level runner
# ---------------------------------------------------------------------------

def _transform_inputs(sc: dict, feature_set: str) -> dict:
    """Rebuild a scenario's turns with inputs compatible with ``feature_set``.

    For the telematics_only set we restrict the input dicts to the telematics
    features so both feature sets see fair, compatible inputs on the same
    scenarios.
    """
    if feature_set != TELEMATICS_FEATURE_SET:
        return sc
    out = dict(sc)
    turns = []
    for t in sc.get("turns", []):
        t2 = dict(t)
        if isinstance(t.get("inputs"), dict):
            t2["inputs"] = {
                k: v for k, v in t["inputs"].items()
                if k in _TELEMATICS_FEATURES
            }
        turns.append(t2)
    out["turns"] = turns
    return out


def run_scenario_for_variant(
    variant: str,
    sc: dict,
    *,
    carry_session: bool = True,
    feature_set: str = DEFAULT_FEATURE_SET,
) -> dict[str, Any]:
    """Run one scenario through the given variant's service."""
    service_factory = _VARIANT_SERVICES.get(variant)
    if service_factory is None:
        raise ValueError(f"unknown variant {variant!r}")
    service = service_factory()
    sc = _transform_inputs(sc, feature_set)
    return run_scenario_variant(service, sc, carry_session=carry_session)


# ---------------------------------------------------------------------------
# Whole-variant runner
# ---------------------------------------------------------------------------

def run_variant(
    variant: str,
    scenarios: list[dict] | None = None,
    *,
    carry_session: bool = True,
    feature_set: str = DEFAULT_FEATURE_SET,
) -> dict[str, Any]:
    """Run a full variant over the shared scenario set and aggregate."""
    scs = scenarios if scenarios is not None else shared_scenarios()
    results = [
        run_scenario_for_variant(variant, sc, carry_session=carry_session,
                                 feature_set=feature_set)
        for sc in scs
    ]
    agg = aggregate_variant(results)
    return {
        "variant": variant,
        "feature_set": feature_set,
        "carry_session": carry_session,
        "scenarios": results,
        **agg,
    }


# ---------------------------------------------------------------------------
# Variant E — offline response-validation shadow analysis
# ---------------------------------------------------------------------------

def _validation_shadow(service: DriverConversationService, sc: dict) -> dict[str, Any]:
    """Offline shadow analysis for Variant E.

    Runs the real full-pipeline service (so real tools execute) and inspects
    the validator's findings on the RAW generated response. The unvalidated
    text is never delivered; we only measure what the validator detects.
    """
    session = DriverSession(feature_set=service.default_feature_set)
    turns = []
    for i, turn_spec in enumerate(sc.get("turns", []), start=1):
        resp = service.handle(session, turn_spec["message"], turn_spec.get("inputs"))
        raw = (resp.grounded.driver_version if resp.grounded else "") or ""
        issues = resp.validation_issues or []
        failed = [x for x in issues if not x.passed]
        error_failed = [x for x in failed if x.severity == "error"]
        turns.append({
            "turn": i,
            "message": turn_spec["message"],
            "intent": resp.driver_intent,
            "expected_intent": turn_spec["expected_intent"],
            "delivered": resp.delivered,
            "validation_blocked": resp.validation_blocked,
            "raw_feedback_len": len(raw),
            "presented_feedback_len": len(resp.driver_feedback),
            "total_validation_checks": len(issues),
            "failed_checks": len(failed),
            "error_severity_failed": len(error_failed),
            "failed_check_ids": sorted({x.check_id for x in failed}),
        })
    return {
        "scenario": sc["name"],
        "turns": turns,
        "totals": {
            "turns": len(turns),
            "delivered": sum(1 for t in turns if t["delivered"]),
            "blocked": sum(1 for t in turns if t["validation_blocked"]),
            "failed_checks": sum(t["failed_checks"] for t in turns),
            "error_severity_failed": sum(t["error_severity_failed"] for t in turns),
        },
    }


def run_validation_shadow(scenarios: list[dict] | None = None) -> dict[str, Any]:
    """Run Variant E over the shared set using the real full-pipeline service."""
    service = _base_service()
    scs = scenarios if scenarios is not None else shared_scenarios()
    per_scenario = [_validation_shadow(service, sc) for sc in scs]
    totals = {
        "scenarios": len(per_scenario),
        "turns": sum(s["totals"]["turns"] for s in per_scenario),
        "delivered": sum(s["totals"]["delivered"] for s in per_scenario),
        "blocked": sum(s["totals"]["blocked"] for s in per_scenario),
        "failed_checks": sum(s["totals"]["failed_checks"] for s in per_scenario),
        "error_severity_failed": sum(
            s["totals"]["error_severity_failed"] for s in per_scenario),
    }
    return {
        "variant": "E_without_validation_shadow",
        "unvalidated_never_delivered": True,
        "scenarios": per_scenario,
        "totals": totals,
    }


# ---------------------------------------------------------------------------
# Variant F — feature-set comparison
# ---------------------------------------------------------------------------

def _explanation_text(resp: DriverConversationResponse) -> str:
    tech = resp.technical_report or {}
    expl = tech.get("explanation") or {}
    if not expl:
        return ""
    pos = (expl.get("positive_contributions") or [])[:3]
    neg = (expl.get("negative_contributions") or [])[:3]
    parts = []
    for it in pos:
        parts.append(f"+{it.get('feature')}({it.get('shap')})")
    for it in neg:
        parts.append(f"{it.get('feature')}({it.get('shap')})")
    return ", ".join(parts)


def _collect_explanations(service: DriverConversationService, feature_set: str):
    out = []
    for sc in shared_scenarios():
        for i, t in enumerate(sc.get("turns", []), start=1):
            if t.get("expected_intent") in ("explain_prediction", "predict_fuel"):
                s = DriverSession(feature_set=feature_set)
                inputs = t.get("inputs")
                if isinstance(inputs, dict) and feature_set == TELEMATICS_FEATURE_SET:
                    inputs = {k: v for k, v in inputs.items()
                              if k in _TELEMATICS_FEATURES}
                resp = service.handle(s, t["message"], inputs)
                out.append({
                    "scenario": sc["name"],
                    "turn": i,
                    "intent": resp.driver_intent,
                    "expected": t["expected_intent"],
                    "delivered": resp.delivered,
                    "explanation": _explanation_text(resp),
                })
    return out


def compare_feature_sets() -> dict[str, Any]:
    """Variant F: full_operational vs telematics_only on compatible inputs.

    Model/tool-level facts are reused from M5.3 (not recomputed); this
    comparison measures the resulting driver-facing feedback/explanation.
    """
    full = run_variant("A_full_pipeline", feature_set=DEFAULT_FEATURE_SET)
    tele = run_variant("F_telematics_only", feature_set=TELEMATICS_FEATURE_SET)
    full_expl = _collect_explanations(_base_service(), DEFAULT_FEATURE_SET)
    tele_expl = _collect_explanations(_telematics_service(), TELEMATICS_FEATURE_SET)
    return {
        "variant": "F_feature_set_comparison",
        "model_tool_facts_reused": {
            "source": "M5.3 summary (tool-layer determinism/agreement/reconstruction)",
            "note": (
                "Both feature sets' tools already pass M5.3 reliability checks; "
                "no model-level accuracy is recomputed here."
            ),
        },
        "full_operational": {
            "overall": full["evaluation"]["overall_score"],
            "dimensions": full["evaluation"]["dimensions"],
            "scenarios_with_safety_critical_failures": full["evaluation"][
                "scenarios_with_safety_critical_failures"],
        },
        "telematics_only": {
            "overall": tele["evaluation"]["overall_score"],
            "dimensions": tele["evaluation"]["dimensions"],
            "scenarios_with_safety_critical_failures": tele["evaluation"][
                "scenarios_with_safety_critical_failures"],
        },
        "explanation_driver_facing": {
            "full_operational": full_expl,
            "telematics_only": tele_expl,
        },
    }


# ---------------------------------------------------------------------------
# Full study + reporting
# ---------------------------------------------------------------------------

def run_full_study() -> dict[str, Any]:
    """Run all variants over the shared deterministic scenario set."""
    scs = shared_scenarios()
    baseline = run_variant("A_full_pipeline", scs, feature_set=DEFAULT_FEATURE_SET)
    no_session = run_variant(
        "A_full_pipeline", scs, carry_session=False,
        feature_set=DEFAULT_FEATURE_SET)
    no_retrieval = run_variant(
        "C_without_retrieval", scs, feature_set=DEFAULT_FEATURE_SET)
    no_explanation = run_variant(
        "D_without_explanation", scs, feature_set=DEFAULT_FEATURE_SET)
    shadow = run_validation_shadow(scs)
    feature_sets = compare_feature_sets()

    return {
        "experiment": "M5.5_ablation_study",
        "milestone": "M5.5",
        "seed": SEED,
        "baseline_variant": "A_full_pipeline",
        "baseline_reference": "M5.4 corrected & frozen (185/201 applicable)",
        "variants": {
            "A_full_pipeline": {
                "description": "full pipeline: session + router + predict + explain + retrieval + generate + validate",
                "results": _variant_summary(baseline),
                "scenarios": baseline["scenarios"],
            },
            "B_without_session_context": {
                "description": "same service, each turn processed on a fresh session (no carried state)",
                "results": _variant_summary(no_session),
                "scenarios": no_session["scenarios"],
            },
            "C_without_retrieval_grounding": {
                "description": "real service with retrieval grounding removed (empty retrieval stub)",
                "results": _variant_summary(no_retrieval),
                "scenarios": no_retrieval["scenarios"],
            },
            "D_without_shap_explanation": {
                "description": "real service with SHAP explanation removed (no-op explanation stub)",
                "results": _variant_summary(no_explanation),
                "scenarios": no_explanation["scenarios"],
            },
            "E_without_response_validation": {
                "description": "offline shadow only; unvalidated text never delivered",
                "shadow": shadow["scenarios"],
                "totals": shadow["totals"],
                "unvalidated_never_delivered": True,
            },
            "F_feature_set_comparison": {
                "description": "full_operational vs telematics_only on compatible inputs (M5.3 facts reused)",
                "full_operational": feature_sets["full_operational"],
                "telematics_only": feature_sets["telematics_only"],
                "explanation_driver_facing": feature_sets["explanation_driver_facing"],
                "model_tool_facts_reused": feature_sets["model_tool_facts_reused"],
            },
        },
        "differences": {
            "B_vs_A": _diff_from_baseline(baseline, no_session),
            "C_vs_A": _diff_from_baseline(baseline, no_retrieval),
            "D_vs_A": _diff_from_baseline(baseline, no_explanation),
        },
        "limitations": [
            "This is an engineering ablation study using the frozen M5.4 rubric; "
            "it does not measure real driver satisfaction or comprehension.",
            "All telemetry and prediction targets are synthetic; results are not "
            "real-world truck validation.",
            "Score differences are reported as absolute percentage-point deltas "
            "and are not claimed to be statistically significant (no justified "
            "statistical test is available on this deterministic rubric).",
            "Variant E is an offline shadow; unvalidated responses are never "
            "delivered to a user and production safety behavior is untouched.",
            "Variant F compares feature sets only on compatible inputs and reuses "
            "M5.3 model/tool facts; no model-level accuracy is recomputed and no "
            "superiority is claimed from incomparable input sets.",
        ],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _variant_summary(res: dict) -> dict:
    return {
        "overall_score": res["evaluation"]["overall_score"],
        "applicable_criteria": res["evaluation"]["applicable_criteria"],
        "passed_criteria": res["evaluation"]["passed_criteria"],
        "dimensions": res["evaluation"]["dimensions"],
        "scenarios_total": res["evaluation"]["scenarios_total"],
        "scenarios_with_safety_critical_failures": res["evaluation"][
            "scenarios_with_safety_critical_failures"],
        "safety_critical_violations": res["safety_critical_violations"],
    }


def _diff_from_baseline(baseline: dict, variant: dict) -> dict:
    base = baseline["evaluation"]
    var = variant["evaluation"]
    dims = {}
    for dim in DIMENSIONS:
        b = base["dimensions"][dim]["score"]
        v = var["dimensions"][dim]["score"]
        dims[dim] = {
            "baseline": b,
            "variant": v,
            "delta_pp": (v - b) * 100 if (b is not None and v is not None) else None,
        }
    b_o, v_o = base["overall_score"], var["overall_score"]
    return {
        "overall": {
            "baseline": b_o,
            "variant": v_o,
            "delta_pp": (v_o - b_o) * 100 if (b_o is not None and v_o is not None) else None,
        },
        "dimensions": dims,
        "safety_critical": {
            "baseline": base["scenarios_with_safety_critical_failures"],
            "variant": var["scenarios_with_safety_critical_failures"],
        },
    }


def build_report(report_dir: Path = DEFAULT_REPORT_DIR) -> dict[str, Any]:
    """Run the full study and write evaluation.json + summary.md."""
    report = run_full_study()
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "evaluation.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    _write_markdown(report, report_dir)
    return report


def _write_markdown(report: dict[str, Any], report_dir: Path) -> None:
    lines: list[str] = []
    add = lines.append
    add("# M5.5 Comparative Experiments & Ablation Study\n")
    add(f"**Milestone:** {report['milestone']}")
    add(f"**Baseline variant:** {report['baseline_variant']} (full pipeline, == M5.4)")
    add(f"**Seed:** {report['seed']}")
    add("**Caveat:** engineering ablation study; synthetic data; not human-subject, not real-world validation\n")

    add("## Variant overview\n")
    add("| Variant | overall | applicable | passed | safety-critical scenarios |")
    add("| --- | --- | --- | --- | --- |")
    for vid, v in report["variants"].items():
        if "results" not in v:
            if vid == "E_without_response_validation":
                add(f"| {vid} | shadow only | — | — | — |")
            else:
                add(f"| {vid} | see below | — | — | — |")
            continue
        r = v["results"]
        score = f"{r['overall_score']:.3f}" if r["overall_score"] is not None else "n/a"
        add(f"| {vid} | {score} | {r['applicable_criteria']} | {r['passed_criteria']} | "
            f"{r['scenarios_with_safety_critical_failures']} |")
    add("")

    add("## Differences from baseline (absolute percentage points)\n")
    add("| Variant | overall Δpp | relevance | actionability | clarity | evidence | safety | consistency |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for vid, diff in report["differences"].items():
        o = diff["overall"]["delta_pp"]
        o_s = f"{o:+.2f}" if o is not None else "n/a"
        row = [f"{diff['dimensions'][d]['delta_pp']:+.2f}" if diff["dimensions"][d]["delta_pp"] is not None else "n/a"
               for d in DIMENSIONS]
        add(f"| {vid} | {o_s} | " + " | ".join(row) + " |")
    add("")

    add("## Per-variant details\n")
    for vid, v in report["variants"].items():
        add(f"### {vid}\n")
        add(f"- {v['description']}\n")
        if "results" in v:
            r = v["results"]
            score = f"{r['overall_score']:.3f}" if r["overall_score"] is not None else "n/a"
            add(f"- Overall: **{score}** ({r['passed_criteria']}/{r['applicable_criteria']} applicable)")
            add("- Per-dimension:")
            for dim, d in r["dimensions"].items():
                ds = f"{d['score']:.3f}" if d["score"] is not None else "n/a"
                add(f"  - {dim}: {d['passed']}/{d['applicable']} ({ds})")
            add(f"- Safety-critical scenario count: {r['scenarios_with_safety_critical_failures']}")
            if r["safety_critical_violations"]:
                for viol in r["safety_critical_violations"]:
                    add(f"  - **{viol['scenario']}**")
                    for f in viol["failures"]:
                        add(f"    - {f['criterion_id']} (turn {f['turn']}): {f['evidence']}")
            add("")
        elif vid == "E_without_response_validation":
            t = v["totals"]
            add(f"- Turns evaluated: {t['turns']}")
            add(f"- Delivered (validated): {t['delivered']} / blocked: {t['blocked']}")
            add(f"- Validator failed checks in shadow: {t['failed_checks']} "
                f"(error-severity: {t['error_severity_failed']})")
            add(f"- Unvalidated responses are never delivered: {v['unvalidated_never_delivered']}\n")
        elif vid == "F_feature_set_comparison":
            for fs in ("full_operational", "telematics_only"):
                fo = v[fs]["overall"]
                fscore = f"{fo:.3f}" if fo is not None else "n/a"
                add(f"- **{fs}** overall: {fscore}")
                add(f"  - Safety-critical scenarios: {v[fs]['scenarios_with_safety_critical_failures']}")
            add("- Explanation (driver-facing) on predict/explain turns:")
            fe = v["explanation_driver_facing"]
            for full, tele in zip(fe["full_operational"], fe["telematics_only"]):
                add(f"  - {full['scenario']} turn {full['turn']} (expected {full['expected']}):")
                add(f"      full_operational: {full['explanation'] or '(none)'}")
                add(f"      telematics_only: {tele['explanation'] or '(none)'}")
            add("")

    add("## Limitations\n")
    for lim in report["limitations"]:
        add(f"- {lim}")

    path = report_dir / "summary.md"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
