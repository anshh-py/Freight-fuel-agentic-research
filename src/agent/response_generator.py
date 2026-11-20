"""M2.3 — grounded response generation for the M2.2 agent.

This module turns the structured ``AgentResponse`` produced by
``src.agent.orchestrator`` into clear, evidence-grounded, human-readable
responses for heavy-duty truck drivers and fleet operators.

Design principles
-----------------
* It consumes the *actual* ``AgentResponse`` and tool outputs — it never calls
  the tools itself and never fabricates values.
* Every response separates, at all times:
    A. model prediction (value, feature set, model identity, synthetic disclaimer),
    B. model explanation (instance-level SHAP with per-feature contributions),
    C. retrieved evidence (document id/title, excerpt, source provenance,
       similarity score),
    D. recommendations — only what the evidence supports, and explicitly split
       into evidence-backed / general / unsupported.
* Two renderings are produced:
    - driver-facing (readable, low jargon, ``driver_version``),
    - technical / fleet-operator (structured dict with feature names, signed
      SHAP values, source ids and detailed caveats, ``technical_version``).
* It is deterministic and independent of any LLM provider. There is no LLM
  configured; nothing here claims LLM-generated text or autonomous reasoning.
* Safety: it never invents a driver name, truck identity, route, observed fuel
  consumption, manufacturer specification, maintenance limit, or historical
  baseline. SHAP is always described as model attribution, never causality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.agent.intents import Intent


# ---------------------------------------------------------------------------
# Extracting tool results from an AgentResponse
# ---------------------------------------------------------------------------
def _ok_result(response: Any, tool: str) -> dict | None:
    """Return the successful result dict for ``tool``, or None."""
    for r in getattr(response, "tool_results", []):
        if r.get("tool") == tool:
            if r.get("status") == "ok" and isinstance(r.get("result"), dict):
                return r["result"]
            return None
    return None


def _failed_tools(response: Any) -> list[str]:
    return [
        r.get("tool")
        for r in getattr(response, "tool_results", [])
        if r.get("status") != "ok"
    ]


@dataclass
class GroundedResponse:
    """A grounded, validated response assembled from an ``AgentResponse``."""

    intent: str
    detected_intent: str
    router_backend: str
    selected_tools: list[str] = field(default_factory=list)
    # A. Prediction block
    prediction: dict | None = None
    # B. Explanation block
    explanation: dict | None = None
    # C. Retrieved-evidence block
    evidence: list[dict] = field(default_factory=list)
    evidence_insufficient: bool = False
    # D. Recommendations (three separate buckets)
    recommendations_evidence_backed: list[dict] = field(default_factory=list)
    recommendations_general_suggestions: list[dict] = field(default_factory=list)
    recommendations_unsupported: list[dict] = field(default_factory=list)
    # Cross-cutting
    tool_failures: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    driver_version: str = ""
    technical_version: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "detected_intent": self.detected_intent,
            "router_backend": self.router_backend,
            "selected_tools": self.selected_tools,
            "prediction": self.prediction,
            "explanation": self.explanation,
            "evidence": self.evidence,
            "evidence_insufficient": self.evidence_insufficient,
            "recommendations": {
                "evidence_backed": self.recommendations_evidence_backed,
                "general_suggestions": self.recommendations_general_suggestions,
                "unsupported": self.recommendations_unsupported,
            },
            "tool_failures": self.tool_failures,
            "limitations": self.limitations,
            "warnings": self.warnings,
            "driver_version": self.driver_version,
            "technical_version": self.technical_version,
        }


# Fixed, honest phrasing used by the generator. These are the only claims the
# response makes; the validator relies on them to detect any editing that would
# remove a required disclaimer.
SYNTHETIC_DISCLAIMER = (
    "This is an estimate from a model trained on SYNTHETIC telemetry, not a "
    "validated real-world fuel measurement. Do not use it for billing, "
    "compliance, or vehicle-performance claims."
)
CAUSAL_DISCLAIMER = (
    "SHAP attribution describes how the model composes its estimate from the "
    "input features. It is NOT a causal explanation: it does not prove that "
    "changing a feature will change real fuel consumption by any amount."
)
CORPUS_DISCLAIMER = (
    "Retrieved passages are SYNTHETIC educational material (not OEM or "
    "manufacturer documentation). Retrieval similarity is not proof of "
    "factual correctness."
)


class ResponseGenerator:
    """Deterministic builder of grounded responses from an ``AgentResponse``."""

    def __init__(self, inputs: dict | None = None) -> None:
        # Optional original (validated) vehicle inputs, used only to attach the
        # actual feature value to each SHAP contribution "where available".
        self._inputs = inputs or {}

    # -- extraction ------------------------------------------------------
    def _extract(self, response: Any) -> None:
        self._pred = _ok_result(response, "fuel_rate_prediction")
        self._expl = _ok_result(response, "prediction_explanation")
        self._ret = _ok_result(response, "knowledge_retrieval")
        self._failures = _failed_tools(response)

    # -- feature-value lookup --------------------------------------------
    def _feature_value(self, column: str):
        lookup = self._inputs.get(column)
        if lookup is None:
            return None
        if isinstance(lookup, (int, float)):
            return round(float(lookup), 2)
        return lookup

    # -- D: recommendations ----------------------------------------------
    def _build_recommendations(self, out: GroundedResponse) -> None:
        # Evidence-backed recommendations are only derived when the retrieved
        # evidence was judged *sufficient*. If it was insufficient, we say so
        # explicitly and refuse to turn weak similarity into guidance.
        if not out.evidence_insufficient:
            seen: set[str] = set()
            for p in out.evidence:
                doc_id = p.get("document_id")
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                title = p.get("title") or "untitled passage"
                out.recommendations_evidence_backed.append(
                    {
                        "statement": (
                            "The retrieved material noted the following (synthetic "
                            "educational content, not a verified measurement)."
                        ),
                        "source_document_id": doc_id,
                        "source_title": title,
                        "similarity_score": p.get("score"),
                        "excerpt": p.get("content_excerpt"),
                    }
                )

        # Unsupported: things the evidence/model cannot answer. We only ever say
        # what we *cannot* support; we never fabricate a recommendation.
        if out.recommendations_evidence_backed:
            out.recommendations_unsupported.append(
                {
                    "topic": "quantified savings from changing a driver action",
                    "reason": (
                        "The model's SHAP contributions are not causal and the "
                        "synthetic corpus provides no calibrated real-world "
                        "savings figures; no quantitative driver instruction "
                        "can be derived from this evidence."
                    ),
                }
            )
        else:
            out.recommendations_unsupported.append(
                {
                    "topic": "operational driver recommendation",
                    "reason": (
                        "No sufficiently relevant corpus evidence was retrieved, "
                        "so no evidence-backed operational recommendation can be "
                        "given without inventing guidance."
                    ),
                }
            )

    # -- main entry ------------------------------------------------------
    def generate(self, response: Any) -> GroundedResponse:
        self._extract(response)

        intent = getattr(response, "intent", Intent.AMBIGUOUS)
        intent_val = intent.value if isinstance(intent, Intent) else str(intent)
        backend = ""
        router = getattr(response, "router", None)
        if router is not None:
            backend = getattr(router, "backend", "") or ""

        out = GroundedResponse(
            intent=intent_val,
            detected_intent=intent_val,
            router_backend=backend,
            selected_tools=list(getattr(response, "selected_tools", [])),
            limitations=list(getattr(response, "limitations", [])),
            tool_failures=self._failures,
        )

        # --- A: prediction ---
        if self._pred is not None:
            out.prediction = {
                "predicted_fuel_rate_lph": self._pred.get("predicted_fuel_rate_lph"),
                "units": self._pred.get("units", "L/h"),
                "feature_set": self._pred.get("feature_set"),
                "model_name": (self._pred.get("model") or {}).get("name"),
                "model_artifact": (self._pred.get("model") or {}).get("artifact"),
                "synthetic_model_estimate": True,
            }

        # --- B: explanation ---
        if self._expl is not None:
            mi = (self._expl.get("most_influential_features") or {})
            pos = list(mi.get("top_positive_shap") or [])
            neg = list(mi.get("top_negative_shap") or [])
            out.explanation = {
                "predicted_fuel_rate_lph": self._expl.get("predicted_fuel_rate_lph"),
                "feature_set": self._expl.get("feature_set"),
                "model_name": (self._expl.get("model") or {}).get("name"),
                "explanation_kind": self._expl.get("explanation_kind"),
                "expected_value": self._expl.get("expected_value"),
                "reconstruction": self._expl.get("reconstruction"),
                "positive_contributions": [
                    {
                        "feature": it.get("readable_label"),
                        "column": it.get("column"),
                        "shap": it.get("shap"),
                        "actual_input_value": self._feature_value(it.get("column")),
                    }
                    for it in pos
                ],
                "negative_contributions": [
                    {
                        "feature": it.get("readable_label"),
                        "column": it.get("column"),
                        "shap": it.get("shap"),
                        "actual_input_value": self._feature_value(it.get("column")),
                    }
                    for it in neg
                ],
            }

        # --- C: evidence ---
        if self._ret is not None:
            sufficiency = self._ret.get("evidence_sufficiency") or {}
            out.evidence_insufficient = bool(sufficiency.get("insufficient"))
            out.evidence = list(self._ret.get("passages") or [])

        self._build_recommendations(out)
        out.driver_version = self._render_driver(out)
        out.technical_version = self._render_technical(out)
        return out

    # -- rendering -------------------------------------------------------
    def _render_driver(self, out: GroundedResponse) -> str:
        lines: list[str] = []
        intent = out.intent

        # Unsupported / ambiguous.
        if intent in (Intent.UNSUPPORTED.value, Intent.AMBIGUOUS.value):
            if intent == Intent.UNSUPPORTED.value:
                lines.append(
                    "I can't answer that within this project's scope: the agent "
                    "only works with synthetic fuel-rate prediction, retrieval "
                    "over a synthetic knowledge corpus, and model explanation. "
                    "It does not provide real-world, manufacturer, or "
                    "maintenance information."
                )
            else:
                lines.append(
                    "I couldn't pin down a single request. Please ask for a fuel "
                    "prediction (with vehicle inputs), a knowledge question, an "
                    "explanation, or a combination."
                )
            lines.append(
                "Routing is a deterministic rule-based baseline, not LLM "
                "reasoning or autonomous reasoning."
            )
            return "\n\n".join(lines)

        # Tool failures.
        if out.tool_failures:
            lines.append(
                "One or more underlying tools failed, so I cannot provide a "
                "complete trusted answer. The failed component(s): "
                + ", ".join(out.tool_failures)
                + ". No output from a failed tool is presented as a success."
            )

        # A. prediction
        if out.prediction is not None:
            p = out.prediction
            lines.append(
                f"Estimated fuel rate: {p['predicted_fuel_rate_lph']} L/h "
                f"(model: {p.get('model_name')}, feature set: "
                f"{p.get('feature_set')})."
            )
            lines.append(SYNTHETIC_DISCLAIMER)

        # B. explanation
        if out.explanation is not None:
            e = out.explanation
            pos = e["positive_contributions"]
            neg = e["negative_contributions"]
            lines.append("What influenced the model:")
            if pos:
                top = pos[0]
                lines.append(
                    f"  - The model estimate was pushed upward most by "
                    f"{top['feature']} (contribution +{top['shap']:.2f} L/h"
                    + (f", actual input {top['actual_input_value']}" if top["actual_input_value"] is not None else "")
                    + ")."
                )
            if neg:
                top = neg[0]
                lines.append(
                    f"  - It was pulled downward most by {top['feature']} "
                    f"(contribution {top['shap']:.2f} L/h"
                    + (f", actual input {top['actual_input_value']}" if top["actual_input_value"] is not None else "")
                    + ")."
                )
            lines.append(CAUSAL_DISCLAIMER)

        # C. evidence
        if self._ret is not None:
            if out.evidence_insufficient:
                lines.append(
                    "Relevant knowledge: no sufficiently supportive passage was "
                    "retrieved from the synthetic corpus for this question. I "
                    "will not invent information to fill the gap."
                )
            elif out.evidence:
                lines.append("Relevant knowledge (retrieved passages):")
                for p in out.evidence[:3]:
                    title = p.get("title") or "untitled passage"
                    doc_id = p.get("document_id") or "?"
                    sim = p.get("score")
                    lines.append(
                        f"  - [{doc_id}] {title} (topic: {p.get('topic')}; "
                        f"similarity {sim:.3f})."
                    )
                lines.append(CORPUS_DISCLAIMER)

        # D. recommendations (evidence-backed only, never unverified numbers).
        if out.recommendations_evidence_backed:
            lines.append("Operational considerations (evidence-backed):")
            for rec in out.recommendations_evidence_backed:
                lines.append(
                    f"  - Per retrieved material [{rec['source_document_id']}]: "
                    f"{rec['excerpt'][:220]}"
                )
        if out.recommendations_unsupported:
            u = out.recommendations_unsupported[0]
            lines.append(
                f"Not supported by the evidence: {u['topic']} — {u['reason']}"
            )

        # Synthesis of limitations.
        if out.limitations:
            dedup = list(dict.fromkeys(out.limitations))
            lines.append("Important limitations: " + " ".join(dedup))

        return "\n\n".join(lines).strip()

    def _render_technical(self, out: GroundedResponse) -> dict:
        """Structured fleet-operator view with feature names, signed SHAP and ids."""
        tech: dict[str, Any] = {
            "detected_intent": out.detected_intent,
            "router_backend": out.router_backend,
            "selected_tools": out.selected_tools,
            "tool_failures": out.tool_failures,
        }
        if out.prediction is not None:
            tech["prediction"] = out.prediction
        if out.explanation is not None:
            tech["explanation"] = out.explanation
        if self._ret is not None:
            tech["retrieval"] = {
                "evidence_insufficient": out.evidence_insufficient,
                "passages": [
                    {
                        "document_id": p.get("document_id"),
                        "chunk_id": p.get("chunk_id"),
                        "title": p.get("title"),
                        "topic": p.get("topic"),
                        "source_reference": p.get("source_reference"),
                        "is_oem_manual": p.get("is_oem_manual"),
                        "similarity_score": p.get("score"),
                    }
                    for p in out.evidence
                ],
            }
        tech["recommendations"] = {
            "evidence_backed": out.recommendations_evidence_backed,
            "general_suggestions": out.recommendations_general_suggestions,
            "unsupported": out.recommendations_unsupported,
        }
        tech["claims_and_provenance"] = {
            "model_prediction_is_synthetic_estimate": True,
            "shap_is_attribution_not_causality": True,
            "corpus_is_synthetic_educational_material": True,
            "retrieval_similarity_is_not_factual_proof": True,
            "no_invented_driver_or_truck_identity": True,
            "no_invented_oem_or_maintenance_specs": True,
        }
        tech["limitations"] = list(dict.fromkeys(out.limitations))
        return tech
