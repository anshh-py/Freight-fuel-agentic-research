"""M2.3 — response validation layer.

Deterministic, rule-based checks that detect prohibited claims in generated
responses. These are used both as guardrails on the generator's output and as
the scoring machinery for the response-evaluation benchmark (Phase 6).

Every check returns a list of ``ValidationIssue``:
    check_id   — machine-readable check name
    passed     — bool
    detail     — human-readable explanation
    severity   — "error" (prohibited / must-fix) or "warning"

The checks are deliberately deterministic and documented; they are NOT an LLM
judge and do not attempt to score subjective quality. Subjective criteria are
only reported through documented manual-review notes (see the benchmark).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.agent.response_generator import (
    CAUSAL_DISCLAIMER,
    CORPUS_DISCLAIMER,
    SYNTHETIC_DISCLAIMER,
)

# Word-level causal terms only count when used as a NON-negated assertion
# (e.g. "torque caused the higher fuel rate"). The generator's own protective
# phrasing ("is NOT a causal explanation", "contributions are not causal") must
# not be mistaken for a claim.
_CAUSAL_WORD_PATTERNS = [
    re.compile(r"\bcauses?\b", re.I),
    re.compile(r"\bcaused\b", re.I),
    re.compile(r"\bcausal\b", re.I),
]
# Unambiguous assertion frames: presence always indicates a prohibited claim.
_CAUSAL_FRAME_PATTERNS = [
    re.compile(r"\bproves?\b", re.I),
    re.compile(r"\bis the reason (?:the )?(?:model )?prediction is (?:higher|lower)\b", re.I),
    re.compile(r"\bwill (?:reduce|lower|increase|raise) (?:fuel consumption|fuel rate) by\b", re.I),
    re.compile(r"\bguarantee[ds]?\b", re.I),
]
# Negation tokens. Multi-word negations ("does not", "is not", "was not") are
# detected by their terminal token ("not"); the auxiliary verb is not itself a
# negation token. A double negative cancels out.
_NEGATION_TOKENS = frozenset({
    "not", "no", "never", "nor",
    "cannot", "can't", "doesn't", "isn't", "aren't", "don't", "won't",
    "couldn't", "shouldn't", "mustn't", "wouldn't", "hasn't", "hadn't",
    "haven't", "wasn't", "weren't",
})

# Patterns that imply OEM/manufacturer authority (prohibited for synthetic corpus).
_OEM_PATTERNS = [
    re.compile(r"\boem\b", re.I),
    re.compile(r"\bmanufacturer (?:spec|specification|limit|guideline|manual|data)\b", re.I),
    re.compile(r"\bofficial (?:spec|torque|limit)\b", re.I),
    re.compile(r"\bfactory (?:spec|limit|manual)\b", re.I),
]

# Numeric-ish spec tokens that would indicate fabricated concrete numbers.
_SPEC_NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\s*(?:L/h|N·m|Nm|bar|°C|°F|psi|rpm|kg|percent|%)\b", re.I)


@dataclass
class ValidationIssue:
    check_id: str
    passed: bool
    detail: str
    severity: str = "error"


def _has(text: str, patterns: list[re.Pattern]) -> bool:
    return any(p.search(text or "") for p in patterns)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------
def check_synthetic_disclaimer(text: str) -> ValidationIssue:
    """Prediction text must carry the synthetic-trained model disclaimer."""
    has_prediction = bool(re.search(r"\d+(?:\.\d+)?\s*L/h", text or ""))
    missing = has_prediction and not ("SYNTHETIC" in (text or "").upper() and "estimate" in (text or "").lower())
    return ValidationIssue(
        "synthetic_disclaimer",
        not missing,
        ("prediction present without a synthetic-trained-model disclaimer" if missing
         else "prediction carries the synthetic disclaimer"),
        "error" if missing else "warning",
    )


def _is_negated_context(text: str, pos: int) -> bool:
    """True if the word at ``pos`` sits in a negated verb phrase.

    Recognizes a negation governing the word, including when an article or
    adjective separates them ("NOT a causal explanation", "does not prove",
    "is not really a causal contribution"). A double negative cancels out.
    """
    pre = text[max(0, pos - 60):pos].lower()
    tokens = re.findall(r"[a-z']+", pre)
    window = tokens[-5:]
    for i in range(len(window) - 1, -1, -1):
        if window[i] in _NEGATION_TOKENS:
            # "not ... not" cancels out; do not treat as negated.
            return not (i > 0 and window[i - 1] in _NEGATION_TOKENS)
    return False


def check_causal_language(text: str, explanation_present: bool = False) -> ValidationIssue:
    """Explanation text must not assert a causal/real-world effect of a feature."""
    text = text or ""
    # Strip the generator's own fixed protective sentences: they are
    # anti-causal statements, not claims the response is making.
    work = text
    for d in (CAUSAL_DISCLAIMER, CORPUS_DISCLAIMER, SYNTHETIC_DISCLAIMER):
        work = work.replace(d, "")
    # Both assertion frames and word-level causal terms count only when used as
    # a NON-negated assertion ("this feature causes higher fuel consumption",
    # "SHAP proves this feature increases consumption"). Protective phrasing that
    # is negated ("NOT a causal explanation", "does not prove") must not be
    # flagged.
    for pat in _CAUSAL_FRAME_PATTERNS + _CAUSAL_WORD_PATTERNS:
        for m in pat.finditer(work):
            if not _is_negated_context(work, m.start()):
                return ValidationIssue(
                    "causal_language", False,
                    "response asserts a causal/real-world effect of a feature",
                    "error",
                )
    # Any explanation must carry the non-causal disclaimer in the real text.
    if explanation_present and "NOT a causal explanation" not in text and "not causality" not in text.lower():
        return ValidationIssue(
            "causal_language", False,
            "explanation present but the non-causal disclaimer is missing",
            "error",
        )
    return ValidationIssue(
        "causal_language", True,
        "no causal language detected; explanation carries the non-causal disclaimer",
        "warning",
    )


def check_oem_mislabeling(text: str) -> ValidationIssue:
    """Corpus content must never be presented as OEM/manufacturer documentation."""
    text = text or ""
    # A negation right before an OEM term is allowed (e.g. "not OEM documentation").
    cleaned = re.sub(r"not (?:OEM|manufacturer|official|factory)", "", text, flags=re.I)
    if _has(cleaned, _OEM_PATTERNS):
        return ValidationIssue(
            "oem_mislabeling", False,
            "response presents synthetic corpus content as OEM/manufacturer documentation",
            "error",
        )
    return ValidationIssue(
        "oem_mislabeling", True,
        "no OEM/authoritative framing of synthetic corpus content",
        "warning",
    )


def check_synthetic_corpus_notice(text: str, mentions_passage: bool = False) -> ValidationIssue:
    """Any reference to a retrieved passage must note synthetic-educational provenance."""
    text = text or ""
    has_notice = "SYNTHETIC educational material" in text.upper() or "synthetic educational" in text.lower()
    if mentions_passage and not has_notice:
        return ValidationIssue(
            "synthetic_corpus_notice", False,
            "passage cited without the synthetic-educational provenance notice",
            "error",
        )
    return ValidationIssue(
        "synthetic_corpus_notice", True,
        "corpus provenance notice present where passages are cited",
        "warning",
    )


def check_source_attribution(text: str, known_doc_ids: list[str]) -> ValidationIssue:
    """Every cited source must correspond to a real retrieved document id."""
    text = text or ""
    cited = set(re.findall(r"\[(doc_\d+)\]", text))
    fabricated = cited - set(known_doc_ids)
    if fabricated:
        return ValidationIssue(
            "source_attribution", False,
            f"cited source id(s) not present in retrieved evidence: {sorted(fabricated)}",
            "error",
        )
    return ValidationIssue(
        "source_attribution", True,
        "all cited sources correspond to retrieved document ids",
        "warning",
    )


def check_unsupported_not_answered(text: str, minimal: bool = False) -> ValidationIssue:
    """An unsupported/ambiguous/unanswerable query must not receive fabricated facts."""
    text = text or ""
    if minimal and _SPEC_NUMBER_PATTERN.search(text) and "can't answer" not in text.lower():
        return ValidationIssue(
            "unsupported_not_answered", False,
            "unsupported query produced concrete specifications/numbers",
            "error",
        )
    return ValidationIssue(
        "unsupported_not_answered", True,
        "unsupported query handled without fabricating specifications",
        "warning",
    )


def check_insufficient_evidence_stated(out: Any) -> ValidationIssue:
    """If retrieval evidence is insufficient, the response must say so explicitly."""
    stated = out.evidence_insufficient and "no sufficiently supportive passage" in out.driver_version
    if out.evidence_insufficient and not stated:
        return ValidationIssue(
            "insufficient_evidence_stated", False,
            "evidence was insufficient but the response did not say so",
            "error",
        )
    return ValidationIssue(
        "insufficient_evidence_stated", True,
        "insufficient evidence is stated explicitly",
        "warning",
    )


def check_failure_not_presented_as_success(out: Any) -> ValidationIssue:
    """A failed tool must never be presented as a successful result."""
    if out.tool_failures:
        ok = "failed" in out.driver_version
        return ValidationIssue(
            "failure_not_presented_as_success", ok,
            ("tool failure not disclosed in the response" if not ok
             else "tool failure is disclosed, not presented as success"),
            "error" if not ok else "warning",
        )
    return ValidationIssue(
        "failure_not_presented_as_success", True,
        "no tool failures to disclose",
        "warning",
    )


def check_prediction_value_preserved(out: Any, source: dict | None) -> ValidationIssue:
    """The rendered answer must preserve the exact predicted L/h value."""
    if source is None:
        return ValidationIssue("prediction_value_preserved", True, "no prediction block", "warning")
    val = source.get("predicted_fuel_rate_lph")
    if val is None:
        return ValidationIssue("prediction_value_preserved", False, "source prediction missing value", "error")
    ok = f"{val}" in out.driver_version
    return ValidationIssue(
        "prediction_value_preserved", ok,
        ("predicted value {val} preserved" if ok else f"predicted value {val} missing from response"),
        "warning" if ok else "error",
    )


def check_shap_value_preserved(out: Any, source: dict | None) -> ValidationIssue:
    """The top signed SHAP values must be preserved in the rendered answer."""
    if source is None:
        return ValidationIssue("shap_value_preserved", True, "no explanation block", "warning")
    mi = source.get("most_influential_features") or {}
    top = (mi.get("top_positive_shap") or [])
    if not top:
        return ValidationIssue("shap_value_preserved", False, "explanation has no positive contributions", "error")
    val = top[0].get("shap")
    tok = f"+{val:.2f}" if val is not None else None
    ok = tok is not None and tok in out.driver_version
    return ValidationIssue(
        "shap_value_preserved", ok,
        ("top SHAP value %s preserved" % tok if ok else f"top SHAP value {tok} missing"),
        "warning" if ok else "error",
    )


# ---------------------------------------------------------------------------
# Full classifier
# ---------------------------------------------------------------------------
def validate_grounded_response(
    out: Any,
    *,
    source_prediction: dict | None = None,
    source_explanation: dict | None = None,
    known_doc_ids: list[str] | None = None,
) -> list[ValidationIssue]:
    """Run the full battery over a ``GroundedResponse`` (or compatible object)."""
    text = out.driver_version
    known_doc_ids = known_doc_ids or []
    results = [
        check_synthetic_disclaimer(text),
        check_causal_language(text, explanation_present=out.explanation is not None),
        check_oem_mislabeling(text),
        check_synthetic_corpus_notice(text, mentions_passage=bool(out.evidence)),
        check_source_attribution(text, known_doc_ids),
        check_insufficient_evidence_stated(out),
        check_failure_not_presented_as_success(out),
        check_prediction_value_preserved(out, source_prediction),
        check_shap_value_preserved(out, source_explanation),
    ]
    if out.intent in ("unsupported", "ambiguous"):
        results.append(check_unsupported_not_answered(text, minimal=True))
    return results


def summary(issues: list[ValidationIssue]) -> dict:
    """Aggregate check results into counts and a pass/fail verdict."""
    failed = [i for i in issues if not i.passed]
    return {
        "total_checks": len(issues),
        "passed": len(issues) - len(failed),
        "failed": len(failed),
        "all_passed": not failed,
        "failed_checks": sorted({i.check_id for i in failed}),
        "issues": [{"check_id": i.check_id, "passed": i.passed, "detail": i.detail}
                   for i in issues],
    }
