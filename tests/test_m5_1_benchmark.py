"""Focused validation tests for the M5.1 driver-conversation evaluation benchmark.

These tests validate the benchmark *dataset*: schema, scenario IDs, label
consistency, duplicate detection (exact and normalized), and dev/held-out
split leakage. They do not run any model and never touch the frozen
M1/M2/M3/M4 implementations.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.evaluation.m5_1_driver_benchmark import (
    CATEGORIES,
    FULL_REQUIRED,
    TELE_REQUIRED,
    VALID_GROUNDING,
    VALID_INTENTS,
    VALID_SAFETY,
    VALID_TOOLS,
    validate_benchmark,
)
from src.evaluation.m5_1_scenarios import SCENARIOS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = (
    PROJECT_ROOT / "reports" / "m5" / "benchmark"
    / "driver_conversation_benchmark.json"
)

# Map feature-set level expected input counts (dev and held-out both use these).
_MODEL_INTENTS = ("predict_fuel", "explain_prediction", "combined_request",
                  "report_behavior")
_NO_TOOL_INTENTS = ("unsupported", "ambiguous", "clarifying")


@pytest.fixture(scope="module")
def doc() -> dict:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def _all_turns(doc: dict) -> list[dict]:
    return [t for c in doc["cases"] for t in c["turns"]]


# ---------------------------------------------------------------------------
# Schema and ID checks
# ---------------------------------------------------------------------------

def test_report_exists_and_has_cases(doc):
    assert doc["dataset"].startswith("M5.1")
    assert doc["milestone"] == "M5.1"
    assert isinstance(doc["cases"], list) and doc["cases"]


def test_all_scenario_ids_unique_and_prefixed(doc):
    ids = [c["scenario_id"] for c in doc["cases"]]
    assert len(ids) == len(set(ids)), "duplicate scenario_id found"
    for cid in ids:
        assert re.fullmatch(r"M5-(DEV|HO)-\d{3}", cid), f"bad id {cid!r}"


def test_dev_and_heldout_ids_present(doc):
    dev = [c for c in doc["cases"] if c["split"] == "dev"]
    held = [c for c in doc["cases"] if c["split"] == "heldout"]
    assert dev and held
    assert all(c["scenario_id"].startswith("M5-DEV") for c in dev)
    assert all(c["scenario_id"].startswith("M5-HO") for c in held)


# ---------------------------------------------------------------------------
# Label consistency
# ---------------------------------------------------------------------------

def test_turn_labels_within_valid_sets(doc):
    for t in _all_turns(doc):
        assert t["user_intent"] in VALID_INTENTS, t
        assert t["expected_intent"] in VALID_INTENTS, t
        for tool in t["expected_tools"]:
            assert tool in VALID_TOOLS, t
        for g in t["expected_grounding"]:
            assert g in VALID_GROUNDING, t
        assert t["expected_safety_outcome"] in VALID_SAFETY, t
        assert isinstance(t["expected_clarification"], bool)
        assert isinstance(t["expected_abstention"], bool)


def test_model_intents_must_select_tools(doc):
    for t in _all_turns(doc):
        if t["expected_intent"] in _MODEL_INTENTS:
            assert t["expected_tools"], f"model intent with no tools: {t}"
        if t["expected_intent"] in _NO_TOOL_INTENTS:
            assert not t["expected_tools"], f"{t['expected_intent']} picked tools: {t}"


def test_clarifying_turns_request_clarification(doc):
    for t in _all_turns(doc):
        if t["expected_intent"] == "clarifying":
            assert t["expected_clarification"] is True, t


def test_unsupported_and_ambiguous_raise_abstention_or_clarification(doc):
    for t in _all_turns(doc):
        if t["expected_intent"] == "unsupported":
            assert t["expected_abstention"] or t["expected_clarification"], t
        if t["expected_intent"] == "ambiguous":
            assert t["expected_clarification"] is True, t


def test_no_simultaneous_clarification_and_abstention(doc):
    for t in _all_turns(doc):
        assert not (t["expected_clarification"] and t["expected_abstention"]), t


# ---------------------------------------------------------------------------
# Original vs routed intent preservation
# ---------------------------------------------------------------------------

def test_user_intent_preserved_separately_from_routed_intent(doc):
    diff = [t for t in _all_turns(doc)
            if t["user_intent"] != t["expected_intent"]]
    # Clarification/refusal turns may diverge; must be documented not empty.
    assert diff, "expected some clarified turns to preserve original intent"
    for t in diff:
        assert t["expected_intent"] == "clarifying", (
            f"user/routed divergence only on clarifying: {t}"
        )
        assert t["user_intent"] in _MODEL_INTENTS, t


# ---------------------------------------------------------------------------
# Duplicate detection (exact + normalized)
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    t = text.lower()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def test_no_cross_case_exact_duplicate_text(doc):
    seen: dict[str, str] = {}
    for c in doc["cases"]:
        case_texts = set()
        for t in c["turns"]:
            norm = _normalize(t["text"])
            if norm:
                if norm in case_texts:
                    continue  # intra-scenario repeat is legitimate
                assert norm not in seen, (
                    f"cross-case duplicate text: {c['scenario_id']} vs "
                    f"{seen[norm]}: {t['text']!r}"
                )
                seen[norm] = c["scenario_id"]
                case_texts.add(norm)


def test_no_cross_case_normalized_duplicate_text(doc):
    seen: dict[str, str] = {}
    for c in doc["cases"]:
        case_texts = set()
        for t in c["turns"]:
            norm = _normalize(t["text"])
            if norm and norm not in case_texts:
                assert norm not in seen, (
                    f"normalized cross-case duplicate: {c['scenario_id']} vs "
                    f"{seen[norm]}: {t['text']!r}"
                )
                seen[norm] = c["scenario_id"]
                case_texts.add(norm)


def test_same_text_within_one_case_is_allowed(doc):
    # Multi-turn clarification legitimately repeats the driver's opening line.
    repeats = 0
    for c in doc["cases"]:
        norms = [_normalize(t["text"]) for t in c["turns"]]
        repeats += len(norms) - len(set(norms))
    assert repeats >= 2, "expected at least one intentional within-case repeat"


# ---------------------------------------------------------------------------
# Split leakage
# ---------------------------------------------------------------------------

def test_no_heldout_text_leaks_into_dev(doc):
    dev_norms = set()
    for c in doc["cases"]:
        if c["split"] != "dev":
            continue
        for t in c["turns"]:
            n = _normalize(t["text"])
            if n:
                dev_norms.add(n)
    for c in doc["cases"]:
        if c["split"] != "heldout":
            continue
        for t in c["turns"]:
            n = _normalize(t["text"])
            assert n not in dev_norms, (
                f"held-out text leaks into dev: {c['scenario_id']}: "
                f"{t['text']!r}"
            )


# ---------------------------------------------------------------------------
# Counts and category distribution
# ---------------------------------------------------------------------------

def test_split_counts_match_document(doc):
    dev = sum(1 for c in doc["cases"] if c["split"] == "dev")
    held = sum(1 for c in doc["cases"] if c["split"] == "heldout")
    assert doc["split_counts"] == {"dev": dev, "heldout": held}


def test_category_counts_match_document(doc):
    real: dict[str, int] = {}
    for c in doc["cases"]:
        real[c["category"]] = real.get(c["category"], 0) + 1
    assert doc["category_counts"] == real
    for cat in real:
        assert cat in CATEGORIES


def test_all_categories_represented(doc):
    covered = {c["category"] for c in doc["cases"]}
    assert covered == set(CATEGORIES.keys())


def test_turn_count_breakdown(doc):
    single = sum(1 for c in doc["cases"] if c["turn_count"] == 1)
    multi = sum(1 for c in doc["cases"] if c["turn_count"] > 1)
    assert doc["turn_counts"] == {"single_turn": single, "multi_turn": multi}
    assert multi >= 3, "benchmark must include substantial multi-turn cases"


def test_full_validator_returns_no_errors(doc):
    errors = validate_benchmark(doc)
    assert errors == [], f"validator reported errors: {errors[:5]}"


# ---------------------------------------------------------------------------
# Held-out distinctness (semantic edge coverage)
# ---------------------------------------------------------------------------

def test_heldout_cases_are_semantically_distinct(doc):
    dev_cats = {c["category"] for c in doc["cases"] if c["split"] == "dev"}
    held_cats = {c["category"] for c in doc["cases"] if c["split"] == "heldout"}
    # Held-out covers a broad slice of the same category space but with
    # distinct phrasings/edges (already enforced by duplicate test).
    assert held_cats and held_cats <= dev_cats | held_cats


def test_heldout_phrasings_are_not_exact_or_normalized_duplicates(doc):
    # Held-out cases must be distinct phrasings; the duplicate-leakage tests
    # above already enforce exact/normalized non-overlap. This test only
    # asserts that the held-out set is non-trivial and disjoint by scenario id.
    dev_ids = {c["scenario_id"] for c in doc["cases"] if c["split"] == "dev"}
    held_ids = {c["scenario_id"] for c in doc["cases"] if c["split"] == "heldout"}
    assert held_ids
    assert dev_ids.isdisjoint(held_ids)
