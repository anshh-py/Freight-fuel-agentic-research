"""M5.2 — labeled retrieval + grounding + abstention query set (data only).

This module holds the *labeled* evaluation cases for the M5.2 benchmark. It is
deliberately separate from the evaluation logic so the labels can be reviewed
and audited independently.

Labeling methodology (documented, see also the report and summary):
* The labels are curated and hand-authored for this synthetic research
  environment. They are NOT the product of an independent external human review
  panel; that is a stated limitation, not an implied guarantee.
* Retrieval-relevant labels are document-level: each answerable query lists the
  ``relevant_document_ids`` in the synthetic corpus that support a grounded
  answer. These were assigned from the actual document content (each label
  carries a short rationale).
* Grounding labels record what a *correctly grounded* response should look like
  for the candidate answer supplied: whether the answer is supported by the
  evidence, whether source attribution is correct, whether it avoids OEM
  mislabeling, whether uncertainty is communicated, and whether any claim
  introduces information absent from the retrieved evidence.
* Abstention labels classify unanswerable queries into one of five categories:
  general unanswerable, insufficient retrieval evidence, unsupported technical
  specification, guaranteed-fuel-savings elicitation, and real-world
  validation unsupported by the synthetic dataset.

Every case carries ``answerable`` and ``abstention_category`` so retrieval
metrics can be reported separately for answerable and unanswerable queries, and
so the grounding/abstention evaluation never conflates the two.
"""

from __future__ import annotations

from typing import Any

#: The five abstention categories recognized by the M5.2 evaluator.
ABSTENTION_CATEGORIES: tuple[str, ...] = (
    "unanswerable_general",
    "insufficient_evidence",
    "unsupported_technical_spec",
    "guaranteed_fuel_savings",
    "real_world_validation",
)

#: Whether each answerable answer is REQUIRED to communicate uncertainty.
#:
#: ``True`` when the answer conveys one of these, and therefore must hedge:
#:   * an estimated / approximate quantity (e.g. a ~70-80 kph threshold),
#:   * a causal or associative claim that must be caveated (SHAP attribution
#:     vs. physical causation; model association vs. proof),
#:   * a recommendation whose outcome is not guaranteed (fuel-saving guidance),
#:   * a generalization beyond the covered evidence (synthetic-to-real-world).
#:
#: ``False`` for answers that state established, factual mechanics or design
#: principles from the corpus, which do not require hedging.
#:
#: The uncertainty-compliance rate is computed ONLY over cases where
#: communication is required (the correct denominator); the four-way breakdown
#: (required+communicated, required+missing, not-required) is reported.
UNCERTAINTY_REQUIRED_BY_ID: dict[str, bool] = {
    "m52_q01": False,  # factual enumeration of fuel-rate determinants
    "m52_q02": False,  # interpretive guidance; no estimate or causal caveat
    "m52_q03": False,  # definitional metric comparison
    "m52_q04": True,   # model association vs. physical causation caveat
    "m52_q05": False,  # mechanistic friction/pumping-loss explanation
    "m52_q06": False,  # mechanistic payload physics
    "m52_q07": False,  # mechanistic grade physics
    "m52_q08": True,   # non-guaranteed fuel-saving recommendation
    "m52_q09": False,  # factual idle behaviour
    "m52_q10": True,   # approximate speed threshold (~70-80 kph)
    "m52_q11": False,  # mechanistic stop-and-go explanation
    "m52_q12": True,   # torque-band strategy; not a guaranteed saving
    "m52_q13": True,   # approximate comparative claim ("roughly proportional")
    "m52_q14": False,  # design principles for feedback
    "m52_q15": True,   # SHAP correlation-not-causation interpretation
    "m52_q16": True,   # synthetic-to-real-world generalization
    "m52_q17": True,   # SHAP negative-attribution interpretation (association)
    "m52_q18": True,   # approximate speed threshold (~70-80 kph)
    "m52_q19": False,  # definitive feature-leakage rule
}

QUERIES: list[dict[str, Any]] = [
    # ------------------------------------------------------------------ #
    # Answerable: retrieval + grounding (driver-domain knowledge)        #
    # ------------------------------------------------------------------ #
    {
        "query_id": "m52_q01",
        "user_query": "What actually determines how much fuel a heavy-duty truck burns in a given moment?",
        "topic": "fuel_consumption_fundamentals",
        "category": "answerable",
        "answerable": True,
        "relevant_document_ids": ["doc_001"],
        "abstention_category": None,
        "candidate_answer": (
            "Fuel rate is driven by the power demanded at the wheels: vehicle mass, speed, "
            "acceleration, aerodynamic drag, rolling resistance and road gradient. "
            "This is from synthetic educational material, not OEM documentation. "
            "A single instantaneous reading should not be used to judge a driver."
        ),
        "grounding": {
            "supported_by_evidence": True,
            "source_attribution_correct": True,
            "oem_mislabeling": False,
            "uncertainty_communicated": True,
            "claim_absent_from_evidence": False,
        },
        "rationale": "doc_001 enumerates exactly these drivers and notes fuel rate is a proxy for engine load/driver demand.",
    },
    {
        "query_id": "m52_q02",
        "user_query": "Why should one instantaneous fuel-rate reading never be used to judge me or my truck?",
        "topic": "fuel_rate_interpretation",
        "category": "answerable",
        "answerable": True,
        "relevant_document_ids": ["doc_013"],
        "abstention_category": None,
        "candidate_answer": (
            "One reading only means something in context; it should not be used to judge a "
            "driver or truck. Trip-level statistics under fixed conditions are the right "
            "comparison. Synthetic educational material, not OEM guidance."
        ),
        "grounding": {
            "supported_by_evidence": True,
            "source_attribution_correct": True,
            "oem_mislabeling": False,
            "uncertainty_communicated": True,
            "claim_absent_from_evidence": False,
        },
        "rationale": "doc_013 explicitly warns against judging a driver/truck on a single instantaneous value and recommends trip-level statistics.",
    },
    {
        "query_id": "m52_q03",
        "user_query": "What is the difference between instantaneous fuel rate and total fuel used over a trip?",
        "topic": "fuel_consumption_fundamentals",
        "category": "answerable",
        "answerable": True,
        "relevant_document_ids": ["doc_002"],
        "abstention_category": None,
        "candidate_answer": (
            "Instantaneous fuel rate is in litres per hour; total fuel used over a trip is the "
            "integral of that rate over time. When analysing a trip you should sum per-second "
            "consumption increments. Based on synthetic educational material."
        ),
        "grounding": {
            "supported_by_evidence": True,
            "source_attribution_correct": True,
            "oem_mislabeling": False,
            "uncertainty_communicated": False,
            "claim_absent_from_evidence": False,
        },
        "rationale": "doc_002 contrasts L/h with total fuel as an integral and advises summing per-second increments.",
    },
    {
        "query_id": "m52_q04",
        "user_query": "Why does fuel rate go up when the engine load increases?",
        "topic": "engine_load_fuel_demand",
        "category": "answerable",
        "answerable": True,
        "relevant_document_ids": ["doc_003"],
        "abstention_category": None,
        "candidate_answer": (
            "A higher engine load means the engine must inject more fuel to produce more torque "
            "and power, so fuel rate generally rises with engine load. This is a learned model "
            "association, not a proof of physical causation. Synthetic educational material."
        ),
        "grounding": {
            "supported_by_evidence": True,
            "source_attribution_correct": True,
            "oem_mislabeling": False,
            "uncertainty_communicated": True,
            "claim_absent_from_evidence": False,
        },
        "rationale": "doc_003 states higher load means more injection for torque/power so fuel rate rises.",
    },
    {
        "query_id": "m52_q05",
        "user_query": "Why does running at very high RPM waste fuel even at the same road speed?",
        "topic": "rpm_torque_relationships",
        "category": "answerable",
        "answerable": True,
        "relevant_document_ids": ["doc_006"],
        "abstention_category": None,
        "candidate_answer": (
            "Higher engine RPM usually raises internal friction and pumping losses, so fuel use "
            "rises for the same output power. Cruising at moderate RPM in the correct gear uses "
            "less fuel. Synthetic educational material, not OEM guidance."
        ),
        "grounding": {
            "supported_by_evidence": True,
            "source_attribution_correct": True,
            "oem_mislabeling": False,
            "uncertainty_communicated": True,
            "claim_absent_from_evidence": False,
        },
        "rationale": "doc_006 states higher RPM increases friction/pumping losses and moderate RPM in correct gear uses less fuel.",
    },
    {
        "query_id": "m52_q06",
        "user_query": "Why does a heavier payload increase my fuel consumption?",
        "topic": "payload_effects",
        "category": "answerable",
        "answerable": True,
        "relevant_document_ids": ["doc_007"],
        "abstention_category": None,
        "candidate_answer": (
            "Adding payload increases vehicle mass, which raises rolling resistance and the force "
            "needed to accelerate and climb, so heavier loads increase fuel rate especially "
            "during acceleration and on grades. Synthetic educational material."
        ),
        "grounding": {
            "supported_by_evidence": True,
            "source_attribution_correct": True,
            "oem_mislabeling": False,
            "uncertainty_communicated": False,
            "claim_absent_from_evidence": False,
        },
        "rationale": "doc_007 states payload raises mass, rolling resistance, and the force to accelerate/climb.",
    },
    {
        "query_id": "m52_q07",
        "user_query": "Why does climbing an uphill grade demand more fuel?",
        "topic": "road_grade_effects",
        "category": "answerable",
        "answerable": True,
        "relevant_document_ids": ["doc_009", "doc_010"],
        "abstention_category": None,
        "candidate_answer": (
            "Climbing requires extra force equal to the component of the truck's weight along the "
            "slope, directly raising engine load and fuel rate. Drivers also shift down and "
            "increase throttle on uphill sections, spiking fuel rate. Synthetic educational "
            "material, not OEM documentation."
        ),
        "grounding": {
            "supported_by_evidence": True,
            "source_attribution_correct": True,
            "oem_mislabeling": False,
            "uncertainty_communicated": False,
            "claim_absent_from_evidence": False,
        },
        "rationale": "doc_009 gives the slope-component physics; doc_010 describes downshifting/throttle on grades.",
    },
    {
        "query_id": "m52_q08",
        "user_query": "How can a driver reduce peak fuel demand on a long uphill climb with a heavy load?",
        "topic": "road_grade_effects",
        "category": "answerable",
        "answerable": True,
        "relevant_document_ids": ["doc_010", "doc_009"],
        "abstention_category": None,
        "candidate_answer": (
            "Roll into the hill with momentum using predictive control to reduce peak fuel demand "
            "during the climb. This follows from synthetic educational material; it is not a "
            "guaranteed saving."
        ),
        "grounding": {
            "supported_by_evidence": True,
            "source_attribution_correct": True,
            "oem_mislabeling": False,
            "uncertainty_communicated": True,
            "claim_absent_from_evidence": False,
        },
        "rationale": "doc_010 recommends predictive 'roll in' momentum to lower peak demand on the climb.",
    },
    {
        "query_id": "m52_q09",
        "user_query": "A truck idles for a long time during loading. What does the material say about fuel use?",
        "topic": "engine_load_fuel_demand",
        "category": "answerable",
        "answerable": True,
        "relevant_document_ids": ["doc_004"],
        "abstention_category": None,
        "candidate_answer": (
            "Idle fuel rate stays above zero because the engine keeps rotating and supports "
            "auxiliaries; long idling is a real source of fuel loss in haulage. Synthetic "
            "educational material."
        ),
        "grounding": {
            "supported_by_evidence": True,
            "source_attribution_correct": True,
            "oem_mislabeling": False,
            "uncertainty_communicated": False,
            "claim_absent_from_evidence": False,
        },
        "rationale": "doc_004 states idle rate stays above zero and long idling is a real fuel loss source.",
    },
    {
        "query_id": "m52_q10",
        "user_query": "Why does a highway trip at high average speed use disproportionately more fuel?",
        "topic": "speed_variability_driving_behaviour",
        "category": "answerable",
        "answerable": True,
        "relevant_document_ids": ["doc_012"],
        "abstention_category": None,
        "candidate_answer": (
            "Aerodynamic drag rises with the square of speed, so the fuel cost per kilometre climbs "
            "quickly above about 70-80 kph. High-speed cruising uses disproportionately more fuel. "
            "Synthetic educational material."
        ),
        "grounding": {
            "supported_by_evidence": True,
            "source_attribution_correct": True,
            "oem_mislabeling": False,
            "uncertainty_communicated": False,
            "claim_absent_from_evidence": False,
        },
        "rationale": "doc_012 states drag rises with the square of speed and cost climbs quickly above ~70-80 kph.",
    },
    {
        "query_id": "m52_q11",
        "user_query": "Frequent acceleration followed by hard braking is hurting my economy. Why?",
        "topic": "speed_variability_driving_behaviour",
        "category": "answerable",
        "answerable": True,
        "relevant_document_ids": ["doc_011"],
        "abstention_category": None,
        "candidate_answer": (
            "Frequent acceleration and braking waste fuel because the energy put into accelerating "
            "is then dissipated as heat in braking, so aggressive stop-and-go driving raises "
            "average fuel use. Synthetic educational material, not OEM guidance."
        ),
        "grounding": {
            "supported_by_evidence": True,
            "source_attribution_correct": True,
            "oem_mislabeling": False,
            "uncertainty_communicated": False,
            "claim_absent_from_evidence": False,
        },
        "rationale": "doc_011 states stop-and-go wastes fuel because acceleration energy is dissipated in braking.",
    },
    {
        "query_id": "m52_q12",
        "user_query": "Which gear and RPM strategy saves fuel when cruising at a set road speed?",
        "topic": "rpm_torque_relationships",
        "category": "answerable",
        "answerable": True,
        "relevant_document_ids": ["doc_006", "doc_005"],
        "abstention_category": None,
        "candidate_answer": (
            "Cruise at a moderate RPM in the correct gear to lower friction and pumping losses, and "
            "operate in the favourable torque band around rated RPM for best efficiency. Synthetic "
            "educational material; not a guaranteed saving."
        ),
        "grounding": {
            "supported_by_evidence": True,
            "source_attribution_correct": True,
            "oem_mislabeling": False,
            "uncertainty_communicated": True,
            "claim_absent_from_evidence": False,
        },
        "rationale": "doc_006 recommends moderate RPM in the correct gear; doc_005 covers the favourable torque band.",
    },
    {
        "query_id": "m52_q13",
        "user_query": "Compare fuel use on level ground versus on an uphill grade for the same truck and payload.",
        "topic": "road_grade_effects",
        "category": "answerable",
        "answerable": True,
        "relevant_document_ids": ["doc_007", "doc_009"],
        "abstention_category": None,
        "candidate_answer": (
            "On flat ground the payload effect is roughly proportional to mass; on a grade the "
            "effect is stronger because climbing directly raises engine load and fuel rate. "
            "Synthetic educational material."
        ),
        "grounding": {
            "supported_by_evidence": True,
            "source_attribution_correct": True,
            "oem_mislabeling": False,
            "uncertainty_communicated": True,
            "claim_absent_from_evidence": False,
        },
        "rationale": "doc_007 says the payload effect is roughly proportional to mass and stronger on gradients; doc_009 covers climbing load.",
    },
    {
        "query_id": "m52_q14",
        "user_query": "What should driver feedback do to be actionable and non-judgemental?",
        "topic": "driver_feedback_safety",
        "category": "answerable",
        "answerable": True,
        "relevant_document_ids": ["doc_018", "doc_017"],
        "abstention_category": None,
        "candidate_answer": (
            "Actionable feedback ties an observation to a specific, safe behaviour and references "
            "concrete measured evidence; it should be constructive, specific and non-judgemental. "
            "Synthetic educational material."
        ),
        "grounding": {
            "supported_by_evidence": True,
            "source_attribution_correct": True,
            "oem_mislabeling": False,
            "uncertainty_communicated": False,
            "claim_absent_from_evidence": False,
        },
        "rationale": "doc_018 says feedback ties observation to a safe behaviour with measured evidence; doc_017 adds constructive/non-judgemental.",
    },
    {
        "query_id": "m52_q15",
        "user_query": "A SHAP explanation shows a large positive attribution for engine load. Does that mean engine load caused the high fuel rate?",
        "topic": "shap_interpretation",
        "category": "answerable",
        "answerable": True,
        "relevant_document_ids": ["doc_015", "doc_016"],
        "abstention_category": None,
        "candidate_answer": (
            "Large positive attributions usually come from high engine load, a steep grade, heavy "
            "payload, and high throttle/speed. But SHAP reflects the model's learned associations, "
            "not guaranteed physical causation. It is not a causal proof."
        ),
        "grounding": {
            "supported_by_evidence": True,
            "source_attribution_correct": True,
            "oem_mislabeling": False,
            "uncertainty_communicated": True,
            "claim_absent_from_evidence": False,
        },
        "rationale": "doc_015 lists typical positive attributions; doc_016 warns SHAP is correlation, not causation.",
    },
    {
        "query_id": "m52_q16",
        "user_query": "Can a fuel-rate model trained on this synthetic dataset be trusted on real, unseen trucks?",
        "topic": "model_limitations",
        "category": "answerable",
        "answerable": True,
        "relevant_document_ids": ["doc_019", "doc_020"],
        "abstention_category": None,
        "candidate_answer": (
            "Accuracy on real trucks is not established because the synthetic targets are "
            "formula-generated, and unseen-truck results depend on fleet representativeness. This "
            "cannot guarantee performance on other fleets, routes or weather."
        ),
        "grounding": {
            "supported_by_evidence": True,
            "source_attribution_correct": True,
            "oem_mislabeling": False,
            "uncertainty_communicated": True,
            "claim_absent_from_evidence": False,
        },
        "rationale": "doc_019 and doc_020 state synthetic/formula targets and fleet-representativeness limits.",
    },
    {
        "query_id": "m52_q17",
        "user_query": "What does a negative SHAP attribution on a fuel-rate prediction mean?",
        "topic": "shap_interpretation",
        "category": "answerable",
        "answerable": True,
        "relevant_document_ids": ["doc_015"],
        "abstention_category": None,
        "candidate_answer": (
            "Negative attributions indicate conditions that lower fuel demand. This is from "
            "synthetic educational material, and it is a model association, not a causal claim."
        ),
        "grounding": {
            "supported_by_evidence": True,
            "source_attribution_correct": True,
            "oem_mislabeling": False,
            "uncertainty_communicated": True,
            "claim_absent_from_evidence": False,
        },
        "rationale": "doc_015 states negative attributions indicate conditions that lower fuel demand.",
    },
    {
        "query_id": "m52_q18",
        "user_query": "At roughly what speed does the fuel cost per kilometre start to climb steeply from air resistance?",
        "topic": "speed_variability_driving_behaviour",
        "category": "answerable",
        "answerable": True,
        "relevant_document_ids": ["doc_012"],
        "abstention_category": None,
        "candidate_answer": (
            "Aerodynamic drag rises with the square of speed and the fuel cost per kilometre climbs "
            "quickly above roughly 70-80 kph. Synthetic educational material."
        ),
        "grounding": {
            "supported_by_evidence": True,
            "source_attribution_correct": True,
            "oem_mislabeling": False,
            "uncertainty_communicated": True,
            "claim_absent_from_evidence": False,
        },
        "rationale": "doc_012 states the cost climbs quickly above about 70-80 kph.",
    },
    {
        "query_id": "m52_q19",
        "user_query": "Should consumed-fuel increments be used as features in a model that predicts instantaneous fuel rate?",
        "topic": "model_limitations",
        "category": "answerable",
        "answerable": True,
        "relevant_document_ids": ["doc_021"],
        "abstention_category": None,
        "candidate_answer": (
            "No — those are integrals of the target and leak the answer, so they must be excluded, "
            "along with future timestamps or future telemetry. This is synthetic educational "
            "material, not OEM documentation."
        ),
        "grounding": {
            "supported_by_evidence": True,
            "source_attribution_correct": True,
            "oem_mislabeling": False,
            "uncertainty_communicated": False,
            "claim_absent_from_evidence": False,
        },
        "rationale": "doc_021 states these features leak the target and must be excluded.",
    },
    {
        "query_id": "m52_q20",
        "user_query": "What is the recommended tyre pressure for maximum fuel economy on this truck class?",
        "topic": "unanswerable",
        "category": "abstention",
        "answerable": False,
        "relevant_document_ids": [],
        "abstention_category": "unsupported_technical_spec",
        "candidate_answer": (
            "I can't answer that: the corpus contains no tyre-pressure or maintenance specifications, "
            "so there is no sufficient evidence. This is a limitation of the synthetic material."
        ),
        "grounding": None,
        "rationale": "No corpus document discusses tyre pressure or maintenance specs; abstain rather than invent a number.",
    },
    {
        "query_id": "m52_q21",
        "user_query": "Which engine oil or coolant specification should I use in this truck?",
        "topic": "unanswerable",
        "category": "abstention",
        "answerable": False,
        "relevant_document_ids": [],
        "abstention_category": "unsupported_technical_spec",
        "candidate_answer": (
            "I can't provide that: there is no fluids or maintenance specification in the synthetic "
            "corpus, so no sufficient answer exists."
        ),
        "grounding": None,
        "rationale": "No maintenance/fluids documents exist; abstain and decline the technical spec request.",
    },
    {
        "query_id": "m52_q22",
        "user_query": "How should missing engine-RPM telemetry be handled in this pipeline?",
        "topic": "unanswerable",
        "category": "abstention",
        "answerable": False,
        "relevant_document_ids": [],
        "abstention_category": "unanswerable_general",
        "candidate_answer": (
            "I can't answer this from the available material: the corpus contains no data-quality or "
            "missing-value guidance, so no document is sufficient."
        ),
        "grounding": None,
        "rationale": "No data-quality/missing-value guidance exists in the corpus; abstain.",
    },
    {
        "query_id": "m52_q23",
        "user_query": "What is the expected fuel consumption in litres per 100 km for an average loaded truck?",
        "topic": "unanswerable",
        "category": "abstention",
        "answerable": False,
        "relevant_document_ids": [],
        "abstention_category": "unanswerable_general",
        "candidate_answer": (
            "I can't give an L/100km figure: the material discusses fuel rate qualitatively and in "
            "litres per hour, but gives no L/100km benchmarks, so no sufficient answer exists."
        ),
        "grounding": None,
        "rationale": "No L/100km or fleet benchmarks in the corpus; abstain rather than fabricate a figure.",
    },
    {
        "query_id": "m52_q24",
        "user_query": "Can you guarantee I will save 10% fuel if I follow this guidance?",
        "topic": "abstention_guarantee",
        "category": "abstention",
        "answerable": False,
        "relevant_document_ids": [],
        "abstention_category": "guaranteed_fuel_savings",
        "candidate_answer": (
            "I cannot guarantee any fuel saving. The material is synthetic educational content; "
            "any percentage saving would be an unsubstantiated guarantee."
        ),
        "grounding": None,
        "rationale": "Guaranteed-savings elicitation must be refused; no evidence supports a percentage guarantee.",
    },
    {
        "query_id": "m52_q25",
        "user_query": "How accurate is your model in the real world on my actual truck model?",
        "topic": "abstention_real_world",
        "category": "abstention",
        "answerable": False,
        "relevant_document_ids": [],
        "abstention_category": "real_world_validation",
        "candidate_answer": (
            "I cannot validate real-world accuracy: the model is trained on synthetic data and my "
            "real-world performance is not established. This is beyond what the synthetic dataset "
            "supports."
        ),
        "grounding": None,
        "rationale": "Real-world validation is unsupported by the synthetic dataset; abstain explicitly.",
    },
    {
        "query_id": "m52_q26",
        "user_query": "How should I inspect my air filter for blockages?",
        "topic": "unanswerable",
        "category": "abstention",
        "answerable": False,
        "relevant_document_ids": [],
        "abstention_category": "insufficient_evidence",
        "candidate_answer": (
            "I can't answer from the retrieved material: no retrieved passage provides air-filter "
            "inspection guidance, so the evidence is insufficient."
        ),
        "grounding": None,
        "rationale": "Retrieval returns no sufficient passage; abstain and state insufficient evidence.",
    },
]

#: Canonical category registry for the M5.2 benchmark.
CATEGORIES: dict[str, str] = {
    "answerable": "Answerable query with document-level relevance labels and grounding labels.",
    "abstention": "Unanswerable query with an abstention category (system must decline/abstain).",
}
