# M7.2 — Academic Literature Review

**Project:** Cross-Truck Fuel-Rate Prediction and Explainable Agentic Driver
Feedback for Heavy-Duty Vehicles

**Stage:** Research-paper phase — literature review only. This document situates
the study described in `docs/M7_RESEARCH_FRAMEWORK.md` within the published
literature and refines the research positioning. It contains **no new
experimental results** and makes **no claim not supported by the project's own
evidence**.

> **Citation & integrity statement.** Every reference below was verified directly
> against the Crossref metadata service on 2026-11-20 (title, authors, journal,
> volume, pages, year, DOI). Where a paper's full text was not accessible, this
> is stated and only bibliographic-level claims are made. No reference is
> invented; no abstract is treated as sufficient evidence for detailed
> methodological claims; the project's synthetic telemetry is **not** described
> as real-world data anywhere in this review.

---

## 1. Heavy-duty vehicle fuel consumption and prediction

Heavy-duty freight vehicles account for a disproportionate share of on-road fuel
use and transport-related emissions, which motivates fuel-consumption modelling
for fleet management, driver feedback, and emissions assessment (Karlsson et al.,
2026). Within this area, a distinction matters between (a) *drive-cycle
development* — constructing representative speed–time profiles used for testing
and emissions estimation — and (b) *instantaneous / trip-level fuel-consumption
prediction* — regressing fuel use on telemetry and operational features.

The reference paper (Gavimath & Ravuri, 2025) belongs to the first category. It
develops real-world drive cycles for heavy-duty freight vehicles from OBD
speed–time data collected over about a month on intercity routes in Karnataka,
India, from two trucks (42 t and 48 t). Microtrips are sequenced with a modified
random-selection approach, and the developed cycles show mean relative errors of
8.34 % and 9.63 % versus the observed data — within the conventional 10 %
acceptance limit — while differing notably from standard Indian and international
heavy-duty cycles in average/maximum speed and duration. Critically for the
present project, that study is **route-matched and truck-specific**: the same
truck and route generate the data used to characterise that vehicle. It does not
address transfer to trucks absent from training. The authors also note the study
was limited to two trucks because of data-collection constraints (low on-road
population of OBD-II-equipped trucks in India and operator unwillingness to allow
device plug-in), underscoring the practical difficulty of collecting multi-truck
real-world heavy-duty datasets.

The second category — fuel-consumption prediction — is the more relevant thread
for our work. Zhao et al. (2023) review data-driven fuel-consumption prediction
models and classify the input data into three aspects: vehicle performance,
driving behaviour, and driving environment. Their review concludes that neural
network and hybrid models tend to achieve coefficient of determination above
0.90 and root-mean-square error below 0.40 in the surveyed studies, and that
hybrid models that combine complementary learners improve accuracy. These figures
concern **in-sample or same-population** settings and are not evidence of
cross-truck transfer.

Recent primary studies on heavy-duty trucks include:

- Zhu et al. (2024) apply LightGBM to heavy-duty truck fuel consumption and
  analyse its influencing factors. *(Bibliographic details verified; full text
  of the article could not be retrieved at review time, so we report only the
  scope indicated by the title and venue rather than detailed figures.)*
- Liu et al. (2025) build CatBoost-based fuel-consumption models for heavy-duty
  diesel trucks and provide an explainable analysis of the impact of engine
  parameters, driving behaviour, and vehicle weight. *(Bibliographic details
  verified; full text not retrieved, so no numeric claims are made here.)*
- Bi et al. (2026) propose temporal feature augmentation for **transient** fuel
  consumption prediction of heavy-duty trucks, targeting the high-dynamics
  regimes that are hardest to model. This is directly relevant to our finding
  that prediction error concentrates in accelerating/highway/uphill conditions.

These works are predominantly **same-fleet** or fleet-level modelling efforts;
they do not systematically hold out entire truck identities to test
generalization to unseen vehicles.

---

## 2. Telematics-based and engine-feature-based prediction

A recurring design question is which features are necessary for accurate fuel
prediction. Many studies rely on rich engine-internal bus signals (engine speed,
load, throttle, torque, coolant temperature). The FEAD line of work exemplifies
the engine-feature-heavy approach: Liu et al. (2023) present fuel-rate prediction
for heavy-duty trucks using engine management system (EMS) telemetry (the source
of the data documented in `DATA_SOURCES.md`). The FEAD dataset includes an
instantaneous fuel-rate label (`x7001`, L/h) and a large feature set drawn from
the vehicle CAN/EMS bus.

The 2026 conversational eco-driving case study (Karlsson et al., 2026) uses
per-segment LightGBM models that predict fuel efficiency from **vehicle-level
telemetry aggregated over weekly-to-monthly windows** for two operational
segments (5XT: 50–59 t, and 7XT: 70–79 t gross combined weight). Its models reach
mean cross-validated R² of 0.796 (7XT) and 0.744 (5XT), above a linear baseline.
Note the different granularity and evaluation style (cross-validation within
segments, aggregated telemetry) compared with our per-second, truck-wise holdout
evaluation on synthetic data.

Our project contributes a controlled feature-set ablation: `full_operational`
(engine-internal signals included) versus `telematics_only` (engine-internal bus
signals removed). On the five held-out synthetic trucks, the two feature sets
produce similar held-out MAE, while SHAP importance is dominated by engine-internal
signals in the full set. The review therefore frames the feature question as an
**empirical difference comparison** (what changes — score and explanation — when
features are removed), not as a claim that either feature set is intrinsically
superior. This is consistent with the reviewed literature, which reports strong
engine-feature performance but rarely isolates the marginal contribution of
engine-internal vs. telematics-only signals under a strict unseen-truck split.

---

## 3. Cross-vehicle and cross-truck generalization

The central novelty of the present study is **cross-truck generalization**:
whether a fuel-rate model trained on some trucks predicts well for trucks absent
from training. The reference drive-cycle paper is route-matched (Gavimath &
Ravuri, 2025); the FEAD paper's defined setting is also route-matched (Liu et
al., 2023). The fleet-level modelling papers above evaluate within their
collected fleet and do not hold out entire truck identities as a systematic
protocol.

We found **little primary literature that operationalizes a strict
leave-trucks-out evaluation** for heavy-duty fuel consumption with per-truck
reporting. One adjacent direction is hierarchical/Bayesian modelling that treats
truck-level heterogeneity explicitly: Lv et al. (2025) propose a hierarchical
Bayesian approach to diesel heavy-duty truck fuel-consumption modelling from
micro- to macro-level, which inherently models between-vehicle variation rather
than reporting a single pooled curve. Such hierarchical approaches are a
complementary strategy to holdout evaluation for handling between-truck
heterogeneity, but they do not by themselves answer the question of how a model
trained on one set of trucks performs on unseen trucks.

The present study addresses this gap with a **truck-wise holdout protocol**
(20 train / 4 validation / 5 held-out test synthetic truck identities), a
leakage denylist, and a trip-clustered bootstrap for uncertainty. The evidence
map in `docs/M7_RESEARCH_FRAMEWORK.md` (§11) ties this directly to
`reports/m1/` and `reports/m3/`. Because the test set contains only five trucks,
the review is explicit that truck-level estimates are descriptive and no
population-level generalization claim is made.

---

## 4. Explainable machine learning and SHAP

Explainability is motivated by the need not only to predict but to explain
fuel-consumption behaviour. Barbado & Corcho (2022) combine unsupervised anomaly
detection with interpretable machine-learning models to explain causes of
abnormal fuel consumption in terms of feature relevance, and generate
recommendations adjusted to two user profiles (fleet managers and fleet
operators). They evaluate their explanations with XAI metrics covering
representativeness, fidelity, stability, contrastiveness, and consistency with
prior beliefs. This is a close precedent for our explanatory framing: SHAP is
used to generate feature-relevance explanations, and we explicitly report the
**non-causal** nature of the attribution.

Liu et al. (2025) and Karlsson et al. (2026) both use SHAP for heavy-duty fuel
modelling. In the 2026 case study, SHAP attributions are computed under
fleet-average and fuel-efficient backgrounds to identify the most influential
behaviours. Our study uses instance-level SHAP on held-out (unseen) synthetic
trucks, with a **reconstruction check** (the SHAP values reconstruct the model
output within tolerance). This reconstruction validation is a deliberate,
evidence-backed operationalization of "effective explanation": it guarantees the
attribution is faithful to the model, while not claiming human-judged
interpretability or causal claims. On the five held-out synthetic trucks the
reconstruction passes (max |dev| < 1.6e-4; fraction within tolerance 1.0).

---

## 5. Retrieval-augmented generation and evidence grounding

Retrieval-augmented generation (RAG) augments a generator with retrieved evidence
to ground responses and reduce hallucination. Chen et al. (2024) propose the
Retrieval-Augmented Generation Benchmark (RGB) and argue that rigorous evaluation
should cover four fundamental abilities: noise robustness, negative rejection,
information integration, and counterfactual robustness. Their evaluation of six
representative LLMs shows that while models exhibit some noise robustness, they
**struggle significantly with negative rejection and information integration**,
and with handling false information.

This benchmark has direct bearing on our agent design. Our driver-feedback agent
is deterministic and retrieval-grounded, and its evaluation is explicitly built
around the failure modes RGB highlights — most importantly **negative rejection**
(refusing to answer unanswerable/unsupported queries) and grounding responses in
a synthetic evidence corpus. The project's evidence shows the agent detects
11/11 negative controls and abstains on 7/7 unanswerable retrieval cases
(`reports/m2/agent_evaluation_report.md`, `reports/m5/m5_2/summary.md`). We do
not claim our deterministic agent matches or exceeds LLM capabilities; rather,
we use RGB's taxonomy to design the guardrail battery and to report abstention
honestly.

---

## 6. AI-based driver feedback and eco-driving assistance

Driver feedback and eco-driving coaching translate predictions/explanations into
actionable advice. The 2026 conversational AI coach (Karlsson et al., 2026) is
the closest analogue to our project and is examined in detail in §9 below. In
short, it combines LightGBM prediction, SHAP attribution, an LLM with RAG and a
persistent memory layer, producing conversational coaching in a mobile prototype
in collaboration with Volvo Group.

Other relevant background: Barbado & Corcho (2022) generate fuel-optimization
recommendations from explanations, adjusted to fleet-manager and fleet-operator
profiles — an early example of explanation-to-recommendation pipelines, though
not conversational and not at second-by-second driver feedback granularity.

Our project's contribution in this space is a **deterministic, rule-based
retrieval-grounded agent** (not an LLM) whose guardrails enforce
non-fabrication, non-causal language, and abstention. It is evaluated on a
manually authored deterministic benchmark (`reports/m2/`, `reports/m5/`) rather
than human subjects. We do not claim the agent is an LLM, autonomous, or
validated by human evaluators.

---

## 7. Evaluation of safety, reliability and unsupported responses

A recurring theme across both RAG and conversational-AI-for-fleets literature is
reliability — especially avoiding fabrication and unsupported advice. Chen et
al. (2024) show LLMs are weak at negative rejection and information integration,
which is precisely where safety-critical driver advice could go wrong. The 2026
eco-driving case study found that **retrieval grounding did not consistently
improve feedback** in its expert evaluation, and that personalization helped only
when the added context was relevant and technically sound — evidence that
grounding alone is not sufficient for reliable advice.

Our agent evaluation explicitly targets this: the benchmark includes negative
controls and unanswerable retrieval cases, and the agent is required to abstain
rather than fabricate. This is a deliberately narrow, engineering-level notion of
reliability; it is **not** a human-subject study and claims no real-world
reliability. The review aligns our evaluation design with the failure modes
documented in RGB (Chen et al., 2024).

---

## 8. Research gap and positioning of the present study

Synthesizing the above:

1. **Drive-cycle development** (Gavimath & Ravuri, 2025) is route-matched and
   truck-specific; it does not test transfer to unseen trucks.
2. **Fleet-level fuel prediction** (Zhao et al., 2023; Zhu et al., 2024; Liu et
   al., 2025; Bi et al., 2026) is accurate within collected populations and is
   increasingly explainable, but does not systematically hold out entire truck
   identities on **cross-truck generalization**.
3. **Explainable/attribution work** (Barbado & Corcho, 2022; Liu et al., 2025)
   validates explanation quality but not under a strict unseen-truck split.
4. **Conversational eco-driving with RAG** exists (Karlsson et al., 2026) and
   combines prediction + SHAP + RAG + coaching — so we are **not** the first to
   combine these technologies.

The **narrow, evidence-supported contribution** of the present study is:

> *A reproducible pipeline-validation study that isolates **cross-truck
> generalization** of instantaneous fuel-rate prediction (truck-wise holdout on
> synthetic data), pairs it with **reconstruction-validated instance-level SHAP**
> on unseen trucks, and evaluates a **deterministic retrieval-grounded agent**
> whose guardrails enforce abstention — reported explicitly as a synthetic-data
> pipeline validation, not as real-world validation.*

This is a *methodological/pipeline* contribution on synthetic telemetry. It does
not claim real-world predictive validity, first use of SHAP/RAG/coaching, or
LLM-level reasoning. Its distinctiveness versus the 2026 coach study is detailed
in §9.

---

## 9. Direct comparison with the 2026 conversational eco-driving study (Karlsson et al., 2026)

Because this paper is the closest analogue, we compare it carefully with our
project. The comparison is grounded in the verified bibliographic record and the
paper's abstract (retrieved from OpenAlex on 2026-11-20); the full text was not
accessible, so only claims supported by the abstract are attributed to it.

| Aspect | Karlsson et al. (2026) | Present study |
| --- | --- | --- |
| Data | Real vehicle-level telemetry aggregated over weekly–monthly windows, two segments (5XT, 7XT), real Volvo Group fleet | Synthetic, formula-generated per-second telemetry, 29 truck identities (20/4/5 split) |
| Prediction target | Fuel *efficiency* over aggregated windows, per segment | Instantaneous `fuel_rate_lph` (L/h), per row |
| Model | LightGBM, per-segment; cross-validated R² 0.796/0.744 | XGBoost; truck-wise holdout; held-out synthetic MAE ≈ 2.57 L/h (R² ≈ 0.988–0.989) |
| Generalization evaluation | Cross-validation within segments; telemetry aggregated per vehicle (not per driver) | Strict leave-trucks-out holdout on unseen truck identities |
| Explainability | SHAP under fleet-average and fuel-efficient backgrounds | Instance-level SHAP on unseen trucks with reconstruction validation |
| Generation | LLM + RAG + persistent memory, conversational mobile prototype | Deterministic rule-based retrieval-grounded agent with abstention guardrails |
| Evaluation of agent | Five domain experts; retrieval grounding did not consistently improve feedback; personalization helped only when relevant/technically sound; experts requested more concise presentation | Manually authored deterministic benchmark + engineering ablation; abstention on negative/unanswerable cases |
| Author-acknowledged limits | Small expert panel; per-vehicle (not per-driver) aggregation | Synthetic data; five-truck test set; no human subjects; M5.5 not formally closed; M6.5 video incomplete |

**Interpretation for positioning.** The 2026 study demonstrates a real integration
of prediction, SHAP, RAG and conversational coaching — we explicitly do not claim
to be first. Its open questions align with our focus: it does not evaluate
cross-truck generalization (its telemetry is aggregated per vehicle and evaluated
via cross-validation), and its expert evaluation found grounding did not
consistently help. Our project targets the cross-truck generalization gap and a
deterministic abstention-oriented agent; the 2026 study targets real-data
integration and expert-perceived usefulness. The two are complementary rather
than overlapping claims.

---

## 10. Remaining literature gaps and unverified items

**Gaps the review could not close (honest limits):**

- **Full text for Zhu et al. (2024) and Liu et al. (2025).** Bibliographic
  details and venue were verified via Crossref, but the article full texts were
  not accessible at review time; we therefore make **no detailed methodological
  or numeric claims** from these two papers and cite them only for scope/venue.
- **Large-scale leave-trucks-out fuel-prediction literature is sparse.** We did
  not locate a primary study that reports a strict truck-wise holdout for
  heavy-duty instantaneous fuel rate with per-truck results; our own work is
  positioned as a synthetic pipeline validation in that gap. The absence of
  comparable real-data studies is itself documented here as a gap, not used to
  claim we fill it with real data.
- **Human-subject evaluation of conversational driver feedback** exists in the
  2026 study but not in ours; the review records this as a limitation of our
  evidence.

**References that could not be independently verified / not used:**

- The original request listed seven starting references; all seven were verified
  against Crossref. No reference was rejected for failing verification.
- The FEAD source paper (Liu et al., 2023) was verified and is cited because it
  is the documented origin of the data lineage in `DATA_SOURCES.md`, with the
  explicit caveat that our modelling uses the **synthetic** suite, not the
  downloaded FEAD archive.

---

## 11. Reference list (all verified against Crossref, 2026-11-20)

1. Barbado, A., Corcho, Ó. (2022). Interpretable machine learning models for
   predicting and explaining vehicle fuel consumption anomalies. *Engineering
   Applications of Artificial Intelligence*, 115, 105222.
   DOI: 10.1016/j.engappai.2022.105222.
2. Bi, D., Song, D., Zeng, X., Liu, Z., Duan, C. (2026). Temporal feature
   augmentation for transient fuel consumption prediction of heavy-duty trucks:
   Toward sustainable road freight and cleaner operations. *Journal of Cleaner
   Production*, 582, 149773. DOI: 10.1016/j.jclepro.2026.149773.
3. Chen, J., Lin, H., Han, X., Sun, L. (2024). Benchmarking Large Language
   Models in Retrieval-Augmented Generation. *Proceedings of the AAAI
   Conference on Artificial Intelligence*, 38(16), 17754–17762.
   DOI: 10.1609/aaai.v38i16.29728.
4. Gavimath, V.V., Ravuri, S. (2025). Developing Real-World Drive Cycles for
   Heavy-Duty Freight Vehicles Using On-Board Diagnostics Data.
   *Transportation Research Procedia*, 82, 158–174.
   DOI: 10.1016/j.trpro.2024.12.035.
5. Karlsson, J., Lundberg, K., Liu, H.X. (2026). An end-to-end conversational
   AI coach for heavy-duty eco-driving: An industrial case study.
   *Transportation Research Interdisciplinary Perspectives*, 40, 102270.
   DOI: 10.1016/j.trip.2026.102270.
6. Liu, A., Fan, P., He, X., Lu, H., Yu, L., Song, G. (2025). CatBoost-based
   fuel consumption modeling and explainable analysis for heavy-duty diesel
   trucks: Impact of engine, driving behavior, and vehicle weight. *Energy*,
   333, 137485. DOI: 10.1016/j.energy.2025.137485.
7. Liu, L., Li, W., Wang, D., Wu, Y., Yang, R., Shi, W. (2023). Fuel Rate
   Prediction for Heavy-Duty Trucks. *IEEE Transactions on Intelligent
   Transportation Systems*, 24(8), 8222–8235.
   DOI: 10.1109/TITS.2023.3265007.
8. Lv, C., Jin, Y., Liu, H. (2025). Micro- to Macro-Level Fuel Consumption
   Modeling of Diesel Heavy-Duty Trucks: A Hierarchical Bayesian Approach.
   *Transportation Research Record: Journal of the Transportation Research
   Board*, 2680(10), 1151–1171. DOI: 10.1177/03611981251368833.
9. Zhao, D., Li, H., Hou, J., Gong, P., Zhong, Y., He, W., Fu, Z. (2023). A
   Review of the Data-Driven Prediction Method of Vehicle Fuel Consumption.
   *Energies*, 16(14), 5258. DOI: 10.3390/en16145258.
10. Zhu, X., Shen, X., Chen, K., Zhang, Z. (2024). Research on the prediction
    and influencing factors of heavy duty truck fuel consumption based on
    LightGBM. *Energy*, 296, 131221. DOI: 10.1016/j.energy.2024.131221.

---

_End of M7.2 literature review. All references verified via Crossref; no new
experimental results, no invented citations, no real-world claims. The review is
positioning only, consistent with `docs/M7_RESEARCH_FRAMEWORK.md`._
