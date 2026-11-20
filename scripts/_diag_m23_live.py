"""Temp read-only diag: live-generate real scenarios and find matched causal patterns."""
import re
import sys
sys.path.insert(0, ".")

from src.agent.intents import Intent
from src.agent.orchestrator import AgentOrchestrator
from src.agent.response_generator import ResponseGenerator, CAUSAL_DISCLAIMER, CORPUS_DISCLAIMER, SYNTHETIC_DISCLAIMER
from src.agent.response_validation import _CAUSAL_PATTERNS
from tests.test_tools import FULL_INPUTS

queries = {
    "K01": ("why does road grade affect fuel consumption on trucks", None),
    "E01": None,  # synthetic, handled below
    "C01": ("predict fuel consumption and explain why engine load affects it", FULL_INPUTS),
    "I01": ("why does road grade affect fuel consumption", None),
}
orch = AgentOrchestrator()
for k, (q, inp) in queries.items():
    if k == "I01":
        from src.tools.knowledge_retrieval import KnowledgeRetrievalTool
        from src.tools.fuel_rate_prediction import FuelRatePredictionTool
        from src.tools.prediction_explanation import PredictionExplanationTool
        tools = {"fuel_rate_prediction": FuelRatePredictionTool(),
                 "knowledge_retrieval": KnowledgeRetrievalTool(min_score=0.99),
                 "prediction_explanation": PredictionExplanationTool()}
        orch = AgentOrchestrator(tools=tools)
    if k == "E01":
        from src.tools.fuel_rate_prediction import FuelRatePredictionTool
        from src.tools.prediction_explanation import PredictionExplanationTool
        from src.agent.orchestrator import AgentResponse
        p = FuelRatePredictionTool().predict(FULL_INPUTS, "full_operational")
        e = PredictionExplanationTool().explain(FULL_INPUTS, "full_operational")
        r = AgentResponse()
        r.intent = Intent.COMBINED_PREDICT_EXPLAIN
        r.selected_tools = ["fuel_rate_prediction", "prediction_explanation"]
        r.tool_results = [{"tool": "fuel_rate_prediction", "status": "ok", "result": p},
                          {"tool": "prediction_explanation", "status": "ok", "result": e}]
        r.router = None
        r.limitations = []
    else:
        r = orch.handle(q, inp)
    g = ResponseGenerator(inputs=FULL_INPUTS).generate(r)
    text = g.driver_version
    print("#" * 40, k, "#" * 40)
    print(text)
    print("---- causal checks ----")
    work = text
    for d in (CAUSAL_DISCLAIMER, CORPUS_DISCLAIMER, SYNTHETIC_DISCLAIMER):
        work = work.replace(d, "")
    for p in _CAUSAL_PATTERNS:
        m = p.search(work)
        if m:
            s = max(0, m.start() - 60)
            print(f"   MATCH {p.pattern!r} -> ...{work[s:m.end()+20]!r}")
