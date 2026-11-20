from src.agent.orchestrator import AgentOrchestrator
from src.agent.response_generator import ResponseGenerator
from src.agent.response_validation import check_causal_language, _CAUSAL_PATTERNS

orch = AgentOrchestrator()
resp = orch.handle("why does road grade affect fuel consumption on trucks")
g = ResponseGenerator().generate(resp)
print("causal check:", check_causal_language(g.driver_version, explanation_present=False))
for p in _CAUSAL_PATTERNS:
    m = p.search(g.driver_version)
    if m:
        print("MATCHED:", p.pattern, "->", repr(m.group(0)))
