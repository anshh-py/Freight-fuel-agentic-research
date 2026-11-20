from src.agent.orchestrator import AgentOrchestrator
from src.agent.response_generator import ResponseGenerator

orch = AgentOrchestrator()
resp = orch.handle("why does road grade affect fuel consumption on trucks")
g = ResponseGenerator().generate(resp)
print("FULL K01 driver_version:")
print(g.driver_version)
