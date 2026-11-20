from copy import deepcopy
from src.agent.orchestrator import AgentOrchestrator
from tests.test_tools import FULL_INPUTS

orch = AgentOrchestrator()
missing = deepcopy(FULL_INPUTS)
del missing["speed_kph"]
resp = orch.handle("predict fuel consumption", missing)
print("intent:", resp.intent)
print("selected_tools:", resp.selected_tools)
for r in resp.tool_results:
    print("tool:", r.get("tool"), "status:", r.get("status"))
    if r.get("status") != "ok":
        print("   error:", r.get("error"))
print("limitations:", resp.limitations)
