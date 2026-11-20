"""M6.1 — thin application gateway around ``DriverConversationService``.

The gateway is an *adapter only*. It does not redesign or re-implement any
part of the frozen research pipeline — it wraps the existing conversation
service, session, router, tools, generator, and validation battery behind a
stable, UI-friendly contract.

Key guarantees:

* feature set is fixed per session (never changed mid-session),
* prediction/explanation run only after the existing strict input validation
  succeeds (delegated to the service/session — never bypassed),
* knowledge queries never trigger prediction,
* every delivered response is the validated ``driver_feedback`` (never an
  unvalidated string), and no failure is presented as a success,
* SHAP technical output is kept in ``technical_report``, separated from the
  driver-facing feedback,
* boundary errors are surfaced without stack traces, secrets, or internal
  paths,
* deterministic and offline — no LLM, no external API, no database.

All telemetry, prediction targets, and the retrieval corpus are synthetic. This
is not a real-world truck validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

from src.agent.driver_conversation import (
    DriverConversationResponse,
    DriverConversationService,
)
from src.agent.driver_session import DriverSession
from src.features.spec import get_feature_set, load_feature_spec
from src.tools.fuel_rate_prediction import FuelRatePredictionTool
from src.tools.knowledge_retrieval import KnowledgeRetrievalTool
from src.tools.prediction_explanation import PredictionExplanationTool

from src.api.schemas import APIResponse, api_error_response
from src.api.session_registry import (
    SessionEntry,
    SessionExpiredError,
    SessionNotFoundError,
    SessionRegistry,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "agent_config.json"

VALID_FEATURE_SETS = frozenset({"full_operational", "telematics_only"})


class GatewayError(RuntimeError):
    """Base error for gateway-level failures."""


class InvalidInputError(GatewayError):
    """The request itself is malformed (non-empty query / inputs shape)."""


class GatewayStartupError(GatewayError):
    """Raised when required artifacts/configuration are not available."""


def _resolve_path(raw: Any, base: Path = PROJECT_ROOT) -> Path:
    p = Path(str(raw))
    if not p.is_absolute():
        p = base / p
    return p.resolve()


class FuelRateFeedbackGateway:
    """Application-level adapter over ``DriverConversationService``."""

    def __init__(
        self,
        service: DriverConversationService | None = None,
        registry: SessionRegistry | None = None,
        config_path: Path | None = None,
        *,
        skip_preflight: bool = False,
        ttl_seconds: float = 1800.0,
    ) -> None:
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.registry = registry or SessionRegistry(ttl_seconds=ttl_seconds)
        self.service = service or self._build_service()
        if not skip_preflight:
            self.preflight()

    # -- config / construction -----------------------------------------
    def load_config(self) -> dict[str, Any]:
        if not self.config_path.exists():
            raise GatewayStartupError(f"agent config not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _build_service(self) -> DriverConversationService:
        cfg = self.load_config()
        tools_cfg = cfg.get("tools", {})

        def _model_dir(key: str) -> Path:
            raw = (tools_cfg.get(key) or {}).get("model_artifacts_dir", "reports/m1/artifacts")
            return _resolve_path(raw)

        predict_cfg = tools_cfg.get("fuel_rate_prediction") or {}
        retrieval_cfg = tools_cfg.get("knowledge_retrieval") or {}
        explain_cfg = tools_cfg.get("prediction_explanation") or {}

        tools: dict[str, Any] = {
            "fuel_rate_prediction": FuelRatePredictionTool(
                model_dir=_model_dir("fuel_rate_prediction"),
            ),
            "knowledge_retrieval": KnowledgeRetrievalTool(
                corpus_path=str(_resolve_path(
                    retrieval_cfg.get("corpus_path", "data/synthetic_heavy_truck_research/rag_knowledge_corpus.jsonl")
                )),
                top_k=int(retrieval_cfg.get("top_k", 5)),
                min_score=float(retrieval_cfg.get("min_score_floor", 0.45)),
            ),
            "prediction_explanation": PredictionExplanationTool(
                model_dir=_model_dir("prediction_explanation"),
                top_features=int(explain_cfg.get("top_features", 5)),
            ),
        }
        return DriverConversationService(tools=tools)

    # -- artifact preflight --------------------------------------------
    def preflight(self) -> list[str]:
        """Check that required artifact paths exist; raise on missing ones.

        Model artifacts are loaded lazily by the tools (loading is heavy); here
        we verify the directories/files exist so startup fails fast with a clear
        message. Deep load errors still surface as tool errors at request time.
        """
        cfg = self.load_config()
        tools_cfg = cfg.get("tools", {})
        problems: list[str] = []

        for key in ("fuel_rate_prediction", "prediction_explanation"):
            raw = (tools_cfg.get(key) or {}).get("model_artifacts_dir")
            if raw:
                d = _resolve_path(raw)
                if not d.exists():
                    problems.append(f"model artifact directory missing for {key}: {d}")

        corpus = (tools_cfg.get("knowledge_retrieval") or {}).get("corpus_path")
        if corpus:
            c = _resolve_path(corpus)
            if not c.exists():
                problems.append(f"corpus file missing: {c}")

        if problems:
            raise GatewayStartupError("artifact preflight failed: " + "; ".join(problems))
        return []

    # -- session lifecycle ---------------------------------------------
    def start_session(self, feature_set: str = "full_operational") -> str:
        """Create a session pinned to ``feature_set``; return its session id."""
        fs = self._normalize_feature_set(feature_set)
        return self.registry.create(feature_set=fs)

    def session_state(self, session_id: str) -> dict[str, Any]:
        """Return the session's current serializable state."""
        session = self._require_session(session_id)
        return session.to_dict()

    def refresh_session(self, session_id: str) -> None:
        """Touch a session so it is not reaped (optional keep-alive)."""
        self.registry.get(session_id)

    # -- request handling ----------------------------------------------
    def send(
        self,
        session_id: str,
        query: str,
        structured_inputs: Mapping[str, Any] | None = None,
    ) -> APIResponse:
        """Process one driver turn and return a serializable ``APIResponse``."""
        # --- session lookup (missing/expired -> clean boundary error) ---
        try:
            entry = self.registry.get(session_id)
        except SessionExpiredError:
            return api_error_response(session_id, code="session_expired", message="session expired")
        except SessionNotFoundError:
            return api_error_response(session_id, code="session_not_found", message="unknown session id")

        # --- request validation -----------------------------------------
        try:
            query = self._check_query(query)
            inputs = self._check_inputs(structured_inputs)
        except InvalidInputError as exc:
            return api_error_response(session_id, code="invalid_input", message=str(exc))

        # --- per-session serialization ----------------------------------
        with entry.lock:
            try:
                resp = self.service.handle(entry.session, query, inputs)
            except GatewayError as exc:
                # Boundary-normalized gateway errors (never internal secrets).
                return api_error_response(
                    session_id,
                    code="gateway_error",
                    message=str(exc) or "gateway error",
                )
            except Exception:  # noqa: BLE001 - boundary catch-all
                # Never leak stack traces, paths, or secrets to callers.
                return api_error_response(
                    session_id,
                    code="internal_error",
                    message="internal processing error; please retry",
                )

        return self._to_api_response(entry, resp)

    # -- helpers --------------------------------------------------------
    def _normalize_feature_set(self, feature_set: str) -> str:
        if feature_set not in VALID_FEATURE_SETS:
            spec = load_feature_spec()
            try:
                get_feature_set(spec, feature_set)
            except ValueError:
                raise InvalidInputError(
                    f"unknown feature set {feature_set!r}; expected one of "
                    f"{sorted(VALID_FEATURE_SETS)}"
                ) from None
        return feature_set

    def _require_session(self, session_id: str) -> DriverSession:
        try:
            return self.registry.get(session_id).session
        except SessionExpiredError as exc:
            raise InvalidInputError(str(exc)) from exc
        except SessionNotFoundError as exc:
            raise InvalidInputError(str(exc)) from exc

    @staticmethod
    def _check_query(query: Any) -> str:
        if not isinstance(query, str) or not query.strip():
            raise InvalidInputError("query must be a non-empty string")
        return query.strip()

    @staticmethod
    def _check_inputs(structured_inputs: Any) -> dict[str, Any] | None:
        if structured_inputs is None:
            return None
        if not isinstance(structured_inputs, Mapping):
            raise InvalidInputError("structured_inputs must be a mapping or null")
        return dict(structured_inputs)

    @staticmethod
    def _to_api_response(entry: SessionEntry, resp: DriverConversationResponse) -> APIResponse:
        issues = tuple(resp.validation_issues or ())
        failed_check_ids = sorted({i.check_id for i in issues if not i.passed})
        technical = dict(resp.technical_report or {})
        technical.setdefault("synthetic_only", True)
        technical["tool_failures"] = [
            r.get("tool") for r in resp.tool_results if r.get("status") != "ok"
        ]
        technical.setdefault("validation", {
            "failed_checks": failed_check_ids,
        })
        return APIResponse(
            session_id=resp.session_id,
            driver_intent=resp.driver_intent,
            conversation_state=resp.conversation_state,
            delivered=bool(resp.delivered),
            validation_blocked=bool(resp.validation_blocked),
            driver_feedback=resp.driver_feedback or "",
            clarification_request=resp.clarification_request,
            missing_fields=list(resp.missing_fields),
            invalid_fields=dict(resp.invalid_fields),
            selected_tools=list(resp.selected_tools),
            executed_tools=list(resp.executed_tools),
            technical_report=technical,
            error=None,
        )
