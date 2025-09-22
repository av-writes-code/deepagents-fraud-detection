"""LangGraph-based orchestrator for the Agentic ID Fraud pilot."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

try:  # pragma: no cover - optional dependency
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import START, END, StateGraph
    from langgraph.graph.state import CompiledStateGraph
except ModuleNotFoundError:  # pragma: no cover - fallback shim for test envs
    from ._graph_shim import CompiledStateGraph, MemorySaver, START, END, StateGraph

from .audit import AuditRenderer
from .config_loader import PilotConfig
from .state import CaseState, OCRResult, PADResult, QualityMetrics
from .tools import Toolset

logger = logging.getLogger(__name__)


class AgenticIDFraudOrchestrator:
    """DeepAgents-style orchestrator compiled as a LangGraph workflow."""

    def __init__(
        self,
        toolset: Toolset,
        config: PilotConfig,
        audit_renderer: AuditRenderer,
    ) -> None:
        self.toolset = toolset
        self.config = config
        self.audit_renderer = audit_renderer
        self.checkpointer = MemorySaver()
        self.workflow = self._build_workflow()
        self._runtime_recapture_supplier: Optional[
            Callable[[CaseState], Sequence[str]]
        ] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_case(
        self,
        images: List[str],
        meta: Dict[str, Any],
        *,
        extra_images: Optional[Iterable[str]] = None,
        recapture_supplier: Optional[Callable[[CaseState], Sequence[str]]] = None,
    ) -> CaseState:
        """Execute the orchestrator for a single case via LangGraph."""

        state = CaseState(images=list(images), meta=dict(meta))
        if state.images:
            state.active_image = state.images[0]
            state.capture_history.append(state.active_image)
        if len(state.images) > 1:
            state.queue_recaptures(state.images[1:])

        recapture_candidates = state.meta.pop("recapture_candidates", None)
        if isinstance(recapture_candidates, Iterable):
            state.queue_recaptures(recapture_candidates)
        if extra_images:
            state.queue_recaptures(extra_images)

        thread_id = str(state.meta.get("case_id", f"case-{id(state)}"))
        config = {"configurable": {"thread_id": thread_id}}

        logger.debug("Starting LangGraph orchestrator thread %s", thread_id)

        self._runtime_recapture_supplier = recapture_supplier
        try:
            return self.workflow.invoke(state, config=config)
        finally:
            self._runtime_recapture_supplier = None

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------
    def _build_workflow(self) -> CompiledStateGraph:
        workflow = StateGraph(CaseState)

        workflow.add_node("capture_ocr", self._node_capture_and_ocr)
        workflow.add_node("pad_primary", self._node_pad_primary)
        workflow.add_node("uncertainty_gate", self._node_uncertainty_gate)
        workflow.add_node("recapture", self._node_recapture)
        workflow.add_node("second_opinion", self._node_second_opinion)
        workflow.add_node("risk_policy", self._node_risk_policy)
        workflow.add_node("audit", self._node_audit)

        workflow.add_edge(START, "capture_ocr")
        workflow.add_edge("capture_ocr", "pad_primary")
        workflow.add_edge("pad_primary", "uncertainty_gate")

        workflow.add_conditional_edges(
            "uncertainty_gate",
            self._route_after_gate,
            {
                "recapture": "recapture",
                "second_opinion": "second_opinion",
                "risk_policy": "risk_policy",
            },
        )

        workflow.add_edge("recapture", "pad_primary")
        workflow.add_edge("second_opinion", "risk_policy")
        workflow.add_edge("risk_policy", "audit")
        workflow.add_edge("audit", END)

        return workflow.compile(checkpointer=self.checkpointer)

    # ------------------------------------------------------------------
    # Graph nodes
    # ------------------------------------------------------------------
    def _node_capture_and_ocr(self, state: CaseState) -> CaseState:
        self._capture_and_ocr(state)
        return state

    def _node_pad_primary(self, state: CaseState) -> CaseState:
        self._run_pad_primary(state)
        return state

    def _node_uncertainty_gate(self, state: CaseState) -> CaseState:
        self._evaluate_uncertainty(state)
        return state

    def _node_recapture(self, state: CaseState) -> CaseState:
        self._perform_recapture(state)
        return state

    def _node_second_opinion(self, state: CaseState) -> CaseState:
        self._run_second_opinion(state)
        return state

    def _node_risk_policy(self, state: CaseState) -> CaseState:
        self._run_risk_policy(state)
        return state

    def _node_audit(self, state: CaseState) -> CaseState:
        self._render_audit(state)
        return state

    # ------------------------------------------------------------------
    # Node implementations
    # ------------------------------------------------------------------
    def _capture_and_ocr(self, state: CaseState) -> None:
        payload = self._select_capture_payload(state)
        state.meta["active_image"] = state.active_image

        try:
            doc_output = self.toolset.doc_detect_rectify(payload, state.meta)
        except Exception as exc:  # pragma: no cover - defensive guard
            message = f"DocDetectRectifyTool failed: {exc}"
            logger.exception(message)
            state.record_error(message)
            return

        if doc_output.get("status") == "error":
            state.record_error(doc_output.get("error", "Unknown capture error"))
            return

        canonical_image = doc_output.get("canonical_image") or doc_output.get("image_rectified")
        if canonical_image:
            state.canonical = canonical_image
            state.meta["canonical_image"] = canonical_image
        else:
            state.record_error("Capture tool did not return a canonical image")

        quality_metrics = doc_output.get("quality_metrics") or {}
        if quality_metrics:
            state.quality = QualityMetrics(
                blur=float(quality_metrics.get("blur", 0.0)),
                glare=float(quality_metrics.get("glare", 0.0)),
                perspective=float(quality_metrics.get("perspective", 0.0)),
                score=float(quality_metrics.get("score", quality_metrics.get("quality", 0.0))),
            )
            state.meta["quality_score"] = state.quality.score

        evidence = doc_output.get("evidence") or {}
        if "corners" in evidence:
            state.evidence.corners = evidence["corners"]
        if "crops" in evidence:
            state.evidence.crops.extend(evidence["crops"])

        if not state.canonical:
            return

        try:
            ocr_output = self.toolset.ocr(state.canonical, state.meta)
        except Exception as exc:  # pragma: no cover - defensive guard
            message = f"OCRTool failed: {exc}"
            logger.exception(message)
            state.record_error(message)
            return

        if ocr_output.get("status") == "error":
            state.record_error(ocr_output.get("error", "Unknown OCR error"))
            return

        state.ocr = OCRResult(
            fields=ocr_output.get("fields") or {},
            conf=float(ocr_output.get("conf", ocr_output.get("confidence", 0.0))),
        )
        state.meta["ocr_conf"] = state.ocr.conf

    def _run_pad_primary(self, state: CaseState) -> None:
        if not state.canonical:
            state.record_error("Cannot run PAD without canonical image")
            return

        try:
            pad_output = self.toolset.pad_primary(state.canonical, state.meta)
        except Exception as exc:  # pragma: no cover - defensive guard
            message = f"PADTool failure: {exc}"
            logger.exception(message)
            state.record_error(message)
            return

        if pad_output.get("status") == "error":
            state.record_error(pad_output.get("error", "Unknown PAD error"))
            return

        state.pad = PADResult(
            score=float(pad_output.get("score", 0.0)),
            label=pad_output.get("label"),
        )
        uncertainty = pad_output.get("uncertainty") or {}
        state.meta.setdefault("uncertainty", {}).update(uncertainty)

        if "tta_var" in pad_output:
            state.meta["tta_var"] = float(pad_output["tta_var"])
        if "frame_agree" in pad_output:
            state.meta["frame_agree"] = float(pad_output["frame_agree"])

    def _evaluate_uncertainty(self, state: CaseState) -> None:
        thresholds = self.config.thresholds
        actions = self.config.actions
        uncertainty_conf = self.config.uncertainty
        band = self.config.borderline_band

        recapture_trigger = False
        if state.quality and state.quality.score < float(thresholds.get("quality_min", 0.0)):
            recapture_trigger = True
        if state.ocr and state.ocr.conf < float(thresholds.get("ocr_conf_low", 0.0)):
            recapture_trigger = True

        recapture_allowed = actions.get("recapture_once", False) and state.recapture_attempts < 1
        recapture_needed = recapture_trigger and recapture_allowed

        needs_second = False
        if state.pad:
            pad_score = state.pad.score
            in_band = band["lower"] <= pad_score <= band["upper"]
            tta_var = float(
                state.meta.get("tta_var", state.meta.get("uncertainty", {}).get("tta_var", 0.0))
            )
            frame_agree = float(
                state.meta.get("frame_agree", state.meta.get("uncertainty", {}).get("frame_agree", 1.0))
            )
            if in_band:
                needs_second = True
            if tta_var > float(uncertainty_conf.get("tta_variance_max", 1.0)):
                needs_second = True
            if frame_agree < float(uncertainty_conf.get("frame_agreement_min", 0.0)):
                needs_second = True

        second_allowed = actions.get("second_opinion_on_borderline", False) and state.pad is not None
        second_needed = needs_second and second_allowed

        decision = "risk_policy"
        if recapture_needed:
            decision = "recapture"
        elif second_needed:
            decision = "second_opinion"

        state.gate = {
            "quality_trigger": recapture_trigger,
            "recapture_allowed": recapture_allowed,
            "second_trigger": needs_second,
            "second_allowed": second_allowed,
            "next": decision,
        }
        state.meta["gate"] = state.gate

    def _route_after_gate(self, state: CaseState) -> str:
        next_step = state.gate.get("next", "risk_policy")
        if next_step == "recapture":
            return "recapture"
        if next_step == "second_opinion":
            return "second_opinion"
        return "risk_policy"

    def _perform_recapture(self, state: CaseState) -> None:
        state.increment_recapture()
        state.meta.setdefault("events", []).append("recapture_triggered")
        next_image = self._next_recapture_image(state)
        if next_image:
            state.active_image = next_image
            state.capture_history.append(next_image)
        else:
            state.meta.setdefault("events", []).append("recapture_reused_original")
        self._capture_and_ocr(state)

    def _run_second_opinion(self, state: CaseState) -> None:
        if not state.canonical:
            return

        try:
            pad_output = self.toolset.pad_secondary(state.canonical, state.meta)
        except Exception as exc:  # pragma: no cover - defensive guard
            message = f"PADToolAlt failure: {exc}"
            logger.exception(message)
            state.record_error(message)
            return

        if pad_output.get("status") == "error":
            state.record_error(pad_output.get("error", "Unknown PAD secondary error"))
            return

        state.pad_secondary = PADResult(
            score=float(pad_output.get("score", 0.0)),
            label=pad_output.get("label"),
        )

        fusion_mode = self.config.fusion.get("pad_final", "max")
        if state.pad is None:
            state.pad = state.pad_secondary
        elif fusion_mode == "max":
            state.pad = PADResult(
                score=max(state.pad.score, state.pad_secondary.score),
                label=state.pad.label or state.pad_secondary.label,
            )
        else:
            state.pad = PADResult(
                score=state.pad_secondary.score,
                label=state.pad_secondary.label,
            )

        state.meta.setdefault("events", []).append("second_opinion_triggered")

    def _run_risk_policy(self, state: CaseState) -> None:
        try:
            features = self.toolset.feature_builder(state)
        except Exception as exc:  # pragma: no cover - defensive guard
            message = f"FeatureBuilderTool failure: {exc}"
            logger.exception(message)
            state.record_error(message)
            features = []
        state.features = features

        try:
            classifier_output = self.toolset.classifier(features)
        except Exception as exc:  # pragma: no cover - defensive guard
            message = f"ClassifierTool failure: {exc}"
            logger.exception(message)
            state.record_error(message)
            classifier_output = {"value": 1.0, "reasons": ["classifier_error"]}

        risk_value = float(classifier_output.get("value", 1.0))
        state.risk = risk_value
        state.meta.setdefault("reasons", []).extend(classifier_output.get("reasons", []))

        pad_score = state.pad.score if state.pad else 1.0
        thresholds = self.config.thresholds
        pad_threshold = float(thresholds.get("pad_review", 1.0))
        risk_threshold = float(thresholds.get("risk_review", 1.0))

        if pad_score < pad_threshold and risk_value < risk_threshold:
            state.decision = "approve"
        else:
            state.decision = "review"

    def _render_audit(self, state: CaseState) -> None:
        try:
            report_payload = self.toolset.reporter(state)
        except Exception as exc:  # pragma: no cover - defensive guard
            message = f"ReporterTool failure: {exc}"
            logger.exception(message)
            state.record_error(message)
            report_payload = {}

        report_uri = report_payload.get("report_uri")
        if not report_uri:
            metadata = {
                "thresholds": self.config.thresholds,
                "actions": self.config.actions,
            }
            report_uri = self.audit_renderer.render(state, metadata)

        state.report_uri = report_uri

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _select_capture_payload(self, state: CaseState) -> List[str]:
        if state.active_image:
            return [state.active_image]
        if state.images:
            return [state.images[0]]
        return []

    def _next_recapture_image(self, state: CaseState) -> Optional[str]:
        if state.recapture_queue:
            return state.recapture_queue.pop(0)

        if self._runtime_recapture_supplier is None:
            return None

        try:
            candidates = self._runtime_recapture_supplier(state)
        except Exception as exc:  # pragma: no cover - defensive guard
            message = f"Recapture supplier failure: {exc}"
            logger.exception(message)
            state.record_error(message)
            return None

        if not candidates:
            return None

        head, *tail = list(candidates)
        state.queue_recaptures(tail)
        return head