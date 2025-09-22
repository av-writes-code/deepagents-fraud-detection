"""Tool adapter interfaces for the Agentic ID Fraud orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from .state import CaseState


DocDetectRectify = Callable[[List[str], Dict[str, Any]], Dict[str, Any]]
OCRTool = Callable[[str, Dict[str, Any]], Dict[str, Any]]
PADTool = Callable[[str, Dict[str, Any]], Dict[str, Any]]
FeatureBuilderTool = Callable[[CaseState], List[float]]
ClassifierTool = Callable[[List[float]], Dict[str, Any]]
ReporterTool = Callable[[CaseState], Dict[str, Any]]


@dataclass
class Toolset:
    """Collection of callables used by the orchestrator."""

    doc_detect_rectify: DocDetectRectify
    ocr: OCRTool
    pad_primary: PADTool
    pad_secondary: PADTool
    feature_builder: FeatureBuilderTool
    classifier: ClassifierTool
    reporter: ReporterTool