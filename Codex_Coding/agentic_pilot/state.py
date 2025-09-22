"""Case state management for the Agentic ID Fraud pilot orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class QualityMetrics:
    """Image quality diagnostics captured by capture/OCR node."""

    blur: float
    glare: float
    perspective: float
    score: float


@dataclass
class OCRResult:
    """Structured OCR extraction output."""

    fields: Dict[str, str]
    conf: float


@dataclass
class PADResult:
    """Primary or secondary presentation attack detection result."""

    score: float
    label: Optional[str] = None


@dataclass
class EvidencePacket:
    """Artifacts and metadata used for analyst review and auditing."""

    corners: Optional[List[Tuple[float, float]]] = None
    crops: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class CaseState:
    """Typed state container shared across orchestrator nodes."""

    images: List[str]
    meta: Dict[str, Any] = field(default_factory=dict)
    canonical: Optional[str] = None
    quality: Optional[QualityMetrics] = None
    ocr: Optional[OCRResult] = None
    pad: Optional[PADResult] = None
    pad_secondary: Optional[PADResult] = None
    features: Optional[List[float]] = None
    risk: Optional[float] = None
    decision: Optional[str] = None
    evidence: EvidencePacket = field(default_factory=EvidencePacket)
    report_uri: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    gate: Dict[str, Any] = field(default_factory=dict)
    active_image: Optional[str] = None
    capture_history: List[str] = field(default_factory=list)
    recapture_queue: List[str] = field(default_factory=list)

    def record_error(self, message: str) -> None:
        """Add an error message to the state."""

        self.errors.append(message)

    @property
    def recapture_attempts(self) -> int:
        """How many recapture loops have already occurred."""

        return int(self.meta.get("recapture_attempts", 0))

    def increment_recapture(self) -> None:
        """Increment the recapture counter in metadata."""

        self.meta["recapture_attempts"] = self.recapture_attempts + 1

    def queue_recaptures(self, images: Iterable[str]) -> None:
        """Add additional candidate images for future recapture attempts."""

        for image in images:
            if image and image not in self.recapture_queue:
                self.recapture_queue.append(image)