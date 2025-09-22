"""Audit artifact generation for the Agentic ID Fraud orchestrator."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from .state import CaseState


class AuditRenderer:
    """Render signed JSON/HTML audit artifacts for downstream consumers."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def render(self, state: CaseState, metadata: Dict[str, str]) -> str:
        """Persist a JSON audit record and return its URI."""

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        case_id = state.meta.get("case_id", "unknown")
        filename = f"audit_{case_id}_{timestamp}.json"
        path = self.output_dir / filename

        payload = {
            "metadata": metadata,
            "state": asdict(state),
        }

        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path.as_uri()