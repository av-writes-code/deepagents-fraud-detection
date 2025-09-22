"""Default tool adapters for running the pilot end to end on a single image."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:  # pragma: no cover - optional dependency
    from PIL import Image, ImageOps
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    Image = None  # type: ignore[assignment]
    ImageOps = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import pytesseract
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pytesseract = None  # type: ignore[assignment]

from .state import CaseState
from .tools import Toolset

LOGGER = logging.getLogger(__name__)


def build_default_toolset() -> Toolset:
    """Return a ``Toolset`` backed by lightweight, dependency-friendly tools."""

    return Toolset(
        doc_detect_rectify=_doc_detect_rectify,
        ocr=_ocr_extract,
        pad_primary=lambda image, meta: _pad_score(image, meta, boost=0.0),
        pad_secondary=lambda image, meta: _pad_score(image, meta, boost=0.08),
        feature_builder=_feature_builder,
        classifier=_risk_classifier,
        reporter=_noop_reporter,
    )


# ---------------------------------------------------------------------------
# Capture / OCR
# ---------------------------------------------------------------------------


def _doc_detect_rectify(images: List[str], meta: Dict[str, Any]) -> Dict[str, Any]:
    if not images:
        return {"status": "error", "error": "no_images_supplied"}

    image_path = Path(images[0])
    try:
        image = _load_image(image_path)
    except FileNotFoundError:
        return {"status": "error", "error": f"image_not_found:{image_path}"}
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.exception("Failed to load image for capture step")
        return {"status": "error", "error": str(exc)}

    gray = np.asarray(image.convert("L"), dtype="float32") / 255.0
    focus_score = _normalise_focus(gray)
    glare_ratio = float(np.mean(gray > 0.92))
    glare_score = 1.0 - glare_ratio
    aspect_ratio = image.width / max(image.height, 1)
    perspective_score = 1.0 - min(1.0, abs(1.0 - aspect_ratio))
    quality_score = float(
        np.clip(0.55 * focus_score + 0.25 * glare_score + 0.2 * perspective_score, 0.0, 1.0)
    )

    meta.update(
        {
            "capture_width": image.width,
            "capture_height": image.height,
            "focus_score": focus_score,
            "glare_ratio": glare_ratio,
        }
    )

    return {
        "status": "success",
        "canonical_image": str(image_path.resolve()),
        "quality_metrics": {
            "blur": 1.0 - focus_score,
            "glare": glare_ratio,
            "perspective": perspective_score,
            "score": quality_score,
        },
        "evidence": {
            "corners": [
                (0.0, 0.0),
                (float(image.width), 0.0),
                (float(image.width), float(image.height)),
                (0.0, float(image.height)),
            ],
            "crops": [str(image_path.resolve())],
        },
    }


def _ocr_extract(image_path: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    text_lines: List[str] = []
    confidences: List[float] = []

    if pytesseract is not None:
        try:
            data = pytesseract.image_to_data(str(image_path), output_type=pytesseract.Output.DICT)
            for text, conf in zip(data.get("text", []), data.get("conf", [])):
                if not text.strip():
                    continue
                text_lines.append(text.strip())
                try:
                    confidences.append(max(0.0, float(conf)) / 100.0)
                except (TypeError, ValueError):
                    continue
        except Exception as exc:  # pragma: no cover - depends on local tesseract
            LOGGER.warning("pytesseract failed: %s", exc)

    confidence = float(np.mean(confidences)) if confidences else 0.0
    fields = {
        "raw_text": "\n".join(text_lines),
        "line_count": len(text_lines),
    }
    if not text_lines:
        fields["note"] = "ocr_stub_used"

    meta["ocr_line_count"] = len(text_lines)

    return {"status": "success", "fields": fields, "conf": confidence}


# ---------------------------------------------------------------------------
# PAD + risk
# ---------------------------------------------------------------------------


def _pad_score(image_path: str, meta: Dict[str, Any], *, boost: float) -> Dict[str, Any]:
    try:
        image = _load_image(Path(image_path))
    except FileNotFoundError:
        return {"status": "error", "error": f"image_not_found:{image_path}"}
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.exception("Failed to open image for PAD step")
        return {"status": "error", "error": str(exc)}

    base_score, tta_scores = _evaluate_pad_scores(image)
    score = float(np.clip(base_score + boost, 0.0, 1.0))
    label = "forged" if score >= 0.85 else "genuine"
    tta_var = float(np.var(tta_scores))
    frame_range = float(max(tta_scores) - min(tta_scores)) if tta_scores else 0.0
    frame_agree = float(max(0.0, 1.0 - min(1.0, frame_range * 2.5)))

    meta.setdefault("pad_features", {})[f"boost_{boost:.2f}"] = {
        "base": base_score,
        "tta_scores": tta_scores,
    }

    return {
        "status": "success",
        "score": score,
        "label": label,
        "tta_var": tta_var,
        "frame_agree": frame_agree,
    }


def _feature_builder(state: CaseState) -> List[float]:
    pad_score = state.pad.score if state.pad else 1.0
    ocr_conf = state.ocr.conf if state.ocr else 0.0
    quality_score = state.quality.score if state.quality else 0.0

    return [pad_score, ocr_conf, quality_score]


def _risk_classifier(features: List[float]) -> Dict[str, Any]:
    if len(features) != 3:
        return {"value": 1.0, "reasons": ["insufficient_features"]}

    pad_score, ocr_conf, quality_score = features
    risk = 0.5 * pad_score + 0.25 * (1.0 - ocr_conf) + 0.25 * (1.0 - quality_score)
    risk = float(np.clip(risk, 0.0, 1.0))

    reasons: List[str] = []
    if pad_score >= 0.85:
        reasons.append("pad_high")
    if ocr_conf < 0.5:
        reasons.append("ocr_low_conf")
    if quality_score < 0.6:
        reasons.append("quality_low")
    if not reasons:
        reasons.append("auto_approve")

    return {"value": risk, "reasons": reasons}


def _noop_reporter(state: CaseState) -> Dict[str, Any]:
    """Return metadata only; ``AuditRenderer`` will persist the full payload."""

    summary = {
        "decision": state.decision,
        "risk": state.risk,
        "pad": state.pad.score if state.pad else None,
        "secondary_pad": state.pad_secondary.score if state.pad_secondary else None,
        "quality": state.quality.score if state.quality else None,
        "ocr_conf": state.ocr.conf if state.ocr else None,
    }

    state.meta.setdefault("report_summary", summary)
    return {}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _load_image(path: Path) -> Image.Image:
    if Image is None:  # pragma: no cover - optional dependency guard
        raise RuntimeError("Pillow is required to run the default toolset. Install 'pillow'.")
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def _normalise_focus(gray: np.ndarray) -> float:
    gy, gx = np.gradient(gray)
    energy = float(np.mean(gx ** 2 + gy ** 2))
    return float(min(1.0, energy / (energy + 35.0)))


def _evaluate_pad_scores(image: Image.Image) -> tuple[float, List[float]]:
    base = _pad_score_for_image(image)
    tta_scores: List[float] = []
    for angle in (0, -5, 5):
        rotated = image.rotate(angle, resample=Image.BILINEAR)
        tta_scores.append(_pad_score_for_image(rotated))
    return base, tta_scores


def _pad_score_for_image(image: Image.Image) -> float:
    arr = np.asarray(image, dtype="float32") / 255.0
    gray = arr.mean(axis=2)
    focus = _normalise_focus(gray)
    glare = float(np.mean(gray > 0.92))
    contrast = float(np.std(gray))
    color_var = float(np.std(arr - gray[..., None]))

    score = 0.35 * glare + 0.3 * (1.0 - focus) + 0.2 * max(0.0, 0.3 - contrast)
    score += 0.15 * max(0.0, 0.25 - color_var)
    return float(np.clip(score, 0.0, 1.0))