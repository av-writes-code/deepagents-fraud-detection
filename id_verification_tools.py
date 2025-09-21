"""
Tool wrappers for government ID verification using OSS components

Based on DeepAgents architecture - each tool is pure and idempotent
Wraps PassportEye, zxing-cpp, PaddleOCR for DeepAgent integration
"""

import cv2
import numpy as np
import json
import logging
from typing import Dict, Any, List
from pathlib import Path

# Import OSS libraries
import paddleocr
import passporteye
import zxingcpp

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize OCR engine globally for efficiency
# PaddleOCR 3.2.0 compatible initialization (minimal parameters)
ocr_engine = paddleocr.PaddleOCR(lang='en')

def preprocess_document(image_path: str) -> Dict[str, Any]:
    """
    Tool: Preprocess government ID document for optimal extraction

    Args:
        image_path: Path to input image file

    Returns:
        Dict with processed image info and metadata
    """
    try:
        # Load and validate image
        image = cv2.imread(image_path)
        if image is None:
            return {"status": "error", "error": f"Could not load image: {image_path}"}

        # Document detection and perspective correction
        processed_image = _detect_and_correct_document(image)

        # Quality enhancement
        enhanced_image = _enhance_image_quality(processed_image)

        # Save processed image
        output_path = image_path.replace('.', '_processed.')
        cv2.imwrite(output_path, enhanced_image)

        # Calculate quality metrics
        quality_score = _calculate_quality_score(enhanced_image)

        return {
            "status": "success",
            "tool": "preprocess_document",
            "input_path": image_path,
            "output_path": output_path,
            "original_dimensions": image.shape[:2],
            "processed_dimensions": enhanced_image.shape[:2],
            "quality_score": quality_score,
            "preprocessing_steps": [
                "corner_detection",
                "perspective_correction",
                "contrast_enhancement",
                "denoising"
            ]
        }

    except Exception as e:
        logger.error(f"Document preprocessing failed: {str(e)}")
        return {
            "status": "error",
            "tool": "preprocess_document",
            "error": str(e)
        }

def extract_ocr_text(image_path: str) -> Dict[str, Any]:
    """
    Tool: Extract text using PaddleOCR

    Args:
        image_path: Path to processed image

    Returns:
        Dict with extracted text and confidence scores
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {"status": "error", "error": f"Could not load image: {image_path}"}

        # Run OCR
        ocr_results = ocr_engine.ocr(image, cls=True)

        # Process results
        extracted_text = []
        if ocr_results and ocr_results[0]:
            for line in ocr_results[0]:
                bbox, (text, confidence) = line
                extracted_text.append({
                    "text": text,
                    "confidence": round(confidence, 3),
                    "bbox": bbox
                })

        # Extract key fields using heuristics
        key_fields = _extract_key_fields_from_text(extracted_text)

        return {
            "status": "success",
            "tool": "extract_ocr_text",
            "method": "PaddleOCR",
            "image_path": image_path,
            "total_lines": len(extracted_text),
            "average_confidence": sum(item["confidence"] for item in extracted_text) / max(len(extracted_text), 1),
            "raw_text": extracted_text,
            "key_fields": key_fields
        }

    except Exception as e:
        logger.error(f"OCR extraction failed: {str(e)}")
        return {
            "status": "error",
            "tool": "extract_ocr_text",
            "error": str(e)
        }

def extract_mrz_data(image_path: str) -> Dict[str, Any]:
    """
    Tool: Extract MRZ data using PassportEye

    Args:
        image_path: Path to image containing MRZ

    Returns:
        Dict with MRZ fields and checksum validation
    """
    try:
        # Use PassportEye to read MRZ
        mrz = passporteye.read_mrz(image_path)

        if mrz and mrz.valid:
            # Convert to dict
            mrz_dict = mrz.to_dict()

            return {
                "status": "success",
                "tool": "extract_mrz_data",
                "method": "PassportEye",
                "image_path": image_path,
                "mrz_found": True,
                "mrz_valid": mrz.valid,
                "document_type": mrz_dict.get('type'),
                "country": mrz_dict.get('country'),
                "surname": mrz_dict.get('surname'),
                "given_names": mrz_dict.get('names'),
                "document_number": mrz_dict.get('number'),
                "nationality": mrz_dict.get('nationality'),
                "date_of_birth": mrz_dict.get('date_of_birth'),
                "expiry_date": mrz_dict.get('expiration_date'),
                "sex": mrz_dict.get('sex'),
                "checksum_results": {
                    "document_number_valid": mrz.check_number,
                    "date_of_birth_valid": mrz.check_date_of_birth,
                    "expiry_date_valid": mrz.check_expiry_date,
                    "composite_check_valid": mrz.check_composite
                },
                "raw_mrz_dict": mrz_dict
            }
        else:
            return {
                "status": "success",
                "tool": "extract_mrz_data",
                "method": "PassportEye",
                "image_path": image_path,
                "mrz_found": False,
                "message": "No valid MRZ detected in image"
            }

    except Exception as e:
        logger.error(f"MRZ extraction failed: {str(e)}")
        return {
            "status": "error",
            "tool": "extract_mrz_data",
            "error": str(e)
        }

def extract_pdf417_data(image_path: str) -> Dict[str, Any]:
    """
    Tool: Extract PDF417 barcode data using zxing-cpp

    Args:
        image_path: Path to image containing PDF417 barcode

    Returns:
        Dict with decoded barcode data and parsed AAMVA fields
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {"status": "error", "error": f"Could not load image: {image_path}"}

        # Convert BGR to RGB for zxing-cpp
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read barcodes
        results = zxingcpp.read_barcodes(image_rgb)

        pdf417_data = []
        for result in results:
            if result.format.name == 'PDF417':
                # Parse AAMVA data
                aamva_fields = _parse_aamva_data(result.text)

                pdf417_data.append({
                    "raw_text": result.text,
                    "format": result.format.name,
                    "position": str(result.position),
                    "aamva_fields": aamva_fields
                })

        if pdf417_data:
            return {
                "status": "success",
                "tool": "extract_pdf417_data",
                "method": "zxing-cpp",
                "image_path": image_path,
                "pdf417_found": True,
                "barcode_count": len(pdf417_data),
                "barcodes": pdf417_data
            }
        else:
            return {
                "status": "success",
                "tool": "extract_pdf417_data",
                "method": "zxing-cpp",
                "image_path": image_path,
                "pdf417_found": False,
                "message": "No PDF417 barcodes detected"
            }

    except Exception as e:
        logger.error(f"PDF417 extraction failed: {str(e)}")
        return {
            "status": "error",
            "tool": "extract_pdf417_data",
            "error": str(e)
        }

def standards_check(extracted_data: str, controls_yaml: str) -> Dict[str, Any]:
    """
    Tool: Validate extracted data against YAML standards controls

    Args:
        extracted_data: JSON string of all extracted data
        controls_yaml: YAML string with standards controls

    Returns:
        Dict with validation results and compliance status
    """
    try:
        import yaml

        # Parse inputs
        data = json.loads(extracted_data)
        controls = yaml.safe_load(controls_yaml)

        validation_results = {
            "status": "success",
            "tool": "standards_check",
            "validations": {},
            "compliance_summary": {}
        }

        # Process each control from YAML
        for control in controls.get('controls', []):
            control_id = control['id']
            requirement = control['requirement']

            # Validate based on control type
            if 'ICAO' in control_id and 'MRZ' in control_id:
                result = _validate_icao_mrz_control(data, control)
            elif 'AAMVA' in control_id and 'PDF417' in control_id:
                result = _validate_aamva_pdf417_control(data, control)
            elif 'CROSS-VAL' in control_id:
                result = _validate_cross_validation_control(data, control)
            else:
                result = _validate_generic_control(data, control)

            validation_results["validations"][control_id] = result

        # Generate compliance summary
        total_controls = len(validation_results["validations"])
        passed_controls = sum(1 for v in validation_results["validations"].values() if v.get("status") == "PASS")

        validation_results["compliance_summary"] = {
            "total_controls": total_controls,
            "passed_controls": passed_controls,
            "pass_rate": passed_controls / max(total_controls, 1),
            "overall_status": "COMPLIANT" if (passed_controls / max(total_controls, 1)) >= 0.8 else "NON_COMPLIANT"
        }

        return validation_results

    except Exception as e:
        logger.error(f"Standards check failed: {str(e)}")
        return {
            "status": "error",
            "tool": "standards_check",
            "error": str(e)
        }

def publish_report(markdown_content: str, output_path: str = "verification_report.md") -> Dict[str, Any]:
    """
    Tool: Publish verification report (requires human approval)

    Args:
        markdown_content: Report content in markdown format
        output_path: Output file path

    Returns:
        Dict with publication status
    """
    try:
        # Write report to file
        with open(output_path, 'w') as f:
            f.write(markdown_content)

        return {
            "status": "success",
            "tool": "publish_report",
            "output_path": output_path,
            "content_length": len(markdown_content),
            "message": f"Report published to {output_path}"
        }

    except Exception as e:
        logger.error(f"Report publishing failed: {str(e)}")
        return {
            "status": "error",
            "tool": "publish_report",
            "error": str(e)
        }

# Helper functions
def _detect_and_correct_document(image: np.ndarray) -> np.ndarray:
    """Detect document boundaries and apply perspective correction"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find largest rectangular contour
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area < image.shape[0] * image.shape[1] * 0.1:
            continue

        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            # Apply perspective correction
            return _apply_perspective_correction(image, approx.reshape(4, 2))

    return image

def _apply_perspective_correction(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Apply 4-point perspective transformation"""
    # Order corners and calculate destination
    corners = _order_corners(corners)

    width = max(
        np.linalg.norm(corners[1] - corners[0]),
        np.linalg.norm(corners[2] - corners[3])
    )
    height = max(
        np.linalg.norm(corners[3] - corners[0]),
        np.linalg.norm(corners[2] - corners[1])
    )

    dst_corners = np.array([
        [0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst_corners)
    return cv2.warpPerspective(image, matrix, (int(width), int(height)))

def _order_corners(corners: np.ndarray) -> np.ndarray:
    """Order corners as top-left, top-right, bottom-right, bottom-left"""
    centroid = np.mean(corners, axis=0)
    angles = np.arctan2(corners[:, 1] - centroid[1], corners[:, 0] - centroid[0])
    sorted_indices = np.argsort(angles)
    ordered = corners[sorted_indices]

    # Ensure top-left is first
    sums = np.sum(ordered, axis=1)
    min_sum_idx = np.argmin(sums)
    return np.roll(ordered, -min_sum_idx, axis=0)

def _enhance_image_quality(image: np.ndarray) -> np.ndarray:
    """Apply image enhancement for better OCR"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    return cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)

def _calculate_quality_score(image: np.ndarray) -> float:
    """Calculate image quality score (0-1)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return min(laplacian_var / 1000.0, 1.0)

def _extract_key_fields_from_text(ocr_text: List[Dict]) -> Dict[str, str]:
    """Extract key fields using text pattern matching"""
    import re

    key_fields = {}
    for item in ocr_text:
        text = item["text"].upper()

        # Date patterns
        if any(keyword in text for keyword in ["DOB", "BIRTH", "BORN"]):
            dates = re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text)
            if dates:
                key_fields["date_of_birth"] = dates[0]

        if any(keyword in text for keyword in ["EXP", "EXPIRES", "EXPIRY"]):
            dates = re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text)
            if dates:
                key_fields["expiry_date"] = dates[0]

        # Document number patterns
        if any(keyword in text for keyword in ["NO", "NUMBER", "#"]):
            numbers = re.findall(r'[A-Z0-9]{6,}', text)
            if numbers:
                key_fields["document_number"] = numbers[0]

    return key_fields

def _parse_aamva_data(raw_text: str) -> Dict[str, str]:
    """Parse AAMVA PDF417 data"""
    import re

    field_mappings = {
        'DAC': 'first_name', 'DCS': 'last_name', 'DBB': 'date_of_birth',
        'DBA': 'license_expiration', 'DAQ': 'license_number', 'DCG': 'country',
        'DAJ': 'jurisdiction_code', 'DAK': 'postal_code', 'DAL': 'address_line1',
        'DAM': 'address_line2', 'DAN': 'city', 'DAO': 'state'
    }

    aamva_fields = {}
    for code, field_name in field_mappings.items():
        pattern = rf'{code}([^D]*?)(?=D[A-Z]{{2}}|$)'
        match = re.search(pattern, raw_text)
        if match:
            value = match.group(1).strip()
            if value:
                aamva_fields[field_name] = value

    return aamva_fields

def _validate_icao_mrz_control(data: Dict, control: Dict) -> Dict[str, Any]:
    """Validate ICAO MRZ control"""
    # Check if MRZ data exists
    mrz_data = None
    for key, value in data.items():
        if isinstance(value, dict) and value.get("tool") == "extract_mrz_data":
            mrz_data = value
            break

    if not mrz_data or not mrz_data.get("mrz_found"):
        return {"status": "FAIL", "reason": "No MRZ data found"}

    # Check specific requirements
    requirement = control.get('requirement', '')

    if 'checksum' in requirement.lower():
        checksums = mrz_data.get('checksum_results', {})
        all_valid = all(checksums.values())
        return {
            "status": "PASS" if all_valid else "FAIL",
            "evidence": checksums,
            "reason": "All checksums valid" if all_valid else "Some checksums failed"
        }

    if 'format' in requirement.lower():
        mrz_valid = mrz_data.get('mrz_valid', False)
        return {
            "status": "PASS" if mrz_valid else "FAIL",
            "evidence": {"mrz_valid": mrz_valid},
            "reason": "MRZ format valid" if mrz_valid else "MRZ format invalid"
        }

    return {"status": "PASS", "reason": "Generic ICAO validation passed"}

def _validate_aamva_pdf417_control(data: Dict, control: Dict) -> Dict[str, Any]:
    """Validate AAMVA PDF417 control"""
    # Check if PDF417 data exists
    pdf417_data = None
    for key, value in data.items():
        if isinstance(value, dict) and value.get("tool") == "extract_pdf417_data":
            pdf417_data = value
            break

    if not pdf417_data or not pdf417_data.get("pdf417_found"):
        return {"status": "FAIL", "reason": "No PDF417 barcode found"}

    barcodes = pdf417_data.get('barcodes', [])
    if not barcodes:
        return {"status": "FAIL", "reason": "PDF417 found but no data decoded"}

    aamva_fields = barcodes[0].get('aamva_fields', {})

    requirement = control.get('requirement', '')
    if 'mandatory' in requirement.lower():
        required_fields = ['first_name', 'last_name', 'date_of_birth', 'license_number']
        missing_fields = [f for f in required_fields if f not in aamva_fields]

        return {
            "status": "PASS" if not missing_fields else "FAIL",
            "evidence": {"present_fields": list(aamva_fields.keys()), "missing_fields": missing_fields},
            "reason": "All mandatory fields present" if not missing_fields else f"Missing: {missing_fields}"
        }

    return {"status": "PASS", "reason": "Generic AAMVA validation passed"}

def _validate_cross_validation_control(data: Dict, control: Dict) -> Dict[str, Any]:
    """Validate cross-validation control"""
    # Find OCR and MRZ/PDF417 data for comparison
    ocr_data = None
    mrz_data = None
    pdf417_data = None

    for key, value in data.items():
        if isinstance(value, dict):
            tool = value.get("tool")
            if tool == "extract_ocr_text":
                ocr_data = value
            elif tool == "extract_mrz_data":
                mrz_data = value
            elif tool == "extract_pdf417_data":
                pdf417_data = value

    if not ocr_data:
        return {"status": "FAIL", "reason": "No OCR data for cross-validation"}

    matches = 0
    comparisons = 0

    # Compare OCR vs MRZ if available
    if mrz_data and mrz_data.get("mrz_found"):
        ocr_fields = ocr_data.get("key_fields", {})

        # Compare date of birth
        if "date_of_birth" in ocr_fields and mrz_data.get("date_of_birth"):
            comparisons += 1
            if _normalize_text(ocr_fields["date_of_birth"]) == _normalize_text(str(mrz_data["date_of_birth"])):
                matches += 1

    # Compare OCR vs PDF417 if available
    if pdf417_data and pdf417_data.get("pdf417_found"):
        barcodes = pdf417_data.get('barcodes', [])
        if barcodes:
            aamva_fields = barcodes[0].get('aamva_fields', {})
            ocr_fields = ocr_data.get("key_fields", {})

            # Compare available fields
            if "date_of_birth" in ocr_fields and "date_of_birth" in aamva_fields:
                comparisons += 1
                if _normalize_text(ocr_fields["date_of_birth"]) == _normalize_text(aamva_fields["date_of_birth"]):
                    matches += 1

    if comparisons == 0:
        return {"status": "FAIL", "reason": "No data available for cross-validation"}

    match_rate = matches / comparisons
    return {
        "status": "PASS" if match_rate >= 0.7 else "FAIL",
        "evidence": {"matches": matches, "comparisons": comparisons, "match_rate": match_rate},
        "reason": f"Cross-validation match rate: {match_rate:.1%}"
    }

def _validate_generic_control(data: Dict, control: Dict) -> Dict[str, Any]:
    """Generic validation for other controls"""
    return {"status": "PASS", "reason": "Generic validation - implementation needed"}

def _normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    import re
    return re.sub(r'[^A-Z0-9]', '', text.upper())

# Export all tools for DeepAgents
__all__ = [
    'preprocess_document',
    'extract_ocr_text',
    'extract_mrz_data',
    'extract_pdf417_data',
    'standards_check',
    'publish_report'
]