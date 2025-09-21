"""
ExtractAgent: Multi-modal data extraction from government IDs

Based on COMPREHENSIVE_PROJECT_SPECIFICATIONS.md#agent-architecture
Handles OCR (PaddleOCR), MRZ parsing (PassportEye), and PDF417 decoding (zxing-cpp)
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Optional
import logging
import json

# Import extraction libraries
import paddleocr
import passporteye
import zxingcpp

class ExtractAgent:
    """Agent for multi-modal data extraction from government ID documents"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Initialize OCR engine
        self.ocr_engine = paddleocr.PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

        self.logger.info("ExtractAgent initialized with PaddleOCR, PassportEye, and zxing-cpp")

    def extract_all_data(self, image: np.ndarray, image_path: str = None) -> Dict[str, Any]:
        """
        Extract all available data from government ID using multiple methods

        Args:
            image: Processed image from PreprocessAgent
            image_path: Original image path for PassportEye

        Returns:
            Dict containing all extracted data and metadata
        """
        try:
            results = {
                "status": "success",
                "agent": "ExtractAgent",
                "extractions": {}
            }

            # 1. OCR Text Extraction
            ocr_results = self._extract_ocr_text(image)
            results["extractions"]["ocr"] = ocr_results

            # 2. MRZ Processing (if passport/ID card)
            if image_path:
                mrz_results = self._extract_mrz_data(image_path)
                results["extractions"]["mrz"] = mrz_results

            # 3. PDF417 Barcode Decoding (if driver's license)
            pdf417_results = self._extract_pdf417_data(image)
            results["extractions"]["pdf417"] = pdf417_results

            # 4. Face Detection and Extraction
            face_results = self._extract_face_data(image)
            results["extractions"]["face"] = face_results

            # Generate extraction summary
            results["summary"] = self._generate_extraction_summary(results["extractions"])

            return results

        except Exception as e:
            self.logger.error(f"Data extraction failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "agent": "ExtractAgent"
            }

    def _extract_ocr_text(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract text using PaddleOCR"""

        try:
            # Run OCR
            ocr_results = self.ocr_engine.ocr(image, cls=True)

            # Process results
            extracted_text = []
            text_boxes = []

            if ocr_results and ocr_results[0]:
                for line in ocr_results[0]:
                    bbox, (text, confidence) = line

                    extracted_text.append({
                        "text": text,
                        "confidence": round(confidence, 3),
                        "bbox": bbox
                    })

                    text_boxes.append({
                        "text": text,
                        "bbox": bbox
                    })

            # Extract potential key fields using heuristics
            key_fields = self._extract_key_fields_from_ocr(extracted_text)

            return {
                "status": "success",
                "method": "PaddleOCR",
                "raw_text": extracted_text,
                "text_boxes": text_boxes,
                "key_fields": key_fields,
                "total_lines": len(extracted_text)
            }

        except Exception as e:
            self.logger.error(f"OCR extraction failed: {str(e)}")
            return {
                "status": "error",
                "method": "PaddleOCR",
                "error": str(e)
            }

    def _extract_mrz_data(self, image_path: str) -> Dict[str, Any]:
        """Extract MRZ data using PassportEye"""

        try:
            # Use PassportEye to read MRZ
            mrz = passporteye.read_mrz(image_path)

            if mrz and mrz.valid:
                # Convert to dict and extract key information
                mrz_dict = mrz.to_dict()

                return {
                    "status": "success",
                    "method": "PassportEye",
                    "mrz_found": True,
                    "mrz_valid": mrz.valid,
                    "raw_mrz": mrz_dict,
                    "key_fields": {
                        "document_type": mrz_dict.get('type'),
                        "country": mrz_dict.get('country'),
                        "surname": mrz_dict.get('surname'),
                        "given_names": mrz_dict.get('names'),
                        "document_number": mrz_dict.get('number'),
                        "nationality": mrz_dict.get('nationality'),
                        "date_of_birth": mrz_dict.get('date_of_birth'),
                        "expiry_date": mrz_dict.get('expiration_date'),
                        "sex": mrz_dict.get('sex')
                    },
                    "checksum_results": {
                        "document_number_valid": mrz.check_number,
                        "date_of_birth_valid": mrz.check_date_of_birth,
                        "expiry_date_valid": mrz.check_expiry_date,
                        "final_check_valid": mrz.check_composite
                    }
                }
            else:
                return {
                    "status": "success",
                    "method": "PassportEye",
                    "mrz_found": False,
                    "message": "No valid MRZ detected in image"
                }

        except Exception as e:
            self.logger.error(f"MRZ extraction failed: {str(e)}")
            return {
                "status": "error",
                "method": "PassportEye",
                "error": str(e)
            }

    def _extract_pdf417_data(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract PDF417 barcode data using zxing-cpp"""

        try:
            # Convert image to format expected by zxing-cpp
            if len(image.shape) == 3:
                # Convert BGR to RGB for zxing-cpp
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image

            # Read barcodes from image
            results = zxingcpp.read_barcodes(image_rgb)

            pdf417_data = []
            for result in results:
                if result.format.name == 'PDF417':
                    # Parse AAMVA data if it's a driver's license
                    parsed_data = self._parse_aamva_data(result.text)

                    pdf417_data.append({
                        "raw_text": result.text,
                        "format": result.format.name,
                        "position": str(result.position),
                        "parsed_aamva": parsed_data
                    })

            if pdf417_data:
                return {
                    "status": "success",
                    "method": "zxing-cpp",
                    "pdf417_found": True,
                    "barcodes": pdf417_data,
                    "count": len(pdf417_data)
                }
            else:
                return {
                    "status": "success",
                    "method": "zxing-cpp",
                    "pdf417_found": False,
                    "message": "No PDF417 barcodes detected"
                }

        except Exception as e:
            self.logger.error(f"PDF417 extraction failed: {str(e)}")
            return {
                "status": "error",
                "method": "zxing-cpp",
                "error": str(e)
            }

    def _extract_face_data(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract face information using OpenCV (basic detection)"""

        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Load Haar cascade for face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            # Detect faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            face_regions = []
            for (x, y, w, h) in faces:
                face_regions.append({
                    "bbox": [x, y, w, h],
                    "confidence": 0.8,  # Haar cascades don't provide confidence scores
                    "area": w * h
                })

            return {
                "status": "success",
                "method": "OpenCV_Haar",
                "faces_found": len(face_regions),
                "face_regions": face_regions
            }

        except Exception as e:
            self.logger.error(f"Face extraction failed: {str(e)}")
            return {
                "status": "error",
                "method": "OpenCV_Haar",
                "error": str(e)
            }

    def _extract_key_fields_from_ocr(self, ocr_text: List[Dict]) -> Dict[str, str]:
        """Extract key identity fields from OCR text using heuristics"""

        key_fields = {}

        for item in ocr_text:
            text = item["text"].upper()

            # Look for date patterns
            if any(keyword in text for keyword in ["DOB", "BIRTH", "BORN"]):
                # Extract date pattern
                import re
                date_pattern = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
                dates = re.findall(date_pattern, text)
                if dates:
                    key_fields["date_of_birth"] = dates[0]

            # Look for expiration dates
            if any(keyword in text for keyword in ["EXP", "EXPIRES", "EXPIRY"]):
                import re
                date_pattern = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
                dates = re.findall(date_pattern, text)
                if dates:
                    key_fields["expiry_date"] = dates[0]

            # Look for document numbers (sequences of letters and numbers)
            if any(keyword in text for keyword in ["NO", "NUMBER", "#"]):
                import re
                number_pattern = r'[A-Z0-9]{6,}'
                numbers = re.findall(number_pattern, text)
                if numbers:
                    key_fields["document_number"] = numbers[0]

        return key_fields

    def _parse_aamva_data(self, raw_text: str) -> Dict[str, Any]:
        """Parse AAMVA PDF417 data according to standard format"""

        try:
            # AAMVA data elements are typically in format: DXX<data>
            # where XX is a two-letter code

            aamva_fields = {}

            # Common AAMVA field mappings
            field_mappings = {
                'DAC': 'first_name',
                'DCS': 'last_name',
                'DBB': 'date_of_birth',
                'DBA': 'license_expiration',
                'DAQ': 'license_number',
                'DCG': 'country',
                'DAJ': 'jurisdiction_code',
                'DAK': 'postal_code',
                'DAL': 'address_line1',
                'DAM': 'address_line2',
                'DAN': 'city',
                'DAO': 'state',
                'DCL': 'race_ethnicity',
                'DCK': 'inventory_control'
            }

            # Parse each field
            for code, field_name in field_mappings.items():
                # Look for pattern: code followed by data until next code or end
                import re
                pattern = rf'{code}([^D]*?)(?=D[A-Z]{{2}}|$)'
                match = re.search(pattern, raw_text)
                if match:
                    value = match.group(1).strip()
                    if value:
                        aamva_fields[field_name] = value

            return {
                "status": "success",
                "parsed_fields": aamva_fields,
                "raw_length": len(raw_text),
                "field_count": len(aamva_fields)
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def _generate_extraction_summary(self, extractions: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of all extraction results"""

        summary = {
            "methods_attempted": list(extractions.keys()),
            "successful_extractions": [],
            "data_sources_found": []
        }

        # Check each extraction method
        for method, result in extractions.items():
            if result.get("status") == "success":
                summary["successful_extractions"].append(method)

                # Check what data sources were found
                if method == "ocr" and result.get("total_lines", 0) > 0:
                    summary["data_sources_found"].append("text_content")

                if method == "mrz" and result.get("mrz_found"):
                    summary["data_sources_found"].append("machine_readable_zone")

                if method == "pdf417" and result.get("pdf417_found"):
                    summary["data_sources_found"].append("pdf417_barcode")

                if method == "face" and result.get("faces_found", 0) > 0:
                    summary["data_sources_found"].append("face_image")

        summary["extraction_completeness"] = len(summary["successful_extractions"]) / len(summary["methods_attempted"])

        return summary