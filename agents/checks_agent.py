"""
ChecksAgent: Rules validation and cross-verification

Based on COMPREHENSIVE_PROJECT_SPECIFICATIONS.md#agent-architecture
Implements NIST/ICAO/AAMVA compliance checks and cross-validation logic
"""

import re
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple
import yaml

class ChecksAgent:
    """Agent for standards-based validation and cross-verification"""

    def __init__(self, standards_config_path: str = None):
        self.logger = logging.getLogger(__name__)

        # Load standards configuration
        if standards_config_path:
            with open(standards_config_path, 'r') as f:
                self.standards_config = yaml.safe_load(f)
        else:
            self.standards_config = None

        self.logger.info("ChecksAgent initialized with standards-based validation")

    def validate_extracted_data(self, extraction_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive validation of extracted data

        Args:
            extraction_results: Output from ExtractAgent

        Returns:
            Dict containing validation results and compliance status
        """
        try:
            results = {
                "status": "success",
                "agent": "ChecksAgent",
                "validations": {},
                "compliance_results": {},
                "cross_validation": {}
            }

            extractions = extraction_results.get("extractions", {})

            # 1. ICAO MRZ Validation
            if "mrz" in extractions:
                mrz_validation = self._validate_mrz_data(extractions["mrz"])
                results["validations"]["mrz"] = mrz_validation

            # 2. AAMVA PDF417 Validation
            if "pdf417" in extractions:
                pdf417_validation = self._validate_pdf417_data(extractions["pdf417"])
                results["validations"]["pdf417"] = pdf417_validation

            # 3. OCR Data Validation
            if "ocr" in extractions:
                ocr_validation = self._validate_ocr_data(extractions["ocr"])
                results["validations"]["ocr"] = ocr_validation

            # 4. Cross-validation between data sources
            cross_validation = self._perform_cross_validation(extractions)
            results["cross_validation"] = cross_validation

            # 5. Standards compliance assessment
            compliance_results = self._assess_standards_compliance(results["validations"])
            results["compliance_results"] = compliance_results

            # 6. Generate overall assessment
            results["overall_assessment"] = self._generate_overall_assessment(results)

            return results

        except Exception as e:
            self.logger.error(f"Validation failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "agent": "ChecksAgent"
            }

    def _validate_mrz_data(self, mrz_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate MRZ data according to ICAO Doc 9303 standards"""

        validation_results = {
            "standard": "ICAO Doc 9303",
            "checks_performed": [],
            "passed_checks": [],
            "failed_checks": [],
            "details": {}
        }

        if not mrz_data.get("mrz_found"):
            return {
                **validation_results,
                "status": "not_applicable",
                "message": "No MRZ data found"
            }

        # Check 1: MRZ Format Validation
        check_name = "ICAO_9303_MRZ_FORMAT"
        validation_results["checks_performed"].append(check_name)

        if mrz_data.get("mrz_valid"):
            validation_results["passed_checks"].append(check_name)
            validation_results["details"][check_name] = "MRZ structure conforms to ICAO format"
        else:
            validation_results["failed_checks"].append(check_name)
            validation_results["details"][check_name] = "MRZ structure does not conform to ICAO format"

        # Check 2: Checksum Validation
        checksum_results = mrz_data.get("checksum_results", {})
        checksum_checks = [
            ("ICAO_9303_DOC_NUMBER_CHECKSUM", "document_number_valid"),
            ("ICAO_9303_DOB_CHECKSUM", "date_of_birth_valid"),
            ("ICAO_9303_EXPIRY_CHECKSUM", "expiry_date_valid"),
            ("ICAO_9303_COMPOSITE_CHECKSUM", "final_check_valid")
        ]

        for check_id, checksum_key in checksum_checks:
            validation_results["checks_performed"].append(check_id)

            if checksum_results.get(checksum_key):
                validation_results["passed_checks"].append(check_id)
                validation_results["details"][check_id] = "Check digit validation passed"
            else:
                validation_results["failed_checks"].append(check_id)
                validation_results["details"][check_id] = "Check digit validation failed"

        # Check 3: Mandatory Field Presence
        mandatory_fields = ["document_type", "country", "document_number", "date_of_birth", "expiry_date"]
        key_fields = mrz_data.get("key_fields", {})

        check_name = "ICAO_9303_MANDATORY_FIELDS"
        validation_results["checks_performed"].append(check_name)

        missing_fields = [field for field in mandatory_fields if not key_fields.get(field)]

        if not missing_fields:
            validation_results["passed_checks"].append(check_name)
            validation_results["details"][check_name] = "All mandatory fields present"
        else:
            validation_results["failed_checks"].append(check_name)
            validation_results["details"][check_name] = f"Missing fields: {missing_fields}"

        # Check 4: Date Format Validation
        check_name = "ICAO_9303_DATE_VALIDATION"
        validation_results["checks_performed"].append(check_name)

        date_validation = self._validate_dates(key_fields)
        if date_validation["valid"]:
            validation_results["passed_checks"].append(check_name)
            validation_results["details"][check_name] = "Date formats and logic valid"
        else:
            validation_results["failed_checks"].append(check_name)
            validation_results["details"][check_name] = date_validation["errors"]

        validation_results["pass_rate"] = len(validation_results["passed_checks"]) / len(validation_results["checks_performed"])

        return validation_results

    def _validate_pdf417_data(self, pdf417_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate PDF417 data according to AAMVA standards"""

        validation_results = {
            "standard": "AAMVA DL/ID Standard 2025",
            "checks_performed": [],
            "passed_checks": [],
            "failed_checks": [],
            "details": {}
        }

        if not pdf417_data.get("pdf417_found"):
            return {
                **validation_results,
                "status": "not_applicable",
                "message": "No PDF417 barcode found"
            }

        barcodes = pdf417_data.get("barcodes", [])
        if not barcodes:
            return {
                **validation_results,
                "status": "error",
                "message": "PDF417 found but no data decoded"
            }

        # Use first barcode for validation
        barcode_data = barcodes[0]
        aamva_data = barcode_data.get("parsed_aamva", {})

        # Check 1: AAMVA Header Validation
        check_name = "AAMVA_PDF417_HEADER"
        validation_results["checks_performed"].append(check_name)

        raw_text = barcode_data.get("raw_text", "")
        if raw_text.startswith("@") and "ANSI" in raw_text:
            validation_results["passed_checks"].append(check_name)
            validation_results["details"][check_name] = "Valid AAMVA header format"
        else:
            validation_results["failed_checks"].append(check_name)
            validation_results["details"][check_name] = "Invalid or missing AAMVA header"

        # Check 2: Mandatory AAMVA Fields
        mandatory_aamva_fields = [
            "first_name", "last_name", "date_of_birth",
            "license_expiration", "license_number", "jurisdiction_code"
        ]

        check_name = "AAMVA_MANDATORY_FIELDS"
        validation_results["checks_performed"].append(check_name)

        if aamva_data.get("status") == "success":
            parsed_fields = aamva_data.get("parsed_fields", {})
            missing_fields = [field for field in mandatory_aamva_fields if not parsed_fields.get(field)]

            if not missing_fields:
                validation_results["passed_checks"].append(check_name)
                validation_results["details"][check_name] = "All mandatory AAMVA fields present"
            else:
                validation_results["failed_checks"].append(check_name)
                validation_results["details"][check_name] = f"Missing AAMVA fields: {missing_fields}"
        else:
            validation_results["failed_checks"].append(check_name)
            validation_results["details"][check_name] = "Failed to parse AAMVA data"

        # Check 3: License Number Format
        check_name = "AAMVA_LICENSE_FORMAT"
        validation_results["checks_performed"].append(check_name)

        license_number = aamva_data.get("parsed_fields", {}).get("license_number")
        if license_number and self._validate_license_number_format(license_number):
            validation_results["passed_checks"].append(check_name)
            validation_results["details"][check_name] = "License number format valid"
        else:
            validation_results["failed_checks"].append(check_name)
            validation_results["details"][check_name] = "License number format invalid or missing"

        validation_results["pass_rate"] = len(validation_results["passed_checks"]) / len(validation_results["checks_performed"])

        return validation_results

    def _validate_ocr_data(self, ocr_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate OCR data quality and content"""

        validation_results = {
            "standard": "OCR Quality Assessment",
            "checks_performed": [],
            "passed_checks": [],
            "failed_checks": [],
            "details": {}
        }

        # Check 1: OCR Confidence Threshold
        check_name = "OCR_CONFIDENCE_THRESHOLD"
        validation_results["checks_performed"].append(check_name)

        raw_text = ocr_data.get("raw_text", [])
        if raw_text:
            avg_confidence = sum(item.get("confidence", 0) for item in raw_text) / len(raw_text)

            if avg_confidence >= 0.7:  # 70% confidence threshold
                validation_results["passed_checks"].append(check_name)
                validation_results["details"][check_name] = f"Average OCR confidence: {avg_confidence:.3f}"
            else:
                validation_results["failed_checks"].append(check_name)
                validation_results["details"][check_name] = f"Low OCR confidence: {avg_confidence:.3f}"

        # Check 2: Text Content Completeness
        check_name = "OCR_CONTENT_COMPLETENESS"
        validation_results["checks_performed"].append(check_name)

        total_lines = ocr_data.get("total_lines", 0)
        if total_lines >= 5:  # Minimum expected lines on ID
            validation_results["passed_checks"].append(check_name)
            validation_results["details"][check_name] = f"Sufficient text content: {total_lines} lines"
        else:
            validation_results["failed_checks"].append(check_name)
            validation_results["details"][check_name] = f"Insufficient text content: {total_lines} lines"

        validation_results["pass_rate"] = len(validation_results["passed_checks"]) / len(validation_results["checks_performed"])

        return validation_results

    def _perform_cross_validation(self, extractions: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-validate data between different extraction methods"""

        cross_validation = {
            "checks_performed": [],
            "matches": [],
            "discrepancies": [],
            "details": {}
        }

        # Cross-validate OCR vs MRZ
        if "ocr" in extractions and "mrz" in extractions:
            ocr_mrz_validation = self._cross_validate_ocr_mrz(extractions["ocr"], extractions["mrz"])
            cross_validation["checks_performed"].append("OCR_MRZ_CROSS_VALIDATION")
            cross_validation["details"]["OCR_MRZ_CROSS_VALIDATION"] = ocr_mrz_validation

            if ocr_mrz_validation.get("overall_match", False):
                cross_validation["matches"].append("OCR_MRZ_CROSS_VALIDATION")
            else:
                cross_validation["discrepancies"].append("OCR_MRZ_CROSS_VALIDATION")

        # Cross-validate Visual vs PDF417
        if "ocr" in extractions and "pdf417" in extractions:
            visual_pdf417_validation = self._cross_validate_visual_pdf417(extractions["ocr"], extractions["pdf417"])
            cross_validation["checks_performed"].append("VISUAL_PDF417_CROSS_VALIDATION")
            cross_validation["details"]["VISUAL_PDF417_CROSS_VALIDATION"] = visual_pdf417_validation

            if visual_pdf417_validation.get("overall_match", False):
                cross_validation["matches"].append("VISUAL_PDF417_CROSS_VALIDATION")
            else:
                cross_validation["discrepancies"].append("VISUAL_PDF417_CROSS_VALIDATION")

        cross_validation["match_rate"] = len(cross_validation["matches"]) / max(len(cross_validation["checks_performed"]), 1)

        return cross_validation

    def _cross_validate_ocr_mrz(self, ocr_data: Dict[str, Any], mrz_data: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-validate OCR text against MRZ data"""

        if not mrz_data.get("mrz_found"):
            return {"status": "not_applicable", "message": "No MRZ data for comparison"}

        mrz_fields = mrz_data.get("key_fields", {})
        ocr_fields = ocr_data.get("key_fields", {})

        comparisons = {}
        matches = 0
        total_comparisons = 0

        # Compare common fields
        field_mappings = [
            ("date_of_birth", "date_of_birth"),
            ("document_number", "document_number"),
            ("expiry_date", "expiry_date")
        ]

        for ocr_field, mrz_field in field_mappings:
            if ocr_field in ocr_fields and mrz_field in mrz_fields:
                total_comparisons += 1
                ocr_value = self._normalize_text(ocr_fields[ocr_field])
                mrz_value = self._normalize_text(str(mrz_fields[mrz_field]))

                similarity = self._calculate_string_similarity(ocr_value, mrz_value)
                comparisons[f"{ocr_field}_vs_{mrz_field}"] = {
                    "ocr_value": ocr_value,
                    "mrz_value": mrz_value,
                    "similarity": similarity,
                    "match": similarity >= 0.8
                }

                if similarity >= 0.8:
                    matches += 1

        return {
            "status": "completed",
            "comparisons": comparisons,
            "match_count": matches,
            "total_comparisons": total_comparisons,
            "match_rate": matches / max(total_comparisons, 1),
            "overall_match": (matches / max(total_comparisons, 1)) >= 0.7
        }

    def _cross_validate_visual_pdf417(self, ocr_data: Dict[str, Any], pdf417_data: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-validate visual OCR against PDF417 barcode data"""

        if not pdf417_data.get("pdf417_found"):
            return {"status": "not_applicable", "message": "No PDF417 data for comparison"}

        barcodes = pdf417_data.get("barcodes", [])
        if not barcodes:
            return {"status": "error", "message": "PDF417 found but no data"}

        aamva_data = barcodes[0].get("parsed_aamva", {})
        if aamva_data.get("status") != "success":
            return {"status": "error", "message": "Failed to parse AAMVA data"}

        aamva_fields = aamva_data.get("parsed_fields", {})
        ocr_fields = ocr_data.get("key_fields", {})

        comparisons = {}
        matches = 0
        total_comparisons = 0

        # Compare available fields
        field_mappings = [
            ("date_of_birth", "date_of_birth"),
            ("document_number", "license_number")
        ]

        for ocr_field, aamva_field in field_mappings:
            if ocr_field in ocr_fields and aamva_field in aamva_fields:
                total_comparisons += 1
                ocr_value = self._normalize_text(ocr_fields[ocr_field])
                aamva_value = self._normalize_text(aamva_fields[aamva_field])

                similarity = self._calculate_string_similarity(ocr_value, aamva_value)
                comparisons[f"{ocr_field}_vs_{aamva_field}"] = {
                    "visual_value": ocr_value,
                    "barcode_value": aamva_value,
                    "similarity": similarity,
                    "match": similarity >= 0.8
                }

                if similarity >= 0.8:
                    matches += 1

        return {
            "status": "completed",
            "comparisons": comparisons,
            "match_count": matches,
            "total_comparisons": total_comparisons,
            "match_rate": matches / max(total_comparisons, 1),
            "overall_match": (matches / max(total_comparisons, 1)) >= 0.7
        }

    def _assess_standards_compliance(self, validations: Dict[str, Any]) -> Dict[str, Any]:
        """Assess compliance with different standards based on validation results"""

        compliance_results = {}

        # NIST SP 800-63-4 IAL Assessment
        compliance_results["NIST_SP_800_63_4"] = self._assess_nist_compliance(validations)

        # ICAO Doc 9303 Compliance
        if "mrz" in validations:
            compliance_results["ICAO_DOC_9303"] = self._assess_icao_compliance(validations["mrz"])

        # AAMVA Compliance
        if "pdf417" in validations:
            compliance_results["AAMVA_STANDARD"] = self._assess_aamva_compliance(validations["pdf417"])

        return compliance_results

    def _assess_nist_compliance(self, validations: Dict[str, Any]) -> Dict[str, Any]:
        """Assess NIST SP 800-63-4 IAL requirements"""

        # For demonstration - not actual NIST compliance
        evidence_strength = "WEAK"
        evidence_count = 0

        if validations.get("mrz", {}).get("pass_rate", 0) >= 0.8:
            evidence_count += 1
            evidence_strength = "FAIR"

        if validations.get("pdf417", {}).get("pass_rate", 0) >= 0.8:
            evidence_count += 1
            evidence_strength = "GOOD"

        if evidence_count >= 2:
            evidence_strength = "SUPERIOR"

        return {
            "assessment": "DEMONSTRATION_ONLY",
            "evidence_strength": evidence_strength,
            "evidence_count": evidence_count,
            "note": "Illustrative mapping to IAL concepts - not formal compliance"
        }

    def _assess_icao_compliance(self, mrz_validation: Dict[str, Any]) -> Dict[str, Any]:
        """Assess ICAO Doc 9303 compliance"""

        pass_rate = mrz_validation.get("pass_rate", 0)

        if pass_rate >= 0.9:
            compliance_level = "HIGH"
        elif pass_rate >= 0.7:
            compliance_level = "MODERATE"
        else:
            compliance_level = "LOW"

        return {
            "compliance_level": compliance_level,
            "pass_rate": pass_rate,
            "status": "checks_consistent_with_icao_guidance"
        }

    def _assess_aamva_compliance(self, pdf417_validation: Dict[str, Any]) -> Dict[str, Any]:
        """Assess AAMVA standard compliance"""

        pass_rate = pdf417_validation.get("pass_rate", 0)

        if pass_rate >= 0.9:
            compliance_level = "HIGH"
        elif pass_rate >= 0.7:
            compliance_level = "MODERATE"
        else:
            compliance_level = "LOW"

        return {
            "compliance_level": compliance_level,
            "pass_rate": pass_rate,
            "status": "checks_consistent_with_aamva_guidance"
        }

    def _generate_overall_assessment(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall assessment of ID verification"""

        validations = validation_results.get("validations", {})
        cross_validation = validation_results.get("cross_validation", {})

        # Calculate overall scores
        validation_scores = [v.get("pass_rate", 0) for v in validations.values() if "pass_rate" in v]
        avg_validation_score = sum(validation_scores) / max(len(validation_scores), 1)

        cross_val_score = cross_validation.get("match_rate", 0)

        overall_score = (avg_validation_score * 0.7) + (cross_val_score * 0.3)

        # Determine risk level
        if overall_score >= 0.8:
            risk_level = "LOW"
        elif overall_score >= 0.6:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        return {
            "overall_score": round(overall_score, 3),
            "validation_score": round(avg_validation_score, 3),
            "cross_validation_score": round(cross_val_score, 3),
            "risk_level": risk_level,
            "recommendation": self._generate_recommendation(overall_score, risk_level)
        }

    def _generate_recommendation(self, score: float, risk_level: str) -> str:
        """Generate recommendation based on assessment"""

        if risk_level == "LOW":
            return "Document shows strong consistency across validation checks"
        elif risk_level == "MEDIUM":
            return "Document has some validation concerns - recommend additional verification"
        else:
            return "Document failed multiple validation checks - recommend rejection or manual review"

    # Utility methods
    def _validate_dates(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Validate date formats and logic"""

        try:
            dob = fields.get("date_of_birth")
            expiry = fields.get("expiry_date")

            errors = []

            if dob:
                # Parse and validate date of birth
                dob_date = self._parse_date(str(dob))
                if not dob_date:
                    errors.append("Invalid date of birth format")
                elif dob_date > datetime.now():
                    errors.append("Date of birth is in the future")

            if expiry:
                # Parse and validate expiry date
                expiry_date = self._parse_date(str(expiry))
                if not expiry_date:
                    errors.append("Invalid expiry date format")
                elif expiry_date < datetime.now():
                    errors.append("Document has expired")

            return {"valid": len(errors) == 0, "errors": errors}

        except Exception as e:
            return {"valid": False, "errors": [f"Date validation error: {str(e)}"]}

    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string in various formats"""

        date_formats = [
            "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%m-%d-%Y", "%d-%m-%Y",
            "%Y%m%d", "%m%d%Y", "%d%m%Y"
        ]

        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        return None

    def _validate_license_number_format(self, license_number: str) -> bool:
        """Validate license number format (basic check)"""

        # Basic format check - alphanumeric, 6-15 characters
        pattern = r'^[A-Z0-9]{6,15}$'
        return bool(re.match(pattern, license_number.upper()))

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""

        return re.sub(r'[^A-Z0-9]', '', text.upper())

    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings"""

        if not str1 or not str2:
            return 0.0

        # Simple Levenshtein distance-based similarity
        len1, len2 = len(str1), len(str2)
        if len1 == 0:
            return 0.0 if len2 > 0 else 1.0

        # Create matrix
        matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]

        # Initialize first row and column
        for i in range(len1 + 1):
            matrix[i][0] = i
        for j in range(len2 + 1):
            matrix[0][j] = j

        # Fill matrix
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if str1[i-1] == str2[j-1] else 1
                matrix[i][j] = min(
                    matrix[i-1][j] + 1,      # deletion
                    matrix[i][j-1] + 1,      # insertion
                    matrix[i-1][j-1] + cost  # substitution
                )

        # Calculate similarity
        max_len = max(len1, len2)
        distance = matrix[len1][len2]
        similarity = 1.0 - (distance / max_len)

        return max(0.0, similarity)