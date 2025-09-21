"""
PreprocessAgent: Document cleanup and preparation for ID verification

Based on COMPREHENSIVE_PROJECT_SPECIFICATIONS.md#agent-architecture
Uses OpenCV for corner detection, 4-point transform, and denoising
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple
import logging

class PreprocessAgent:
    """Agent for document preprocessing and cleanup operations"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def preprocess_document(self, image_path: str) -> Dict[str, Any]:
        """
        Main preprocessing pipeline for government ID documents

        Args:
            image_path: Path to input image

        Returns:
            Dict containing processed image and metadata
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")

            # Document detection and perspective correction
            corrected_image = self._detect_and_correct_document(image)

            # Quality enhancement
            enhanced_image = self._enhance_image_quality(corrected_image)

            # Generate preprocessing metadata
            metadata = self._generate_metadata(image, enhanced_image)

            return {
                "status": "success",
                "original_image": image,
                "processed_image": enhanced_image,
                "metadata": metadata,
                "agent": "PreprocessAgent"
            }

        except Exception as e:
            self.logger.error(f"Preprocessing failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "agent": "PreprocessAgent"
            }

    def _detect_and_correct_document(self, image: np.ndarray) -> np.ndarray:
        """Detect document corners and apply perspective correction"""

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find largest rectangular contour (document boundary)
        document_contour = self._find_document_contour(contours, image.shape)

        if document_contour is not None:
            # Apply perspective correction
            return self._apply_perspective_correction(image, document_contour)
        else:
            # Return original if no document boundary found
            return image

    def _find_document_contour(self, contours: list, image_shape: Tuple) -> np.ndarray:
        """Find the largest rectangular contour representing document boundary"""

        min_area = image_shape[0] * image_shape[1] * 0.1  # Minimum 10% of image area

        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(contour)

            if area < min_area:
                continue

            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check if approximation has 4 corners (rectangle)
            if len(approx) == 4:
                return approx.reshape(4, 2)

        return None

    def _apply_perspective_correction(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Apply 4-point perspective transformation"""

        # Order corners: top-left, top-right, bottom-right, bottom-left
        corners = self._order_corners(corners)

        # Calculate dimensions of corrected image
        width = max(
            np.linalg.norm(corners[1] - corners[0]),  # top edge
            np.linalg.norm(corners[2] - corners[3])   # bottom edge
        )
        height = max(
            np.linalg.norm(corners[3] - corners[0]),  # left edge
            np.linalg.norm(corners[2] - corners[1])   # right edge
        )

        # Define destination points
        dst_corners = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)

        # Calculate perspective transformation matrix
        matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst_corners)

        # Apply transformation
        corrected = cv2.warpPerspective(image, matrix, (int(width), int(height)))

        return corrected

    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """Order corners as: top-left, top-right, bottom-right, bottom-left"""

        # Calculate centroid
        centroid = np.mean(corners, axis=0)

        # Sort by angle from centroid
        angles = np.arctan2(corners[:, 1] - centroid[1], corners[:, 0] - centroid[0])
        sorted_indices = np.argsort(angles)

        # Reorder to start from top-left
        ordered = corners[sorted_indices]

        # Ensure top-left is first (smallest sum of coordinates)
        sums = np.sum(ordered, axis=1)
        min_sum_idx = np.argmin(sums)
        ordered = np.roll(ordered, -min_sum_idx, axis=0)

        return ordered

    def _enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """Apply image enhancement for better OCR/analysis"""

        # Convert to LAB color space for better contrast adjustment
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # Merge channels and convert back to BGR
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        # Apply slight denoising
        denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)

        return denoised

    def _generate_metadata(self, original: np.ndarray, processed: np.ndarray) -> Dict[str, Any]:
        """Generate preprocessing metadata for traceability"""

        return {
            "original_dimensions": original.shape[:2],
            "processed_dimensions": processed.shape[:2],
            "preprocessing_steps": [
                "corner_detection",
                "perspective_correction",
                "contrast_enhancement",
                "denoising"
            ],
            "opencv_version": cv2.__version__,
            "quality_score": self._calculate_quality_score(processed)
        }

    def _calculate_quality_score(self, image: np.ndarray) -> float:
        """Calculate basic image quality score (0-1)"""

        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate Laplacian variance (focus measure)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Normalize to 0-1 range (higher values indicate sharper images)
        quality_score = min(laplacian_var / 1000.0, 1.0)

        return round(quality_score, 3)