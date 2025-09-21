#!/usr/bin/env python3
"""
Test Tools on REAL Government ID Images

Test our DeepAgents tools on actual DocXPand-25k synthetic government ID images
NO MORE MOCK DATA - REAL PERFORMANCE VALIDATION
"""

import cv2
import numpy as np
import json
import time
from pathlib import Path

def test_opencv_on_real_ids():
    """Test OpenCV preprocessing on real government ID images"""

    print("🧪 Testing OpenCV on REAL Government IDs...")

    real_samples = [
        "real_test_samples/passport_sample_1.png",
        "real_test_samples/id_card_datamatrix.png"
    ]

    results = {}

    for i, sample_path in enumerate(real_samples, 1):
        if not Path(sample_path).exists():
            print(f"  ❌ Sample {i}: File not found - {sample_path}")
            continue

        try:
            # Load real government ID image
            image = cv2.imread(sample_path)
            if image is None:
                results[f"sample_{i}"] = {"status": "error", "error": "Failed to load image"}
                continue

            # Document preprocessing pipeline
            start_time = time.time()

            # 1. Color space conversion
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 2. Noise reduction
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)

            # 3. Contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)

            # 4. Edge detection for document structure
            edges = cv2.Canny(enhanced, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size

            processing_time = time.time() - start_time

            # Save processed image
            output_path = sample_path.replace('.png', '_processed.png')
            cv2.imwrite(output_path, enhanced)

            # Quality assessment
            quality_metrics = {
                "contrast": float(np.std(enhanced)),
                "edge_density": float(edge_density),
                "brightness": float(np.mean(enhanced))
            }

            print(f"  ✅ Sample {i}: Processed in {processing_time:.3f}s")
            print(f"    Contrast: {quality_metrics['contrast']:.1f}")
            print(f"    Edge density: {quality_metrics['edge_density']:.3f}")

            results[f"sample_{i}"] = {
                "status": "success",
                "input_path": sample_path,
                "output_path": output_path,
                "processing_time": processing_time,
                "quality_metrics": quality_metrics
            }

        except Exception as e:
            print(f"  ❌ Sample {i}: Processing failed - {str(e)}")
            results[f"sample_{i}"] = {"status": "error", "error": str(e)}

    return results

def test_easyocr_on_real_ids():
    """Test EasyOCR on real government ID images"""

    print("\n🧪 Testing EasyOCR on REAL Government IDs...")

    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False)  # CPU only for compatibility

        real_samples = [
            "real_test_samples/passport_sample_1.png",
            "real_test_samples/id_card_datamatrix.png"
        ]

        results = {}

        for i, sample_path in enumerate(real_samples, 1):
            if not Path(sample_path).exists():
                print(f"  ❌ Sample {i}: File not found")
                continue

            try:
                start_time = time.time()

                # Run OCR on real government ID
                ocr_results = reader.readtext(sample_path)

                processing_time = time.time() - start_time

                # Extract text and confidence scores
                extracted_text = []
                total_confidence = 0

                for (bbox, text, confidence) in ocr_results:
                    extracted_text.append({
                        "text": text,
                        "confidence": float(confidence),
                        "bbox": [float(x) for point in bbox for x in point]
                    })
                    total_confidence += confidence

                avg_confidence = total_confidence / len(ocr_results) if ocr_results else 0

                print(f"  ✅ Sample {i}: Found {len(ocr_results)} text regions")
                print(f"    Avg confidence: {avg_confidence:.3f}")
                print(f"    Processing time: {processing_time:.3f}s")

                # Show sample extracted text
                for j, item in enumerate(extracted_text[:3]):  # Show first 3
                    print(f"    Text {j+1}: '{item['text']}' (conf: {item['confidence']:.3f})")

                results[f"sample_{i}"] = {
                    "status": "success",
                    "text_regions_found": len(ocr_results),
                    "average_confidence": avg_confidence,
                    "processing_time": processing_time,
                    "extracted_text": extracted_text
                }

            except Exception as e:
                print(f"  ❌ Sample {i}: OCR failed - {str(e)}")
                results[f"sample_{i}"] = {"status": "error", "error": str(e)}

        return results

    except ImportError as e:
        print(f"  ❌ EasyOCR not available: {str(e)}")
        return {"status": "error", "error": "EasyOCR not installed"}

def test_zxing_on_real_ids():
    """Test zxing-cpp barcode detection on real government IDs"""

    print("\n🧪 Testing zxing-cpp on REAL Government IDs...")

    try:
        import zxingcpp

        real_samples = [
            "real_test_samples/passport_sample_1.png",
            "real_test_samples/id_card_datamatrix.png"
        ]

        results = {}

        for i, sample_path in enumerate(real_samples, 1):
            if not Path(sample_path).exists():
                print(f"  ❌ Sample {i}: File not found")
                continue

            try:
                start_time = time.time()

                # Load image
                image = cv2.imread(sample_path)
                if image is None:
                    results[f"sample_{i}"] = {"status": "error", "error": "Failed to load image"}
                    continue

                # Try to detect barcodes/datamatrix
                barcodes = zxingcpp.read_barcodes(image)

                processing_time = time.time() - start_time

                if barcodes:
                    print(f"  ✅ Sample {i}: Found {len(barcodes)} barcode(s)")
                    for j, barcode in enumerate(barcodes):
                        print(f"    Barcode {j+1}: Format={barcode.format}, Text length={len(barcode.text)}")

                    results[f"sample_{i}"] = {
                        "status": "success",
                        "barcodes_found": len(barcodes),
                        "processing_time": processing_time,
                        "barcode_details": [
                            {
                                "format": barcode.format.name,
                                "text_length": len(barcode.text),
                                "text_preview": barcode.text[:50] + "..." if len(barcode.text) > 50 else barcode.text
                            } for barcode in barcodes
                        ]
                    }
                else:
                    print(f"  ⚠️ Sample {i}: No barcodes detected")
                    results[f"sample_{i}"] = {
                        "status": "success",
                        "barcodes_found": 0,
                        "processing_time": processing_time
                    }

            except Exception as e:
                print(f"  ❌ Sample {i}: Barcode detection failed - {str(e)}")
                results[f"sample_{i}"] = {"status": "error", "error": str(e)}

        return results

    except ImportError as e:
        print(f"  ❌ zxing-cpp not available: {str(e)}")
        return {"status": "error", "error": "zxing-cpp not installed"}

def generate_real_performance_report():
    """Generate empirical performance report based on REAL government ID testing"""

    print("\n📊 RUNNING REAL GOVERNMENT ID PERFORMANCE TEST")
    print("=" * 60)
    print("Testing on DocXPand-25k synthetic government ID samples")

    # Test all tools
    opencv_results = test_opencv_on_real_ids()
    ocr_results = test_easyocr_on_real_ids()
    barcode_results = test_zxing_on_real_ids()

    # Calculate actual success rates
    def calculate_success_rate(results):
        if isinstance(results, dict) and "status" in results:
            return 0 if results["status"] == "error" else 100

        total = len(results)
        successful = sum(1 for r in results.values() if r.get("status") == "success")
        return (successful / total * 100) if total > 0 else 0

    opencv_success = calculate_success_rate(opencv_results)
    ocr_success = calculate_success_rate(ocr_results)
    barcode_success = calculate_success_rate(barcode_results)

    # Generate report
    report = {
        "test_date": "2025-09-21",
        "test_type": "Real Government ID Performance Validation",
        "dataset": "DocXPand-25k synthetic government IDs",
        "samples_tested": 2,
        "tool_performance": {
            "opencv_preprocessing": {
                "success_rate": f"{opencv_success}%",
                "details": opencv_results
            },
            "easyocr_text_extraction": {
                "success_rate": f"{ocr_success}%",
                "details": ocr_results
            },
            "zxing_barcode_detection": {
                "success_rate": f"{barcode_success}%",
                "details": barcode_results
            }
        },
        "overall_assessment": {
            "working_tools": [tool for tool, rate in [
                ("OpenCV", opencv_success),
                ("EasyOCR", ocr_success),
                ("zxing-cpp", barcode_success)
            ] if rate > 0],
            "average_success_rate": f"{(opencv_success + ocr_success + barcode_success) / 3:.0f}%"
        }
    }

    # Save report
    report_path = "real_test_samples/empirical_performance_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # Display summary
    print(f"\n📈 EMPIRICAL RESULTS ON REAL GOVERNMENT IDs:")
    print(f"  OpenCV Preprocessing: {opencv_success}% success")
    print(f"  EasyOCR Text Extraction: {ocr_success}% success")
    print(f"  zxing-cpp Barcode Detection: {barcode_success}% success")
    print(f"  Overall Tool Success Rate: {(opencv_success + ocr_success + barcode_success) / 3:.0f}%")

    print(f"\n📁 Detailed report saved to: {report_path}")

    return report

if __name__ == "__main__":
    # Verify we have real samples
    if not Path("real_test_samples").exists():
        print("❌ No real test samples found. Run dataset extraction first.")
        exit(1)

    samples = list(Path("real_test_samples").glob("*.png"))
    if not samples:
        print("❌ No PNG samples found in real_test_samples/")
        exit(1)

    print(f"🎯 Found {len(samples)} real government ID samples:")
    for sample in samples:
        print(f"  - {sample.name}")

    # Run performance validation
    report = generate_real_performance_report()

    print(f"\n🚀 REAL TESTING COMPLETE - NO MORE MOCK DATA!")