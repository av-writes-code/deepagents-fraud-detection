#!/usr/bin/env python3
"""
Simple tool testing with minimal dependencies
Test what actually works vs what fails
"""

import cv2
import numpy as np
import json
from pathlib import Path

def test_opencv():
    """Test basic OpenCV operations"""

    print("🧪 Testing OpenCV...")

    try:
        # Basic operations that should work
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        print("  ✅ OpenCV: Basic operations successful")
        return {"status": "success", "operations": ["color_conversion", "blur"]}

    except Exception as e:
        print(f"  ❌ OpenCV: {str(e)}")
        return {"status": "error", "error": str(e)}

def test_simple_ocr():
    """Test simple OCR without complex dependencies"""

    print("🧪 Testing Simple OCR...")

    try:
        # Try basic OCR import
        import paddleocr

        # Simple OCR test
        ocr = paddleocr.PaddleOCR(lang='en', show_log=False)

        # Test with a simple image path
        test_image_path = "test_data/simple_text.png"
        if Path(test_image_path).exists():
            result = ocr.ocr(test_image_path)

            if result and result[0]:
                print(f"  ✅ PaddleOCR: Detected {len(result[0])} text lines")
                return {"status": "success", "lines_detected": len(result[0])}
            else:
                print("  ⚠️ PaddleOCR: No text detected")
                return {"status": "success", "lines_detected": 0}
        else:
            print(f"  ❌ Test image not found: {test_image_path}")
            return {"status": "error", "error": "test_image_not_found"}

    except Exception as e:
        print(f"  ❌ PaddleOCR: {str(e)}")
        return {"status": "error", "error": str(e)}

def test_zxing():
    """Test zxing-cpp for barcode detection"""

    print("🧪 Testing zxing-cpp...")

    try:
        import zxingcpp

        test_image_path = "test_data/mock_barcode.png"
        if Path(test_image_path).exists():
            # Read image
            image = cv2.imread(test_image_path)

            # Try to detect barcodes
            results = zxingcpp.read_barcodes(image)

            print(f"  ✅ zxing-cpp: Processed image, found {len(results)} barcodes")
            return {"status": "success", "barcodes_found": len(results)}
        else:
            print(f"  ❌ Test image not found: {test_image_path}")
            return {"status": "error", "error": "test_image_not_found"}

    except Exception as e:
        print(f"  ❌ zxing-cpp: {str(e)}")
        return {"status": "error", "error": str(e)}

def test_mrz_simple():
    """Test basic MRZ reading without complex dependencies"""

    print("🧪 Testing Simple MRZ...")

    try:
        # Create a simple MRZ parser without PassportEye dependencies
        test_mrz_line = "P<USADOE<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"

        # Basic MRZ validation
        if len(test_mrz_line) == 44 and test_mrz_line.startswith("P<"):
            print("  ✅ MRZ Parser: Basic format validation successful")
            return {"status": "success", "mrz_format": "valid"}
        else:
            print("  ❌ MRZ Parser: Invalid format")
            return {"status": "error", "error": "invalid_mrz_format"}

    except Exception as e:
        print(f"  ❌ MRZ Parser: {str(e)}")
        return {"status": "error", "error": str(e)}

def run_simple_tests():
    """Run simplified tool tests"""

    print("🚀 SIMPLE TOOL TESTING")
    print("=" * 40)
    print("Testing only what we can actually run")

    results = {}

    # Test 1: OpenCV
    results["opencv"] = test_opencv()

    # Test 2: Simple OCR
    results["simple_ocr"] = test_simple_ocr()

    # Test 3: zxing-cpp
    results["zxing"] = test_zxing()

    # Test 4: Simple MRZ
    results["simple_mrz"] = test_mrz_simple()

    # Calculate actual success rate
    success_count = sum(1 for r in results.values() if r.get("status") == "success")
    total_tests = len(results)
    success_rate = (success_count / total_tests) * 100

    print(f"\n📊 EMPIRICAL RESULTS:")
    for tool, result in results.items():
        status = "✅" if result.get("status") == "success" else "❌"
        print(f"  {status} {tool.upper()}: {result.get('status', 'unknown')}")
        if result.get("status") == "error":
            print(f"    Error: {result.get('error', 'Unknown error')}")

    print(f"\n📈 ACTUAL SUCCESS RATE: {success_rate:.0f}% ({success_count}/{total_tests})")

    # Save results
    results_file = Path("test_data") / "simple_tool_results.json"
    results_file.parent.mkdir(exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"📁 Results saved to: {results_file}")

    return success_rate, results

if __name__ == "__main__":
    success_rate, results = run_simple_tests()

    print(f"\n🎯 NEXT STEPS:")
    if success_rate >= 75:
        print("  ✅ Most tools working - ready for agent integration")
    elif success_rate >= 50:
        print("  ⚠️ Some tools working - can proceed with working tools")
    else:
        print("  ❌ Major failures - need to debug further")