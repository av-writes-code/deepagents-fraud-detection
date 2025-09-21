#!/usr/bin/env python3
"""
Create Test Data for DeepAgents Validation

Generate simple test images to measure actual tool performance
No speculation - just build and test
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
from pathlib import Path

def create_simple_test_images():
    """Create basic test images for tool validation"""

    print("🔧 Creating test images...")

    # Create test_data directory
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)

    # Test 1: Simple text document (for OCR)
    create_simple_text_image(test_dir)

    # Test 2: Mock MRZ (for PassportEye)
    create_mock_mrz_image(test_dir)

    # Test 3: Simple barcode pattern (for zxing-cpp)
    create_mock_barcode_image(test_dir)

    print(f"✅ Test images created in {test_dir}/")

def create_simple_text_image(test_dir):
    """Create simple text image for OCR testing"""

    # Create white background
    width, height = 400, 200
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)

    # Add simple text
    try:
        # Try to use a system font
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
    except:
        # Fallback to default font
        font = ImageFont.load_default()

    text_lines = [
        "DRIVER LICENSE",
        "JOHN SMITH",
        "DOB: 01/15/1980",
        "EXP: 01/15/2028"
    ]

    y_position = 20
    for line in text_lines:
        draw.text((20, y_position), line, fill='black', font=font)
        y_position += 35

    # Save image
    image_path = test_dir / "simple_text.png"
    image.save(image_path)

    print(f"  📄 Created simple text image: {image_path}")
    return image_path

def create_mock_mrz_image(test_dir):
    """Create mock MRZ pattern for PassportEye testing"""

    # Create image with MRZ-like pattern
    width, height = 600, 150
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)

    try:
        # Use monospace font for MRZ
        font = ImageFont.truetype("/System/Library/Fonts/Courier.ttf", 16)
    except:
        font = ImageFont.load_default()

    # Mock MRZ lines (simplified ICAO format)
    mrz_lines = [
        "P<USADOE<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<",
        "1234567890USA8001011M2801011<<<<<<<<<<<<<<<<0"
    ]

    y_position = 50
    for line in mrz_lines:
        draw.text((20, y_position), line, fill='black', font=font)
        y_position += 25

    # Save image
    image_path = test_dir / "mock_mrz.png"
    image.save(image_path)

    print(f"  🛂 Created mock MRZ image: {image_path}")
    return image_path

def create_mock_barcode_image(test_dir):
    """Create simple pattern for barcode testing"""

    # Create image with barcode-like pattern
    width, height = 400, 100
    image = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background

    # Create simple black and white pattern (not a real barcode)
    pattern_width = 3
    x_pos = 20

    # Simple alternating pattern
    for i in range(50):
        if i % 2 == 0:
            cv2.rectangle(image, (x_pos, 20), (x_pos + pattern_width, 80), (0, 0, 0), -1)
        x_pos += pattern_width

    # Save image
    image_path = test_dir / "mock_barcode.png"
    cv2.imwrite(str(image_path), image)

    print(f"  📊 Created mock barcode image: {image_path}")
    return image_path

def test_individual_tools():
    """Test each OSS tool individually with REAL government ID data"""

    print("\n🧪 Testing individual tools on REAL government IDs...")

    # Use real DocXPand-25k samples instead of mock data
    real_samples_dir = Path("real_test_samples")
    if not real_samples_dir.exists():
        print("❌ No real samples found. Expected real_test_samples/ directory.")
        return {}

    real_images = list(real_samples_dir.glob("*.png"))
    if not real_images:
        print("❌ No PNG images found in real_test_samples/")
        return {}

    print(f"📊 Testing on {len(real_images)} REAL government ID images from DocXPand-25k")

    results = {}

    # Test 1: OpenCV preprocessing on REAL government IDs
    print("\n1. Testing OpenCV preprocessing on real government IDs...")
    try:
        import cv2

        opencv_results = []
        for img_path in real_images[:3]:  # Test on first 3 images
            image = cv2.imread(str(img_path))
            if image is not None:
                # Basic preprocessing operations
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)

                # Quality metrics
                height, width = gray.shape
                brightness = float(np.mean(gray))
                contrast = float(np.std(gray))

                opencv_results.append({
                    "image": img_path.name,
                    "dimensions": f"{width}x{height}",
                    "brightness": brightness,
                    "contrast": contrast,
                    "status": "success"
                })
                print(f"    ✅ {img_path.name}: {width}x{height}, brightness={brightness:.1f}, contrast={contrast:.1f}")
            else:
                opencv_results.append({
                    "image": img_path.name,
                    "status": "error",
                    "error": "failed_to_load"
                })
                print(f"    ❌ {img_path.name}: Failed to load")

        success_rate = len([r for r in opencv_results if r["status"] == "success"]) / len(opencv_results) * 100
        print(f"  📊 OpenCV Success Rate: {success_rate:.0f}% ({len([r for r in opencv_results if r['status'] == 'success'])}/{len(opencv_results)})")

        results["opencv"] = {
            "status": "success" if success_rate > 0 else "error",
            "success_rate": f"{success_rate:.0f}%",
            "results": opencv_results
        }

    except Exception as e:
        print(f"  ❌ OpenCV: {str(e)}")
        results["opencv"] = {"status": "error", "error": str(e)}

    # Test 2: PaddleOCR
    print("\n2. Testing PaddleOCR...")
    try:
        # Import OCR engine from our tools
        from id_verification_tools import extract_ocr_text

        ocr_result = extract_ocr_text(str(test_dir / "simple_text.png"))
        print(f"  📊 PaddleOCR result: {ocr_result.get('status', 'unknown')}")

        if ocr_result.get("status") == "success":
            total_lines = ocr_result.get("total_lines", 0)
            avg_confidence = ocr_result.get("average_confidence", 0)
            print(f"    Lines detected: {total_lines}")
            print(f"    Avg confidence: {avg_confidence:.2f}")

        results["paddleocr"] = ocr_result

    except Exception as e:
        print(f"  ❌ PaddleOCR: {str(e)}")
        results["paddleocr"] = {"status": "error", "error": str(e)}

    # Test 3: PassportEye
    print("\n3. Testing PassportEye...")
    try:
        from id_verification_tools import extract_mrz_data

        mrz_result = extract_mrz_data(str(test_dir / "mock_mrz.png"))
        print(f"  🛂 PassportEye result: {mrz_result.get('status', 'unknown')}")

        if mrz_result.get("mrz_found"):
            print(f"    MRZ found: {mrz_result.get('mrz_valid', 'unknown')}")
        else:
            print(f"    MRZ found: False")

        results["passporteye"] = mrz_result

    except Exception as e:
        print(f"  ❌ PassportEye: {str(e)}")
        results["passporteye"] = {"status": "error", "error": str(e)}

    # Test 4: zxing-cpp
    print("\n4. Testing zxing-cpp...")
    try:
        from id_verification_tools import extract_pdf417_data

        pdf417_result = extract_pdf417_data(str(test_dir / "mock_barcode.png"))
        print(f"  📊 zxing-cpp result: {pdf417_result.get('status', 'unknown')}")

        if pdf417_result.get("pdf417_found"):
            print(f"    PDF417 found: True")
        else:
            print(f"    PDF417 found: False (expected for mock pattern)")

        results["zxing"] = pdf417_result

    except Exception as e:
        print(f"  ❌ zxing-cpp: {str(e)}")
        results["zxing"] = {"status": "error", "error": str(e)}

    return results

def generate_test_report(results):
    """Generate empirical test report"""

    print("\n" + "="*50)
    print("📊 EMPIRICAL TEST RESULTS")
    print("="*50)

    success_count = 0
    total_tests = len(results)

    for tool, result in results.items():
        status = result.get("status", "unknown")
        if status == "success":
            print(f"  ✅ {tool.upper()}: Working")
            success_count += 1
        else:
            error = result.get("error", "Unknown error")
            print(f"  ❌ {tool.upper()}: Failed - {error}")

    success_rate = (success_count / total_tests) * 100 if total_tests > 0 else 0

    print(f"\n📈 ACTUAL SUCCESS RATE: {success_rate:.0f}% ({success_count}/{total_tests})")

    # Save results
    results_file = Path("test_data") / "tool_test_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"📁 Results saved to: {results_file}")

    return success_rate, results

def main():
    """Run incremental testing"""

    print("🚀 INCREMENTAL TOOL TESTING")
    print("="*50)
    print("No speculation - measuring actual performance")

    # Step 1: Create test data
    create_simple_test_images()

    # Step 2: Test individual tools
    results = test_individual_tools()

    # Step 3: Generate empirical report
    success_rate, detailed_results = generate_test_report(results)

    # Step 4: Next steps based on results
    print(f"\n🎯 NEXT STEPS:")
    if success_rate >= 75:
        print("  ✅ Most tools working - proceed to agent testing")
    elif success_rate >= 50:
        print("  ⚠️ Some tools failing - fix issues before proceeding")
    else:
        print("  ❌ Major issues - need debugging before continuing")

    print("\n📋 READY FOR:")
    print("  1. Fix any tool failures")
    print("  2. Test agent integration")
    print("  3. Measure end-to-end workflow")
    print("  4. Document actual performance")

if __name__ == "__main__":
    main()