#!/usr/bin/env python3
"""
Basic tool integration test

Test each OSS component individually before full workflow
Start simple, build incrementally
"""

import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test 1: Verify all imports work"""
    print("🔧 Testing imports...")

    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV failed: {e}")
        return False

    try:
        import paddleocr
        print("✅ PaddleOCR: imported successfully")
    except ImportError as e:
        print(f"❌ PaddleOCR failed: {e}")
        return False

    try:
        import passporteye
        print("✅ PassportEye: imported successfully")
    except ImportError as e:
        print(f"❌ PassportEye failed: {e}")
        return False

    try:
        import zxingcpp
        print("✅ zxing-cpp: imported successfully")
    except ImportError as e:
        print(f"❌ zxing-cpp failed: {e}")
        return False

    try:
        import langgraph
        print("✅ LangGraph: imported successfully")
    except ImportError as e:
        print(f"❌ LangGraph failed: {e}")
        return False

    return True

def test_basic_opencv():
    """Test 2: Basic OpenCV functionality"""
    print("\n📷 Testing OpenCV basics...")

    import cv2
    import numpy as np

    # Create a simple test image
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    test_image[25:75, 25:75] = [255, 255, 255]  # White square

    # Test basic operations
    gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    print(f"✅ Created test image: {test_image.shape}")
    print(f"✅ Grayscale conversion: {gray.shape}")
    print(f"✅ Edge detection: found {np.sum(edges > 0)} edge pixels")

    return True

def test_basic_paddleocr():
    """Test 3: Basic PaddleOCR functionality"""
    print("\n🔤 Testing PaddleOCR basics...")

    try:
        import paddleocr
        import numpy as np

        # Create simple text image for testing
        # Using numpy to create a mock image with "TEST" text
        test_image = np.ones((50, 200, 3), dtype=np.uint8) * 255  # White background

        # Initialize OCR (this will download models on first run)
        ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

        print("✅ PaddleOCR initialized successfully")
        print("✅ Ready for text extraction")

        return True

    except Exception as e:
        print(f"❌ PaddleOCR test failed: {e}")
        return False

def test_standards_yaml():
    """Test 4: Standards YAML loading"""
    print("\n📋 Testing standards YAML...")

    try:
        import yaml

        # Check if standards file exists
        yaml_file = Path("standards_crosswalk.yaml")
        if not yaml_file.exists():
            print(f"❌ Standards file not found: {yaml_file}")
            return False

        # Load and validate YAML
        with open(yaml_file, 'r') as f:
            standards = yaml.safe_load(f)

        print(f"✅ YAML loaded successfully")
        print(f"✅ Found {len(standards.get('controls', []))} controls")

        # Check for required structure
        required_keys = ['meta', 'controls', 'agent_responsibilities']
        missing_keys = [key for key in required_keys if key not in standards]

        if missing_keys:
            print(f"⚠️ Missing YAML keys: {missing_keys}")
        else:
            print("✅ YAML structure valid")

        return True

    except Exception as e:
        print(f"❌ YAML test failed: {e}")
        return False

def test_tool_imports():
    """Test 5: Our custom tool imports"""
    print("\n🛠️ Testing custom tools...")

    try:
        from id_verification_tools import (
            preprocess_document,
            extract_ocr_text,
            extract_mrz_data,
            extract_pdf417_data,
            standards_check,
            publish_report
        )

        print("✅ All custom tools imported successfully")

        # Test function signatures
        import inspect

        tools = [preprocess_document, extract_ocr_text, extract_mrz_data,
                extract_pdf417_data, standards_check, publish_report]

        for tool in tools:
            sig = inspect.signature(tool)
            print(f"✅ {tool.__name__}: {sig}")

        return True

    except Exception as e:
        print(f"❌ Custom tools test failed: {e}")
        return False

def main():
    """Run all basic tests"""
    print("🚀 Starting basic tool integration tests...\n")

    tests = [
        ("Import Test", test_imports),
        ("OpenCV Basic", test_basic_opencv),
        ("PaddleOCR Basic", test_basic_paddleocr),
        ("Standards YAML", test_standards_yaml),
        ("Custom Tools", test_tool_imports)
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False

    # Summary
    print(f"\n{'='*50}")
    print("📊 TEST SUMMARY")
    print(f"{'='*50}")

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<30} {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All basic tests passed! Ready for incremental development.")
        return True
    else:
        print("⚠️ Some tests failed. Fix issues before proceeding.")
        return False

if __name__ == "__main__":
    main()