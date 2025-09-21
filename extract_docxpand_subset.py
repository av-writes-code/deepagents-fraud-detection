#!/usr/bin/env python3
"""
Extract and Prepare DocXPand Dataset Subset
Work with partial dataset to test our DeepAgents pipeline
"""

import os
import json
import shutil
from pathlib import Path
import tarfile

def extract_partial_dataset():
    """Extract what we can from the partial DocXPand file"""

    print("🔧 Extracting DocXPand-25k subset...")

    # Try to extract as much as possible
    try:
        with tarfile.open("DocXPand-25k.tar.gz.00", "r:gz") as tar:
            # Extract to a temporary directory first
            temp_dir = Path("temp_extract")
            temp_dir.mkdir(exist_ok=True)

            # Extract only files that are complete
            extracted_files = []
            for member in tar.getmembers():
                try:
                    tar.extract(member, path=temp_dir)
                    extracted_files.append(member.name)
                    print(f"  ✅ Extracted: {member.name}")
                except Exception as e:
                    print(f"  ❌ Failed: {member.name} - {str(e)}")
                    break

            print(f"📊 Successfully extracted {len(extracted_files)} files")
            return temp_dir, extracted_files

    except Exception as e:
        print(f"❌ Extraction failed: {str(e)}")
        return None, []

def create_test_dataset():
    """Create a usable test dataset from extracted files"""

    print("\n🔧 Creating test dataset structure...")

    # Create dataset directory
    dataset_dir = Path("docxpand_test_data")
    dataset_dir.mkdir(exist_ok=True)

    # Check what we have in temp_extract
    temp_dir = Path("temp_extract")
    if not temp_dir.exists():
        print("❌ No extracted data found")
        return

    # Look for images and labels
    extracted_items = list(temp_dir.rglob("*"))

    images = [f for f in extracted_items if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
    json_files = [f for f in extracted_items if f.suffix.lower() == '.json']

    print(f"📊 Found {len(images)} images and {len(json_files)} JSON files")

    # Copy images to test dataset
    if images:
        images_dir = dataset_dir / "images"
        images_dir.mkdir(exist_ok=True)

        for i, img_path in enumerate(images[:50]):  # Take first 50 images
            dest_path = images_dir / f"test_doc_{i:03d}{img_path.suffix}"
            shutil.copy2(img_path, dest_path)
            print(f"  📄 Copied: {img_path.name} -> {dest_path.name}")

    # Copy and examine JSON files
    if json_files:
        labels_dir = dataset_dir / "labels"
        labels_dir.mkdir(exist_ok=True)

        for json_path in json_files[:10]:  # Take first 10 JSON files
            dest_path = labels_dir / json_path.name
            shutil.copy2(json_path, dest_path)

            # Examine structure
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                print(f"  📋 JSON structure in {json_path.name}: {list(data.keys())}")
            except Exception as e:
                print(f"  ❌ Failed to read {json_path.name}: {str(e)}")

    print(f"✅ Test dataset created in {dataset_dir}/")
    return dataset_dir

def prepare_for_testing():
    """Prepare the dataset for our DeepAgents testing"""

    print("\n🎯 Preparing for DeepAgents testing...")

    dataset_dir = Path("docxpand_test_data")
    if not dataset_dir.exists():
        print("❌ No test dataset found")
        return

    # Create manifest for testing
    manifest = {
        "dataset_name": "DocXPand-25k subset",
        "source": "Partial extraction from DocXPand-25k.tar.gz.00",
        "purpose": "DeepAgents empirical testing",
        "test_categories": []
    }

    images_dir = dataset_dir / "images"
    if images_dir.exists():
        images = list(images_dir.glob("*"))
        manifest["total_images"] = len(images)

        # Categorize by type for testing
        for img in images:
            test_case = {
                "image_path": str(img),
                "filename": img.name,
                "tests": ["OCR extraction", "Document preprocessing", "Quality assessment"],
                "expected_elements": ["Text fields", "Document structure"]
            }
            manifest["test_categories"].append(test_case)

    # Save manifest
    manifest_path = dataset_dir / "test_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"📋 Test manifest created: {manifest_path}")
    print(f"🚀 Ready to test with {manifest.get('total_images', 0)} images")

    return manifest

def main():
    """Main extraction and preparation"""

    print("🚀 DOCXPAND DATASET PREPARATION")
    print("=" * 50)

    # Step 1: Extract partial dataset
    temp_dir, extracted_files = extract_partial_dataset()

    if not extracted_files:
        print("❌ No files extracted. Using mock data for testing.")
        return

    # Step 2: Create test dataset
    dataset_dir = create_test_dataset()

    # Step 3: Prepare for testing
    manifest = prepare_for_testing()

    # Cleanup
    if temp_dir and temp_dir.exists():
        shutil.rmtree(temp_dir)
        print("🧹 Cleaned up temporary files")

    print("\n🎯 NEXT STEPS:")
    print("1. ✅ DocXPand subset ready for testing")
    print("2. 🧪 Run individual tool tests on real data")
    print("3. 📊 Measure actual performance vs speculation")
    print("4. 🚀 Push to GitHub when complete")

if __name__ == "__main__":
    main()