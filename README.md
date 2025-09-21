# DeepAgents Government ID Verification System

A standards-compliant, vendor-neutral government ID verification system using DeepAgents pattern with LangGraph orchestration.

## 🚨 Current Status: Active Development

**Working Components:**
- ✅ Standards crosswalk (NIST/ICAO/AAMVA/ISO compliance)
- ✅ OSS tool integration (PaddleOCR, PassportEye, zxing-cpp)
- ✅ LangGraph workflow state management
- ✅ Real government ID testing on DocXPand-25k samples
- ✅ DataMatrix barcode extraction with empirical results

**Known Limitations:**
- PaddleOCR requires specific parameter configuration for v3.2.0
- PassportEye needs Tesseract dependency setup
- Workflow tested on field extracts, not complete documents
- Performance metrics based on limited test samples

## Architecture

### DeepAgents Pattern
- **Planning Agent**: Document analysis and routing coordination
- **Extraction Agent**: Specialized data extraction (OCR, barcode, MRZ)
- **Verification Agent**: Standards compliance checking
- **Critique Agent**: Result validation and quality assessment

### LangGraph Integration
- State management with checkpointer configuration
- Multi-agent workflow orchestration
- Error handling and recovery patterns

## Real Test Results

**DataMatrix Extraction (zxing-cpp)**:
```
Input: id_card_datamatrix.png
Output: "ID/BRD/TROST/RUDOLPH/19911216/WY980UN8849"
Success Rate: 100% (1/1 tested)
Tool: zxing-cpp v1.4.0
```

**OCR Extraction (PaddleOCR)**:
```
Input: passport_1.png, passport_2.png, id_card_1.png
Status: Tool initializes successfully, requires text region validation
Tool: PaddleOCR v3.2.0
```

## Quick Start

### Prerequisites
```bash
Python 3.9+
Virtual environment setup
```

### Installation
```bash
git clone https://github.com/Avyas11/deepagents-fraud-detection
cd deepagents-fraud-detection
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### Basic Usage
```python
from deepagents_workflow import DeepAgentsIDVerification

# Initialize verification system
verifier = DeepAgentsIDVerification()

# Process government ID
result = verifier.process_document("path/to/government_id.png")
print(result)
```

## Standards Compliance

Implements controls from:
- **NIST SP 800-63-4**: Digital Identity Guidelines
- **ICAO Doc 9303**: Machine Readable Travel Documents
- **AAMVA PDF417**: Driver License Data Format
- **ISO/IEC 30107-3**: Biometric Presentation Attack Detection

See `standards_crosswalk.yaml` for complete mapping.

## Key Files

- `deepagents_workflow.py` - Main LangGraph workflow implementation
- `id_verification_tools.py` - OSS tool integrations
- `standards_crosswalk.yaml` - Compliance requirements mapping
- `CODING_MISTAKES_MADE.md` - Development lessons learned
- `real_test_samples/` - Government ID test images (DocXPand-25k)

## Development Notes

This project emphasizes:
- **Empirical Testing**: Real government ID images, not simulations
- **Standards Compliance**: Vendor-neutral, regulation-aligned approach
- **Honest Assessment**: Documented limitations and known issues
- **Quality Over Speed**: Systematic debugging and proper foundations

See `CODING_MISTAKES_MADE.md` for critical development lessons.

## Contributing

1. Read `CODING_MISTAKES_MADE.md` for development patterns to avoid
2. Test on real government ID samples in `real_test_samples/`
3. Ensure standards compliance using `standards_crosswalk.yaml`
4. Document empirical results, not simulations

## License

MIT License - See LICENSE file for details

## Disclaimer

This system is for research and educational purposes. Not certified for production identity verification. Always comply with local privacy and data protection regulations.