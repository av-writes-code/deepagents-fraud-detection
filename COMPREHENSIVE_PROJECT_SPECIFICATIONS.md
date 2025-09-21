# Comprehensive Project Specifications: Government ID Verification with DeepAgents

## Table of Contents
1. [Bill of Materials & Component Specifications](#bill-of-materials)
2. [Standards Framework & Compliance](#standards-framework)
3. [Technical Implementation Plan](#technical-implementation)
4. [Legal & Compliance Guidelines](#legal-compliance)
5. [Agent Architecture Specifications](#agent-architecture)
6. [Implementation Timeline & Deliverables](#timeline-deliverables)

---

## Bill of Materials & Component Specifications {#bill-of-materials}

*Source: Original specification provided - "reuse-only bill of materials you can drop into a LangGraph + DeepAgent project to check fake IDs without retraining anything"*

### Orchestration (LangGraph + DeepAgents)

**LangGraph (stateful multi-agent graphs)**
- Official library + quickstart examples
- GitHub: https://github.com/langchain-ai/langgraph
- Purpose: State management for multi-agent workflows

**DeepAgents (plan→execute with sub-agents & tools)**
- Docs + sample code patterns
- GitHub: https://github.com/langchain-ai/deepagents
- Purpose: Hierarchical planning with sub-agent delegation

**FastAPI + LangGraph deployment template**
- Production-ready scaffold (Docker, Langfuse, auth, metrics)
- GitHub: Referenced for production deployment
- Purpose: Enterprise-scalable deployment

### Pre-processing (crop, deskew, denoise)

**OpenCV document scanner pipelines**
- Corner detect + 4-point transform; pure Python
- Source: learnopencv.com
- Purpose: Document preparation and cleanup

**Optional illumination fix: UDoc-GAN**
- Script + weights for uneven lighting
- Source: arXiv paper
- Purpose: One-click enhancement before OCR

### OCR & Key-Value Extraction (no training)

**PaddleOCR 3.0**
- Batteries-included OCR; PP-Structure/KIE and PP-ChatOCRv4
- High-level key extraction out-of-the-box
- Source: paddlepaddle.github.io
- GitHub: https://github.com/PaddlePaddle/PaddleOCR
- Purpose: Primary OCR solution - "Great default"

**MMOCR**
- Plug-and-play end-to-end OCR with KIE (SDMGR)
- One command demo pipeline
- GitHub: Referenced
- Source: mmocr.readthedocs.io
- Purpose: Alternative OCR solution

**EasyOCR**
- Dead-simple OCR for lightest dependency set
- GitHub: Referenced
- Purpose: Minimal dependency OCR fallback

### MRZ (passport/ID) parsing & validation

**PassportEye**
- Detects MRZ, runs OCR, returns parsed dict with checksum validation
- One-liner: `read_mrz(...).to_dict()`
- PyPI: https://pypi.org/project/PassportEye/
- Docs: passporteye.readthedocs.io
- Purpose: Primary MRZ processing

**mrz (standalone)**
- Standalone MRZ checker/generator that validates fields per ICAO 9303
- PyPI: Referenced
- Purpose: Cross-check validation

### PDF417 (US/CA driver's license back) + AAMVA parsing

**zxing-cpp (Python bindings)**
- Robust PDF417 decoder
- Installation: `pip install zxing-cpp`
- PyPI: Referenced
- Purpose: Primary PDF417 decoding

**pdf417decoder (pure Python)**
- Fallback if you want zero native deps
- PyPI: Referenced
- Purpose: Backup PDF417 decoder

**AAMVA barcode field parser (Python)**
- Ready-made parser for decoded strings (names, DOB, DL number, expiry)
- GitHub: Referenced
- Purpose: Parse decoded PDF417 content

**AAMVA 2020 spec**
- Standard reference to power rule checks (no dev needed)
- Source: AAMVA official documentation
- Purpose: Compliance validation rules

### Face match (selfie ↔ ID portrait)

**InsightFace**
- SOTA face embeddings & verification with pretrained models (ArcFace)
- Simple Python API
- GitHub: Referenced
- Purpose: Primary face matching solution

**DeepFace**
- One-function verification wrapper (VGG-Face/ArcFace/Facenet, all pretrained)
- Source: barkoder (referenced)
- Purpose: Alternative face matching

**Anti-spoofing (optional): Silent-Face MiniFASNet**
- Repo includes usable pretrained weights
- Source: Clarifai (often mirrored off Baidu; see issues thread)
- Purpose: Liveness detection

### Image tamper / forgery signals (no training)

*Use these as red-flag sub-agents (heatmaps + scores) on the portrait crop and full ID*

**TruFor**
- Modern, robust manipulation detection/localization with pretrained weights + script
- GitHub: Referenced
- Purpose: Primary tamper detection

**ManTraNet**
- Classic pixel-level forgery localization
- Repo + notebook with pretrained
- GitHub: Referenced
- Google Colab: Referenced
- Purpose: Secondary tamper detection

**Noiseprint**
- Camera-model "fingerprint" maps to spot splices/inconsistencies
- Includes demo code
- GitHub: Referenced
- Purpose: Camera consistency analysis

### EXIF/metadata sanity checks

**PyExifTool or piexif**
- Quick pulls of camera + editing traces (presence/absence anomalies)
- PyPI: Referenced
- Purpose: Metadata integrity analysis

### Ready-to-use datasets for demoing (no training)

**IDNet (synthetic)**
- Huge synthetic ID corpus for evaluation demos
- Kaggle: Referenced
- Purpose: Testing dataset

**DocXPand-25k (synthetic)**
- 24,994 labeled ID images for benchmarks
- arXiv: Referenced
- Purpose: Evaluation dataset

**Roboflow MRZ / License detection**
- Small, easy demo sets/APIs to validate your pipeline
- Roboflow: Referenced
- Purpose: Validation dataset

---

## Standards Framework & Compliance {#standards-framework}

*Source: "Short answer: you're not overthinking. The LoR is production-grade..." - comprehensive standards alignment specification*

### A. Standards to Cite & Operationalize

**NIST SP 800-63-4 (digital identity)**
- IAL/AAL/FAL components for identity proofing & auth
- Source: NIST Pages (https://pages.nist.gov/800-63-4/)
- **CRITICAL**: Use 800-63-**4** (Aug 1, 2025), NOT 63-3
- Purpose: Digital identity framework compliance

**ICAO Doc 9303**
- MRZ layout & check-digit rules for passports/ID cards
- Source: ICAO official documentation
- Purpose: International passport/ID standards

**AAMVA DL/ID Standard**
- Mandatory PDF417 on US/CA driver's licenses + field specs
- Source: AAMVA official documentation
- **Updated**: AAMVA DL/ID Card Design Standard (2025)
- Purpose: North American driver's license compliance

**ISO/IEC 30107-3**
- PAD test methodology (APCER/BPCER) to frame liveness/anti-spoofing
- Source: ISO official documentation
- Purpose: Presentation attack detection standards

### B. One YAML "Control Crosswalk" (Core NIW Artifact)

*This is the vendor-neutral, reusable mapping from public standards → concrete checks/tools*

```yaml
schema: v0
standards:
  - id: ICAO_9303_P3_MRZ_CHECK
    requires: ["mrz_checksum_valid", "mrz_format_ok"]
    toolchain: ["passport_eye.mrz_extract", "passport_eye.checksum"]
    evidence: ["image_front.png"]
  - id: AAMVA_DLID_PDF417_MANDATORY
    requires: ["pdf417_decoded", "aamva_fields_present", "aamva_dates_valid"]
    toolchain: ["zxing.decode_pdf417", "aamva.parse"]
    evidence: ["image_back.png"]
  - id: NIST_800_63A_IAL2_DOC_PROOFING
    requires: ["doc_image_quality_ok", "name_dob_match_ocr_barcode"]
    toolchain: ["paddleocr.extract", "aamva.parse", "rules.cross_check"]
  - id: ISO_30107_3_PAD_REPORT
    requires: ["pad_metric_apcer", "pad_metric_bpcer"]
    toolchain: ["minifasnet.score", "pad.report"]
report:
  format: "jsonl"   # also render to PDF template
```

**Key Properties:**
- Single file = "automated control mapping"
- Change YAML → system adapts to different regime without retraining
- References public standards for vendor neutrality

---

## Technical Implementation Plan {#technical-implementation}

### Agent Architecture Snap-Together Design

**PreprocessAgent**
- Tool: OpenCV deskew/denoise
- Output: Clean crop
- Source: learnopencv.com

**OCRAgent**
- Tool: PaddleOCR/MMOCR
- Output: Text lines + boxes
- Source: paddlepaddle.github.io

**MRZAgent**
- Tool: PassportEye/mrz
- Output: Fields + checksum pass/fail
- Source: PyPI

**BarcodeAgent**
- Tool: zxing-cpp/pdf417decoder → AAMVA parser
- Output: DL fields
- Sources: PyPI (zxing-cpp), PyPI (pdf417decoder)

**FaceAgent**
- Tool: InsightFace/DeepFace match
- Optional: Anti-spoof score
- Sources: GitHub (InsightFace), barkoder (DeepFace)

**ForgeryAgent**
- Tools: TruFor + ManTraNet + Noiseprint
- Output: Heatmaps & integrity score
- Sources: GitHub (TruFor), GitHub (ManTraNet)

**RulesAgent**
- Function: Cross-checks (MRZ ↔ OCR ↔ AAMVA)
- Validation: Date/format validators per AAMVA/ICAO
- Source: AAMVA standards

**Coordinator (DeepAgent)**
- Function: Plan/route, request human-in-the-loop when confidence < threshold
- Framework: LangGraph + DeepAgents pattern
- Source: GitHub (langchain-ai/deepagents)

### Graph Structure: 5 Minimal Nodes

1. **Preprocess** (OpenCV) → crops/deskews
2. **Extract** → OCR + MRZ + PDF417/AAMVA parse
3. **Checks** → rule functions (checksum, field presence, date math)
4. **Critic** (LLM) → verifies every required control has evidence, flags omissions/redundancy, normalizes terminology → emits traceability matrix (Control → Tool → Evidence → Pass/Fail)
5. **Planner/Orchestrator** (LLM via LangGraph) → reads YAML, schedules nodes, short-circuits if must-pass control fails

### Outputs (standards-aligned, forkable)

**Machine-readable**: `verification.jsonl` with control IDs from YAML
**Human report**: Templated PDF with sections:
- IAL mapping (NIST 800-63)
- MRZ evidence (ICAO 9303)
- AAMVA barcode checklist
- PAD metrics (ISO 30107-3)
**Transparency appendix**: Model+tool versions, config hash, deterministic seeds

---

## Legal & Compliance Guidelines {#legal-compliance}

### Safe Compliance Language (Copy-Paste Approved)

**Compliance Posture for Medium Article:**
> *This is a research **prototype** that demonstrates a **standards-aligned** verification workflow for government IDs. It **implements checks consistent with** NIST SP 800-63-4 (identity proofing concepts), **applies MRZ rules from ICAO Doc 9303**, **parses PDF417 per AAMVA DL/ID guidance**, and **illustrates PAD reporting terminology from ISO/IEC 30107-3**. It is **not** a certified or accredited system; no claim of formal **compliance**, **conformance**, or **approval** is made. Any production deployment must undergo independent security, privacy, and standards assessments.*

**Vendor-Neutrality / No Lock-In:**
> *All controls are declared in a portable YAML "crosswalk" that references public standards (NIST 800-63-4, ICAO 9303, AAMVA, ISO/IEC 30107-3). Individual tools (OCR, MRZ, PDF417, face-match) are swappable OSS components; no proprietary SDKs are required for the demo.*

### Claims to Avoid vs Correct Language

❌ **AVOID**: "Compliant with NIST/ISO"
✅ **USE**: "implements checks consistent with NIST/ISO guidance"

❌ **AVOID**: "Meets IAL2/3"
✅ **USE**: "maps prototype evidence to IAL control **objectives** for illustration"

❌ **AVOID**: "Certified / Approved / FedRAMP-ready"
✅ **USE**: "research demo; not evaluated or accredited"

### Required Legal Boilerplate

> *Legal & Standards Notice — This prototype is an educational demonstration. It **references** NIST SP 800-63-4/63A-4, ICAO Doc 9303, AAMVA DL/ID standards, and ISO/IEC 30107-3 **to organize checks and reporting**, but it has **not** undergone any formal evaluation, certification, or accreditation against these or any other standards. All validations (e.g., MRZ check-digits, PDF417 field presence, OCR cross-checks, illustrative PAD metrics) are **for demonstration only** and do not constitute compliance or legal advice. Organizations considering production use should perform independent security, privacy, and standards assessments and consult applicable regulators.*

### Self-Audit Checklist Before Publishing

- [ ] Cite **800-63-4**, not 63-3, and avoid "compliant" language
- [ ] MRZ check-digit math is shown (inputs & computed digits)
- [ ] PDF417 decode + AAMVA field mapping is shown
- [ ] PAD is clearly labeled **illustrative** (no ISO compliance claim)
- [ ] YAML crosswalk uses **standard IDs** in control names (e.g., `NIST_63A4_...`, `ICAO_9303_...`)
- [ ] Report includes **Traceability Matrix** and **Tooling BOM** (with links/versions)
- [ ] Article includes **Legal & Standards Notice**

---

## Agent Architecture Specifications {#agent-architecture}

### Core DeepAgent Pattern Implementation

**Requirements:**
- Follow DeepAgents and sub-agents architecture in LangGraph
- Do not deviate from established patterns
- Focus on 3-4 core agents initially
- Implement one agent successfully at a time

**Target Architecture: 4-Agent System**
1. **PreprocessAgent**: Document cleanup and preparation
2. **ExtractAgent**: Multi-modal data extraction (OCR/MRZ/PDF417)
3. **ChecksAgent**: Rules validation and cross-verification
4. **CriticAgent**: LLM-powered traceability and completeness verification

### Comparison Framework: Agentic AI vs Traditional ML Pipelines

**Traditional ML Pipeline:**
- Monolithic model training
- Fixed feature extraction
- Static rule sets
- Difficult to adapt to new requirements

**Agentic AI Pipeline:**
- Modular specialist agents
- Dynamic planning and routing
- Standards-driven configuration
- Vendor-neutral tool swapping

### Practical Implementation Notes

**PDF417 Preference**: Use zxing-cpp (ZBar doesn't read PDF417)
**Licensing Considerations**: InsightFace & some forensics repos have specific licenses—okay for OSS demos, verify for commercial use
**Domain Gap**: Generic forgery models (TruFor/ManTraNet/Noiseprint) aren't ID-specific; use as signals, not sole deciders

### Quick "Happy Path" to Ship

Use FastAPI LangGraph Template → add sub-agents as tools (each exposing simple `analyze(image)` or `parse(text)` function)

**Default Stack**: OpenCV → PaddleOCR → PassportEye + zxing-cpp + AAMVA parser → InsightFace → TruFor → Rules

*All components are pretrained/turnkey*

---

## Timeline & Deliverables {#timeline-deliverables}

### 5-Hour Implementation Schedule

**Target Audience**: Medium article readers who are technical business-savvy people (thinking technical product managers)

**Article Title**: "Building State-of-the-Art ID Verification with Open-Source DeepAgents"

**Key Deliverables:**
1. Working 4-agent DeepAgent system
2. Standards-aligned YAML crosswalk
3. Traceability matrix generation
4. Medium article with technical implementation
5. Comparison analysis: Agentic AI vs Traditional ML
6. Legal compliance documentation

### Medium Article Requirements

**Content Specifications:**
- Short "Why this matters nationally" box tying demo to NIST 800-63 IAL proofing and AAMVA consistency across states
- One diagram of 5-node graph
- 1-page traceability matrix screenshot
- Code snippet showing YAML + tiny LangGraph snippet that reads it and schedules tools

### Evidence Pack Requirements

**Must Generate:**
1. **Standards Crosswalk (YAML)** → control IDs reference standards
2. **Traceability Matrix** → each control shows: Tool → Input evidence → Output → Pass/Fail → Rationale
3. **Raw artifacts** → MRZ dict + check-digit results; PDF417 raw string + parsed fields; OCR text; (optional) PAD scores
4. **Tooling BOM** → versions + licenses
5. **Scope line** → "Prototype for research; not certified"

### Success Metrics

✅ **Technical**: Working 4-agent system processing government IDs
✅ **Compliance**: Standards-aligned reporting with legal boilerplate
✅ **Reusability**: YAML-driven, vendor-neutral architecture
✅ **Documentation**: Medium article showing agentic AI advantages
✅ **Performance**: Demonstrable improvement over traditional ML approaches

---

## How Implementation Ties Back to LoR (Point-by-Point)

*Direct mapping to Letter of Recommendation claims*

**Hierarchical multi-agent planning** → YAML-driven **planner** (regime → steps) + **specialist tools**

**Critic agents & closed loop** → Dedicated **critic** enforces traceability, missing-control detection, and terminology consistency—exactly the "self-audit" capability

**Automated control mapping / crosswalks** → YAML maps **policy → controls → tools**, reusable across frameworks (swap ICAO/AAMVA/NIST blocks)

**Vendor-neutral & no lock-in** → All checks reference public standards; OSS tools are swappable; LLM is abstracted (local or API)

**Nationwide utility** → Aligns with **NIST SP 800-63** identity proofing, **AAMVA** driver's licenses, **ICAO 9303** MRZ rules, and **ISO 30107-3** PAD—standards used across US agencies & DMVs

---

*This comprehensive specification captures all requirements, technical details, compliance frameworks, and implementation guidelines for the government ID verification system using DeepAgents + LangGraph.*