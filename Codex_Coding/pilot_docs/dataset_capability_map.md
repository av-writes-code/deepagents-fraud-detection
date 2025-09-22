# Dataset-to-Capability Map

This document captures the first milestone from the validation plan: grounding the pilot in publicly available datasets that exercise each branch of the agentic workflow without any model retraining.

## End-to-End Capabilities We Can Validate Today

| Capability | Dataset Support | Why It's Realistic |
| --- | --- | --- |
| **Bona-fide vs forged (binary PAD)** | **SIDTD** pairs MIDV-2020 bona-fide documents with forged variants (composite, print, screen) and includes JSON metadata plus video clips for printed PAIs. | Direct supervision for the PAD boundary with both still images and mobile capture footage. |
| **Mobile capture & "recapture" loop** | **MIDV-500/2019/2020** provide multi-frame videos across varied capture conditions with frame-level document corners and field boxes/values. | Allows deterministic "try again" loops by selecting higher-quality frames without fabricating data. |
| **OCR confidence & quality gating** | **MIDV-500** exposes per-frame quadrangles and field annotations; **MIDV-2019** stresses OCR with low-light and perspective variants. | Enables computing OCR confidence and quality diagnostics that drive the recapture gate. |
| **Borderline second-opinion PAD** | The existing **PAD secondary** (vendor or heuristic ELA/PRNU bundle) can be run on MIDV/SIDTD borderline slices without retraining. | Pure orchestration—demonstrates agentic lift at a fixed review rate. |

## Known Gaps and Mitigations

| Gap | Reality | Patch/Workaround |
| --- | --- | --- |
| **Micro-tamper (character-level) labels** | **SIDTD** lacks per-character bounding boxes for tampered regions. | Use **FCD-P/D** (≈15k images each) which include forged-character boxes for ROI evaluation or heuristic second-opinion checks. |
| **Real government documents** | Public datasets remain mock/synthetic for privacy, including MIDV and SIDTD bona-fide sets. | Treat results as **relative lift** versus the legacy flow; do not claim absolute production error rates. |
| **US-heavy coverage** | SIDTD emphasizes 10 European nationalities; MIDV is synthetic but globally themed. | Incorporate **IDNet** to add U.S. and EU synthetic appearance diversity (evaluation only). |
| **Face/liveness/registry checks** | Datasets omit selfie streams and registry ground truth. | Limit pilot scope to document PAD + OCR agentic actions; defer liveness/registry to production data. |

## Labels and Metadata Available

- **SIDTD**: forged/genuine flags, PAI type (`composite`, `print`, `screen`), manipulation JSON, and predefined split manifests (including video for printed forgeries).
- **MIDV-500/2019**: 500+ videos with document corners per frame, field boxes/values, and capture conditions (table/keyboard/hand/partial/clutter, plus challenging lighting).
- **MIDV-2020**: 1000 unique mock IDs with videos, scans, and photos, annotated for detection, field extraction, and face crops.
- **FCD-P/D**: 15k forged-ID images per subset with bounding boxes for modified characters.
- **IDNet** (optional): synthetic corpus spanning 10 U.S. states and 10 EU countries (>500k images) for appearance diversity.

## Feasible Pilot Claim

> Using MIDV and SIDTD we can demonstrate agentic gains—targeting a 15–20% false-negative reduction at a 5% review rate—by triggering recapture on low-quality/OCR-low submissions and invoking second-opinion PAD on borderline scores, all without retraining. Micro-tamper sensitivity is stress-tested on FCD-P/D; IDNet extends appearance diversity when needed. Results are interpreted as relative lift given synthetic provenance.

## Next Actions

1. Link each dataset manifest into the synthetic testbed harness.
2. Script deterministic recapture frame selection using MIDV metadata.
3. Define borderline PAD slices from SIDTD for secondary-model evaluation.