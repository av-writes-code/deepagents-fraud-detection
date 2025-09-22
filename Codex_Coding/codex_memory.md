# Codex Memory

## Current Focus
- Align the agentic pilot orchestrator with DeepAgents/LangGraph patterns using explicit sub-agent nodes for capture/OCR, PAD, uncertainty gate, recapture, second opinion, risk/policy, and audit.
- Ground evaluation and testing strategy in publicly available datasets (MIDV, SIDTD, FCD-P/D, IDNet) without any model retraining.
- Centralise the pilot artifacts, documentation, and guidance inside the `Codex_Coding` workspace for easier iteration.

## Key Constraints
- No model retraining; treat OCR, PAD, and risk models as black-box tools.
- Keep recapture loops to a single attempt and use max fusion for PAD second opinions.
- Maintain auditability with persisted node inputs/outputs and signed reports.

## Next Steps
1. Document the remaining blockers that prevent true end-to-end system testing and prioritise fixes.
2. Build dataset capability mapping artifacts per the pilot validation plan and link them to concrete manifest entries.
3. Expand synthetic testbed coverage with recapture simulations and second-opinion triggers after bucket scaffolding is populated.