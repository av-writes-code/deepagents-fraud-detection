# System Testing Gap Analysis

This note captures the concrete blockers that keep the current pilot scaffolding from running a realistic end-to-end (E2E) system test.

## 1. Missing production-grade tool adapters *(Resolved)*
- Added `agentic_pilot/default_toolset.py`, a dependency-light tool bundle that performs capture diagnostics, OCR (with graceful Tesseract fallback), PAD scoring, feature building, risk scoring, and reporter stubbing so the orchestrator can run against actual image files without bespoke wiring.【F:agentic_pilot/default_toolset.py†L1-L197】
- The new CLI and manifest runner automatically inject this toolset, providing a first-class default assembly when executing single cases or bucket sweeps.【F:agentic_pilot/cli.py†L20-L155】【F:agentic_pilot/testbed/runner.py†L1-L161】

## 2. Recapture loop has no way to source new evidence *(Resolved)*
- `CaseState` now tracks the active image, capture history, and a recapture queue; helper methods let callers seed alternative frames programmatically.【F:agentic_pilot/state.py†L37-L66】
- The orchestrator switches payloads during recapture, pulling from queued frames or an optional supplier callback so fresh evidence is processed instead of replaying the original capture.【F:agentic_pilot/orchestrator.py†L43-L180】

## 3. Testbed harness stops at manifest parsing *(Resolved)*
- Introduced `agentic_pilot/testbed/runner.py`, which loads images, metadata, trigger hints, and expected outputs for each case, executes the orchestrator, and records pass/fail comparisons.【F:agentic_pilot/testbed/runner.py†L1-L161】
- Added CLI support for `run-manifest` so teams can execute an entire synthetic bucket in one command and capture JSON summaries of the outcomes.【F:agentic_pilot/cli.py†L70-L155】

## 4. No orchestration entry point or CLI *(Resolved)*
- Added `agentic_pilot/cli.py` with `run-case` and `run-manifest` subcommands that wire config loading, toolset creation, orchestrator execution, and JSON summary export for immediate hands-on evaluation.【F:agentic_pilot/cli.py†L1-L175】

Addressing these gaps—by wiring real tool adapters, enabling recapture input swapping, implementing a manifest-driven executor, and exposing a runnable entry point—will unlock a reproducible end-to-end system test for the pilot.