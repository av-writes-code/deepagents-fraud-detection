# Pilot Synthetic Testbed

This folder captures the scaffolding for the PRD Section 8.2 synthetic testbed.
We begin by materialising the **bucket structure** so we can add cases for each
slice (genuine vs forged, clean vs borderline) while keeping manifests
automatable.

```
pilot_testbed/
  cases/
    genuine_clean/
    genuine_borderline/
    forged_clear/
    forged_borderline/
```

Use the manifest helpers in `agentic_pilot.testbed` to declare the cases that
live in each bucket. The default expectation is that each case folder contains
`input/`, `meta.json`, `label.json`, `triggers.json`, and `expected.json` as
outlined in the PRD.

## Next Steps

1. Populate the buckets with real cases from the public datasets.
2. Capture manifests (see `manifest_template.json`) that drive automated runs.
3. Extend the harness with recapture simulations and second-opinion triggers.