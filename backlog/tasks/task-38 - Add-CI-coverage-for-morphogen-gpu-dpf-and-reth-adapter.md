---
id: TASK-38
title: Add CI coverage for morphogen-gpu-dpf and reth-adapter
status: Done
assignee: []
created_date: '2026-02-22 10:42'
updated_date: '2026-02-23 20:53'
labels:
  - ci
  - gpu
  - reth
  - production
dependencies: []
references:
  - .github/workflows/ci.yml
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Current CI excludes GPU DPF and reth-adapter crates from workspace checks.

Production confidence requires at least build/test/lint coverage strategy for these components.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Add CI jobs (or scheduled/self-hosted variants) that cover morphogen-gpu-dpf compilation and key tests
- [x] #2 Add CI coverage for reth-adapter build/tests
- [x] #3 Document expected runner capabilities and failure triage workflow for these jobs
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-02-23: Started implementation for CI coverage of morphogen-gpu-dpf and reth-adapter, including runner capability/triage documentation.

2026-02-23: Added dedicated CI jobs for morphogen-gpu-dpf (CPU path), reth-adapter (build/tests + --features reth compile), and optional self-hosted CUDA-path coverage gated by ENABLE_GPU_CI. Added docs/CI_RETH_GPU_COVERAGE.md for runner requirements and triage workflow. Fixed existing reth-adapter --features reth compile/warnings issues so CI coverage can run with -Dwarnings.

2026-02-23: Addressed roborev findings by adding explicit -Dwarnings flags to CUDA CI cargo steps and adding a positive runtime-config test for oversized row size when page PIR is disabled. Re-ran targeted tests and roborev dirty review job 874, which reported no issues.
<!-- SECTION:NOTES:END -->
