---
id: TASK-38
title: Add CI coverage for morphogen-gpu-dpf and reth-adapter
status: In Progress
assignee: []
created_date: '2026-02-22 10:42'
updated_date: '2026-02-23 19:49'
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
- [ ] #1 Add CI jobs (or scheduled/self-hosted variants) that cover morphogen-gpu-dpf compilation and key tests
- [ ] #2 Add CI coverage for reth-adapter build/tests
- [ ] #3 Document expected runner capabilities and failure triage workflow for these jobs
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-02-23: Started implementation for CI coverage of morphogen-gpu-dpf and reth-adapter, including runner capability/triage documentation.
<!-- SECTION:NOTES:END -->
