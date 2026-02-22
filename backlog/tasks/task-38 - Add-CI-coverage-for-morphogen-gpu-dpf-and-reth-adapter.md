---
id: TASK-38
title: Add CI coverage for morphogen-gpu-dpf and reth-adapter
status: To Do
assignee: []
created_date: '2026-02-22 10:42'
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
