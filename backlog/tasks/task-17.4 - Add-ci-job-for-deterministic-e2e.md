---
id: TASK-17.4
title: Add CI job for deterministic E2E
status: Done
assignee: []
created_date: '2026-02-11 20:42'
updated_date: '2026-02-11 22:55'
labels:
  - rpc
  - testing
  - integration
  - e2e
  - ci
dependencies:
  - TASK-17.2
parent_task_id: TASK-17
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a PR-gating CI job that runs deterministic E2E against subset fixtures.

Requirements:
- execute full local stack startup in CI
- run deterministic E2E assertions
- archive logs/artifacts on failure for debugging
- keep runtime within practical CI budget

Acceptance:
- CI blocks merges on deterministic E2E failure
- job output is sufficient to diagnose failures without rerunning locally
<!-- SECTION:DESCRIPTION:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a new CI job `Deterministic E2E` in `.github/workflows/ci.yml` that runs
`bash scripts/test_rpc_e2e.sh` on every push/PR.
<!-- SECTION:FINAL_SUMMARY:END -->
