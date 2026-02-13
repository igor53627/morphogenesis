---
id: TASK-17.5
title: Add nightly live-upstream smoke tests
status: Done
assignee: []
created_date: '2026-02-11 20:42'
updated_date: '2026-02-11 23:20'
labels:
  - rpc
  - testing
  - integration
  - e2e
  - ci
dependencies:
  - TASK-17.3
parent_task_id: TASK-17
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add non-gating nightly smoke tests against live upstream RPC to catch integration drift.

Scope:
- reuse wallet/filter smoke scripts where appropriate
- run with `--fallback-to-upstream` and track failure trends
- do not block PR merges; alert on regressions

Acceptance:
- nightly workflow runs on schedule
- failures are visible with enough context for triage
<!-- SECTION:DESCRIPTION:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added `.github/workflows/nightly-smoke.yml`:
- scheduled nightly + manual dispatch
- runs `scripts/test_wallet_compat.sh` and `scripts/test_filter_e2e.sh`
- uploads logs as workflow artifacts for triage

This workflow is non-PR-gating by design (no push/pull_request trigger).
<!-- SECTION:FINAL_SUMMARY:END -->
