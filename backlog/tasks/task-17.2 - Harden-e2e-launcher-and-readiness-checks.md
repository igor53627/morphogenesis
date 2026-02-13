---
id: TASK-17.2
title: Harden E2E launcher and readiness checks
status: Done
assignee: []
created_date: '2026-02-11 20:42'
updated_date: '2026-02-11 22:55'
labels:
  - rpc
  - testing
  - integration
  - e2e
dependencies:
  - TASK-17.1
parent_task_id: TASK-17
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Stabilize E2E process orchestration for local and CI runs:
- explicit readiness probes for PIR servers, adapter, dictionary service, and CAS
- deterministic ports and collision handling
- strict cleanup/trap behavior with useful diagnostics on failure
- remove fixed sleeps where possible and replace with bounded polling

Acceptance:
- E2E launcher fails fast with clear root cause when a service is not ready
- repeated local runs are stable and do not leave orphaned processes
<!-- SECTION:DESCRIPTION:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reworked `scripts/test_rpc_e2e.sh` to be deterministic and robust:
- fixture-based execution (no ad-hoc mock CAS server)
- explicit readiness checks for PIR servers and adapter
- strict cleanup and log capture on failures
- deterministic assertions for core private-path methods
<!-- SECTION:FINAL_SUMMARY:END -->
