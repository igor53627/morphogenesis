---
id: TASK-17
title: Datadog service map and E2E monitoring baseline
status: To Do
assignee: []
created_date: '2026-02-15 00:00'
labels:
  - observability
  - datadog
  - e2e
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Establish a production-like Datadog observability baseline for full E2E protocol runs on real hardware.

### Scope

1. Validate distributed traces and service map path:
   - `morphogen-e2e-client -> morphogen-rpc-adapter`
2. Add automated bootstrap for Datadog assets (requires `DD_API_KEY` + `DD_APP_KEY`):
   - one baseline dashboard for E2E runs
   - a small monitor set for latency/errors/fallback behavior
3. Keep setup idempotent and scriptable for temporary environments (Modal + VM).

### Suggested Deliverables

- `scripts/datadog_bootstrap_e2e.*` (API-driven asset creation/update)
- `docs/DATADOG_E2E.md` section with bootstrap and validation flow
- Exported dashboard/monitor JSON templates checked into repo

### Acceptance Criteria

- Running E2E with Datadog enabled produces traces for both services and a visible dependency edge.
- Bootstrap script creates or updates dashboard and monitors without duplicates.
- Alert thresholds and tags are documented (`env`, `service`, `version`, `run_id`).
<!-- SECTION:DESCRIPTION:END -->
