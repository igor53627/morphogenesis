---
id: TASK-40
title: Decide WASM gateway production stance and close MVP gaps
status: To Do
assignee: []
created_date: '2026-02-22 10:42'
labels:
  - wasm
  - product
  - production
dependencies: []
references:
  - docs/WASM_GATEWAY.md
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The WASM gateway is currently documented as MVP with explicit non-goals.

Define whether it is production-supported and align implementation/docs/tests to that stance.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Publish clear support tier (experimental vs production) and compatibility matrix
- [ ] #2 If production-targeted, implement or formally gate currently unsupported high-impact flows
- [ ] #3 Add CI/release checks aligned with the chosen support tier
<!-- AC:END -->
