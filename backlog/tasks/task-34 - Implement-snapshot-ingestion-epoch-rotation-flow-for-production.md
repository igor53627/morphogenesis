---
id: TASK-34
title: Implement snapshot ingestion + epoch rotation flow for production
status: To Do
assignee: []
created_date: '2026-02-22 10:42'
labels:
  - production
  - server
  - rotation
dependencies: []
references:
  - crates/morphogen-server/src/network/api.rs
  - crates/morphogen-server/src/epoch.rs
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Snapshot/rotation APIs exist but are not production-complete.

We need an operationally safe path to ingest new snapshots, advance epoch metadata, and keep CPU/GPU matrices consistent during rotations.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Implement /admin/snapshot end-to-end request handling with robust validation and error semantics
- [ ] #2 Implement GPU matrix swap path during snapshot submit/rotation with correctness tests
- [ ] #3 Document and test rollback/failure behavior so partial rotations cannot serve inconsistent state
<!-- AC:END -->
