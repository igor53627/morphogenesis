---
id: TASK-19
title: Fix parallel scan batching path (implement or remove dead parameter)
status: To Do
assignee: []
created_date: '2026-02-18 19:52'
labels:
  - performance
  - server
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The parallel scan API currently accepts batch_size but the batched path is a stub and batch_size is ignored. This adds complexity without performance benefit.

Scope:
- Decide and implement one path:
  1) Implement true batched processing in scan_main_matrix_parallel_batched, or
  2) Remove batch_size plumbing and dead batched wrapper to simplify API.
- Update callers in crates/morphogen-server/src/server.rs and related exports accordingly.
- Keep external behavior explicit and tested.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No dead batching API remains (either real batching exists and is tested, or batch_size path is removed)
- [ ] #2 All call sites and public exports compile cleanly with chosen direction
- [ ] #3 Tests cover chosen behavior (batch semantics or simplified non-batch path)
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Server crate tests pass with parallel feature
- [ ] #2 Task notes record chosen direction and rationale
<!-- DOD:END -->
