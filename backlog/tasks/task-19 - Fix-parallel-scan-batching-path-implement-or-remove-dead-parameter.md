---
id: TASK-19
title: Fix parallel scan batching path (implement or remove dead parameter)
status: Done
assignee: []
created_date: '2026-02-18 19:52'
updated_date: '2026-02-18 20:38'
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
- [x] #1 No dead batching API remains (either real batching exists and is tested, or batch_size path is removed)
- [x] #2 All call sites and public exports compile cleanly with chosen direction
- [x] #3 Tests cover chosen behavior (batch semantics or simplified non-batch path)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Chose simplification path (Scope option 2): removed dead batch_size plumbing and deleted scan_main_matrix_parallel_batched stub, which previously ignored batch_size.
Updated scan_consistent_parallel and scan_consistent_parallel_with_max_retries signatures/call path to invoke scan_main_matrix_parallel directly.
Updated MorphogenServer::scan_parallel caller accordingly.
Added regression test server::tests::scan_parallel_matches_scan to keep simplified behavior explicit.

2026-02-19: Addressed roborev job 784 findings by adding deprecated compatibility wrappers for the old batch-size signatures (to avoid external API breakage) while using the no-batch API internally, and by strengthening scan_parallel_matches_scan with seeded non-zero matrix data, a pending delta, and deterministic RNG.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Server crate tests pass with parallel feature
- [x] #2 Task notes record chosen direction and rationale
<!-- DOD:END -->
