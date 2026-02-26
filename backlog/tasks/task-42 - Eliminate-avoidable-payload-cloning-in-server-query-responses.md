---
id: TASK-42
title: Eliminate avoidable payload cloning in server query responses
status: Done
assignee: []
created_date: '2026-02-25 13:13'
updated_date: '2026-02-25 19:02'
labels: []
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace deep copies (`to_vec`) with ownership moves in hot row/page query handlers and batch response assembly paths.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Row/query handlers avoid deep-cloning [Vec<u8>;3] payload arrays
- [x] #2 Batch query response construction uses move semantics
- [x] #3 Existing API response tests remain green
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-02-25: Replaced hot-path payload array deep copies (to_vec) with move-based conversion in query, batch, and page response assembly paths via payload_array_into_vec, including snapshot batch scan output handling. Added regression test payload_array_into_vec_moves_without_copying and revalidated targeted morphogen-server network tests for query and batch paths.

2026-02-25: Revalidated TASK-42 lifecycle fields through backlog CLI during review follow-up.
<!-- SECTION:NOTES:END -->
