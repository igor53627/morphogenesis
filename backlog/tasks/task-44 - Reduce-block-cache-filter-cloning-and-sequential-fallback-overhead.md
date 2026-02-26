---
id: TASK-44
title: Reduce block-cache/filter cloning and sequential fallback overhead
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
Optimize RPC adapter block-cache/filter paths to reduce avoidable cloning and improve receipt fallback fetch efficiency under load.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Log/filter retrieval path avoids unnecessary filter reconstruction clones
- [x] #2 Receipt fallback path supports bounded concurrency
- [x] #3 Existing filter/cache behavior is preserved in tests
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-02-25: Removed per-call LogFilter reconstruction in filter polling and log retrieval by scanning with explicit block bounds via get_logs_in_range while preserving cursor semantics. Reworked receipt fallback to bounded concurrent fetching via fetch_receipts_fallback_bounded with deterministic output ordering and bounded in-flight requests. Added async regression test fetch_receipts_fallback_is_bounded_and_preserves_order and revalidated block-cache filter tests.

2026-02-25: Revalidated TASK-44 lifecycle fields through backlog CLI during review follow-up.
<!-- SECTION:NOTES:END -->
