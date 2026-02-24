---
id: TASK-43
title: Move blocking scan retries off async runtime threads
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
Ensure CPU scan/retry loops execute on blocking threads so Tokio worker threads are not blocked during epoch mismatch retry backoff.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 HTTP query handlers offload scan_consistent-style work via spawn_blocking
- [x] #2 Add regression test proving runtime timers are not starved
- [x] #3 No behavior regression in status codes/output payloads
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-02-25: Offloaded retrying scan paths from async handlers onto blocking threads using tokio::task::spawn_blocking in query, batch, and page handlers. Added regression test query_handler_retry_backoff_does_not_starve_runtime_timers (current-thread runtime) to validate timer progress during retry backoff, and confirmed handler behavior remains consistent on targeted tests.

2026-02-25: Revalidated TASK-43 lifecycle fields through backlog CLI during review follow-up.
<!-- SECTION:NOTES:END -->
