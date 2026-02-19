---
id: TASK-21
title: Research server-side multi-query scan batching ROI
status: Done
assignee: []
created_date: '2026-02-18 21:44'
updated_date: '2026-02-18 21:45'
labels:
  - performance
  - research
  - server
dependencies: []
documentation:
  - docs/plans/2026-02-18-task-21-scan-batching-research.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate whether implementing true scan-level batching in morphogen-server (single matrix pass serving multiple PIR queries) provides meaningful throughput/latency gains versus current per-query scan loop in /query/batch.

Scope:
- Document current behavior and complexity in batch_query_handler and scan paths.
- Capture baseline local measurements for current scan path.
- Propose at least one concrete implementation strategy, risks, and rollout plan.
- Define go/no-go criteria for implementation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Research note includes current-state analysis with code references
- [x] #2 Baseline measurement(s) captured and recorded
- [x] #3 Implementation options + go/no-go criteria documented
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Completed research spike with current-state code-path analysis and baseline measurements.
Baseline (parallel bench_scan, rows=262144, iterations=20): ~107 ms/query, ~0.58 GB/s; confirms current /query/batch still performs per-query scan passes.
Recommendation: prototype fused multi-query scan behind flag (portable path first) and gate rollout on >=1.5x throughput gain at batch size 8 with no Q=1 regression.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Backlog task notes summarize recommendation and next step
<!-- DOD:END -->
