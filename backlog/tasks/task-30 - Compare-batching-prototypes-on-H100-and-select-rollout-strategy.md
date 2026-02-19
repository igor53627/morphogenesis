---
id: TASK-30
title: Compare batching prototypes on H100 and select rollout strategy
status: To Do
assignee: []
created_date: '2026-02-19 12:05'
labels:
  - performance
  - cuda
  - batching
  - research
dependencies:
  - TASK-26
  - TASK-27
  - TASK-28
  - TASK-29
documentation:
  - docs/plans/2026-02-19-task-24-gpu-page-batch-endpoint.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Run a controlled benchmark matrix across batching prototypes and decide production direction.

Compare baseline, adaptive micro-batch, multi-stream dispatch, CUDA Graph replay, and tiled kernel on H100 with identical datasets and query mixes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Decision report includes latency, throughput, and correctness parity for Q=1,2,4,8,16,32
- [ ] #2 Recommendation includes go/no-go and fallback strategy
- [ ] #3 Selected strategy has explicit rollout guard (feature flag or config)
<!-- AC:END -->
