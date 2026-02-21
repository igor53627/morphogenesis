---
id: TASK-31
title: Prototype per-stream CUDA workspace pool for concurrent page-PIR scans
status: To Do
assignee: []
created_date: '2026-02-20 11:30'
labels:
  - performance
  - cuda
  - server
dependencies: []
documentation:
  - docs/benchmarks/2026-02-19-task-25-page-gpu-batch-h100-mainnet-volume.csv
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Current GPU scan path reuses a single shared batch workspace, which serializes concurrent requests and likely limits throughput under load.

Implement a per-stream (or pooled) CUDA workspace strategy so concurrent requests avoid lock contention on temporary device/host buffers while preserving correctness and bounded memory use.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Workspace allocation strategy supports concurrent GPU requests without a single global buffer lock hot path
- [ ] #2 Correctness parity holds vs current path (matching checksums/results across concurrency tests)
- [ ] #3 H100 benchmark reports throughput/latency at concurrency 1,8,16 and compares against current baseline
<!-- AC:END -->
