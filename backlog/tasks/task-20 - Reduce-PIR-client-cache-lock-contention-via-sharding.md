---
id: TASK-20
title: Reduce PIR client cache lock contention via sharding
status: To Do
assignee: []
created_date: '2026-02-18 19:52'
labels:
  - performance
  - rpc
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PirClient currently uses a single async Mutex over the whole cache, which can serialize concurrent query paths under load.

Scope:
- Replace single-lock cache design with sharded locking or equivalent low-contention approach.
- Preserve epoch-aware cache invalidation semantics and eviction behavior.
- Validate correctness under concurrent single and batch query flows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 PirClient cache no longer relies on a single global Mutex for all cache operations
- [ ] #2 Epoch-change invalidation and eviction semantics are preserved
- [ ] #3 Concurrency-focused tests added/updated for single and batch query paths
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 morphogen-client tests pass
- [ ] #2 Task notes include contention rationale and measured/observed impact
<!-- DOD:END -->
