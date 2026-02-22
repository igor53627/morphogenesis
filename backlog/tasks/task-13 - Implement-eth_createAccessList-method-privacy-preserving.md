---
id: TASK-13
title: Implement eth_createAccessList method (privacy-preserving)
status: Done
assignee: []
created_date: '2026-02-03 14:38'
updated_date: '2026-02-22 03:54'
labels:
  - rpc
  - research
  - optimization
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement `eth_createAccessList` in the RPC adapter as an optional optimization path for callers that want precomputed access lists.

Constraints:
- Keep wallet compatibility unaffected when this method is absent.
- Avoid privacy regressions (do not leak call intent to upstream by default).
- Integrate with existing local EVM / PIR-backed execution where feasible.
<!-- SECTION:DESCRIPTION:END -->
