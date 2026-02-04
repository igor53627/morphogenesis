---
id: TASK-1
title: Implement eth_getStorageAt RPC method
status: Done
assignee: []
created_date: '2026-02-03 14:29'
updated_date: '2026-02-04 14:05'
labels: []
dependencies: []
references:
  - 3112620 - Add tag tests and legacy fallback
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Enable private ERC20 balance checks by implementing eth_getStorageAt. Storage data already exists in 68GB database (1.5B slots). Queries should use 8-byte Cuckoo tag keys (keccak(address||slot)[0..8]) for addressing; tag verification still uses the full 52-byte key.
<!-- SECTION:DESCRIPTION:END -->
