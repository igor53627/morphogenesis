---
id: TASK-17.1
title: Build subset fixtures for deterministic E2E
status: Done
assignee: []
created_date: '2026-02-11 20:42'
updated_date: '2026-02-11 22:55'
labels:
  - rpc
  - testing
  - integration
  - e2e
dependencies: []
parent_task_id: TASK-17
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a reproducible, checked-in subset fixture set for E2E:
- account rows (balance, nonce, code_id)
- storage rows for selected slots
- dictionary file mapping code_id -> code_hash
- CAS bytecode blobs
- minimal block/receipt/log fixtures for cache/filter tests

Acceptance:
- fixture generation is scripted and deterministic
- fixture version/checksum is documented
- fixtures are small enough for CI
<!-- SECTION:DESCRIPTION:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added deterministic subset fixtures under `fixtures/e2e/`:
- `mainnet_compact.dict` with CodeID 1 mapped to hash `0xaa...aa`
- CAS blob `fixtures/e2e/cas/aa/aa/aaaaaaaa....bin` with bytecode `0x60016001`
<!-- SECTION:FINAL_SUMMARY:END -->
