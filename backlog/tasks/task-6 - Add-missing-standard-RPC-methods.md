---
id: TASK-6
title: Add missing standard RPC methods
status: Done
assignee: []
created_date: '2026-02-03 14:38'
updated_date: '2026-02-03 15:06'
labels:
  - rpc
  - immediate
dependencies: []
references:
  - aa13844 - Initial implementation
  - afa4322 - Security fixes (roborev job 21)
  - 9596c98 - Test refactoring (roborev job 22)
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement missing standard Ethereum RPC methods as passthroughs: eth_getStorageAt, eth_getBlockByHash, eth_getProof, wallet signing methods (eth_accounts, eth_sign, eth_signTransaction), and filter APIs (eth_newFilter, eth_getFilterChanges).
<!-- SECTION:DESCRIPTION:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Successfully implemented missing RPC methods with security review and fixes. Added 11 new passthrough methods including storage, filter APIs. Removed unsafe signing methods based on security review. Added test coverage. All roborev findings addressed.
<!-- SECTION:FINAL_SUMMARY:END -->
