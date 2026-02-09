---
id: TASK-4
title: Integration testing with MetaMask/Rabby
status: Done
assignee: []
created_date: '2026-02-03 14:30'
labels: []
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Test RPC adapter with real wallets: connection, transaction flow, block data, logs.

Implemented as `scripts/test_wallet_compat.sh` — a cast-based test script that simulates
all JSON-RPC methods wallets call (chain ID, blocks, account state, gas estimation,
transaction lookup, eth_call, logs, negative tests). Uses `--fallback-to-upstream` against
publicnode.com so no mock PIR servers are needed.
<!-- SECTION:DESCRIPTION:END -->
