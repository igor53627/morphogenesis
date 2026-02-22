---
id: TASK-33
title: Harden morphogen-server production binary for real deployments
status: Done
assignee: []
created_date: '2026-02-22 10:42'
updated_date: '2026-02-22 13:07'
labels:
  - production
  - server
  - ops
dependencies: []
references:
  - crates/morphogen-server/src/bin/server.rs
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The current production binary is scaffold-level (hardcoded config, demo matrix path, no explicit lifecycle management).

Production rollout needs deterministic config loading, validated startup, background worker lifecycle, and graceful shutdown semantics.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Load runtime config from CLI/env/file (bind address, matrix source, chunk sizing, page settings) with validation and explicit startup errors
- [x] #2 Replace demo/empty data initialization with real matrix load + optional GPU preload/sync path
- [x] #3 Add graceful shutdown (signal handling) and lifecycle wiring for merge workers/background tasks
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-02-22: Addressed roborev job 843 findings by making terminal ctrl-c failure behavior platform-aware (non-Unix forces shutdown, Unix disables ctrl-c waiter and continues relying on SIGTERM) and adding assertions for injected retry-sleep backoff durations in async ctrl-c loop tests.\n\nVerification (fifth follow-up): cargo test -p morphogen-server --features network --bin server; cargo test -p morphogen-server --features network; cargo clippy --package morphogen-server --features network --bin server -- -D warnings
<!-- SECTION:NOTES:END -->
