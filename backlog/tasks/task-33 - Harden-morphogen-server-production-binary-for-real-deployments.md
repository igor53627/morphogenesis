---
id: TASK-33
title: Harden morphogen-server production binary for real deployments
status: Done
assignee: []
created_date: '2026-02-22 10:42'
updated_date: '2026-02-22 10:54'
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
2026-02-22: Started implementation. Refactoring server binary to load runtime config from CLI/env, add validated startup path, and wire graceful shutdown/background worker lifecycle.

2026-02-22: Completed. Replaced scaffold server startup with resolved runtime config pipeline (CLI > env > JSON file), explicit validation errors, real matrix file loading, optional synthetic matrix generation guard, and page-PIR config parsing/validation.

2026-02-22: Added CUDA startup options for device selection and optional CPU->GPU preload sync path; non-preload mode allocates empty GPU matrix explicitly.

2026-02-22: Wired merge worker lifecycle and graceful shutdown: signal handling (Ctrl+C/SIGTERM), watch-channel stop for merge worker, and clean join on shutdown.

2026-02-22: Added unit tests for runtime config precedence, matrix source validation, PRG key parsing/validation, and matrix-file alignment errors in crates/morphogen-server/src/bin/server.rs.

Verification: cargo test -p morphogen-server --features network --bin server; cargo test -p morphogen-server --features network; cargo clippy --package morphogen-server --features network --bin server -- -D warnings
<!-- SECTION:NOTES:END -->
