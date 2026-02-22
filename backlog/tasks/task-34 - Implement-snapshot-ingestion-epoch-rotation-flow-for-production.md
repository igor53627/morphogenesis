---
id: TASK-34
title: Implement snapshot ingestion + epoch rotation flow for production
status: Done
assignee: []
created_date: '2026-02-22 10:42'
updated_date: '2026-02-22 15:58'
labels:
  - production
  - server
  - rotation
dependencies: []
references:
  - crates/morphogen-server/src/network/api.rs
  - crates/morphogen-server/src/epoch.rs
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Snapshot/rotation APIs exist but are not production-complete.

We need an operationally safe path to ingest new snapshots, advance epoch metadata, and keep CPU/GPU matrices consistent during rotations.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implement /admin/snapshot end-to-end request handling with robust validation and error semantics
- [x] #2 Implement GPU matrix swap path during snapshot submit/rotation with correctness tests
- [x] #3 Document and test rollback/failure behavior so partial rotations cannot serve inconsistent state
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-02-22: Started implementation of snapshot ingestion + epoch rotation flow. Scoping /admin/snapshot, EpochManager::submit_snapshot, and metadata wiring for safe rotation semantics with tests-first approach.

2026-02-22: Completed TASK-34.
- Implemented /admin/snapshot end-to-end with robust validation/error semantics (empty/unsupported URL, local file errors, misaligned matrix size) and metadata updates via epoch watch channel.
- Added router integration tests for snapshot success and missing-file failure, plus unit tests for handler validation and state-preserving failure behavior.
- Wired AppState snapshot control surfaces (EpochManager + epoch_tx) and updated server/test/bench constructors.
- Implemented production snapshot submit flow in EpochManager: prepare CUDA replacement matrix before publish, drain pending deltas to next epoch, publish CPU snapshot, then swap prepared GPU matrix when present.
- Added submit_snapshot correctness/failure tests and failure-safety documentation in epoch.rs.
- Added ChunkedMatrix::from_bytes for ingestion plus storage-level tests.

Verification: cargo build --package morphogen-server --features network; cargo test --package morphogen-server --features network; cargo clippy --package morphogen-server --features network -- -D warnings

2026-02-22: Follow-up hardening after roborev job 845. Added fail-closed admin auth token checks for /admin/snapshot, host/local-source restrictions, snapshot size limits, and serialized rotation lock to eliminate metadata races across concurrent submissions. Adjusted tests for auth headers and source-policy validation.\n\n2026-02-22: Hardened submit_snapshot failure ordering on CUDA path by holding GPU matrix lock across preparation+publish+swap, avoiding post-publish lock acquisition failures that could diverge CPU/GPU serving state.

Verification (hardening follow-up): cargo test --package morphogen-server --features network; cargo clippy --package morphogen-server --features network -- -D warnings

2026-02-22: Addressed roborev job 846 follow-up by disabling HTTP redirect following for snapshot fetch, enforcing in-flight streaming byte caps for HTTP/local snapshot reads (no post-read-only size checks), and adding regression tests for redirect rejection and oversize chunked HTTP bodies.

Verification (second hardening follow-up): cargo test --package morphogen-server --features network; cargo clippy --package morphogen-server --features network -- -D warnings

2026-02-22: Addressed roborev page-PIR compatibility and diagnostics follow-up by validating snapshot page alignment/chunk alignment/row-capacity invariants before rotation when page PIR is enabled, adding regression test coverage for incompatible page capacity rejection, and introducing MergeError::GpuUploadFailed for clearer CUDA rotation diagnostics.

Verification (third hardening follow-up): cargo test --package morphogen-server --features network; cargo clippy --package morphogen-server --features network -- -D warnings
<!-- SECTION:NOTES:END -->
