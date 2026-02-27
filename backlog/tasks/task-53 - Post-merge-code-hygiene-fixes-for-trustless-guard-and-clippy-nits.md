---
id: TASK-53
title: Post-merge code hygiene fixes for trustless guard and clippy nits
status: Done
assignee: []
created_date: '2026-03-01 11:36'
updated_date: '2026-03-01 11:40'
labels: []
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Apply low-risk cleanup after PR #18: fail fast on unimplemented trustless extraction path, remove unsafe unwrap-invariant in client batch reconstruction, and address clippy nits (div_ceil, io::Error::other).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Trustless mode no longer silently produces output
- [x] #2 Batch result reconstruction avoids unwrap panic on missing entry
- [x] #3 Targeted clippy warnings in touched files are resolved
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented trustless-mode fail-fast guard in reth-adapter (library + CLI), added panic regression test, replaced batch reconstruction unwraps with explicit internal error path + tests, and applied div_ceil/io::Error::other cleanups. Validation: cargo test -p reth-adapter build_matrix_trustless_mode_panics_until_implemented; cargo test -p morphogen-client collect_batch_results_; cargo test -p morphogen-server --features network --bin server wait_for_ctrl_c_signal_with_; cargo test -p morphogen-server --features network gpu_micro_batch_ranges; cargo clippy --package morphogen-server --features network --all-targets; cargo clippy --package morphogen-client --all-targets; cargo clippy --package reth-adapter --all-targets.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed post-merge code hygiene pass: trustless extraction no longer silently proceeds, batch result reconstruction no longer panics on missing rows, and targeted clippy nits were resolved in touched code paths. All targeted tests and clippy runs passed.
<!-- SECTION:FINAL_SUMMARY:END -->
