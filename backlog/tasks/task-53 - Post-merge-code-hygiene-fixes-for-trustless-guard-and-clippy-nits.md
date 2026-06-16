---
id: TASK-53
title: Post-merge code hygiene fixes for trustless guard and clippy nits
status: Done
assignee: []
created_date: '2026-03-01 11:36'
updated_date: '2026-06-17 21:30'
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

2026-03-02 follow-up: addressing RoboRev PR #19 findings by replacing panic-style trustless guard in reth-adapter APIs with explicit Result error path and adding CLI trustless integration coverage.

2026-03-02 completed follow-up: replaced panic assertions with typed ExtractionError in trustless path, propagated Result to CLI, added CLI integration test for --trustless exit code 2, and validated with cargo fmt/test/clippy plus RoboRev job 1016 (no issues).

2026-06-17 follow-up: closed RoboRev 2×2 (jobs 7794/7795/7796/7797 — codex+claude-code × security+design) findings on the dirty follow-up. Security clean from both agents. Design Medium (both agents): breaking public-API change undocumented + ExtractionError lacked #[non_exhaustive]. Fixed by adding rustdoc to ExtractionError/ExtractionResult/dump_reth_to_matrix/build_matrix documenting the tuple→Result migration, and #[non_exhaustive] on ExtractionError so external match arms need a `_` fallback ahead of future I/O/DB variants. Codex-only Medium on .roborev.toml scoping ambiguity fixed by rewriting the header comment to state explicitly that top-level model applies to the post-commit hook only and [ci] agents ignore it (plus added trailing newline). Validated: cargo test -p reth-adapter (3 unit + 1 CLI integration green), cargo clippy -p reth-adapter --all-targets clean, cargo fmt -p reth-adapter clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed post-merge code hygiene pass: trustless extraction no longer silently proceeds, batch result reconstruction no longer panics on missing rows, and targeted clippy nits were resolved in touched code paths. All targeted tests and clippy runs passed.

2026-03-02 follow-up pass closed RoboRev concerns on PR #19; branch task-53-cleanup now points to 4ed6ab7 and PR updated.

2026-06-17: closing change for this follow-up is jj change_id `yyzmulzv` (git SHA is mutable until merged; self-reference by SHA is intentionally omitted). Two confirmation RoboRev 2x2 passes (jobs 7794-7797 on dirty, 7798-7801 on the committed change): security clean from codex+claude-code; all design findings from the first pass closed. Remaining low/doc findings from the confirmation pass addressed in the same change (derive-drop rustdoc corrected to all four derives + assert_eq!->matches! note) or deferred as non-blocking.
<!-- SECTION:FINAL_SUMMARY:END -->
