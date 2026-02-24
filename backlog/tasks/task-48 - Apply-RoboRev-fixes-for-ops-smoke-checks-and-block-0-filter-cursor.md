---
id: TASK-48
title: Apply RoboRev fixes for ops smoke checks and block-0 filter cursor
status: Done
assignee: []
created_date: '2026-02-26 17:53'
updated_date: '2026-02-27 11:25'
labels: []
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address RoboRev findings: remove masked docker run failures in ops smoke script and fix block-cache filter cursor edge case for block 0 after empty first poll.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Regression test covers block-0-after-empty-poll scenario
- [x] #2 Block cache logic returns block 0 correctly after delayed insertion
- [x] #3 Ops smoke script fails on broken runtime startup instead of masking failures
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started TDD fix pass for RoboRev findings on block_cache cursor and ops smoke script strictness.

Added regression test filter_log_get_changes_includes_block_zero_after_empty_initial_poll and implemented cursor guard so sentinel is preserved until block 0 is actually cached.

Removed '|| true' masking from CI docker --help runtime smoke checks in scripts/test_ops_artifacts.sh.

Validation: cargo test -p morphogen-rpc-adapter filter_log_get_changes_includes_block_zero ; cargo test -p morphogen-rpc-adapter block_cache::tests::filter_log_get_changes ; bash scripts/test_ops_artifacts.sh.

Post-fix local roborev fast pass no longer reports the two targeted findings; it still reports unrelated/pre-existing findings in server env var precedence and telemetry path redaction.

Reopened to address remaining RoboRev findings from follow-up pass: (1) explicit env var ambiguity handling for MORPHOGEN_SERVER_{A,B}_MAX_CONCURRENT_SCANS, and (2) delayed block-0 handling when latest_block advances before block 0 appears.

Follow-up pass addressed remaining and newly surfaced RoboRev findings: (1) parse_env_usize_any now enforces conflict rejection across preferred prefixed keys (A/B) while preserving legacy fallback semantics, and (2) log filter state now tracks pending late block-0 delivery so higher blocks continue streaming while block 0 remains pending, then emits block 0 exactly once when available. Also adjusted get_logs_in_range to avoid early break on out-of-order inserts and preserved chronological ordering by prepending delayed block-0 logs when mixed with newer logs in one poll.\n\nAdded regression tests: parse_env_usize_any_rejects_conflicting_prefixed_values; filter_log_get_changes_includes_block_zero_after_latest_advances_before_cache_warmup; filter_log_get_changes_returns_higher_blocks_while_block_zero_pending; filter_log_get_changes_keeps_block_zero_before_newer_logs_when_returned_together.\n\nValidation reruns: cargo test -p morphogen-server --features network --bin server parse_env_usize_any ; cargo test -p morphogen-rpc-adapter block_cache::tests::filter_log_get_changes ; cargo test -p morphogen-rpc-adapter telemetry::tests::redact_url_for_effective_config ; cargo test -p morphogen-rpc-adapter --test config_env `effective_config_` ; cargo test -p morphogen-server --features network ws_single_query_retry_backoff_does_not_starve_runtime_timers ; cargo test -p morphogen-server --features network ws_batch_query_retry_backoff_does_not_starve_runtime_timers ; roborev review --dirty --local --fast.

Final RoboRev closure pass: addressed follow-up WS concerns by adding bounded in-flight websocket queueing (), enforcing size checks on queued text frames, and preserving socket continuity after  /  errors. Added websocket integration tests for queue overflow and oversized queued-frame handling while a request is in-flight. Also tightened telemetry redaction to preserve only explicit safe path segments (including ) and added explicit runtime-config test that preferred scan-concurrency env values win over conflicting legacy fallback values.\n\nFinal verification: cargo test -p morphogen-server --features network ws_query_rejects_unbounded_queue_growth ; cargo test -p morphogen-server --features network ws_query_rejects_oversized_message_while_request_in_flight ; cargo test -p morphogen-server --features network --bin server parse_env_usize_any ; cargo test -p morphogen-rpc-adapter telemetry::tests::redact_url_for_effective_config ; roborev review --dirty --local --fast (No issues found).

Correction: WS follow-up hardening added bounded in-flight websocket queueing via MAX_WS_QUEUED_MESSAGES, enforced queued-frame size checks with message_too_large handling, and preserved socket continuity after message_too_large / too_many_messages errors. Added websocket integration tests for queue overflow and oversized queued-frame behavior during in-flight request processing. Telemetry redaction now preserves only explicit safe path segments, including mainnet_compact.dict. Added explicit runtime-config test that preferred scan-concurrency env values win over conflicting legacy fallback values. Final verification includes targeted cargo tests and roborev review --dirty --local --fast with no findings.

Post-completion metadata reconciliation: RoboRev daemon still tracked historical dirty-review jobs as unaddressed. Ran roborev compact --wait --limit 100 (job 938) to verify 32 prior jobs against current tree, auto-mark legacy jobs addressed, then explicitly marked consolidated job 938 addressed to clear UI status. Verification now returns 'No unaddressed jobs found'.

Packaging current TASK-42/43/44+RoboRev follow-up changes for PR publication (jj commit/push + GitHub PR creation).

Published PR for current change set: https://github.com/igor53627/morphogenesis/pull/18 (branch: task-42-43-44-opt -> main).

2026-02-27: Adjusted websocket oversized-message tests to accept protocol-level close/reset when Axum WebSocketUpgrade enforces max frame/message size before app-level queue handling. Verified with: cargo test -p morphogen-server --features network websocket_query::.

2026-02-27: Resolved post-rebase conflict integration for websocket/API and adapter hot-path fixes, preserving protocol-level WS size-limit behavior and block-0 filter cursor semantics; verified via package tests.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Applied RoboRev-targeted fixes:
1) Fixed block-cache log filter edge case where first empty poll could skip later block 0 insertion.
2) Restored strict CI smoke behavior by removing docker run failure masking in ops artifact script.
All targeted validations passed.

Reopened follow-up complete: all remaining RoboRev findings (including second-pass regressions) are resolved with targeted tests and code fixes. Final local RoboRev fast review reports no issues.

Completed additional iterative RoboRev remediations (WS queue bounds/continuity, queued oversize guard, telemetry allowlist hardening, precedence-policy test coverage). Final RoboRev local fast review reports no findings.
<!-- SECTION:FINAL_SUMMARY:END -->
