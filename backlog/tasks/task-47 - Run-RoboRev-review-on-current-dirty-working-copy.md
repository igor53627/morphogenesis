---
id: TASK-47
title: Run RoboRev review on current dirty working copy
status: Done
assignee: []
created_date: '2026-02-26 17:36'
updated_date: '2026-02-27 11:25'
labels: []
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute roborev review and capture findings for current uncommitted changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 RoboRev executed against dirty tree
- [x] #2 Findings summarized with file/line references
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Ran roborev local dirty review (fast reasoning) and captured findings for server env var parsing, WS test coverage, and ops smoke script strictness.

Daemon review job 934 also completed: findings were Medium on scripts/test_ops_artifacts.sh CI check masking with || true, and Low on block_cache from_block=0 sentinel edge case.

Audit trail (backlog CLI):
- `backlog task list --plain`
- `backlog task view TASK-47 --plain`
- `backlog task edit TASK-47 --status Done --append-notes "..."`

2026-02-27: Rebased change onto main@origin after PR #17 merge to remove divergent-history merge conflicts; resolved six file conflicts and revalidated RoboRev clean state (roborev compact --dry-run => no unaddressed jobs).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
RoboRev findings summary:
1) Medium: env var conflict behavior in server max-concurrent-scans parsing can cause startup regression when both A/B vars are present with different values (`crates/morphogen-server/src/bin/server.rs`, around `EnvConfig::from_process_env` and `parse_env_usize_any`).
2) Low: WS path lacks starvation/error regression tests after async + spawn_blocking refactor (`crates/morphogen-server/src/network/api.rs`, around `handle_ws_single_query` / `handle_ws_batch_query` test coverage).
3) Low: `test_ops_artifacts` docker `--help` check is masked by `|| true` and may hide failures (`scripts/test_ops_artifacts.sh`, around CI `docker run ... --help` smoke checks).

Daemon RoboRev job 934 findings:
1) Medium: `scripts/test_ops_artifacts.sh` uses `docker run ... --help || true`, masking startup failures in smoke checks (`scripts/test_ops_artifacts.sh`, CI image smoke-check section).
2) Low: `block_cache` filter cursor edge case may skip block 0 if polled before cache initialization (`crates/morphogen-rpc-adapter/src/block_cache.rs`, around `get_filter_changes` log-filter cursor advancement).
<!-- SECTION:FINAL_SUMMARY:END -->
