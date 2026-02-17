# KANBAN (Archived)

Active task tracking has migrated to `backlog/` and should be managed with the `backlog` CLI.
This file is retained only for historical notes recorded before migration.

## 2026-02-11

- [x] Address pending roborev review items (448, 449, 450)
- [x] Add backlog epic TASK-17 for deterministic E2E suite with subset data
- [x] Split TASK-17 into subtasks TASK-17.1 through TASK-17.5
- [x] TASK-17 foundation: fixtures, deterministic E2E harness, and CI job
- [x] TASK-17.3 complete: deterministic `eth_call`/`eth_estimateGas` and log/filter assertions
- [x] TASK-17.5 complete: nightly live-upstream smoke workflow with log artifacts

## 2026-02-13

- [x] Address PR #6 review findings (file URL hardening, E2E script robustness, cleanup artifacts)
- [x] Address roborev job 594 findings (explicit file URL root, nightly pipefail, extra resolver tests)
- [x] Address roborev job 605 findings (CAS symlink escape guard, CI e2e log artifacts, explicit cache readiness timeout)
- [x] Address roborev job 607 findings (safe LOG_DIR cleanup, error trap diagnostics, port preflight/env overrides)
- [x] Address roborev job 608 finding (`ss` fallback port-check logic fix)
- [x] roborev job 609 re-review: no new findings
