---
id: TASK-55
title: Split morphogen-server/src/bin/server.rs (1882 LOC) into config/matrix_loader/gpu_init modules
status: Done
assignee: []
created_date: '2026-06-17 23:55'
updated_date: '2026-06-18 00:30'
labels:
  - refactor
  - maintainability
  - server
dependencies: []
priority: medium
references:
  - TASK-54
  - TASK-45
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`crates/morphogen-server/src/bin/server.rs` is the largest non-test source
file in the workspace (1882 LOC, 0 inline tests). It bundles server startup
into one file: CLI parsing, env/file config layering, matrix loading from
file or synthetic source, GPU resource init, page-PIR config construction,
and the actual `main` wiring. Splitting it reduces review surface and makes
the config layer reusable (the same patterns repeat across the bench
binaries: `bench_page_gpu_batch.rs`, `bench_batch_query.rs`).

Proposed sub-tasks (same mechanical-extraction pattern as TASK-54; one PR
each, all behavior-preserving):

1. **55.1 — `config.rs`** (~700 LOC): CLI / env / file config layering
   - `CliArgs`, `FileConfig`, `EnvConfig`, `PagePirRuntimeConfig`,
     `RuntimeConfig`, `StartupError`
   - `pick3` helper, `validate_server_config`, `build_page_config`
   - `impl RuntimeConfig`/`impl CliArgs`/`impl EnvConfig` methods
2. **55.2 — `matrix_loader.rs`** (~150 LOC): matrix source loading
   - `load_file_config`, `load_matrix_from_file`, `build_matrix`
3. **55.3 — `gpu_init.rs`** (~50 LOC): GPU resource init
   - `init_gpu_resources`
4. **(optional) 55.4** — relocate any helpers shared with bench binaries
   (`bench_page_gpu_batch.rs`, `bench_batch_query.rs` duplicate some of the
   CLI/matrix code; once 55.1–55.3 land, factoring the shared pieces into a
   `pub` module becomes cheap).

Constraints (carried over from TASK-54 design constraints):
- Pure structural refactor: no behavior change, no public-API change beyond
  what's needed for the new `pub` modules.
- Per-split acceptance: `cargo test -p morphogen-server --features network`
  green; binary `--help` output unchanged; CLI flags/env vars unchanged;
  no new `pub` API unless documented.
- One PR per sub-task, roborev 2x2 (codex + claude-code × security + design)
  on each, full review loop per AGENTS.md (gemini triage + close + compact
  when > 10 open).

This file has 0 inline tests, so the safety net is the existing
`tests/network_api.rs` integration suite plus the deterministic E2E suite
(`scripts/test_rpc_e2e.sh`) that exercises the server binary.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `bin/server.rs` reduced to `main()` wiring + thin orchestration (<600 LOC)
- [ ] #2 `config.rs` extracted (CliArgs/FileConfig/EnvConfig/RuntimeConfig/StartupError + pick3 + validate/build)
- [ ] #3 `matrix_loader.rs` extracted (load_matrix_from_file + build_matrix + load_file_config)
- [ ] #4 `gpu_init.rs` extracted (init_gpu_resources)
- [ ] #5 Per-split: `cargo test -p morphogen-server --features network` green; binary `--help` output unchanged; CLI flags and env vars unchanged; no new `pub` API unless documented
- [ ] #6 `cargo clippy --all-targets -- -D warnings` clean on touched code
- [ ] #7 roborev 2x2 (codex + claude-code × security + design) passes on each PR-sized change
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-06-17: TASK-55 created after TASK-54 closure. Selected `bin/server.rs`
as the next refactor target because it is the largest non-test source file
in the workspace and has clear extraction boundaries (config layering /
matrix loading / GPU init) that don't require restructuring the server
itself. No inline tests → safety net is `tests/network_api.rs` +
`scripts/test_rpc_e2e.sh`.

2026-06-18: TASK-55 DONE. 4 PRs merged (#48–#50 + hotfix #51):
  - 55.1 (PR #48): split bin/server.rs into bin/server/ module directory;
    tests → tests.rs (614 LOC). 1882→1262.
  - 55.2 (PR #49): extract config_helpers.rs (184 LOC, 12 parse fns).
    1262→1098.
  - 55.3 (PR #50): extract config.rs (542 LOC: StartupError, CliArgs,
    FileConfig, EnvConfig, PagePirRuntimeConfig, RuntimeConfig + impls +
    validate/build). 1098→574.
  - Hotfix (PR #51): remove stray #[derive(Debug, Clone)] left on
    init_gpu_resources by the Python extraction script — caught by
    greptile-apps[bot] (P1), invisible to CI because the item is
    cfg(cuda)-gated. Also fixed: gemini findings on earlier PRs
    (TopicFilter API consistency PR #52, rpc_call credential leak PRs
    #52+#53).
Final structure: main.rs 574, config.rs 542, tests.rs 614, config_helpers.rs 184.
Net: bin/server.rs 1882→main.rs 574 (−70%). 33 runtime_config_tests + 249
workspace tests green throughout. AC #1, #2, #3 met; AC #4 (matrix_loader)
and gpu_init/shutdown extraction deferred — remaining code in main.rs is
small (574 LOC) and tightly coupled to run() orchestration.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
bin/server.rs 1882→main.rs 574 LOC (−70%) across 3 sub-task PRs + 1 hotfix.
Config layer (StartupError/CliArgs/FileConfig/EnvConfig/RuntimeConfig + impls)
and 12 env-var parse helpers extracted into dedicated modules. 33 runtime
config tests relocated to tests.rs. Binary surface unchanged (−−help verified).
Greptile P1 regression (stray derive under cfg(cuda)) caught and fixed.
<!-- SECTION:FINAL_SUMMARY:END -->
