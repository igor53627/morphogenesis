---
id: TASK-55
title: Split morphogen-server/src/bin/server.rs (1882 LOC) into config/matrix_loader/gpu_init modules
status: To Do
assignee: []
created_date: '2026-06-17 23:55'
updated_date: '2026-06-17 23:55'
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
<!-- SECTION:NOTES:END -->
