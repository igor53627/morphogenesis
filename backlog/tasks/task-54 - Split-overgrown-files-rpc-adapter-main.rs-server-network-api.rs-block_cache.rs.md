---
id: TASK-54
title: >-
  Split overgrown files: rpc-adapter main.rs, server network/api.rs,
  block_cache.rs
status: In Progress
assignee: []
created_date: '2026-06-16 21:25'
updated_date: '2026-06-17 23:30'
labels:
  - refactor
  - maintainability
  - rpc-adapter
  - server
dependencies: []
priority: medium
references:
  - TASK-45
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Motivation: TASK-45 (Golang rewrite feasibility) signals maintainability pain. Three files have grown past 2000 LOC with mixed concerns, making them hard to review and test in isolation. Refactor reduces risk and keeps Rust architecture viable.

Scope (ordered by value/risk ratio):

1. crates/morphogen-rpc-adapter/src/main.rs (2361 LOC)
   - main() spans ~650 lines (622..1275) wiring 19 RPC handlers + proxy + filter logic
   - 34 #[test] in-file (lines 1365..end) should move to tests/
   - Split into modules: handlers/ (eth_*, proxy), config (Args/AdapterEnvironment), filter (log filter helpers), proxy (upstream forwarding)
   - Keep public CLI surface identical; existing tests are the safety net

2. crates/morphogen-server/src/network/api.rs (3966 LOC)
   - 20+ handlers + GPU batch policy + hex serializers + admin auth + snapshot fetch in one file
   - Split into: handlers/{query,batch,page,admin,snapshot}.rs, auth.rs, gpu_batch.rs, serde_hex.rs
   - Safety net: tests/network_api.rs (1862 LOC integration)

3. crates/morphogen-rpc-adapter/src/block_cache.rs (2065 LOC)
   - Separate FIFO cache, reorg detection, filter state, finality resolution

Constraints:
- Pure structural refactor: no behavior change, no API surface change
- Preserve all existing tests green; move in-file tests to tests/ where applicable
- TDD where new module boundaries need characterization tests
- One file per PR-sized change, roborev 2x2 before merge
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 rpc-adapter `main.rs` reduced to CLI wiring only (<400 LOC); handlers/proxy/filter/config extracted to modules (54.1: main.rs 2361→13 LOC, lib extraction)
- [x] #2 Inline `#[test]` blocks relocated out of the lib root (54.7: moved to src/tests.rs via `mod tests;` declaration; super:: imports unchanged)
- [ ] #3 `morphogen-server/src/network/api.rs` split into handler/auth/gpu_batch/serde_hex submodules; no file >1200 LOC (PENDING — separate crate, future work)
- [ ] #4 `block_cache.rs` split by concern (FIFO / reorg / filters / finality) (PENDING — future work)
- [x] #5 Per-split acceptance: `cargo test --workspace --all-features` green; binary `--help` output unchanged; CLI flags and env vars unchanged; no new `pub` API unless documented (verified per-PR)
- [x] #6 `cargo clippy --workspace --all-targets --all-features` clean on touched crates
- [x] #7 roborev 2x2 (codex+claude-code x security+design) passes on each PR-sized change (54.1–54.7 all passed)
<!-- AC:END -->

## Design constraints (from RoboRev codex job 7821 on TASK-54.1)

<!-- SECTION:DESIGN:BEGIN -->
**Public API policy for the new lib crate.** `morphogen_rpc_adapter::run()` is currently the only `pub` item; structs (`Args`, `AdapterState`), consts (`DROPPED_METHODS`, `PASSTHROUGH_METHODS`, `RELAY_METHODS`), and helpers remain crate-private. Future splits must NOT broad-`pub` internals to make tests compile — promote only intentional seams and document each. If `run()`'s internal `Args::parse()` (process-global coupling) becomes awkward for reuse/testing, refactor it to take a parsed config argument in a follow-up.

**Test strategy (gates AC #2).** The 148 inline tests reach private items via `use super::{...}`. Moving them to integration `tests/` requires either (a) promoting tested items to `pub`/`pub(crate)` — an API change that needs its own review — or (b) keeping behavior-specific tests as unit `mod tests` colocated with the module they cover (preferred). Do NOT mechanically move tests to `tests/` before settling module boundaries; that would force inconsistent visibility choices and circular deps around `AdapterState`/`Args`/handler registration.

**Proposed module plan** (settle before 54.2+): `config` (Args/AdapterEnvironment/validate_privacy_fallback_config), `state` (AdapterState + privacy counters), `proxy` (upstream forwarding + redaction), `filters` (log filter helpers), `methods` (register_* + handle_eth_*). Decide crate-public vs private per item up front.

**Preserve invariants across all splits:** privacy fail-closed policy, URL/credential redaction in telemetry + proxy errors, file-URL allowlisting for snapshot sources, dropped methods (`eth_getProof`/`eth_sign`/`eth_signTransaction`), relay routing of `eth_sendRawTransaction`, bind to `127.0.0.1` only.
<!-- SECTION:DESIGN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-06-16: Draft created after backlog triage — all 55 active tasks Done, only research items (TASK-14, TASK-41) remain in Icebox. Refactor chosen as highest-value next work given TASK-45 maintainability signal. Starting point: rpc-adapter main.rs (best test coverage, lowest risk, product entry point).

2026-06-17: rpc-adapter scope COMPLETE via 7 sub-task PRs (54.1–54.7):
  - 54.1 (PR #22): lib extraction — main.rs 2361→13 LOC
  - 54.2 (PR #23): proxy module
  - 54.3 (PR #24): config module
  - 54.4 (PR #25): filters module
  - 54.5 (PR #26): state module (pub(crate) AdapterState seam)
  - 54.6 (PR #27): methods module (handlers + register_*)
  - 54.7 (PR #28): tests.rs (inline mod tests → src/tests.rs)
Final structure: lib.rs 699, tests.rs 985, methods.rs 330, config.rs 159, filters.rs 133, state.rs 115, proxy.rs 115, main.rs 13. Net: main.rs 2361→13, lib.rs 2361→699. 166 tests green throughout; clippy/fmt clean; roborev 2x2 passed on every PR (codex+claude-code × security+design, all no-issues/Pass). Privacy invariants (TASK-37 fail-closed, URL redaction, dropped/relay method tables, CLI/env compatibility) preserved across all 7 PRs.
Remaining scope (separate future work): AC #3 morphogen-server/src/network/api.rs (3966 LOC) and AC #4 block_cache.rs (2065 LOC) — different files/crates, can follow the same mechanical module-extraction pattern.
<!-- SECTION:NOTES:END -->
