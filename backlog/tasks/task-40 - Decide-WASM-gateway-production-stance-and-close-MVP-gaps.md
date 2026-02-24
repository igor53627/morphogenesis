---
id: TASK-40
title: Decide WASM gateway production stance and close MVP gaps
status: Done
assignee: []
created_date: '2026-02-22 10:42'
updated_date: '2026-02-24 11:20'
labels:
  - wasm
  - product
  - production
dependencies: []
references:
  - docs/WASM_GATEWAY.md
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The WASM gateway is currently documented as MVP with explicit non-goals.

Define whether it is production-supported and align implementation/docs/tests to that stance.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Publish clear support tier (experimental vs production) and compatibility matrix
- [x] #2 If production-targeted, implement or formally gate currently unsupported high-impact flows
- [x] #3 Add CI/release checks aligned with the chosen support tier
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-02-24: Started TASK-40. Plan is to declare WASM gateway support tier explicitly (experimental), publish compatibility matrix, formalize unsupported high-impact flow gating with regression tests, and add CI checks aligned to this tier.

2026-02-24: Chose EXPERIMENTAL support tier for morphogen-wasm-gateway and documented it with explicit runtime/RPC compatibility matrices plus CI/release policy in docs/WASM_GATEWAY.md. Added regression test high_impact_unsupported_methods_are_rejected to formalize blocking of high-impact unsupported wallet/sign/write/filter methods with -32601 and no passthrough. Added dedicated CI lane WASM Gateway (experimental) in .github/workflows/ci.yml to run cargo test -p morphogen-wasm-gateway and wasm32 target build. Updated README WASM docs pointer text accordingly. Verified with cargo fmt --all -- --check, cargo test -p morphogen-wasm-gateway (all tests), and cargo build -p morphogen-wasm-gateway --target wasm32-unknown-unknown under -Dwarnings.

2026-02-24: Addressed RoboRev follow-ups by expanding high_impact_unsupported_methods_are_rejected coverage to representative methods across all documented blocked families (send/sign/accounts/submit/filter/subscription), and extending the WASM Gateway experimental CI lane with wasm-pack node runtime tests. Re-verified via cargo fmt --all -- --check, cargo test -p morphogen-wasm-gateway, cargo build -p morphogen-wasm-gateway --target wasm32-unknown-unknown, and wasm-pack test --node crates/morphogen-wasm-gateway.

2026-02-24: Addressed additional RoboRev supply-chain hardening note by replacing mutable-tag third-party wasm-pack action with locked-toolchain install step (cargo install wasm-pack --locked --version 0.13.1) in WASM Gateway CI lane.

2026-02-24: Addressed post-push CodeRabbit CI hardening nits by adding timeout-minutes: 20 to wasm-gateway job and bumping wasm-pack install to --version 0.14.0.
<!-- SECTION:NOTES:END -->
