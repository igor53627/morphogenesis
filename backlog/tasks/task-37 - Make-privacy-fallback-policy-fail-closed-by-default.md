---
id: TASK-37
title: Make privacy fallback policy fail-closed by default
status: Done
assignee: []
created_date: '2026-02-22 10:42'
updated_date: '2026-02-22 17:14'
labels:
  - privacy
  - production
  - rpc-adapter
dependencies: []
references:
  - crates/morphogen-rpc-adapter/src/main.rs
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fallback-to-upstream behavior is useful for resilience but can silently degrade privacy characteristics.

Production policy should default to fail-closed for private methods, with explicit opt-in and observability for degraded mode.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Default behavior for private methods is fail-closed when PIR path is unavailable
- [x] #2 If degraded mode is enabled, emit explicit structured metrics/logging on each privacy-degrading fallback
- [x] #3 Add config guardrails (for example production profile warning/error unless explicit override)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented fail-closed privacy fallback policy with explicit config guardrails in crates/morphogen-rpc-adapter/src/main.rs. Private-method fallback remains disabled by default (fallback_to_upstream=false), and production now requires explicit acknowledgement (--allow-privacy-degraded-fallback) when enabling --fallback-to-upstream.

Added structured observability for every privacy-degrading fallback via centralized record_privacy_degrading_fallback() helper, emitting method/reason fields and monotonic privacy_degraded_fallback_total counter on each upstream fallback from private paths.

Added tests for fallback policy defaults and guardrails: default fail-closed, prod requires explicit override, prod override allowed, non-prod behavior, and fallback counter monotonicity.

Verification: cargo test --package morphogen-rpc-adapter; cargo clippy --package morphogen-rpc-adapter -- -D warnings.

Roborev follow-up hardening: extracted explicit fail_closed_if_fallback_disabled() gate and used it on private cache-miss/log-range fallback paths so fail-closed intent is centralized and testable. Added fail-closed gate unit tests to cover disabled/enabled behavior directly.
<!-- SECTION:NOTES:END -->
