---
id: TASK-37
title: Make privacy fallback policy fail-closed by default
status: To Do
assignee: []
created_date: '2026-02-22 10:42'
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
- [ ] #1 Default behavior for private methods is fail-closed when PIR path is unavailable
- [ ] #2 If degraded mode is enabled, emit explicit structured metrics/logging on each privacy-degrading fallback
- [ ] #3 Add config guardrails (for example production profile warning/error unless explicit override)
<!-- AC:END -->
