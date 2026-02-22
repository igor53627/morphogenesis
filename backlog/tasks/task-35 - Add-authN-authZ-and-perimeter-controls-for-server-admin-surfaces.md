---
id: TASK-35
title: Add authN/authZ and perimeter controls for server admin surfaces
status: To Do
assignee: []
created_date: '2026-02-22 10:42'
labels:
  - security
  - production
  - server
dependencies: []
references:
  - crates/morphogen-server/src/network/api.rs
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Admin and high-impact server surfaces need explicit access controls for production.

Current endpoints rely on network placement only; introduce first-class controls and deployment guidance.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Protect admin endpoints with configurable authentication/authorization (for example bearer token or mTLS-backed identity)
- [ ] #2 Define secure defaults for bind/listen strategy and document TLS termination expectations
- [ ] #3 Add tests and operator documentation for both allowed and denied access paths
<!-- AC:END -->
