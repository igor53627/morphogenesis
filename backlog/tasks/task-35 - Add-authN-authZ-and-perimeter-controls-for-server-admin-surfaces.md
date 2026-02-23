---
id: TASK-35
title: Add authN/authZ and perimeter controls for server admin surfaces
status: Done
assignee: []
created_date: '2026-02-22 10:42'
updated_date: '2026-02-22 17:04'
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
- [x] #1 Protect admin endpoints with configurable authentication/authorization (for example bearer token or mTLS-backed identity)
- [x] #2 Define secure defaults for bind/listen strategy and document TLS termination expectations
- [x] #3 Add tests and operator documentation for both allowed and denied access paths
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started TASK-35 implementation. Scoping admin auth methods (bearer + mTLS header), perimeter guardrails, and secure bind defaults with TLS-termination documentation.

Implemented admin authN/authZ controls for /admin/snapshot: Authorization Bearer token support (MORPHOGEN_ADMIN_BEARER_TOKEN, legacy MORPHOGEN_ADMIN_SNAPSHOT_TOKEN still accepted), mTLS-subject allowlist support (MORPHOGEN_ADMIN_MTLS_ALLOWED_SUBJECTS + MORPHOGEN_ADMIN_MTLS_SUBJECT_HEADER), and fail-closed behavior when no admin auth method is configured (403).

Hardened perimeter defaults by changing default bind address to 127.0.0.1:3000 and adding runtime warning when binding non-loopback to reinforce TLS termination + ACL expectations for /admin/* surfaces.

Added tests for allowed and denied admin access paths (bearer allow, mTLS allow, missing/no-config deny) in crates/morphogen-server/src/network/api.rs and crates/morphogen-server/tests/network_api.rs, plus runtime config test for loopback default bind in src/bin/server.rs.

Added operator documentation in docs/ADMIN_SECURITY.md with configuration env vars, TLS termination expectations, and concrete allowed/denied curl paths.

Verification: cargo fmt --package morphogen-server; cargo test --package morphogen-server --features network; cargo clippy --package morphogen-server --features network -- -D warnings.

Roborev follow-up hardening: header-based mTLS auth now requires explicit trusted-proxy opt-in via MORPHOGEN_ADMIN_TRUST_PROXY_HEADERS=true; startup fails closed when mTLS subject allowlist is configured without this opt-in. Added regression tests for lowercase bearer scheme, mixed bearer+legacy token precedence, custom mTLS subject header override, and trust-proxy-disabled denial path.

Addressing roborev job 853 follow-up findings: default-bind migration warning/docs, constant-time bearer-token comparison, and invalid mTLS header-name parsing test coverage.

Roborev 853 follow-up applied: added constant-time admin token comparison helper in network auth path, added runtime test for invalid MORPHOGEN_ADMIN_MTLS_SUBJECT_HEADER parsing, and added bind-source tracking with production warning when default loopback bind is being used.

Added upgrade note to docs/ADMIN_SECURITY.md documenting default bind migration from 0.0.0.0:3000 to 127.0.0.1:3000 and explicit MORPHOGEN_SERVER_BIND_ADDR guidance.

Verification rerun: cargo fmt --package morphogen-server; cargo test --package morphogen-server --features network; cargo clippy --package morphogen-server --features network -- -D warnings.
<!-- SECTION:NOTES:END -->
