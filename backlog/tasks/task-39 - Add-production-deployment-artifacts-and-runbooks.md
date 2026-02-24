---
id: TASK-39
title: Add production deployment artifacts and runbooks
status: Done
assignee: []
created_date: '2026-02-22 10:42'
updated_date: '2026-02-24 14:16'
labels:
  - ops
  - deployment
  - production
dependencies: []
references:
  - ops
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The repository currently lacks deployable packaging and operations runbooks for server/adapter services.

Introduce canonical deployment artifacts plus operational docs for repeatable, auditable releases.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Provide containerization and/or orchestration manifests for core services (server, adapter, supporting dependencies)
- [x] #2 Document environment variables, secret handling, startup ordering, and health checks
- [x] #3 Add runbooks for deploy, rollback, incident response, and routine maintenance
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-02-24: Started TASK-39. Building canonical ops bundle with container artifacts, production compose orchestration, env/secret/startup-order/health documentation, and explicit deploy/rollback/incident/maintenance runbooks.

2026-02-24: Added canonical production ops bundle under ops/: multi-stage Dockerfiles for morphogen-server and morphogen-rpc-adapter, compose orchestration manifest with two server replicas (A/B), adapter, and supporting code-data service plus health checks and startup dependencies (depends_on with service_healthy). Added env templates for server A/B and adapter in ops/env/ and documented required env vars, secret handling, startup ordering, and health verification in ops/README.md. Added runbooks for deploy, rollback, incident response, and routine maintenance under ops/runbooks/. Added artifact validation script scripts/test_ops_artifacts.sh and followed red->green flow (initially failed on missing ops/README.md, then passed after implementation). Validated compose manifest rendering via docker compose config with required env vars set.

2026-02-24: Addressed RoboRev findings by making code-data optional via compose profile/local-code-data with non-blocking dependency, fixing deploy runbook health checks to use docker compose exec for server A/B, aligning server env templates with compose A/B-prefixed variables, hardening runtime images to run as non-root uid 10001, and extending scripts/test_ops_artifacts.sh with compose config validation and additional sanity checks.

2026-02-24: Follow-up RoboRev fixes: required DICT_URL/CAS_URL at compose render time (with explicit local-code-data values), updated deploy runbook and ops README to reflect required adapter inputs and A/B-prefixed server envs, replaced example PRG keys with explicit placeholders to prevent accidental reuse, and made scripts/test_ops_artifacts.sh fail in CI when Docker is unavailable for compose validation.

2026-02-24: Final hardening pass after additional RoboRev findings: switched compose to immutable image refs via MORPHOGEN_SERVER_IMAGE/MORPHOGEN_RPC_ADAPTER_IMAGE with pull+--no-build runbook flow, defaulted RPC host bind to loopback, added local-code-data compose override to gate adapter startup on code-data health in local mode, strengthened code-data/JSON health checks, added secure env-file workflow with ops/env/morphogen-prod.env.example, and updated morphogen-rpc-adapter to accept UPSTREAM_RPC_URL/DICT_URL/CAS_URL from environment (plus test coverage) so sensitive upstream URL values are no longer passed as CLI args in compose.

2026-02-24: Completed final RoboRev hardening cycle (jobs 904-913), including immutable/pull-based deploy flow, env-file runbook consistency, local-code-data override gating, loopback RPC bind default, digest-pinned base/code-data images, and adapter env URL parsing with subprocess integration tests. Final dirty RoboRev job 913 reported no issues.
<!-- SECTION:NOTES:END -->
