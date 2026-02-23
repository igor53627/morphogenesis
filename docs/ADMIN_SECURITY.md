# Admin Surface Security

This document defines production expectations for `morphogen-server` admin endpoints.

## Secure Defaults

- Default bind address is loopback: `127.0.0.1:3000`.
- Admin routes fail closed when no auth method is configured.
- If you bind to a non-loopback address, run behind TLS termination and perimeter ACLs.

## Upgrade Note

- As of February 22, 2026, the server default bind changed from `0.0.0.0:3000` to `127.0.0.1:3000`.
- If remote reachability was previously relying on defaults, set `MORPHOGEN_SERVER_BIND_ADDR` explicitly during upgrade.

## Admin Authentication Modes

`POST /admin/snapshot` supports either of these auth methods:

1. Bearer token
   - Preferred env var: `MORPHOGEN_ADMIN_BEARER_TOKEN`
   - Legacy alias still accepted: `MORPHOGEN_ADMIN_SNAPSHOT_TOKEN`
   - Request header: `Authorization: Bearer <token>`
   - Backward-compatible header: `x-admin-token: <token>`

2. mTLS-backed identity (via trusted proxy header)
   - Env var: `MORPHOGEN_ADMIN_MTLS_ALLOWED_SUBJECTS` (comma-separated allowlist)
   - Required opt-in: `MORPHOGEN_ADMIN_TRUST_PROXY_HEADERS=true`
   - Optional header name override: `MORPHOGEN_ADMIN_MTLS_SUBJECT_HEADER`
   - Default header name: `x-mtls-subject`
   - Startup fails closed if allowlisted subjects are configured without trusted-proxy opt-in.

If both methods are configured, either one can authorize the request.

## TLS Termination Expectation

When exposing `morphogen-server` beyond localhost (`MORPHOGEN_SERVER_BIND_ADDR` not loopback):

- Terminate TLS at an ingress/reverse proxy (Nginx/Envoy/ALB/etc).
- Restrict `/admin/*` to trusted operator networks/identities.
- Treat mTLS identity headers as trusted only when injected by that proxy.

## Example Outcomes

Allowed (bearer):

```bash
curl -X POST http://127.0.0.1:3000/admin/snapshot \
  -H 'Authorization: Bearer <token>' \
  -H 'Content-Type: application/json' \
  -d '{"r2_url":"https://example.com/snapshot.bin"}'
```

Allowed (mTLS subject header):

```bash
curl -X POST http://127.0.0.1:3000/admin/snapshot \
  -H 'x-mtls-subject: spiffe://morphogenesis/control-plane' \
  -H 'Content-Type: application/json' \
  -d '{"r2_url":"https://example.com/snapshot.bin"}'
```

Denied:

- `401 Unauthorized`: invalid/missing credentials when auth is configured.
- `403 Forbidden`: no admin auth method configured (fail-closed policy).
