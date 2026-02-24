#!/usr/bin/env bash
set -euo pipefail

required_files=(
  "ops/README.md"
  "ops/compose/docker-compose.prod.yml"
  "ops/compose/docker-compose.prod.local-code-data.yml"
  "ops/docker/Dockerfile.server"
  "ops/docker/Dockerfile.rpc-adapter"
  "ops/env/morphogen-prod.env.example"
  "ops/env/morphogen-server-a.env.example"
  "ops/env/morphogen-server-b.env.example"
  "ops/env/morphogen-rpc-adapter.env.example"
  "ops/runbooks/deploy.md"
  "ops/runbooks/rollback.md"
  "ops/runbooks/incident-response.md"
  "ops/runbooks/maintenance.md"
)

for file in "${required_files[@]}"; do
  if [[ ! -f "${file}" ]]; then
    echo "missing required artifact: ${file}" >&2
    exit 1
  fi
done

derive_test_prg_key() {
  local label="$1"
  local digest

  if command -v shasum >/dev/null 2>&1; then
    digest="$(printf '%s' "${label}" | shasum -a 256 | awk '{print $1}')"
  elif command -v sha256sum >/dev/null 2>&1; then
    digest="$(printf '%s' "${label}" | sha256sum | awk '{print $1}')"
  else
    echo "shasum or sha256sum is required to derive deterministic test keys" >&2
    exit 1
  fi

  printf '0x%s' "${digest:0:32}"
}

grep -q "morphogen-server-a" ops/compose/docker-compose.prod.yml
grep -q "morphogen-server-b" ops/compose/docker-compose.prod.yml
grep -q "morphogen-rpc-adapter" ops/compose/docker-compose.prod.yml
grep -q "MORPHOGEN_SERVER_IMAGE" ops/compose/docker-compose.prod.yml
grep -q "MORPHOGEN_RPC_ADAPTER_IMAGE" ops/compose/docker-compose.prod.yml
grep -Eq 'profiles:.*local-code-data|-[[:space:]]+"?local-code-data"?' ops/compose/docker-compose.prod.yml
grep -q "/health" ops/compose/docker-compose.prod.yml
grep -q "web3_clientVersion" ops/compose/docker-compose.prod.yml
grep -q 'RPC_BIND_HOST:-127.0.0.1' ops/compose/docker-compose.prod.yml
grep -q "MORPHOGEN_SERVER_A_PAGE_PRG_KEY_0" ops/compose/docker-compose.prod.yml
grep -q "MORPHOGEN_SERVER_B_PAGE_PRG_KEY_0" ops/compose/docker-compose.prod.yml
grep -q '\${DICT_URL:?' ops/compose/docker-compose.prod.yml
grep -q '\${CAS_URL:?' ops/compose/docker-compose.prod.yml
grep -q "code-data" ops/compose/docker-compose.prod.local-code-data.yml
grep -q "condition: service_healthy" ops/compose/docker-compose.prod.local-code-data.yml

grep -q "startup ordering" ops/README.md
grep -q "secret" ops/README.md
grep -q "health" ops/README.md

grep -q "Deploy" ops/runbooks/deploy.md
grep -q "local-code-data" ops/runbooks/deploy.md
grep -q "exec -T morphogen-server-a" ops/runbooks/deploy.md
grep -q "exec -T morphogen-server-b" ops/runbooks/deploy.md
grep -q "docker-compose.prod.local-code-data.yml" ops/runbooks/deploy.md
grep -q -- "--env-file /secure/path/morphogen-prod.env" ops/runbooks/deploy.md
grep -q "DICT_URL" ops/runbooks/deploy.md
grep -q "CAS_URL" ops/runbooks/deploy.md
grep -q "Rollback" ops/runbooks/rollback.md
grep -q "local-code-data" ops/runbooks/rollback.md
grep -q "docker-compose.prod.local-code-data.yml" ops/runbooks/rollback.md
grep -q -- "--env-file /secure/path/morphogen-prod.env" ops/runbooks/rollback.md
grep -q "Incident" ops/runbooks/incident-response.md
grep -q "Maintenance" ops/runbooks/maintenance.md

grep -q "MORPHOGEN_SERVER_IMAGE" ops/env/morphogen-prod.env.example
grep -q "MORPHOGEN_RPC_ADAPTER_IMAGE" ops/env/morphogen-prod.env.example
grep -q "UPSTREAM_RPC_URL" ops/env/morphogen-prod.env.example
grep -q "DICT_URL" ops/env/morphogen-prod.env.example
grep -q "CAS_URL" ops/env/morphogen-prod.env.example
grep -q "MORPHOGEN_SERVER_A_PAGE_PRG_KEY_0" ops/env/morphogen-server-a.env.example
grep -q "MORPHOGEN_SERVER_A_PAGE_PRG_KEY_1" ops/env/morphogen-server-a.env.example
grep -q "MORPHOGEN_SERVER_B_PAGE_PRG_KEY_0" ops/env/morphogen-server-b.env.example
grep -q "MORPHOGEN_SERVER_B_PAGE_PRG_KEY_1" ops/env/morphogen-server-b.env.example
grep -q "USER morphogen:morphogen" ops/docker/Dockerfile.server
grep -q "USER morphogen:morphogen" ops/docker/Dockerfile.rpc-adapter

if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  test_prg_key_a_0="$(derive_test_prg_key "ops-artifacts-server-a-key-0")"
  test_prg_key_a_1="$(derive_test_prg_key "ops-artifacts-server-a-key-1")"
  test_prg_key_b_0="$(derive_test_prg_key "ops-artifacts-server-b-key-0")"
  test_prg_key_b_1="$(derive_test_prg_key "ops-artifacts-server-b-key-1")"

  MORPHOGEN_SERVER_IMAGE="morphogenesis/server:test" \
    MORPHOGEN_RPC_ADAPTER_IMAGE="morphogenesis/rpc-adapter:test" \
    UPSTREAM_RPC_URL="https://rpc.example.invalid" \
    DICT_URL="https://dict.example.invalid/mainnet_compact.dict" \
    CAS_URL="https://dict.example.invalid/cas" \
    MORPHOGEN_SERVER_A_PAGE_PRG_KEY_0="${test_prg_key_a_0}" \
    MORPHOGEN_SERVER_A_PAGE_PRG_KEY_1="${test_prg_key_a_1}" \
    MORPHOGEN_SERVER_B_PAGE_PRG_KEY_0="${test_prg_key_b_0}" \
    MORPHOGEN_SERVER_B_PAGE_PRG_KEY_1="${test_prg_key_b_1}" \
    docker compose -f ops/compose/docker-compose.prod.yml config >/dev/null

  local_cfg="$(mktemp)"
  trap 'rm -f "${local_cfg:-}"' EXIT
  MORPHOGEN_SERVER_IMAGE="morphogenesis/server:test" \
    MORPHOGEN_RPC_ADAPTER_IMAGE="morphogenesis/rpc-adapter:test" \
    UPSTREAM_RPC_URL="https://rpc.example.invalid" \
    DICT_URL="http://code-data/mainnet_compact.dict" \
    CAS_URL="http://code-data/cas" \
    MORPHOGEN_SERVER_A_PAGE_PRG_KEY_0="${test_prg_key_a_0}" \
    MORPHOGEN_SERVER_A_PAGE_PRG_KEY_1="${test_prg_key_a_1}" \
    MORPHOGEN_SERVER_B_PAGE_PRG_KEY_0="${test_prg_key_b_0}" \
    MORPHOGEN_SERVER_B_PAGE_PRG_KEY_1="${test_prg_key_b_1}" \
    docker compose \
    -f ops/compose/docker-compose.prod.yml \
    -f ops/compose/docker-compose.prod.local-code-data.yml \
    --profile local-code-data \
    config >"${local_cfg}"
  grep -q "code-data:" "${local_cfg}"
  rm -f "${local_cfg}"
  trap - EXIT

  if [[ -n "${CI:-}" ]]; then
    trap 'docker rmi morphogenesis/server:ops-smoke morphogenesis/rpc-adapter:ops-smoke >/dev/null 2>&1 || true' EXIT
    docker build -f ops/docker/Dockerfile.server -t morphogenesis/server:ops-smoke . >/dev/null
    docker build -f ops/docker/Dockerfile.rpc-adapter -t morphogenesis/rpc-adapter:ops-smoke . >/dev/null
    docker run --rm --entrypoint sh morphogenesis/server:ops-smoke -c 'id -u | grep -q "^10001$"'
    docker run --rm --entrypoint sh morphogenesis/rpc-adapter:ops-smoke -c 'id -u | grep -q "^10001$"'
    docker run --rm morphogenesis/server:ops-smoke --help >/dev/null || true
    docker run --rm morphogenesis/rpc-adapter:ops-smoke --help >/dev/null || true
    trap - EXIT
    docker rmi morphogenesis/server:ops-smoke morphogenesis/rpc-adapter:ops-smoke >/dev/null 2>&1 || true
  fi
elif [[ -n "${CI:-}" ]]; then
  echo "docker with compose plugin is required in CI to validate compose configuration" >&2
  exit 1
else
  echo "warning: docker compose is not available; skipping compose config validation" >&2
fi

echo "ops artifact checks passed"
