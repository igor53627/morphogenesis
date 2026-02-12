#!/bin/bash
set -Eeuo pipefail

# Deterministic end-to-end tests for the RPC adapter using local subset fixtures.
# This script runs a full local stack:
#   - test_server A
#   - test_server B
#   - morphogen-rpc-adapter
# and validates deterministic private-path RPC responses.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT_A="${PORT_A:-3000}"
PORT_B="${PORT_B:-3001}"
ADAPTER_PORT="${ADAPTER_PORT:-8545}"
UPSTREAM_PORT="${UPSTREAM_PORT:-18545}"
RPC_URL="http://127.0.0.1:${ADAPTER_PORT}"
UPSTREAM_URL="http://127.0.0.1:${UPSTREAM_PORT}"
FIXTURE_DIR="${ROOT_DIR}/fixtures/e2e"
DICT_URL="file://${FIXTURE_DIR}/mainnet_compact.dict"
CAS_URL="file://${FIXTURE_DIR}/cas"

LOG_DIR_CREATED_BY_SCRIPT=0
if [ -n "${LOG_DIR:-}" ]; then
    LOG_DIR="${LOG_DIR}"
else
    LOG_DIR="$(mktemp -d "${TMPDIR:-/tmp}/morphogen_rpc_e2e.XXXXXX")"
    LOG_DIR_CREATED_BY_SCRIPT=1
fi
mkdir -p "${LOG_DIR}"
KEEP_LOGS_ON_FAILURE=0
KEEP_LOGS_ON_SUCCESS="${KEEP_LOGS_ON_SUCCESS:-0}"

SERVER_A_PID=""
SERVER_B_PID=""
ADAPTER_PID=""
UPSTREAM_PID=""

green() { printf "\033[32m%s\033[0m\n" "$1"; }
red() { printf "\033[31m%s\033[0m\n" "$1"; }
yellow() { printf "\033[33m%s\033[0m\n" "$1"; }

fail_and_exit() {
    trap - ERR
    red "❌ $1"
    KEEP_LOGS_ON_FAILURE=1
    echo "Logs:"
    for f in "${LOG_DIR}"/*.log; do
        [ -f "$f" ] || continue
        echo "---- $f (tail -40) ----"
        tail -40 "$f" || true
    done
    exit 1
}

on_unexpected_error() {
    local exit_code=$?
    local line_no="${1:-unknown}"
    fail_and_exit "Unexpected error at line ${line_no} (exit ${exit_code})"
}
trap 'on_unexpected_error ${LINENO}' ERR

cleanup() {
    set +e
    [ -n "${ADAPTER_PID}" ] && kill "${ADAPTER_PID}" 2>/dev/null || true
    [ -n "${UPSTREAM_PID}" ] && kill "${UPSTREAM_PID}" 2>/dev/null || true
    [ -n "${SERVER_A_PID}" ] && kill "${SERVER_A_PID}" 2>/dev/null || true
    [ -n "${SERVER_B_PID}" ] && kill "${SERVER_B_PID}" 2>/dev/null || true
    [ -n "${ADAPTER_PID}" ] && wait "${ADAPTER_PID}" 2>/dev/null || true
    [ -n "${UPSTREAM_PID}" ] && wait "${UPSTREAM_PID}" 2>/dev/null || true
    [ -n "${SERVER_A_PID}" ] && wait "${SERVER_A_PID}" 2>/dev/null || true
    [ -n "${SERVER_B_PID}" ] && wait "${SERVER_B_PID}" 2>/dev/null || true
    if [ "${KEEP_LOGS_ON_FAILURE}" -eq 0 ] && [ "${KEEP_LOGS_ON_SUCCESS}" -eq 0 ]; then
        if [ "${LOG_DIR_CREATED_BY_SCRIPT}" -eq 1 ]; then
            rm -rf "${LOG_DIR}"
        else
            rm -f "${LOG_DIR}/server_a.log" "${LOG_DIR}/server_b.log" \
                "${LOG_DIR}/upstream.log" "${LOG_DIR}/adapter.log"
        fi
    else
        yellow "Logs preserved at ${LOG_DIR}"
    fi
}
trap cleanup EXIT

port_in_use() {
    local port="$1"
    if command -v lsof >/dev/null 2>&1; then
        lsof -nP -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1
        return $?
    fi
    if command -v ss >/dev/null 2>&1; then
        ss -ltn "sport = :${port}" | awk 'NR>1 {found=1} END {exit(found?0:1)}'
        return $?
    fi
    if command -v netstat >/dev/null 2>&1; then
        netstat -an 2>/dev/null | grep -E "[\.:]${port}[[:space:]].*LISTEN" >/dev/null 2>&1
        return $?
    fi
    return 1
}

ensure_port_free() {
    local port="$1"
    local label="$2"
    local env_name="$3"
    if port_in_use "${port}"; then
        fail_and_exit "Port ${port} already in use (${label}). Override with ${env_name}=<port>."
    fi
}

wait_for_url() {
    local url="$1"
    local label="$2"
    local attempts="${3:-50}"
    local sleep_s="${4:-0.2}"

    for _ in $(seq 1 "${attempts}"); do
        if curl -sSf "${url}" >/dev/null 2>&1; then
            return 0
        fi
        sleep "${sleep_s}"
    done
    fail_and_exit "Timed out waiting for ${label} at ${url}"
}

rpc_call() {
    local method="$1"
    local params_json="$2"
    rpc_call_url "${RPC_URL}" "${method}" "${params_json}"
}

rpc_call_url() {
    local url="$1"
    local method="$2"
    local params_json="$3"
    curl -s -X POST "${url}" \
        -H "Content-Type: application/json" \
        -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"${method}\",\"params\":${params_json}}"
}

is_valid_json() {
    echo "$1" | python3 -c '
import json, sys
json.loads(sys.stdin.read())
'
}

json_len() {
    echo "$1" | python3 -c '
import json, sys
v = json.loads(sys.stdin.read())
if isinstance(v, list):
    print(len(v))
else:
    print(0)
'
}

result_of() {
    # Print string result directly; print JSON for array/object; print empty if no result.
    echo "$1" | python3 -c '
import json, sys
r = json.load(sys.stdin)
v = r.get("result")
if v is None:
    print("")
elif isinstance(v, str):
    print(v)
else:
    print(json.dumps(v, separators=(",", ":")))
'
}

error_of() {
    echo "$1" | python3 -c '
import json, sys
r = json.load(sys.stdin)
e = r.get("error") or {}
print(e.get("message", ""))
'
}

assert_eq() {
    local actual="$1"
    local expected="$2"
    local label="$3"
    if [ "${actual}" = "${expected}" ]; then
        green "✅ ${label}: ${actual}"
    else
        fail_and_exit "${label} mismatch. expected='${expected}' actual='${actual}'"
    fi
}

echo "Building components..."
cargo build -p morphogen-server --bin test_server --features network >/dev/null
cargo build -p morphogen-rpc-adapter --bin morphogen-rpc-adapter --bin mock_upstream >/dev/null

ensure_port_free "${PORT_A}" "PIR server A" "PORT_A"
ensure_port_free "${PORT_B}" "PIR server B" "PORT_B"
ensure_port_free "${UPSTREAM_PORT}" "mock upstream" "UPSTREAM_PORT"
ensure_port_free "${ADAPTER_PORT}" "RPC adapter" "ADAPTER_PORT"

if [ ! -f "${FIXTURE_DIR}/mainnet_compact.dict" ]; then
    fail_and_exit "Missing fixture dictionary: ${FIXTURE_DIR}/mainnet_compact.dict"
fi
if [ ! -f "${FIXTURE_DIR}/cas/aa/aa/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.bin" ]; then
    fail_and_exit "Missing fixture CAS blob for CodeID=1"
fi

echo "Starting PIR test servers..."
"${ROOT_DIR}/target/debug/test_server" --port "${PORT_A}" --domain 10 \
    >"${LOG_DIR}/server_a.log" 2>&1 &
SERVER_A_PID=$!

"${ROOT_DIR}/target/debug/test_server" --port "${PORT_B}" --domain 10 \
    >"${LOG_DIR}/server_b.log" 2>&1 &
SERVER_B_PID=$!

wait_for_url "http://127.0.0.1:${PORT_A}/epoch" "PIR server A"
wait_for_url "http://127.0.0.1:${PORT_B}/epoch" "PIR server B"
green "PIR servers are ready."

echo "Starting mock upstream..."
"${ROOT_DIR}/target/debug/mock_upstream" --port "${UPSTREAM_PORT}" \
    >"${LOG_DIR}/upstream.log" 2>&1 &
UPSTREAM_PID=$!
for _ in $(seq 1 60); do
    RESP="$(rpc_call_url "${UPSTREAM_URL}" "eth_blockNumber" "[]")" || true
    if [ -z "${RESP}" ] || ! is_valid_json "${RESP}" >/dev/null 2>&1; then
        sleep 0.2
        continue
    fi
    HEAD="$(result_of "${RESP}")"
    if [ -n "${HEAD}" ]; then
        break
    fi
    sleep 0.2
done
if [ -z "${HEAD:-}" ]; then
    fail_and_exit "Mock upstream did not become ready"
fi
green "Mock upstream is ready."

echo "Starting RPC adapter..."
"${ROOT_DIR}/target/debug/morphogen-rpc-adapter" \
    --port "${ADAPTER_PORT}" \
    --upstream "${UPSTREAM_URL}" \
    --pir-server-a "http://127.0.0.1:${PORT_A}" \
    --pir-server-b "http://127.0.0.1:${PORT_B}" \
    --dict-url "${DICT_URL}" \
    --cas-url "${CAS_URL}" \
    --file-url-root "${FIXTURE_DIR}" \
    >"${LOG_DIR}/adapter.log" 2>&1 &
ADAPTER_PID=$!

# Adapter readiness: first successful private balance query.
for _ in $(seq 1 80); do
    RESP="$(rpc_call "eth_getBalance" '["0x000000000000000000000000000000000000031c","latest"]')" || true
    if [ -z "${RESP}" ] || ! is_valid_json "${RESP}" >/dev/null 2>&1; then
        sleep 0.25
        continue
    fi
    ERR="$(error_of "${RESP}")"
    BAL="$(result_of "${RESP}")"
    if [ -z "${ERR}" ] && [ -n "${BAL}" ]; then
        break
    fi
    sleep 0.25
done

if [ -z "${BAL:-}" ]; then
    fail_and_exit "RPC adapter did not become ready for private queries"
fi
green "RPC adapter is ready."

# Wait until block cache has block 1 so log/filter checks are deterministic.
CACHE_READY=0
for _ in $(seq 1 80); do
    RESP="$(rpc_call "eth_getLogs" '[{"fromBlock":"0x1","toBlock":"0x1"}]')"
    if [ -z "${RESP}" ] || ! is_valid_json "${RESP}" >/dev/null 2>&1; then
        sleep 0.25
        continue
    fi
    ERR="$(error_of "${RESP}")"
    if [ -z "${ERR}" ]; then
        CACHE_READY=1
        break
    fi
    sleep 0.25
done
if [ "${CACHE_READY}" -ne 1 ]; then
    fail_and_exit "Timed out waiting for block cache readiness at block 1"
fi

echo "Running deterministic E2E assertions..."
TEST_ADDR="0x000000000000000000000000000000000000031c"
ZERO_SLOT="0x0"
TRANSFER_TOPIC="0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"

RESP="$(rpc_call "eth_getBalance" "[\"${TEST_ADDR}\",\"latest\"]")"
ERR="$(error_of "${RESP}")"
[ -z "${ERR}" ] || fail_and_exit "eth_getBalance returned error: ${ERR}"
BALANCE="$(result_of "${RESP}")"
assert_eq "${BALANCE}" "0x56bc75e2d63100000" "eth_getBalance"

RESP="$(rpc_call "eth_getTransactionCount" "[\"${TEST_ADDR}\",\"latest\"]")"
ERR="$(error_of "${RESP}")"
[ -z "${ERR}" ] || fail_and_exit "eth_getTransactionCount returned error: ${ERR}"
NONCE="$(result_of "${RESP}")"
assert_eq "${NONCE}" "0x7b" "eth_getTransactionCount"

RESP="$(rpc_call "eth_getCode" "[\"${TEST_ADDR}\",\"latest\"]")"
ERR="$(error_of "${RESP}")"
[ -z "${ERR}" ] || fail_and_exit "eth_getCode returned error: ${ERR}"
CODE="$(result_of "${RESP}")"
assert_eq "${CODE}" "0x60016001" "eth_getCode"

RESP="$(rpc_call "eth_getStorageAt" "[\"${TEST_ADDR}\",\"${ZERO_SLOT}\",\"latest\"]")"
ERR="$(error_of "${RESP}")"
[ -z "${ERR}" ] || fail_and_exit "eth_getStorageAt returned error: ${ERR}"
STORAGE="$(result_of "${RESP}")"
assert_eq "${STORAGE}" "0x0000000000000000000000000000000000000000000000000000000000000000" "eth_getStorageAt(slot=0)"

RESP="$(rpc_call "eth_call" "[{\"to\":\"${TEST_ADDR}\",\"data\":\"0x\"},\"latest\"]")"
ERR="$(error_of "${RESP}")"
[ -z "${ERR}" ] || fail_and_exit "eth_call returned error: ${ERR}"
CALL_OUT="$(result_of "${RESP}")"
assert_eq "${CALL_OUT}" "0x" "eth_call"

RESP="$(rpc_call "eth_estimateGas" "[{\"to\":\"${TEST_ADDR}\",\"data\":\"0x\"},\"latest\"]")"
ERR="$(error_of "${RESP}")"
[ -z "${ERR}" ] || fail_and_exit "eth_estimateGas returned error: ${ERR}"
ESTIMATE_HEX="$(result_of "${RESP}")"
if [[ ! "${ESTIMATE_HEX}" =~ ^0x[0-9a-fA-F]+$ ]]; then
    fail_and_exit "eth_estimateGas returned invalid hex quantity: ${ESTIMATE_HEX}"
fi
ESTIMATE_DEC="$(echo "${ESTIMATE_HEX}" | python3 -c 'import sys; print(int(sys.stdin.read().strip(),16))')"
if [ "${ESTIMATE_DEC}" -le 0 ]; then
    fail_and_exit "eth_estimateGas must be > 0, got ${ESTIMATE_HEX}"
fi
green "✅ eth_estimateGas: ${ESTIMATE_HEX} (${ESTIMATE_DEC})"

RESP="$(rpc_call "eth_getLogs" "[{\"fromBlock\":\"0x1\",\"toBlock\":\"0x1\",\"address\":\"${TEST_ADDR}\",\"topics\":[\"${TRANSFER_TOPIC}\"]}]")"
ERR="$(error_of "${RESP}")"
[ -z "${ERR}" ] || fail_and_exit "eth_getLogs returned error: ${ERR}"
LOGS_JSON="$(result_of "${RESP}")"
LOG_COUNT="$(json_len "${LOGS_JSON}")"
assert_eq "${LOG_COUNT}" "1" "eth_getLogs cached range"

RESP="$(rpc_call "eth_newFilter" "[{\"fromBlock\":\"0x1\",\"toBlock\":\"latest\",\"address\":\"${TEST_ADDR}\",\"topics\":[\"${TRANSFER_TOPIC}\"]}]")"
ERR="$(error_of "${RESP}")"
[ -z "${ERR}" ] || fail_and_exit "eth_newFilter returned error: ${ERR}"
LOG_FILTER_ID="$(result_of "${RESP}")"
if [[ ! "${LOG_FILTER_ID}" =~ ^0x[0-9a-fA-F]+$ ]]; then
    fail_and_exit "eth_newFilter returned invalid filter id: ${LOG_FILTER_ID}"
fi
green "✅ eth_newFilter: ${LOG_FILTER_ID}"

RESP="$(rpc_call "eth_getFilterLogs" "[\"${LOG_FILTER_ID}\"]")"
ERR="$(error_of "${RESP}")"
[ -z "${ERR}" ] || fail_and_exit "eth_getFilterLogs returned error: ${ERR}"
FILTER_LOGS_COUNT="$(json_len "$(result_of "${RESP}")")"
assert_eq "${FILTER_LOGS_COUNT}" "1" "eth_getFilterLogs initial"

RESP="$(rpc_call "eth_getFilterChanges" "[\"${LOG_FILTER_ID}\"]")"
ERR="$(error_of "${RESP}")"
[ -z "${ERR}" ] || fail_and_exit "eth_getFilterChanges (first) returned error: ${ERR}"
FILTER_CHANGES_1="$(json_len "$(result_of "${RESP}")")"
assert_eq "${FILTER_CHANGES_1}" "1" "eth_getFilterChanges first poll"

RESP="$(rpc_call "eth_getFilterChanges" "[\"${LOG_FILTER_ID}\"]")"
ERR="$(error_of "${RESP}")"
[ -z "${ERR}" ] || fail_and_exit "eth_getFilterChanges (second) returned error: ${ERR}"
FILTER_CHANGES_2="$(json_len "$(result_of "${RESP}")")"
assert_eq "${FILTER_CHANGES_2}" "0" "eth_getFilterChanges second poll"

RESP="$(rpc_call "eth_newBlockFilter" "[]")"
ERR="$(error_of "${RESP}")"
[ -z "${ERR}" ] || fail_and_exit "eth_newBlockFilter returned error: ${ERR}"
BLOCK_FILTER_ID="$(result_of "${RESP}")"
if [[ ! "${BLOCK_FILTER_ID}" =~ ^0x[0-9a-fA-F]+$ ]]; then
    fail_and_exit "eth_newBlockFilter returned invalid filter id: ${BLOCK_FILTER_ID}"
fi
green "✅ eth_newBlockFilter: ${BLOCK_FILTER_ID}"

# Mine one upstream block and wait for poller ingestion.
rpc_call_url "${UPSTREAM_URL}" "evm_mine" "[1]" >/dev/null || fail_and_exit "evm_mine failed on mock upstream"
for _ in $(seq 1 40); do
    RESP="$(rpc_call "eth_getFilterChanges" "[\"${BLOCK_FILTER_ID}\"]")"
    ERR="$(error_of "${RESP}")"
    [ -z "${ERR}" ] || fail_and_exit "eth_getFilterChanges (block filter) returned error: ${ERR}"
    BLOCK_CHANGES="$(result_of "${RESP}")"
    BLOCK_CHANGES_COUNT="$(json_len "${BLOCK_CHANGES}")"
    if [ "${BLOCK_CHANGES_COUNT}" -ge 1 ]; then
        break
    fi
    sleep 0.25
done
if [ "${BLOCK_CHANGES_COUNT:-0}" -lt 1 ]; then
    fail_and_exit "eth_getFilterChanges block filter did not observe newly mined block"
fi
green "✅ eth_getFilterChanges block filter observed ${BLOCK_CHANGES_COUNT} block(s)"

green "ALL DETERMINISTIC E2E ASSERTIONS PASSED."
