#!/bin/bash
set -e

# Wallet Compatibility Test Script
# Simulates the JSON-RPC calls that wallets (MetaMask, Rabby, etc.) make
# when connecting to and operating on the Ethereum network.
#
# Uses --fallback-to-upstream so PIR methods degrade to upstream RPC.
# No mock PIR servers needed - this tests the adapter's passthrough + fallback paths.

PORT=8546
RPC_URL="http://127.0.0.1:$PORT"
PASS=0
FAIL=0
SKIP=0

# Well-known addresses
VITALIK="0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
USDT="0xdAC17F958D2ee523a2206206994597C13D831ec7"
# Known mainnet tx (USDT transfer)
KNOWN_TX="0x5c504ed432cb51138bcf09aa5e8a410dd4a1e204ef84bfed1be16dfba1b22060"

green() { printf "\033[32m%s\033[0m\n" "$1"; }
red()   { printf "\033[31m%s\033[0m\n" "$1"; }
yellow(){ printf "\033[33m%s\033[0m\n" "$1"; }

pass() { PASS=$((PASS + 1)); green "  PASS: $1"; }
fail() { FAIL=$((FAIL + 1)); red   "  FAIL: $1 — $2"; }
skip() { SKIP=$((SKIP + 1)); yellow "  SKIP: $1 — $2"; }

# --- Build ---
echo "Building RPC adapter..."
cargo build -p morphogen-rpc-adapter 2>&1 | tail -1

# --- Start adapter with fallback ---
echo "Starting RPC adapter on port $PORT with --fallback-to-upstream..."
./target/debug/morphogen-rpc-adapter \
    --port "$PORT" \
    --fallback-to-upstream &
ADAPTER_PID=$!

cleanup() {
    echo ""
    echo "Cleaning up..."
    kill "$ADAPTER_PID" 2>/dev/null || true
    wait "$ADAPTER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Wait for adapter to be ready
echo "Waiting for adapter..."
for i in $(seq 1 30); do
    if cast chain-id --rpc-url "$RPC_URL" >/dev/null 2>&1; then
        echo "Adapter ready."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "Adapter failed to start within 30 seconds."
        exit 1
    fi
    sleep 1
done

echo ""
echo "============================================"
echo " Wallet Compatibility Tests"
echo "============================================"

# =============================================================================
# 1. Chain identification (first thing wallets do on connect)
# =============================================================================
echo ""
echo "--- 1. Chain Identification ---"

# eth_chainId
RESULT=$(cast chain-id --rpc-url "$RPC_URL" 2>&1) && RC=$? || RC=$?
if [ $RC -eq 0 ] && [ "$RESULT" = "1" ]; then
    pass "eth_chainId = 1"
else
    fail "eth_chainId" "expected 1, got '$RESULT' (rc=$RC)"
fi

# net_version
RESULT=$(cast rpc net_version --rpc-url "$RPC_URL" 2>&1) && RC=$? || RC=$?
if [ $RC -eq 0 ]; then
    # cast rpc returns JSON-quoted string
    CLEAN=$(echo "$RESULT" | tr -d '"')
    if [ "$CLEAN" = "1" ]; then
        pass "net_version = \"1\""
    else
        fail "net_version" "expected \"1\", got $RESULT"
    fi
else
    fail "net_version" "call failed (rc=$RC): $RESULT"
fi

# web3_clientVersion
RESULT=$(cast rpc web3_clientVersion --rpc-url "$RPC_URL" 2>&1) && RC=$? || RC=$?
if [ $RC -eq 0 ] && [ -n "$RESULT" ] && [ "$RESULT" != '""' ]; then
    pass "web3_clientVersion returned non-empty string"
else
    fail "web3_clientVersion" "expected non-empty, got '$RESULT' (rc=$RC)"
fi

# =============================================================================
# 2. Block data (wallets poll continuously)
# =============================================================================
echo ""
echo "--- 2. Block Data ---"

# eth_blockNumber
RESULT=$(cast block-number --rpc-url "$RPC_URL" 2>&1) && RC=$? || RC=$?
if [ $RC -eq 0 ] && [ "$RESULT" -gt 0 ] 2>/dev/null; then
    pass "eth_blockNumber = $RESULT"
else
    fail "eth_blockNumber" "expected > 0, got '$RESULT' (rc=$RC)"
fi

# eth_getBlockByNumber
RESULT=$(cast block latest --rpc-url "$RPC_URL" 2>&1) && RC=$? || RC=$?
if [ $RC -eq 0 ] && echo "$RESULT" | grep -q "hash"; then
    pass "eth_getBlockByNumber (latest) returned block with hash"
else
    fail "eth_getBlockByNumber" "missing hash field (rc=$RC)"
fi

# eth_gasPrice
RESULT=$(cast gas-price --rpc-url "$RPC_URL" 2>&1) && RC=$? || RC=$?
if [ $RC -eq 0 ] && [ "$RESULT" -gt 0 ] 2>/dev/null; then
    pass "eth_gasPrice = $RESULT"
else
    fail "eth_gasPrice" "expected > 0, got '$RESULT' (rc=$RC)"
fi

# eth_feeHistory
RESULT=$(cast rpc eth_feeHistory '"0x4"' '"latest"' '[]' --rpc-url "$RPC_URL" 2>&1) && RC=$? || RC=$?
if [ $RC -eq 0 ] && echo "$RESULT" | grep -q "baseFeePerGas"; then
    pass "eth_feeHistory returned baseFeePerGas"
else
    fail "eth_feeHistory" "unexpected response (rc=$RC): $(echo "$RESULT" | head -1)"
fi

# =============================================================================
# 3. Account queries (wallet shows balance/state)
# =============================================================================
echo ""
echo "--- 3. Account Queries ---"

# eth_getBalance (EOA - Vitalik)
RESULT=$(cast balance "$VITALIK" --rpc-url "$RPC_URL" 2>&1) && RC=$? || RC=$?
if [ $RC -eq 0 ] && [ -n "$RESULT" ]; then
    pass "eth_getBalance($VITALIK) = $RESULT"
else
    fail "eth_getBalance" "got '$RESULT' (rc=$RC)"
fi

# eth_getTransactionCount (EOA - Vitalik)
RESULT=$(cast nonce "$VITALIK" --rpc-url "$RPC_URL" 2>&1) && RC=$? || RC=$?
if [ $RC -eq 0 ] && [ "$RESULT" -ge 0 ] 2>/dev/null; then
    pass "eth_getTransactionCount($VITALIK) = $RESULT"
else
    fail "eth_getTransactionCount" "got '$RESULT' (rc=$RC)"
fi

# eth_getCode (contract - USDT)
RESULT=$(cast code "$USDT" --rpc-url "$RPC_URL" 2>&1) && RC=$? || RC=$?
if [ $RC -eq 0 ] && [ ${#RESULT} -gt 4 ]; then
    pass "eth_getCode($USDT) returned ${#RESULT} chars of bytecode"
else
    fail "eth_getCode" "expected bytecode, got '${RESULT:0:40}...' (rc=$RC)"
fi

# eth_getCode (EOA - should return 0x or delegation code if EIP-7702)
RESULT=$(cast code "$VITALIK" --rpc-url "$RPC_URL" 2>&1) && RC=$? || RC=$?
if [ $RC -eq 0 ] && echo "$RESULT" | grep -qE '^0x'; then
    pass "eth_getCode(EOA) = ${RESULT:0:20}..."
else
    fail "eth_getCode(EOA)" "expected hex, got '$RESULT' (rc=$RC)"
fi

# eth_getStorageAt (USDT slot 0)
RESULT=$(cast storage "$USDT" 0x0 --rpc-url "$RPC_URL" 2>&1) && RC=$? || RC=$?
if [ $RC -eq 0 ] && echo "$RESULT" | grep -qE '^0x[0-9a-fA-F]{64}$'; then
    pass "eth_getStorageAt($USDT, 0) = $RESULT"
else
    fail "eth_getStorageAt" "expected 32-byte hex, got '$RESULT' (rc=$RC)"
fi

# =============================================================================
# 4. Gas estimation (before sending tx)
# =============================================================================
echo ""
echo "--- 4. Gas Estimation ---"

# eth_maxPriorityFeePerGas
RESULT=$(cast rpc eth_maxPriorityFeePerGas --rpc-url "$RPC_URL" 2>&1) && RC=$? || RC=$?
if [ $RC -eq 0 ] && echo "$RESULT" | grep -qE '^"0x[0-9a-fA-F]+"$'; then
    pass "eth_maxPriorityFeePerGas = $RESULT"
else
    fail "eth_maxPriorityFeePerGas" "expected hex value, got '$RESULT' (rc=$RC)"
fi

# eth_estimateGas (simple ETH transfer)
RESULT=$(cast estimate "$VITALIK" --value 0 --rpc-url "$RPC_URL" --from "$VITALIK" 2>&1) && RC=$? || RC=$?
if [ $RC -eq 0 ] && [ "$RESULT" -gt 0 ] 2>/dev/null; then
    pass "eth_estimateGas = $RESULT"
else
    fail "eth_estimateGas" "expected gas value, got '$RESULT' (rc=$RC)"
fi

# =============================================================================
# 5. Transaction lookup (wallet tracks tx status)
# =============================================================================
echo ""
echo "--- 5. Transaction Lookup ---"

# eth_getTransactionByHash
RESULT=$(cast tx "$KNOWN_TX" --rpc-url "$RPC_URL" 2>&1) && RC=$? || RC=$?
if [ $RC -eq 0 ] && echo "$RESULT" | grep -q "blockNumber"; then
    pass "eth_getTransactionByHash returned tx with blockNumber"
else
    fail "eth_getTransactionByHash" "unexpected response (rc=$RC)"
fi

# eth_getTransactionReceipt
RESULT=$(cast receipt "$KNOWN_TX" --rpc-url "$RPC_URL" 2>&1) && RC=$? || RC=$?
if [ $RC -eq 0 ] && echo "$RESULT" | grep -q "status"; then
    pass "eth_getTransactionReceipt returned receipt with status"
else
    fail "eth_getTransactionReceipt" "unexpected response (rc=$RC)"
fi

# =============================================================================
# 6. eth_call (token balance checks, contract reads)
# =============================================================================
echo ""
echo "--- 6. Contract Reads (eth_call) ---"

# balanceOf on USDT for Vitalik
RESULT=$(cast call "$USDT" "balanceOf(address)(uint256)" "$VITALIK" --rpc-url "$RPC_URL" 2>&1) && RC=$? || RC=$?
if [ $RC -eq 0 ] && [ -n "$RESULT" ]; then
    pass "eth_call USDT.balanceOf($VITALIK) = $RESULT"
else
    fail "eth_call balanceOf" "got '$RESULT' (rc=$RC)"
fi

# =============================================================================
# 7. Logs (event history)
# =============================================================================
echo ""
echo "--- 7. Logs ---"

# Get a recent block number and query a small range
BLOCK_NUM=$(cast block-number --rpc-url "$RPC_URL" 2>&1) && RC=$? || RC=$?
if [ $RC -eq 0 ] && [ "$BLOCK_NUM" -gt 0 ] 2>/dev/null; then
    TARGET_BLOCK=$((BLOCK_NUM - 1))
    RESULT=$(cast logs --from-block "$TARGET_BLOCK" --to-block "$TARGET_BLOCK" --rpc-url "$RPC_URL" 2>&1) && RC=$? || RC=$?
    if [ $RC -eq 0 ]; then
        pass "eth_getLogs returned successfully for block $TARGET_BLOCK"
    else
        fail "eth_getLogs" "call failed (rc=$RC): $(echo "$RESULT" | head -1)"
    fi
else
    skip "eth_getLogs" "could not determine block number"
fi

# =============================================================================
# 8. Negative tests
# =============================================================================
echo ""
echo "--- 8. Negative / Edge Cases ---"

# eth_accounts — should return empty array or "deprecated" error (both acceptable)
RESULT=$(cast rpc eth_accounts --rpc-url "$RPC_URL" 2>&1) && RC=$? || RC=$?
if [ $RC -eq 0 ] && [ "$RESULT" = "[]" ]; then
    pass "eth_accounts = []"
elif echo "$RESULT" | grep -qi "deprecated\|not found\|not supported"; then
    pass "eth_accounts correctly unsupported"
else
    fail "eth_accounts" "unexpected response: '$RESULT' (rc=$RC)"
fi

# eth_sign — should return method-not-found or error
RESULT=$(cast rpc eth_sign --rpc-url "$RPC_URL" 2>&1) && RC=$? || RC=$?
if [ $RC -ne 0 ]; then
    pass "eth_sign correctly rejected"
else
    fail "eth_sign" "expected error, but got success: $RESULT"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================"
TOTAL=$((PASS + FAIL + SKIP))
echo " Results: $PASS passed, $FAIL failed, $SKIP skipped (out of $TOTAL)"
echo "============================================"

if [ $FAIL -gt 0 ]; then
    red "SOME TESTS FAILED"
    exit 1
fi

green "ALL TESTS PASSED"
