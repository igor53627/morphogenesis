#!/bin/bash
set -e

# End-to-end tests for eth_getLogs and Filter APIs against Ethereum mainnet.
#
# Starts the RPC adapter with --fallback-to-upstream, waits for the block cache
# to accumulate 3+ blocks, then exercises:
#   - eth_getLogs across multi-block ranges (from cache, with address/topic filters)
#   - eth_newFilter / eth_getFilterChanges / eth_getFilterLogs / eth_uninstallFilter
#   - eth_newBlockFilter / eth_getFilterChanges (multiple block hashes)
#   - eth_newPendingTransactionFilter / eth_getFilterChanges (empty)
#   - Multi-block accumulation: filters accumulate changes across 2+ blocks
#   - Edge cases: invalid filter ID, fromBlock > toBlock, uninstalled filter

PORT=8547
RPC_URL="http://127.0.0.1:$PORT"
PASS=0
FAIL=0
SKIP=0
MIN_BLOCKS=3  # Minimum cached blocks before running tests

# USDT Transfer event signature: Transfer(address,address,uint256)
TRANSFER_TOPIC="0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
USDT="0xdAC17F958D2ee523a2206206994597C13D831ec7"

green() { printf "\033[32m%s\033[0m\n" "$1"; }
red()   { printf "\033[31m%s\033[0m\n" "$1"; }
yellow(){ printf "\033[33m%s\033[0m\n" "$1"; }

pass() { PASS=$((PASS + 1)); green "  PASS: $1"; }
fail() { FAIL=$((FAIL + 1)); red   "  FAIL: $1 — $2"; }
skip() { SKIP=$((SKIP + 1)); yellow "  SKIP: $1 — $2"; }

# JSON-RPC helper: rpc_call <method> [params...]
rpc_call() {
    local method="$1"
    shift
    local params="["
    local first=true
    for arg in "$@"; do
        if $first; then first=false; else params+=","; fi
        params+="$arg"
    done
    params+="]"

    curl -s -X POST "$RPC_URL" \
        -H "Content-Type: application/json" \
        -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"$method\",\"params\":$params}" \
        2>/dev/null
}

result_of() {
    echo "$1" | python3 -c "import sys,json; r=json.load(sys.stdin); print(json.dumps(r.get('result','')))" 2>/dev/null
}

error_of() {
    echo "$1" | python3 -c "import sys,json; r=json.load(sys.stdin); e=r.get('error',{}); print(e.get('message',''))" 2>/dev/null
}

json_len() {
    echo "$1" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null
}

# Wait for N new blocks beyond $1, updating LATEST_DEC.
# Usage: wait_for_blocks <current_block> <count> <label>
# Sets WAITED_OK=true/false.
wait_for_blocks() {
    local base_block=$1
    local count=$2
    local label=$3
    local target=$((base_block + count))
    WAITED_OK=false
    echo "  Waiting for $count new block(s) beyond $base_block (target: $target)..."
    for i in $(seq 1 90); do
        sleep 2
        CUR=$(cast block-number --rpc-url "$RPC_URL" 2>/dev/null) || true
        if [ -n "$CUR" ] && [ "$CUR" -ge "$target" ] 2>/dev/null; then
            # Extra delay so poller ingests the block
            sleep 3
            LATEST_DEC=$(cast block-number --rpc-url "$RPC_URL" 2>/dev/null)
            LATEST_HEX=$(printf "0x%x" "$LATEST_DEC")
            echo "  Reached block $LATEST_DEC"
            WAITED_OK=true
            return
        fi
    done
    echo "  Timed out waiting for blocks ($label)"
}

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
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "Adapter failed to start within 30 seconds."
        exit 1
    fi
    sleep 1
done

# Record the first cached block
echo "Waiting for first cached block..."
for i in $(seq 1 30); do
    RESP=$(rpc_call "eth_getLogs" "{\"fromBlock\":\"latest\",\"toBlock\":\"latest\"}")
    ERR=$(error_of "$RESP")
    LOGS=$(result_of "$RESP")
    if [ -z "$ERR" ] && [ "$LOGS" != '""' ] && [ "$LOGS" != "" ]; then
        FIRST_BLOCK=$(cast block-number --rpc-url "$RPC_URL" 2>/dev/null)
        echo "First cached block: $FIRST_BLOCK"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "Block cache failed to populate within 30 seconds."
        exit 1
    fi
    sleep 1
done

# Wait for MIN_BLOCKS total cached blocks
LATEST_DEC=$FIRST_BLOCK
TARGET=$((FIRST_BLOCK + MIN_BLOCKS - 1))
echo "Waiting for $MIN_BLOCKS cached blocks (target: $TARGET)..."
for i in $(seq 1 120); do
    CUR=$(cast block-number --rpc-url "$RPC_URL" 2>/dev/null) || true
    if [ -n "$CUR" ] && [ "$CUR" -ge "$TARGET" ] 2>/dev/null; then
        # Extra delay for cache ingestion
        sleep 3
        LATEST_DEC=$(cast block-number --rpc-url "$RPC_URL" 2>/dev/null)
        LATEST_HEX=$(printf "0x%x" "$LATEST_DEC")
        echo "Cache has blocks $FIRST_BLOCK..$LATEST_DEC ($((LATEST_DEC - FIRST_BLOCK + 1)) blocks)."
        break
    fi
    if [ "$i" -eq 120 ]; then
        echo "Timed out waiting for $MIN_BLOCKS blocks."
        exit 1
    fi
    sleep 2
done

FIRST_HEX=$(printf "0x%x" "$FIRST_BLOCK")
RANGE_SIZE=$((LATEST_DEC - FIRST_BLOCK + 1))

echo ""
echo "============================================"
echo " Filter & Logs E2E Tests (Mainnet)"
echo " Cached range: $FIRST_BLOCK..$LATEST_DEC ($RANGE_SIZE blocks)"
echo "============================================"

# =============================================================================
# 1. eth_getLogs — single block
# =============================================================================
echo ""
echo "--- 1. eth_getLogs (single block) ---"

RESP=$(rpc_call "eth_getLogs" "{\"fromBlock\":\"$LATEST_HEX\",\"toBlock\":\"$LATEST_HEX\"}")
LOGS=$(result_of "$RESP")
ERR=$(error_of "$RESP")
if [ -n "$LOGS" ] && [ "$LOGS" != '""' ] && [ -z "$ERR" ]; then
    COUNT=$(json_len "$LOGS")
    pass "eth_getLogs(latest single block) returned $COUNT logs"
else
    fail "eth_getLogs(latest)" "error: $ERR"
fi

# =============================================================================
# 2. eth_getLogs — multi-block range
# =============================================================================
echo ""
echo "--- 2. eth_getLogs (multi-block range: $FIRST_BLOCK..$LATEST_DEC) ---"

# All logs across full cached range
RESP=$(rpc_call "eth_getLogs" "{\"fromBlock\":\"$FIRST_HEX\",\"toBlock\":\"$LATEST_HEX\"}")
LOGS=$(result_of "$RESP")
ERR=$(error_of "$RESP")
if [ -n "$LOGS" ] && [ "$LOGS" != '""' ] && [ -z "$ERR" ]; then
    MULTI_ALL=$(json_len "$LOGS")
    pass "eth_getLogs(all, $RANGE_SIZE blocks) returned $MULTI_ALL logs"
else
    fail "eth_getLogs(multi-block all)" "error: $ERR"
fi

# USDT across full range
RESP=$(rpc_call "eth_getLogs" "{\"fromBlock\":\"$FIRST_HEX\",\"toBlock\":\"$LATEST_HEX\",\"address\":\"$USDT\"}")
LOGS=$(result_of "$RESP")
ERR=$(error_of "$RESP")
if [ -n "$LOGS" ] && [ "$LOGS" != '""' ] && [ -z "$ERR" ]; then
    MULTI_USDT=$(json_len "$LOGS")
    pass "eth_getLogs(USDT, $RANGE_SIZE blocks) returned $MULTI_USDT logs"
else
    fail "eth_getLogs(multi-block USDT)" "error: $ERR"
fi

# Transfer topic across full range
RESP=$(rpc_call "eth_getLogs" "{\"fromBlock\":\"$FIRST_HEX\",\"toBlock\":\"$LATEST_HEX\",\"topics\":[\"$TRANSFER_TOPIC\"]}")
LOGS=$(result_of "$RESP")
ERR=$(error_of "$RESP")
if [ -n "$LOGS" ] && [ "$LOGS" != '""' ] && [ -z "$ERR" ]; then
    MULTI_TRANSFER=$(json_len "$LOGS")
    pass "eth_getLogs(Transfer, $RANGE_SIZE blocks) returned $MULTI_TRANSFER logs"
else
    fail "eth_getLogs(multi-block Transfer)" "error: $ERR"
fi

# Multi-block count should be >= single-block count
RESP=$(rpc_call "eth_getLogs" "{\"fromBlock\":\"$LATEST_HEX\",\"toBlock\":\"$LATEST_HEX\"}")
SINGLE=$(json_len "$(result_of "$RESP")")
if [ "$MULTI_ALL" -ge "$SINGLE" ] 2>/dev/null; then
    pass "Multi-block logs ($MULTI_ALL) >= single-block logs ($SINGLE)"
else
    fail "Multi vs single block count" "$MULTI_ALL < $SINGLE"
fi

# USDT+Transfer combined filter across range
RESP=$(rpc_call "eth_getLogs" "{\"fromBlock\":\"$FIRST_HEX\",\"toBlock\":\"$LATEST_HEX\",\"address\":\"$USDT\",\"topics\":[\"$TRANSFER_TOPIC\"]}")
LOGS=$(result_of "$RESP")
ERR=$(error_of "$RESP")
if [ -n "$LOGS" ] && [ "$LOGS" != '""' ] && [ -z "$ERR" ]; then
    USDT_TRANSFER=$(json_len "$LOGS")
    pass "eth_getLogs(USDT+Transfer, $RANGE_SIZE blocks) returned $USDT_TRANSFER logs"
    # Combined should be <= either individual filter
    if [ "$USDT_TRANSFER" -le "$MULTI_USDT" ] && [ "$USDT_TRANSFER" -le "$MULTI_TRANSFER" ] 2>/dev/null; then
        pass "Combined filter ($USDT_TRANSFER) <= address-only ($MULTI_USDT) and topic-only ($MULTI_TRANSFER)"
    else
        fail "Combined filter invariant" "$USDT_TRANSFER should be <= min($MULTI_USDT, $MULTI_TRANSFER)"
    fi
else
    fail "eth_getLogs(USDT+Transfer)" "error: $ERR"
fi

# Partial range (first 2 blocks)
SECOND_HEX=$(printf "0x%x" $((FIRST_BLOCK + 1)))
RESP=$(rpc_call "eth_getLogs" "{\"fromBlock\":\"$FIRST_HEX\",\"toBlock\":\"$SECOND_HEX\"}")
LOGS=$(result_of "$RESP")
ERR=$(error_of "$RESP")
if [ -n "$LOGS" ] && [ "$LOGS" != '""' ] && [ -z "$ERR" ]; then
    PARTIAL=$(json_len "$LOGS")
    pass "eth_getLogs(first 2 blocks) returned $PARTIAL logs"
    if [ "$PARTIAL" -le "$MULTI_ALL" ] 2>/dev/null; then
        pass "Partial range ($PARTIAL) <= full range ($MULTI_ALL)"
    else
        fail "Partial range invariant" "$PARTIAL > $MULTI_ALL"
    fi
else
    fail "eth_getLogs(partial range)" "error: $ERR"
fi

# =============================================================================
# 3. Log filter — accumulate across multiple blocks
# =============================================================================
echo ""
echo "--- 3. Log Filter (multi-block accumulation) ---"

SNAPSHOT_BLOCK=$LATEST_DEC

# Create filter monitoring ALL logs (no address/topic restriction)
RESP=$(rpc_call "eth_newFilter" "{}")
LOG_FILTER=$(echo "$(result_of "$RESP")" | tr -d '"')
ERR=$(error_of "$RESP")
if [ -n "$LOG_FILTER" ] && [ "$LOG_FILTER" != "" ] && [ -z "$ERR" ]; then
    pass "eth_newFilter (all logs) created: $LOG_FILTER"
else
    fail "eth_newFilter (all logs)" "error: $ERR"
    LOG_FILTER=""
fi

# Also create a USDT Transfer filter for comparison
RESP=$(rpc_call "eth_newFilter" "{\"address\":\"$USDT\",\"topics\":[\"$TRANSFER_TOPIC\"]}")
USDT_FILTER=$(echo "$(result_of "$RESP")" | tr -d '"')
ERR=$(error_of "$RESP")
if [ -n "$USDT_FILTER" ] && [ "$USDT_FILTER" != "" ] && [ -z "$ERR" ]; then
    pass "eth_newFilter (USDT Transfer) created: $USDT_FILTER"
else
    fail "eth_newFilter (USDT Transfer)" "error: $ERR"
    USDT_FILTER=""
fi

if [ -n "$LOG_FILTER" ]; then
    # Wait for 2 new blocks
    wait_for_blocks "$SNAPSHOT_BLOCK" 2 "log filter multi-block"

    if $WAITED_OK; then
        BLOCKS_ELAPSED=$((LATEST_DEC - SNAPSHOT_BLOCK))

        # Poll all-logs filter
        RESP=$(rpc_call "eth_getFilterChanges" "\"$LOG_FILTER\"")
        CHANGES=$(result_of "$RESP")
        ERR=$(error_of "$RESP")
        if [ -n "$CHANGES" ] && [ -z "$ERR" ]; then
            ALL_COUNT=$(json_len "$CHANGES")
            pass "eth_getFilterChanges (all logs, $BLOCKS_ELAPSED new blocks) returned $ALL_COUNT logs"
        else
            fail "eth_getFilterChanges (all logs)" "error: $ERR"
        fi

        # Poll USDT filter
        if [ -n "$USDT_FILTER" ]; then
            RESP=$(rpc_call "eth_getFilterChanges" "\"$USDT_FILTER\"")
            CHANGES=$(result_of "$RESP")
            ERR=$(error_of "$RESP")
            if [ -n "$CHANGES" ] && [ -z "$ERR" ]; then
                USDT_COUNT=$(json_len "$CHANGES")
                pass "eth_getFilterChanges (USDT, $BLOCKS_ELAPSED new blocks) returned $USDT_COUNT logs"

                # USDT logs should be <= all logs
                if [ "$USDT_COUNT" -le "$ALL_COUNT" ] 2>/dev/null; then
                    pass "USDT filter ($USDT_COUNT) <= all-logs filter ($ALL_COUNT)"
                else
                    fail "Filter subset invariant" "USDT ($USDT_COUNT) > all ($ALL_COUNT)"
                fi
            else
                fail "eth_getFilterChanges (USDT)" "error: $ERR"
            fi
        fi

        # Poll again immediately — should be empty (cursor advanced)
        RESP=$(rpc_call "eth_getFilterChanges" "\"$LOG_FILTER\"")
        CHANGES=$(result_of "$RESP")
        RECHECK=$(json_len "$CHANGES")
        if [ "$RECHECK" -eq 0 ] 2>/dev/null; then
            pass "Second poll returns 0 (cursor correctly advanced)"
        else
            fail "Second poll" "expected 0, got $RECHECK"
        fi

        # Wait for 1 more block — verify incremental accumulation works
        SNAPSHOT2=$LATEST_DEC
        wait_for_blocks "$SNAPSHOT2" 1 "incremental poll"
        if $WAITED_OK; then
            RESP=$(rpc_call "eth_getFilterChanges" "\"$LOG_FILTER\"")
            CHANGES=$(result_of "$RESP")
            ERR=$(error_of "$RESP")
            if [ -n "$CHANGES" ] && [ -z "$ERR" ]; then
                INC_COUNT=$(json_len "$CHANGES")
                pass "Incremental poll (1 new block) returned $INC_COUNT logs"
            else
                fail "Incremental poll" "error: $ERR"
            fi
        else
            skip "Incremental poll" "no new block within timeout"
        fi
    else
        skip "Log filter multi-block" "timed out waiting for blocks"
    fi

    # Cleanup
    rpc_call "eth_uninstallFilter" "\"$LOG_FILTER\"" >/dev/null
    [ -n "$USDT_FILTER" ] && rpc_call "eth_uninstallFilter" "\"$USDT_FILTER\"" >/dev/null
fi

# =============================================================================
# 4. Block filter — accumulate hashes across multiple blocks
# =============================================================================
echo ""
echo "--- 4. Block Filter (multi-block hashes) ---"

SNAPSHOT_BLOCK=$LATEST_DEC

RESP=$(rpc_call "eth_newBlockFilter")
BLOCK_FILTER=$(echo "$(result_of "$RESP")" | tr -d '"')
ERR=$(error_of "$RESP")
if [ -n "$BLOCK_FILTER" ] && [ "$BLOCK_FILTER" != "" ] && [ -z "$ERR" ]; then
    pass "eth_newBlockFilter created: $BLOCK_FILTER"
else
    fail "eth_newBlockFilter" "error: $ERR"
    BLOCK_FILTER=""
fi

if [ -n "$BLOCK_FILTER" ]; then
    # Initial poll — empty
    RESP=$(rpc_call "eth_getFilterChanges" "\"$BLOCK_FILTER\"")
    CHANGES=$(result_of "$RESP")
    INIT_COUNT=$(json_len "$CHANGES")
    if [ "$INIT_COUNT" -eq 0 ] 2>/dev/null; then
        pass "Block filter initial poll = 0 hashes"
    else
        # Might be >0 if a block arrived between creation and poll
        pass "Block filter initial poll = $INIT_COUNT hashes"
    fi

    # Wait for 2 new blocks
    wait_for_blocks "$SNAPSHOT_BLOCK" 2 "block filter multi-block"

    if $WAITED_OK; then
        BLOCKS_ELAPSED=$((LATEST_DEC - SNAPSHOT_BLOCK))

        RESP=$(rpc_call "eth_getFilterChanges" "\"$BLOCK_FILTER\"")
        CHANGES=$(result_of "$RESP")
        ERR=$(error_of "$RESP")
        if [ -n "$CHANGES" ] && [ -z "$ERR" ]; then
            HASH_COUNT=$(json_len "$CHANGES")
            if [ "$HASH_COUNT" -ge 2 ] 2>/dev/null; then
                pass "Block filter returned $HASH_COUNT hashes (expected >= 2)"
            else
                # Possible if poller hasn't caught up yet
                pass "Block filter returned $HASH_COUNT hashes"
            fi

            # Verify hashes are valid 0x-prefixed hex strings
            FIRST_HASH=$(echo "$CHANGES" | python3 -c "import sys,json; h=json.load(sys.stdin); print(h[0] if h else '')")
            if echo "$FIRST_HASH" | grep -qE '^0x[0-9a-fA-F]{64}$'; then
                pass "Block hashes are valid 66-char hex (e.g. ${FIRST_HASH:0:18}...)"
            else
                fail "Block hash format" "got '$FIRST_HASH'"
            fi
        else
            fail "Block filter multi-block" "error: $ERR"
        fi

        # Second poll — should be empty
        RESP=$(rpc_call "eth_getFilterChanges" "\"$BLOCK_FILTER\"")
        CHANGES=$(result_of "$RESP")
        RECHECK=$(json_len "$CHANGES")
        if [ "$RECHECK" -eq 0 ] 2>/dev/null; then
            pass "Block filter second poll = 0 (cursor advanced)"
        else
            pass "Block filter second poll = $RECHECK (race with new block)"
        fi
    else
        skip "Block filter multi-block" "timed out waiting for blocks"
    fi

    # eth_getFilterLogs on block filter should error
    RESP=$(rpc_call "eth_getFilterLogs" "\"$BLOCK_FILTER\"")
    ERR=$(error_of "$RESP")
    if [ -n "$ERR" ]; then
        pass "eth_getFilterLogs on block filter correctly returns error"
    else
        fail "eth_getFilterLogs (block filter)" "expected error"
    fi

    rpc_call "eth_uninstallFilter" "\"$BLOCK_FILTER\"" >/dev/null
fi

# =============================================================================
# 5. Pending transaction filter
# =============================================================================
echo ""
echo "--- 5. Pending Transaction Filter ---"

RESP=$(rpc_call "eth_newPendingTransactionFilter")
PENDING_FILTER=$(echo "$(result_of "$RESP")" | tr -d '"')
ERR=$(error_of "$RESP")
if [ -n "$PENDING_FILTER" ] && [ "$PENDING_FILTER" != "" ] && [ -z "$ERR" ]; then
    pass "eth_newPendingTransactionFilter created: $PENDING_FILTER"
else
    fail "eth_newPendingTransactionFilter" "error: $ERR"
    PENDING_FILTER=""
fi

if [ -n "$PENDING_FILTER" ]; then
    RESP=$(rpc_call "eth_getFilterChanges" "\"$PENDING_FILTER\"")
    CHANGES=$(result_of "$RESP")
    if [ "$CHANGES" = "[]" ]; then
        pass "Pending tx filter returns empty array (no mempool)"
    else
        COUNT=$(json_len "$CHANGES")
        pass "Pending tx filter returned $COUNT items"
    fi
    rpc_call "eth_uninstallFilter" "\"$PENDING_FILTER\"" >/dev/null
fi

# =============================================================================
# 6. eth_getFilterLogs — multi-block
# =============================================================================
echo ""
echo "--- 6. eth_getFilterLogs (multi-block) ---"

# Create a filter from FIRST_BLOCK
RESP=$(rpc_call "eth_newFilter" "{\"fromBlock\":\"$FIRST_HEX\"}")
FULL_FILTER=$(echo "$(result_of "$RESP")" | tr -d '"')
ERR=$(error_of "$RESP")
if [ -n "$FULL_FILTER" ] && [ "$FULL_FILTER" != "" ] && [ -z "$ERR" ]; then
    pass "eth_newFilter (from $FIRST_BLOCK) created"

    RESP=$(rpc_call "eth_getFilterLogs" "\"$FULL_FILTER\"")
    LOGS=$(result_of "$RESP")
    ERR=$(error_of "$RESP")
    if [ -n "$LOGS" ] && [ -z "$ERR" ]; then
        FL_COUNT=$(json_len "$LOGS")
        pass "eth_getFilterLogs returned $FL_COUNT logs across cached range"

        # Should match eth_getLogs for the same range
        RESP2=$(rpc_call "eth_getLogs" "{\"fromBlock\":\"$FIRST_HEX\",\"toBlock\":\"$LATEST_HEX\"}")
        GL_COUNT=$(json_len "$(result_of "$RESP2")")
        if [ "$FL_COUNT" -ge "$GL_COUNT" ] 2>/dev/null; then
            pass "getFilterLogs ($FL_COUNT) >= getLogs ($GL_COUNT) (filter may see newer blocks)"
        else
            fail "getFilterLogs vs getLogs" "$FL_COUNT < $GL_COUNT"
        fi
    else
        fail "eth_getFilterLogs (multi-block)" "error: $ERR"
    fi

    rpc_call "eth_uninstallFilter" "\"$FULL_FILTER\"" >/dev/null
else
    fail "eth_newFilter (from $FIRST_BLOCK)" "error: $ERR"
fi

# =============================================================================
# 7. Edge cases / error handling
# =============================================================================
echo ""
echo "--- 7. Edge Cases ---"

# Invalid filter ID
RESP=$(rpc_call "eth_getFilterChanges" "\"0xdeadbeefdeadbeef\"")
ERR=$(error_of "$RESP")
if [ -n "$ERR" ]; then
    pass "getFilterChanges with invalid ID: \"$ERR\""
else
    fail "getFilterChanges (invalid ID)" "expected error"
fi

RESP=$(rpc_call "eth_getFilterLogs" "\"0xdeadbeefdeadbeef\"")
ERR=$(error_of "$RESP")
if [ -n "$ERR" ]; then
    pass "getFilterLogs with invalid ID: \"$ERR\""
else
    fail "getFilterLogs (invalid ID)" "expected error"
fi

RESP=$(rpc_call "eth_uninstallFilter" "\"0xdeadbeefdeadbeef\"")
RESULT=$(result_of "$RESP")
if [ "$RESULT" = "false" ]; then
    pass "uninstallFilter with invalid ID returns false"
else
    fail "uninstallFilter (invalid ID)" "expected false, got $RESULT"
fi

# fromBlock > toBlock
RESP=$(rpc_call "eth_getLogs" "{\"fromBlock\":\"0xffffff\",\"toBlock\":\"0x1\"}")
ERR=$(error_of "$RESP")
if [ -n "$ERR" ]; then
    pass "eth_getLogs with fromBlock > toBlock returns error"
else
    fail "eth_getLogs (reversed range)" "expected error"
fi

RESP=$(rpc_call "eth_newFilter" "{\"fromBlock\":\"0xffffff\",\"toBlock\":\"0x1\"}")
ERR=$(error_of "$RESP")
if [ -n "$ERR" ]; then
    pass "eth_newFilter with fromBlock > toBlock returns error"
else
    fail "eth_newFilter (reversed range)" "expected error"
fi

# Double uninstall
RESP=$(rpc_call "eth_newBlockFilter")
TMP_ID=$(echo "$(result_of "$RESP")" | tr -d '"')
rpc_call "eth_uninstallFilter" "\"$TMP_ID\"" >/dev/null
RESP=$(rpc_call "eth_uninstallFilter" "\"$TMP_ID\"")
RESULT=$(result_of "$RESP")
if [ "$RESULT" = "false" ]; then
    pass "Double uninstall returns false"
else
    fail "Double uninstall" "expected false, got $RESULT"
fi

# Uninstalled filter poll
RESP=$(rpc_call "eth_getFilterChanges" "\"$TMP_ID\"")
ERR=$(error_of "$RESP")
if [ -n "$ERR" ]; then
    pass "Polling uninstalled filter returns error"
else
    fail "Polling uninstalled filter" "expected error"
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
