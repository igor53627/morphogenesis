#!/bin/bash
set -euo pipefail

ENABLE_DATADOG="${ENABLE_DATADOG:-0}"
OTEL_ENDPOINT="${OTEL_ENDPOINT:-http://127.0.0.1:4317}"
OTEL_SERVICE_NAME="${OTEL_SERVICE_NAME:-morphogen-rpc-adapter}"
OTEL_CLIENT_SERVICE_NAME="${OTEL_CLIENT_SERVICE_NAME:-morphogen-e2e-client}"
OTEL_ENV="${OTEL_ENV:-e2e}"
OTEL_VERSION="${OTEL_VERSION:-local}"

# 1. Build components
echo "Building components..."
cargo build -p morphogen-server --bin test_server --features network
cargo build -p morphogen-rpc-adapter
if [[ "${ENABLE_DATADOG}" == "1" ]]; then
    cargo build -p morphogen-rpc-adapter --bin morphogen-rpc-dd-client
fi

# 2. Setup Mock CAS (Dictionary + Bytecode)
echo "Setting up Mock CAS..."
mkdir -p /tmp/morphogen_cas/cas
cd /tmp/morphogen_cas

# Create Dictionary: Map CodeID 1 -> Hash 0xAAAAAAAAAAAAAAAA...
# CodeID 1 is at offset 32 (1 * 32).
# We pad 32 bytes of zeros (for ID 0), then 32 bytes of 0xAA.
# Use python to generate binary file
python3 -c "import sys; sys.stdout.buffer.write(b'\x00'*32 + b'\xAA'*32)" > mainnet_compact.dict

# Create Bytecode for Hash 0xAAAAAAAAAAAAAAAA...
# Path: cas/aa/aa/aaaaaaaa....bin
HASH="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
SHARD1="aa"
SHARD2="aa"
mkdir -p "cas/$SHARD1/$SHARD2"
# Bytecode: 0x60016001 (PUSH1 01 PUSH1 01)
python3 -c "import sys; sys.stdout.buffer.write(b'\x60\x01\x60\x01')" > "cas/$SHARD1/$SHARD2/$HASH.bin"

# Start Range-Aware HTTP Server for CAS
cat > cas_server.py << 'PY'
import http.server
import os
import re

class RangeRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        range_header = self.headers.get('Range')
        if range_header:
            match = re.search(r'bytes=(\d+)-(\d+)', range_header)
            if match:
                start, end = int(match.group(1)), int(match.group(2))
                path = self.translate_path(self.path)
                if os.path.exists(path):
                    with open(path, 'rb') as f:
                        f.seek(start)
                        chunk = f.read(end - start + 1)
                        self.send_response(206)
                        self.send_header('Content-Type', 'application/octet-stream')
                        self.send_header('Content-Range', f'bytes {start}-{end}/{os.path.getsize(path)}')
                        self.send_header('Content-Length', str(len(chunk)))
                        self.end_headers()
                        self.wfile.write(chunk)
                        return
        super().do_GET()

if __name__ == '__main__':
    http.server.test(HandlerClass=RangeRequestHandler, port=8088)
PY

python3 cas_server.py &
CAS_PID=$!
cd - > /dev/null

# 3. Start Servers
echo "Starting PIR Servers..."
./target/debug/test_server --port 3000 --domain 10 &
SERVER_A_PID=$!
./target/debug/test_server --port 3001 --domain 10 &
SERVER_B_PID=$!

# 4. Start RPC Adapter
echo "Starting RPC Adapter..."
ADAPTER_OTEL_ARGS=()
if [[ "${ENABLE_DATADOG}" == "1" ]]; then
    echo "OpenTelemetry export enabled (endpoint: ${OTEL_ENDPOINT})"
    ADAPTER_OTEL_ARGS=(
        --otel-traces
        --otel-endpoint "${OTEL_ENDPOINT}"
        --otel-service-name "${OTEL_SERVICE_NAME}"
        --otel-env "${OTEL_ENV}"
        --otel-version "${OTEL_VERSION}"
    )
fi

if [[ "${ENABLE_DATADOG}" == "1" ]]; then
    ./target/debug/morphogen-rpc-adapter \
        --port 8545 \
        --pir-server-a http://127.0.0.1:3000 \
        --pir-server-b http://127.0.0.1:3001 \
        --dict-url http://localhost:8088/mainnet_compact.dict \
        --cas-url http://localhost:8088/cas \
        "${ADAPTER_OTEL_ARGS[@]}" &
else
    ./target/debug/morphogen-rpc-adapter \
        --port 8545 \
        --pir-server-a http://127.0.0.1:3000 \
        --pir-server-b http://127.0.0.1:3001 \
        --dict-url http://localhost:8088/mainnet_compact.dict \
        --cas-url http://localhost:8088/cas &
fi
ADAPTER_PID=$!

# Cleanup function
cleanup() {
    echo "Cleaning up..."
    kill $SERVER_A_PID $SERVER_B_PID $ADAPTER_PID $CAS_PID || true
    rm -rf /tmp/morphogen_cas
}
trap cleanup EXIT

sleep 5

# 5. Test queries
TEST_ADDR="0x000000000000000000000000000000000000031c"
echo "Querying $TEST_ADDR via RPC..."

if [[ "${ENABLE_DATADOG}" == "1" ]]; then
    ./target/debug/morphogen-rpc-dd-client \
        --rpc-url http://127.0.0.1:8545 \
        --address "$TEST_ADDR" \
        --otel-traces \
        --otel-endpoint "${OTEL_ENDPOINT}" \
        --otel-service-name "${OTEL_CLIENT_SERVICE_NAME}" \
        --otel-env "${OTEL_ENV}" \
        --otel-version "${OTEL_VERSION}"
    exit 0
fi

# eth_getBalance
BALANCE=$(cast balance $TEST_ADDR --rpc-url http://127.0.0.1:8545)
echo "Balance: $BALANCE"

# eth_getTransactionCount
NONCE=$(cast nonce $TEST_ADDR --rpc-url http://127.0.0.1:8545)
echo "Nonce: $NONCE"

# eth_getCode
CODE=$(cast code $TEST_ADDR --rpc-url http://127.0.0.1:8545)
echo "Code: $CODE"

# Verify Results
# Expected Balance: 100 ETH (approx from u128 literal)
if [[ "$BALANCE" == "100000000000000000000" ]]; then
    echo "✅ SUCCESS: Balance matches!"
else
    echo "❌ FAILURE: Balance mismatch. Got $BALANCE"
    exit 1
fi

# Expected Nonce: 123
if [[ "$NONCE" == "123" ]]; then
    echo "✅ SUCCESS: Nonce matches!"
else
    echo "❌ FAILURE: Nonce mismatch. Got $NONCE"
    exit 1
fi

# Expected Code: 0x60016001
if [[ "$CODE" == "0x60016001" ]]; then
    echo "✅ SUCCESS: Code matches!"
else
    echo "❌ FAILURE: Code mismatch. Got $CODE"
    exit 1
fi

echo "ALL TESTS PASSED."
