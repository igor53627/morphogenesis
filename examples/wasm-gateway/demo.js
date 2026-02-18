import init, { WasmGateway } from "./pkg/morphogen_wasm_gateway.js";

const PRIVATE_METHODS = new Set([
  "eth_getBalance",
  "eth_getTransactionCount",
  "eth_getStorageAt",
  "eth_getCode",
]);

const BASE_PASSTHROUGH_METHODS = new Set([
  "eth_chainId",
  "eth_blockNumber",
  "eth_gasPrice",
]);

const outputEl = document.getElementById("output");
let gateway;
let routedProvider;

function log(line) {
  outputEl.textContent += `${line}\n`;
}

function clearLog() {
  outputEl.textContent = "";
}

function shouldRouteToGateway(method) {
  if (PRIVATE_METHODS.has(method) || BASE_PASSTHROUGH_METHODS.has(method)) {
    return true;
  }
  if (method.startsWith("eth_send") || method.startsWith("eth_sign")) {
    return false;
  }
  return method.startsWith("eth_") || method.startsWith("net_") || method.startsWith("web3_");
}

function getConfig() {
  return {
    upstreamUrl: document.getElementById("upstream").value,
    pirServerA: document.getElementById("pirA").value,
    pirServerB: document.getElementById("pirB").value,
    dictUrl: document.getElementById("dict").value,
    casUrl: document.getElementById("cas").value,
    requestTimeoutMs: 15000,
  };
}

function installProviderBridge() {
  const existing = window.ethereum?.request?.bind(window.ethereum);

  routedProvider = {
    request: async (payload) => {
      const method = payload?.method;
      if (!method) {
        throw new Error("request payload must include method");
      }

      if (shouldRouteToGateway(method)) {
        return gateway.request(payload);
      }

      if (!existing) {
        throw new Error(`No base provider available for method ${method}`);
      }

      return existing(payload);
    },
  };

  window.morphogenProvider = routedProvider;

  if (window.ethereum && typeof window.ethereum.request === "function") {
    window.ethereum.request = routedProvider.request;
    log("Patched window.ethereum.request with Morphogenesis routing.");
  } else {
    log("No injected wallet found. Use window.morphogenProvider.request instead.");
  }
}

async function ensureInitialized() {
  if (gateway) {
    return;
  }

  clearLog();
  log("Loading wasm module...");
  await init();

  const config = getConfig();
  gateway = new WasmGateway(config);
  installProviderBridge();
  log("Gateway initialized.");
}

async function runPrivateCall() {
  await ensureInitialized();

  const address = document.getElementById("address").value;
  const result = await routedProvider.request({
    method: "eth_getBalance",
    params: [address, "latest"],
  });

  log(`eth_getBalance(${address}) => ${result}`);
}

async function runPassthroughCall() {
  await ensureInitialized();

  const result = await routedProvider.request({
    method: "eth_chainId",
    params: [],
  });

  log(`eth_chainId() => ${result}`);
}

async function wrap(label, fn) {
  try {
    await fn();
  } catch (error) {
    const details = typeof error === "object" ? JSON.stringify(error) : String(error);
    log(`${label} failed: ${details}`);
  }
}

document.getElementById("init").addEventListener("click", () => {
  wrap("init", ensureInitialized);
});

document.getElementById("private").addEventListener("click", () => {
  wrap("private call", runPrivateCall);
});

document.getElementById("passthrough").addEventListener("click", () => {
  wrap("passthrough call", runPassthroughCall);
});
