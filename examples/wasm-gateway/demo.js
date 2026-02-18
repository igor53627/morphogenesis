import init, { WasmGateway } from "./pkg/morphogen_wasm_gateway.js";
import { requestWithGatewayFallback } from "./routing.mjs";

const outputEl = document.getElementById("output");
let gateway;
let routedProvider;

function log(line) {
  outputEl.textContent += `${line}\n`;
}

function clearLog() {
  outputEl.textContent = "";
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
      return requestWithGatewayFallback({
        payload,
        gatewayRequest: (requestPayload) => gateway.request(requestPayload),
        walletRequest: existing,
        onFallback: (method) => {
          log(`Gateway does not support ${method}; falling back to wallet provider.`);
        },
      });
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
