export const PRIVATE_METHODS = new Set([
  "eth_getBalance",
  "eth_getTransactionCount",
  "eth_getStorageAt",
  "eth_getCode",
]);

export const BASE_PASSTHROUGH_METHODS = new Set([
  "eth_chainId",
  "eth_blockNumber",
  "eth_gasPrice",
]);

export const WALLET_OWNED_METHODS = new Set([
  "eth_accounts",
  "eth_requestAccounts",
  "eth_sendTransaction",
  "eth_sendRawTransaction",
  "eth_sign",
  "eth_signTransaction",
  "personal_sign",
  "eth_subscribe",
  "eth_unsubscribe",
  "eth_newFilter",
  "eth_newBlockFilter",
  "eth_newPendingTransactionFilter",
  "eth_uninstallFilter",
  "eth_getFilterChanges",
  "eth_getFilterLogs",
  "eth_submitWork",
  "eth_submitHashrate",
]);

export function shouldRouteToGateway(method) {
  if (PRIVATE_METHODS.has(method) || BASE_PASSTHROUGH_METHODS.has(method)) {
    return true;
  }

  if (WALLET_OWNED_METHODS.has(method)) {
    return false;
  }

  return method.startsWith("eth_") || method.startsWith("net_") || method.startsWith("web3_");
}

export async function requestWithGatewayFallback({
  payload,
  gatewayRequest,
  walletRequest,
  onFallback,
}) {
  const method = payload?.method;
  if (!method) {
    throw new Error("request payload must include method");
  }

  if (shouldRouteToGateway(method)) {
    try {
      return await gatewayRequest(payload);
    } catch (error) {
      if (error?.code === -32601 && walletRequest) {
        onFallback?.(method);
        return walletRequest(payload);
      }
      throw error;
    }
  }

  if (!walletRequest) {
    throw new Error(`No base provider available for method ${method}`);
  }

  return walletRequest(payload);
}
