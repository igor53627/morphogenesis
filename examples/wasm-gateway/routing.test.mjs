import assert from "node:assert/strict";
import test from "node:test";

import {
  requestWithGatewayFallback,
  shouldRouteToGateway,
} from "./routing.mjs";

test("wallet-owned methods bypass gateway", async () => {
  const calls = [];

  const result = await requestWithGatewayFallback({
    payload: { method: "eth_accounts", params: [] },
    gatewayRequest: async () => {
      calls.push("gateway");
      return "gateway";
    },
    walletRequest: async () => {
      calls.push("wallet");
      return ["0xabc"];
    },
  });

  assert.deepEqual(result, ["0xabc"]);
  assert.deepEqual(calls, ["wallet"]);
  assert.equal(shouldRouteToGateway("eth_accounts"), false);
});

test("gateway -32601 falls back to wallet provider", async () => {
  const calls = [];

  const result = await requestWithGatewayFallback({
    payload: { method: "eth_getProof", params: ["0xabc", [], "latest"] },
    gatewayRequest: async () => {
      calls.push("gateway");
      throw { code: -32601, message: "Unsupported method" };
    },
    walletRequest: async () => {
      calls.push("wallet");
      return { ok: true };
    },
  });

  assert.deepEqual(result, { ok: true });
  assert.deepEqual(calls, ["gateway", "wallet"]);
});

test("non -32601 gateway errors do not fallback", async () => {
  const calls = [];

  await assert.rejects(
    requestWithGatewayFallback({
      payload: { method: "eth_getBalance", params: ["0xabc", "latest"] },
      gatewayRequest: async () => {
        calls.push("gateway");
        throw { code: -32000, message: "PIR unavailable" };
      },
      walletRequest: async () => {
        calls.push("wallet");
        return "0x0";
      },
    }),
    (error) => {
      assert.equal(error.code, -32000);
      return true;
    },
  );

  assert.deepEqual(calls, ["gateway"]);
});

test("non-string method returns validation error", async () => {
  await assert.rejects(
    requestWithGatewayFallback({
      payload: { method: 123, params: [] },
      gatewayRequest: async () => "gateway",
      walletRequest: async () => "wallet",
    }),
    /request payload must include method/,
  );
});

test("missing wallet provider fails for wallet-owned methods", async () => {
  await assert.rejects(
    requestWithGatewayFallback({
      payload: { method: "eth_accounts", params: [] },
      gatewayRequest: async () => ["0xabc"],
      walletRequest: undefined,
    }),
    /No base provider available for method eth_accounts/,
  );
});
