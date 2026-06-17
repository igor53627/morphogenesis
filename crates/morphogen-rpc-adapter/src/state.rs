//! Shared adapter state, JSON-RPC method routing tables, and the
//! privacy-fallback helpers.
//!
//! Extracted from the crate root in TASK-54.5. This module owns:
//! - [`AdapterState`] — the shared state passed to every JSON-RPC handler
//!   (CLI args, HTTP client, PIR client, code resolver, block cache, and the
//!   privacy-degraded fallback counter).
//! - The method routing tables ([`PASSTHROUGH_METHODS`], [`DROPPED_METHODS`],
//!   [`RELAY_METHODS`]) that `run()` dispatches on.
//! - The fail-closed privacy helpers
//!   ([`fail_closed_if_fallback_disabled`] / [`record_privacy_degrading_fallback`])
//!   that enforce the TASK-37 invariant: private methods must NOT silently
//!   leak to a non-private upstream.
//!
//! Visibility: all items are `pub(crate)`. [`AdapterState`] fields are
//! `pub(crate)` — this is the intentional seam for the upcoming `methods`
//! module (register_*/handle_eth_*) which must read state fields to serve
//! requests. No broad `pub` of internals.

use jsonrpsee::types::ErrorObjectOwned;
use morphogen_client::network::PirClient;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::warn;

use crate::block_cache::BlockCache;
use crate::code_resolver::CodeResolver;
use crate::config::Args;

pub(crate) struct AdapterState {
    pub(crate) args: Args,
    pub(crate) http_client: reqwest::Client,
    pub(crate) pir_client: Arc<PirClient>,
    pub(crate) code_resolver: Arc<CodeResolver>,
    pub(crate) block_cache: Arc<RwLock<BlockCache>>,
    pub(crate) privacy_degraded_fallback_total: AtomicU64,
}

/// Methods forwarded verbatim to the upstream provider. These are either
/// chain-global (no address interest) or already-private-via-cache. See the
/// inline NOTEs for which methods are served privately by other code paths.
pub(crate) const PASSTHROUGH_METHODS: &[&str] = &[
    "eth_blockNumber",
    "eth_chainId",
    "eth_gasPrice",
    // NOTE: eth_sendRawTransaction is relayed to a privacy-preserving endpoint
    // (Flashbots Protect by default) — see RELAY_METHODS below.
    "net_version",
    "web3_clientVersion",
    // Wallet Essentials (History & Status)
    // NOTE: eth_getTransactionByHash and eth_getTransactionReceipt are now
    // served from local block cache (private) with upstream fallback
    // NOTE: eth_estimateGas is now private via local EVM execution
    // NOTE: eth_createAccessList is now private via local EVM execution
    "eth_getBlockByNumber",
    "eth_getBlockByHash",
    "eth_feeHistory",
    "eth_maxPriorityFeePerGas",
    // NOTE: eth_getLogs is now served from local block cache (private) for recent blocks
    // Account queries (read-only, safe to passthrough)
    "eth_accounts",
    // NOTE: Filter APIs are now served locally from the block cache.
    // NOTE: Dropped methods (eth_getProof, eth_sign, eth_signTransaction) return
    // explicit errors — see DROPPED_METHODS below.
];

/// Methods explicitly rejected with a clear error message.
/// These are not proxied to upstream because they either leak private state
/// (defeating PIR) or pose security risks (remote signing).
pub(crate) const DROPPED_METHODS: &[(&str, &str)] = &[
    ("eth_getProof", "eth_getProof is disabled: it leaks account/storage interest to the RPC provider, defeating private state queries"),
    ("eth_sign", "eth_sign is disabled: signing should be done client-side by the wallet"),
    ("eth_signTransaction", "eth_signTransaction is disabled: signing should be done client-side by the wallet"),
];

/// Methods relayed to a privacy-preserving endpoint instead of the regular upstream.
/// eth_sendRawTransaction goes to Flashbots Protect to avoid public mempool exposure.
pub(crate) const RELAY_METHODS: &[&str] = &["eth_sendRawTransaction"];

pub(crate) fn next_privacy_degraded_fallback_count(counter: &AtomicU64) -> u64 {
    counter.fetch_add(1, Ordering::Relaxed) + 1
}

pub(crate) fn record_privacy_degrading_fallback(state: &AdapterState, method: &str, reason: &str) {
    let total = next_privacy_degraded_fallback_count(&state.privacy_degraded_fallback_total);
    warn!(
        rpc.method = %method,
        privacy.degraded = true,
        privacy.fallback_reason = %reason,
        privacy.degraded_fallback_total = total,
        "Proxying private method to upstream (privacy degraded)"
    );
}

/// Fail-closed gate for privacy-degrading fallbacks (TASK-37).
///
/// Returns `Ok(())` when the operator has opted into fallbacks, otherwise an
/// [`ErrorObjectOwned`] with the given code/message. This is the single seam
/// that prevents private methods from silently leaking to a non-private
/// upstream unless `--fallback-to-upstream` (+ `--allow-privacy-degraded-fallback`
/// in prod, enforced separately by [`crate::config::validate_privacy_fallback_config`])
/// is set.
pub(crate) fn fail_closed_if_fallback_disabled(
    fallback_to_upstream: bool,
    code: i32,
    message: &'static str,
) -> Result<(), ErrorObjectOwned> {
    if fallback_to_upstream {
        Ok(())
    } else {
        Err(ErrorObjectOwned::owned(code, message, None::<()>))
    }
}
