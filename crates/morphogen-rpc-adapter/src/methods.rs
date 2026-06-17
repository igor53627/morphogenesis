//! JSON-RPC method handlers and registration helpers.
//!
//! Extracted from the crate root in TASK-54.6. This module owns the private
//! method handlers (EVM execution, log filters) and the `register_*` helpers
//! that wire them into the [`jsonrpsee`] `RpcModule`. `run()` in the crate
//! root still owns dispatch; this module supplies the handler bodies.
//!
//! Privacy invariants (TASK-37 / fail-closed) are enforced via the helpers in
//! [`crate::state`] — non-empty state overrides, uncached log ranges, and
//! blockHash log filters all gate through [`crate::state::fail_closed_if_fallback_disabled`]
//! before any upstream fallback.
//!
//! Visibility: all items are `pub(crate)` — intentional seams per the
//! TASK-54 design constraints (no broad `pub` of internals).

use anyhow::Result;
use jsonrpsee::server::RpcModule;
use jsonrpsee::types::ErrorObjectOwned;
use serde_json::Value;
use std::sync::Arc;
use tracing::{error, info, Instrument};

use crate::evm;
use crate::filters::parse_log_filter_for_rpc;
use crate::proxy::proxy_to_upstream;
use crate::state::{
    fail_closed_if_fallback_disabled, record_privacy_degrading_fallback, AdapterState,
};
use crate::telemetry;

/// True if the third (index 2) RPC param is a non-empty state-override object.
/// Empty/null/absent overrides return false; a non-object value returns an
/// `-32602` params error.
pub(crate) fn has_nonempty_state_overrides(raw_params: &[Value]) -> Result<bool, ErrorObjectOwned> {
    let Some(overrides) = raw_params.get(2) else {
        return Ok(false);
    };
    if overrides.is_null() {
        return Ok(false);
    }
    let Some(overrides_obj) = overrides.as_object() else {
        return Err(ErrorObjectOwned::owned(
            -32602,
            "state overrides must be an object when provided",
            None::<()>,
        ));
    };
    Ok(!overrides_obj.is_empty())
}

/// Generic handler for `eth_call`-like methods that accept optional state
/// overrides (`eth_estimateGas`, `eth_createAccessList`).
///
/// Non-empty state overrides are unsupported in the local EVM path: the call
/// is fail-closed (TASK-37) and only falls back to upstream when the operator
/// opted into `--fallback-to-upstream` (+ `--allow-privacy-degraded-fallback`
/// in prod, enforced by [`crate::config::validate_privacy_fallback_config`]).
#[allow(clippy::too_many_arguments)]
pub(crate) async fn handle_evm_method_with_overrides<
    ExecuteFn,
    ExecuteFuture,
    ExecuteOutput,
    MapSuccessFn,
>(
    state: Arc<AdapterState>,
    method_name: &'static str,
    unsupported_state_overrides_message: &'static str,
    raw: Vec<Value>,
    execute: ExecuteFn,
    map_success: MapSuccessFn,
) -> Result<Value, ErrorObjectOwned>
where
    ExecuteFn: FnOnce(Value, Value) -> ExecuteFuture,
    ExecuteFuture:
        std::future::Future<Output = std::result::Result<ExecuteOutput, evm::EthCallError>>,
    MapSuccessFn: FnOnce(ExecuteOutput) -> Value,
{
    // Accept 1-3 params: (call_obj, [block_tag], [state_overrides])
    if raw.is_empty() || raw.len() > 3 {
        return Err(ErrorObjectOwned::owned(
            -32602,
            format!("expected 1-3 params, got {}", raw.len()),
            None::<()>,
        ));
    }

    // Non-empty state overrides are unsupported in the local EVM path.
    // Use fail-closed behavior unless degraded upstream fallback is explicitly enabled.
    let has_overrides = has_nonempty_state_overrides(&raw)?;
    if has_overrides {
        fail_closed_if_fallback_disabled(
            state.args.fallback_to_upstream,
            -32602,
            unsupported_state_overrides_message,
        )?;
        record_privacy_degrading_fallback(
            state.as_ref(),
            method_name,
            "state_overrides_not_supported_locally",
        );
        return proxy_to_upstream(
            &state.args.upstream,
            &state.http_client,
            method_name,
            Value::Array(raw),
        )
        .await;
    }

    let call_params = raw[0].clone();
    let block = raw
        .get(1)
        .cloned()
        .unwrap_or(Value::String("latest".into()));

    info!("Private {} via local EVM", method_name);

    match execute(call_params, block).await {
        Ok(result) => Ok(map_success(result)),
        Err(evm::EthCallError::InvalidParams(msg)) => {
            Err(ErrorObjectOwned::owned(-32602, msg, None::<()>))
        }
        Err(evm::EthCallError::Internal(e)) => {
            error!("Private {} failed: {}", method_name, e);
            if state.args.fallback_to_upstream {
                record_privacy_degrading_fallback(state.as_ref(), method_name, "local_evm_failed");
                return proxy_to_upstream(
                    &state.args.upstream,
                    &state.http_client,
                    method_name,
                    Value::Array(raw),
                )
                .await;
            }
            Err(ErrorObjectOwned::owned(
                -32000,
                format!("{} failed: {}", method_name, e),
                None::<()>,
            ))
        }
    }
}

pub(crate) fn register_eth_estimate_gas_method(module: &mut RpcModule<AdapterState>) -> Result<()> {
    module.register_async_method("eth_estimateGas", |params, state, extensions| {
        let request_span = telemetry::rpc_server_span("eth_estimateGas", &extensions);
        async move {
            let raw: Vec<Value> = params.parse()?;
            handle_evm_method_with_overrides(
                Arc::clone(&state),
                "eth_estimateGas",
                "state overrides not supported for local eth_estimateGas",
                raw,
                move |call_params, block| async move {
                    evm::execute_eth_estimate_gas(
                        Arc::clone(&state.pir_client),
                        Arc::clone(&state.code_resolver),
                        state.http_client.clone(),
                        state.args.upstream.clone(),
                        &call_params,
                        &block,
                    )
                    .await
                },
                |gas| Value::String(format!("0x{:x}", gas)),
            )
            .await
        }
        .instrument(request_span)
    })?;
    Ok(())
}

pub(crate) fn register_eth_create_access_list_method(
    module: &mut RpcModule<AdapterState>,
) -> Result<()> {
    module.register_async_method("eth_createAccessList", |params, state, extensions| {
        let request_span = telemetry::rpc_server_span("eth_createAccessList", &extensions);
        async move {
            let raw: Vec<Value> = params.parse()?;
            handle_evm_method_with_overrides(
                Arc::clone(&state),
                "eth_createAccessList",
                "state overrides not supported for local eth_createAccessList",
                raw,
                move |call_params, block| async move {
                    evm::execute_eth_create_access_list(
                        Arc::clone(&state.pir_client),
                        Arc::clone(&state.code_resolver),
                        state.http_client.clone(),
                        state.args.upstream.clone(),
                        &call_params,
                        &block,
                    )
                    .await
                },
                std::convert::identity,
            )
            .await
        }
        .instrument(request_span)
    })?;
    Ok(())
}

pub(crate) async fn handle_eth_get_logs(
    raw: Vec<Value>,
    state: Arc<AdapterState>,
) -> Result<Value, ErrorObjectOwned> {
    if raw.len() != 1 {
        return Err(ErrorObjectOwned::owned(
            -32602,
            format!("expected 1 param (filter object), got {}", raw.len()),
            None::<()>,
        ));
    }
    let filter_obj = &raw[0];

    // If blockHash is present, proxy to upstream (out of scope for cache)
    if filter_obj.get("blockHash").is_some() {
        fail_closed_if_fallback_disabled(
            state.args.fallback_to_upstream,
            -32000,
            "eth_getLogs by blockHash not supported without upstream fallback",
        )?;
        record_privacy_degrading_fallback(
            state.as_ref(),
            "eth_getLogs",
            "block_hash_filter_not_supported_locally",
        );
        return proxy_to_upstream(
            &state.args.upstream,
            &state.http_client,
            "eth_getLogs",
            Value::Array(raw),
        )
        .await;
    }

    let cache_latest = state.block_cache.read().await.latest_block();
    let filter = parse_log_filter_for_rpc(
        filter_obj,
        cache_latest,
        &state.http_client,
        &state.args.upstream,
    )
    .await?;

    // Check cache coverage
    let cache = state.block_cache.read().await;
    if cache.has_block_range(filter.from_block, filter.to_block) {
        let logs = cache.get_logs(&filter);
        info!(
            from = filter.from_block,
            to = filter.to_block,
            matched = logs.len(),
            "Serving eth_getLogs from cache (private)"
        );
        Ok(Value::Array(logs))
    } else {
        drop(cache);
        fail_closed_if_fallback_disabled(
            state.args.fallback_to_upstream,
            -32000,
            "requested log range not fully cached",
        )?;
        record_privacy_degrading_fallback(state.as_ref(), "eth_getLogs", "range_not_fully_cached");
        proxy_to_upstream(
            &state.args.upstream,
            &state.http_client,
            "eth_getLogs",
            Value::Array(raw),
        )
        .await
    }
}

pub(crate) async fn handle_eth_new_filter(
    raw: Vec<Value>,
    state: Arc<AdapterState>,
) -> Result<Value, ErrorObjectOwned> {
    if raw.len() != 1 {
        return Err(ErrorObjectOwned::owned(
            -32602,
            format!("expected 1 param (filter object), got {}", raw.len()),
            None::<()>,
        ));
    }
    let filter_obj = &raw[0];

    let cache_latest = state.block_cache.read().await.latest_block();
    let filter = parse_log_filter_for_rpc(
        filter_obj,
        cache_latest,
        &state.http_client,
        &state.args.upstream,
    )
    .await?;

    let id = state.block_cache.write().await.create_log_filter(filter);
    info!("Created private log filter {}", id);
    Ok(Value::String(id))
}

pub(crate) async fn handle_eth_get_filter_logs(
    filter_id: String,
    state: Arc<AdapterState>,
) -> Result<Value, ErrorObjectOwned> {
    let mut cache = state.block_cache.write().await;
    match cache.get_filter_logs(&filter_id) {
        Some(Some(logs)) => {
            info!(
                filter = %filter_id,
                count = logs.len(),
                "Serving eth_getFilterLogs from cache (private)"
            );
            Ok(Value::Array(logs))
        }
        Some(None) => Err(ErrorObjectOwned::owned(
            -32000,
            "filter is not a log filter",
            None::<()>,
        )),
        None => Err(ErrorObjectOwned::owned(
            -32000,
            "filter not found",
            None::<()>,
        )),
    }
}
