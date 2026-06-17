mod block_cache;
mod block_poller;
mod code_resolver;
mod config;
mod evm;
mod filters;
mod log_filter;
mod methods;
mod pir_db;
mod proxy;
mod receipt_fetch;
mod state;
mod telemetry;

use anyhow::Result;
use block_cache::BlockCache;
use clap::Parser;
use code_resolver::CodeResolver;
use config::{validate_privacy_fallback_config, Args};
use jsonrpsee::server::{RpcModule, Server};
use jsonrpsee::types::ErrorObjectOwned;
use morphogen_client::network::PirClient;
use serde_json::Value;
use std::net::SocketAddr;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::time::sleep;
use tower::ServiceBuilder;
use tracing::{error, info, warn, Instrument};

use methods::{
    handle_eth_get_filter_logs, handle_eth_get_logs, handle_eth_new_filter,
    register_eth_create_access_list_method, register_eth_estimate_gas_method,
};
use proxy::proxy_to_upstream;
use state::{
    fail_closed_if_fallback_disabled, record_privacy_degrading_fallback, AdapterState,
    DROPPED_METHODS, PASSTHROUGH_METHODS, RELAY_METHODS,
};

/// Entry point for the RPC adapter server.
///
/// Wrapped by `src/main.rs`'s `#[tokio::main] async fn main`. Kept in the
/// library so it is unit-testable and can be split into modules in follow-up
/// TASK-54 changes.
pub async fn run() -> Result<()> {
    let args = Args::parse();
    if args.print_effective_config {
        println!(
            "{}",
            serde_json::json!({
                "upstream": telemetry::redact_url_for_effective_config(&args.upstream),
                "dict_url": telemetry::redact_url_for_effective_config(&args.dict_url),
                "cas_url": telemetry::redact_url_for_effective_config(&args.cas_url),
            })
        );
        return Ok(());
    }
    let otel = telemetry::OtelSettings {
        enabled: args.otel_traces,
        endpoint: args.otel_endpoint.clone(),
        service_name: args.otel_service_name.clone(),
        environment: args.otel_env.clone(),
        service_version: args.otel_version.clone(),
    };
    let _telemetry_guard = telemetry::init_tracing(otel)?;
    validate_privacy_fallback_config(&args)?;

    let addr = SocketAddr::from(([127, 0, 0, 1], args.port));
    let server = Server::builder()
        .set_http_middleware(ServiceBuilder::new().layer(telemetry::TraceContextLayer))
        .build(addr)
        .await?;

    let pir_client = Arc::new(PirClient::new(
        args.pir_server_a.clone(),
        args.pir_server_b.clone(),
    ));
    let code_resolver = Arc::new(match args.file_url_root.clone() {
        Some(root) => CodeResolver::new_with_file_url_root(
            args.dict_url.clone(),
            args.cas_url.clone(),
            Some(root),
        ),
        None => CodeResolver::new(args.dict_url.clone(), args.cas_url.clone()),
    });

    let http_client = reqwest::Client::builder()
        .timeout(Duration::from_secs(args.upstream_timeout))
        .connect_timeout(Duration::from_secs(5))
        .build()?;

    let block_cache = Arc::new(RwLock::new(BlockCache::new()));

    let state = Arc::new(AdapterState {
        args: args.clone(),
        http_client,
        pir_client,
        code_resolver,
        block_cache: block_cache.clone(),
        privacy_degraded_fallback_total: AtomicU64::new(0),
    });

    // Background task for block cache polling
    block_cache::start_block_poller(
        block_cache,
        state.http_client.clone(),
        args.upstream.clone(),
    );

    // Background task for metadata refresh
    let state_clone = state.clone();
    tokio::spawn(async move {
        let mut consecutive_failures = 0u32;
        loop {
            match state_clone.pir_client.update_metadata().await {
                Ok(m) => {
                    if consecutive_failures > 0 {
                        info!(
                            "PIR metadata recovered after {} failures: epoch={}, block={}",
                            consecutive_failures, m.epoch_id, m.block_number
                        );
                    }
                    consecutive_failures = 0;
                }
                Err(e) => {
                    consecutive_failures += 1;
                    if consecutive_failures <= 3 {
                        warn!(consecutive_failures, "Failed to update PIR metadata: {}", e);
                    } else if consecutive_failures.is_multiple_of(10) {
                        error!(
                            consecutive_failures,
                            "PIR metadata refresh repeatedly failing: {}", e
                        );
                    }
                }
            }
            sleep(Duration::from_secs(state_clone.args.refresh_interval)).await;
        }
    });

    let mut module = RpcModule::from_arc(state.clone());

    // Register eth_getBalance (Private)
    module.register_async_method("eth_getBalance", |params, state, extensions| {
        let request_span = telemetry::rpc_server_span("eth_getBalance", &extensions);
        async move {
            let (address_str, block): (String, Value) = params.parse()?;
            let address_hex = address_str.strip_prefix("0x").unwrap_or(&address_str);
            let mut address = [0u8; 20];
            hex::decode_to_slice(address_hex, &mut address).map_err(|e| {
                ErrorObjectOwned::owned(-32602, format!("Invalid address: {}", e), None::<()>)
            })?;

            info!("Private eth_getBalance for 0x{}", address_hex);

            match state.pir_client.query_account(address).await {
                Ok(account) => {
                    Ok::<Value, ErrorObjectOwned>(Value::String(format!("0x{:x}", account.balance)))
                }
                Err(e) => {
                    error!("PIR query failed for eth_getBalance: {}", e);
                    if state.args.fallback_to_upstream {
                        record_privacy_degrading_fallback(
                            state.as_ref(),
                            "eth_getBalance",
                            "pir_query_failed",
                        );
                        let params = serde_json::json!([address_str, block]);
                        return proxy_to_upstream(
                            &state.args.upstream,
                            &state.http_client,
                            "eth_getBalance",
                            params,
                        )
                        .await;
                    }
                    Err(ErrorObjectOwned::owned(
                        -32000,
                        "PIR servers unavailable".to_string(),
                        None::<()>,
                    ))
                }
            }
        }
        .instrument(request_span)
    })?;

    // Register eth_getTransactionCount (Private)
    module.register_async_method("eth_getTransactionCount", |params, state, extensions| {
        let request_span = telemetry::rpc_server_span("eth_getTransactionCount", &extensions);
        async move {
            let (address_str, block): (String, Value) = params.parse()?;
            let address_hex = address_str.strip_prefix("0x").unwrap_or(&address_str);
            let mut address = [0u8; 20];
            hex::decode_to_slice(address_hex, &mut address).map_err(|e| {
                ErrorObjectOwned::owned(-32602, format!("Invalid address: {}", e), None::<()>)
            })?;

            info!("Private eth_getTransactionCount for 0x{}", address_hex);

            match state.pir_client.query_account(address).await {
                Ok(account) => {
                    Ok::<Value, ErrorObjectOwned>(Value::String(format!("0x{:x}", account.nonce)))
                }
                Err(e) => {
                    error!("PIR query failed for eth_getTransactionCount: {}", e);
                    if state.args.fallback_to_upstream {
                        record_privacy_degrading_fallback(
                            state.as_ref(),
                            "eth_getTransactionCount",
                            "pir_query_failed",
                        );
                        let params = serde_json::json!([address_str, block]);
                        return proxy_to_upstream(
                            &state.args.upstream,
                            &state.http_client,
                            "eth_getTransactionCount",
                            params,
                        )
                        .await;
                    }
                    Err(ErrorObjectOwned::owned(
                        -32000,
                        "PIR servers unavailable".to_string(),
                        None::<()>,
                    ))
                }
            }
        }
        .instrument(request_span)
    })?;

    // Register eth_getCode (Private via CAS)
    module.register_async_method("eth_getCode", |params, state, extensions| {
        let request_span = telemetry::rpc_server_span("eth_getCode", &extensions);
        async move {
            let (address_str, block): (String, Value) = params.parse()?;
            let address_hex = address_str.strip_prefix("0x").unwrap_or(&address_str);
            let mut address = [0u8; 20];
            hex::decode_to_slice(address_hex, &mut address).map_err(|e| {
                ErrorObjectOwned::owned(-32602, format!("Invalid address: {}", e), None::<()>)
            })?;

            info!("Private eth_getCode for 0x{}", address_hex);

            // 1. PIR Query for Account Data
            let account = match state.pir_client.query_account(address).await {
                Ok(a) => a,
                Err(e) => {
                    error!("PIR query failed for eth_getCode: {}", e);
                    if state.args.fallback_to_upstream {
                        record_privacy_degrading_fallback(
                            state.as_ref(),
                            "eth_getCode",
                            "pir_query_failed",
                        );
                        let params = serde_json::json!([address_str, block]);
                        return proxy_to_upstream(
                            &state.args.upstream,
                            &state.http_client,
                            "eth_getCode",
                            params,
                        )
                        .await;
                    }
                    return Err(ErrorObjectOwned::owned(
                        -32000,
                        "PIR servers unavailable".to_string(),
                        None::<()>,
                    ));
                }
            };

            // 2. Resolve CodeID -> CodeHash -> Bytecode
            let bytecode = if let Some(code_id) = account.code_id {
                match state.code_resolver.resolve_code_hash(code_id).await {
                    Ok(hash) => match state.code_resolver.fetch_bytecode(hash).await {
                        Ok(code) => code,
                        Err(e) => {
                            error!("CAS fetch failed for code_id {}: {}", code_id, e);
                            return Err(ErrorObjectOwned::owned(
                                -32000,
                                "Failed to fetch bytecode",
                                None::<()>,
                            ));
                        }
                    },
                    Err(e) => {
                        error!("Code resolution failed for code_id {}: {}", code_id, e);
                        return Err(ErrorObjectOwned::owned(
                            -32000,
                            "Failed to resolve code hash",
                            None::<()>,
                        ));
                    }
                }
            } else {
                Vec::new() // EOA or no code
            };

            Ok::<Value, ErrorObjectOwned>(Value::String(format!("0x{}", hex::encode(bytecode))))
        }
        .instrument(request_span)
    })?;

    // Register eth_getStorageAt (Private via PIR)
    module.register_async_method("eth_getStorageAt", |params, state, extensions| {
        let request_span = telemetry::rpc_server_span("eth_getStorageAt", &extensions);
        async move {
            let (address_str, slot_str, block): (String, String, Value) = params.parse()?;
            let address_hex = address_str.strip_prefix("0x").unwrap_or(&address_str);
            let slot_hex = slot_str.strip_prefix("0x").unwrap_or(&slot_str);

            let mut address = [0u8; 20];
            hex::decode_to_slice(address_hex, &mut address).map_err(|e| {
                ErrorObjectOwned::owned(-32602, format!("Invalid address: {}", e), None::<()>)
            })?;

            let mut slot = [0u8; 32];
            // Reject oversized input before decoding (max 64 hex chars = 32 bytes)
            if slot_hex.len() > 64 {
                return Err(ErrorObjectOwned::owned(
                    -32602,
                    "Slot too long (max 32 bytes)".to_string(),
                    None::<()>,
                ));
            }
            // Pad odd-length hex with leading zero for valid decoding
            let slot_hex_padded;
            let slot_hex_final = if slot_hex.len() % 2 != 0 {
                slot_hex_padded = format!("0{}", slot_hex);
                &slot_hex_padded
            } else {
                slot_hex
            };
            let slot_bytes = hex::decode(slot_hex_final).map_err(|e| {
                ErrorObjectOwned::owned(-32602, format!("Invalid slot: {}", e), None::<()>)
            })?;
            // Copy to the end of the array (big-endian padding)
            let offset = 32 - slot_bytes.len();
            slot[offset..].copy_from_slice(&slot_bytes);

            info!(
                "Private eth_getStorageAt for 0x{} slot 0x{}",
                address_hex,
                hex::encode(slot)
            );

            match state.pir_client.query_storage(address, slot).await {
                Ok(storage) => Ok::<Value, ErrorObjectOwned>(Value::String(format!(
                    "0x{}",
                    hex::encode(storage.value)
                ))),
                Err(e) => {
                    error!("PIR storage query failed: {}", e);
                    if state.args.fallback_to_upstream {
                        record_privacy_degrading_fallback(
                            state.as_ref(),
                            "eth_getStorageAt",
                            "pir_query_failed",
                        );
                        let params = serde_json::json!([address_str, slot_str, block]);
                        return proxy_to_upstream(
                            &state.args.upstream,
                            &state.http_client,
                            "eth_getStorageAt",
                            params,
                        )
                        .await;
                    }
                    Err(ErrorObjectOwned::owned(
                        -32000,
                        "PIR servers unavailable".to_string(),
                        None::<()>,
                    ))
                }
            }
        }
        .instrument(request_span)
    })?;

    // Register eth_call (Private via local EVM execution)
    module.register_async_method("eth_call", |params, state, extensions| {
        let request_span = telemetry::rpc_server_span("eth_call", &extensions);
        async move {
            // Block tag is optional; clients may send 1 or 2 params
            let raw: Vec<Value> = params.parse()?;
            if raw.is_empty() || raw.len() > 2 {
                return Err(ErrorObjectOwned::owned(
                    -32602,
                    format!("expected 1-2 params, got {}", raw.len()),
                    None::<()>,
                ));
            }
            let call_params = &raw[0];
            let block = raw
                .get(1)
                .cloned()
                .unwrap_or(Value::String("latest".into()));

            info!("Private eth_call via local EVM");

            match evm::execute_eth_call(
                Arc::clone(&state.pir_client),
                Arc::clone(&state.code_resolver),
                state.http_client.clone(),
                state.args.upstream.clone(),
                call_params,
                &block,
            )
            .await
            {
                Ok(output) => Ok::<Value, ErrorObjectOwned>(Value::String(format!(
                    "0x{}",
                    hex::encode(&output)
                ))),
                Err(evm::EthCallError::InvalidParams(msg)) => {
                    Err(ErrorObjectOwned::owned(-32602, msg, None::<()>))
                }
                Err(evm::EthCallError::Internal(e)) => {
                    error!("Private eth_call failed: {}", e);
                    if state.args.fallback_to_upstream {
                        record_privacy_degrading_fallback(
                            state.as_ref(),
                            "eth_call",
                            "local_evm_failed",
                        );
                        let params = serde_json::json!([call_params, block]);
                        return proxy_to_upstream(
                            &state.args.upstream,
                            &state.http_client,
                            "eth_call",
                            params,
                        )
                        .await;
                    }
                    Err(ErrorObjectOwned::owned(
                        -32000,
                        format!("eth_call failed: {}", e),
                        None::<()>,
                    ))
                }
            }
        }
        .instrument(request_span)
    })?;

    // Register eth_estimateGas and eth_createAccessList (Private via local EVM execution)
    register_eth_estimate_gas_method(&mut module)?;
    register_eth_create_access_list_method(&mut module)?;

    // Register eth_getTransactionByHash (Private via block cache, upstream fallback)
    module.register_async_method("eth_getTransactionByHash", |params, state, extensions| {
        let request_span = telemetry::rpc_server_span("eth_getTransactionByHash", &extensions);
        async move {
            let (hash_str,): (String,) = params.parse()?;
            let hash = block_cache::parse_tx_hash(&hash_str).ok_or_else(|| {
                ErrorObjectOwned::owned(
                    -32602,
                    "Invalid tx hash: expected 32-byte hex string",
                    None::<()>,
                )
            })?;

            // Check local cache first (private)
            {
                let cache = state.block_cache.read().await;
                if let Some(tx) = cache.get_transaction(&hash) {
                    info!("Serving eth_getTransactionByHash from cache (private)");
                    return Ok::<Value, ErrorObjectOwned>(tx.clone());
                }
            }

            // Cache miss: respect fallback setting
            fail_closed_if_fallback_disabled(
                state.args.fallback_to_upstream,
                -32000,
                "transaction not found in local cache",
            )?;
            record_privacy_degrading_fallback(
                state.as_ref(),
                "eth_getTransactionByHash",
                "cache_miss",
            );
            proxy_to_upstream(
                &state.args.upstream,
                &state.http_client,
                "eth_getTransactionByHash",
                serde_json::json!([hash_str]),
            )
            .await
        }
        .instrument(request_span)
    })?;

    // Register eth_getTransactionReceipt (Private via block cache, upstream fallback)
    module.register_async_method("eth_getTransactionReceipt", |params, state, extensions| {
        let request_span = telemetry::rpc_server_span("eth_getTransactionReceipt", &extensions);
        async move {
            let (hash_str,): (String,) = params.parse()?;
            let hash = block_cache::parse_tx_hash(&hash_str).ok_or_else(|| {
                ErrorObjectOwned::owned(
                    -32602,
                    "Invalid tx hash: expected 32-byte hex string",
                    None::<()>,
                )
            })?;

            // Check local cache first (private)
            {
                let cache = state.block_cache.read().await;
                if let Some(receipt) = cache.get_receipt(&hash) {
                    info!("Serving eth_getTransactionReceipt from cache (private)");
                    return Ok::<Value, ErrorObjectOwned>(receipt.clone());
                }
            }

            // Cache miss: respect fallback setting
            fail_closed_if_fallback_disabled(
                state.args.fallback_to_upstream,
                -32000,
                "receipt not found in local cache",
            )?;
            record_privacy_degrading_fallback(
                state.as_ref(),
                "eth_getTransactionReceipt",
                "cache_miss",
            );
            proxy_to_upstream(
                &state.args.upstream,
                &state.http_client,
                "eth_getTransactionReceipt",
                serde_json::json!([hash_str]),
            )
            .await
        }
        .instrument(request_span)
    })?;

    // Register eth_getLogs (Private via block cache for recent blocks, upstream fallback)
    module.register_async_method("eth_getLogs", |params, state, extensions| {
        let request_span = telemetry::rpc_server_span("eth_getLogs", &extensions);
        async move {
            let raw: Vec<Value> = params.parse()?;
            handle_eth_get_logs(raw, state.clone()).await
        }
        .instrument(request_span)
    })?;

    // Register eth_newFilter (Private — filter stored locally)
    module.register_async_method("eth_newFilter", |params, state, extensions| {
        let request_span = telemetry::rpc_server_span("eth_newFilter", &extensions);
        async move {
            let raw: Vec<Value> = params.parse()?;
            handle_eth_new_filter(raw, state.clone()).await
        }
        .instrument(request_span)
    })?;

    // Register eth_newBlockFilter (Private — filter stored locally)
    module.register_async_method("eth_newBlockFilter", |_params, state, extensions| {
        let request_span = telemetry::rpc_server_span("eth_newBlockFilter", &extensions);
        async move {
            let id = state.block_cache.write().await.create_block_filter();
            info!("Created private block filter {}", id);
            Ok::<Value, ErrorObjectOwned>(Value::String(id))
        }
        .instrument(request_span)
    })?;

    // Register eth_newPendingTransactionFilter (Private — filter stored locally)
    module.register_async_method(
        "eth_newPendingTransactionFilter",
        |_params, state, extensions| {
            let request_span =
                telemetry::rpc_server_span("eth_newPendingTransactionFilter", &extensions);
            async move {
                let id = state.block_cache.write().await.create_pending_tx_filter();
                info!("Created private pending tx filter {}", id);
                Ok::<Value, ErrorObjectOwned>(Value::String(id))
            }
            .instrument(request_span)
        },
    )?;

    // Register eth_uninstallFilter (Private)
    module.register_async_method("eth_uninstallFilter", |params, state, extensions| {
        let request_span = telemetry::rpc_server_span("eth_uninstallFilter", &extensions);
        async move {
            let (filter_id,): (String,) = params.parse()?;
            let result = state.block_cache.write().await.uninstall_filter(&filter_id);
            info!("Uninstalled filter {}: {}", filter_id, result);
            Ok::<Value, ErrorObjectOwned>(Value::Bool(result))
        }
        .instrument(request_span)
    })?;

    // Register eth_getFilterChanges (Private — served from cache)
    module.register_async_method("eth_getFilterChanges", |params, state, extensions| {
        let request_span = telemetry::rpc_server_span("eth_getFilterChanges", &extensions);
        async move {
            let (filter_id,): (String,) = params.parse()?;
            let mut cache = state.block_cache.write().await;
            match cache.get_filter_changes(&filter_id) {
            Some(changes) => {
                info!(filter = %filter_id, count = changes.len(), "Serving eth_getFilterChanges from cache (private)");
                Ok::<Value, ErrorObjectOwned>(Value::Array(changes))
            }
            None => {
                Err(ErrorObjectOwned::owned(-32000, "filter not found", None::<()>))
            }
        }
        }
        .instrument(request_span)
    })?;

    // Register eth_getFilterLogs (Private — served from cache)
    module.register_async_method("eth_getFilterLogs", |params, state, extensions| {
        let request_span = telemetry::rpc_server_span("eth_getFilterLogs", &extensions);
        async move {
            let (filter_id,): (String,) = params.parse()?;
            handle_eth_get_filter_logs(filter_id, state.clone()).await
        }
        .instrument(request_span)
    })?;

    // Register passthrough methods
    for method in PASSTHROUGH_METHODS {
        let method_name = *method;
        module.register_async_method(method, move |params, state, extensions| {
            let request_span = telemetry::rpc_server_span(method_name, &extensions);
            async move {
                proxy_to_upstream(
                    &state.args.upstream,
                    &state.http_client,
                    method_name,
                    params.parse()?,
                )
                .await
            }
            .instrument(request_span)
        })?;
    }

    // Register relay methods — sent to privacy relay (Flashbots by default)
    for method in RELAY_METHODS {
        let method_name = *method;
        module.register_async_method(method, move |params, state, extensions| {
            let request_span = telemetry::rpc_server_span(method_name, &extensions);
            async move {
                info!("Relaying {} to privacy relay", method_name);
                proxy_to_upstream(
                    &state.args.tx_relay,
                    &state.http_client,
                    method_name,
                    params.parse()?,
                )
                .await
            }
            .instrument(request_span)
        })?;
    }

    // Register dropped methods — return explicit errors
    for &(method, reason) in DROPPED_METHODS {
        let reason = reason.to_string();
        let method_name = method;
        module.register_async_method(method, move |_params, _state, extensions| {
            let r = reason.clone();
            let request_span = telemetry::rpc_server_span(method_name, &extensions);
            async move {
                Err::<Value, ErrorObjectOwned>(ErrorObjectOwned::owned(-32601, r, None::<()>))
            }
            .instrument(request_span)
        })?;
    }

    let sanitized_upstream = telemetry::sanitize_url_for_telemetry(&args.upstream);
    let sanitized_relay = telemetry::sanitize_url_for_telemetry(&args.tx_relay);
    info!("Morphogenesis RPC Adapter listening on {}", addr);
    info!("Upstream RPC: {}", sanitized_upstream);
    info!("Tx relay: {}", sanitized_relay);
    if args.fallback_to_upstream {
        warn!(
            environment = ?args.environment,
            "Fallback to upstream enabled - privacy will be degraded when PIR is unavailable"
        );
    } else {
        info!("Privacy fallback policy: fail-closed");
    }

    let handle = server.start(module);
    handle.stopped().await;

    Ok(())
}

#[cfg(test)]
mod tests;
