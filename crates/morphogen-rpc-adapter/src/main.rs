mod block_cache;
mod code_resolver;
mod evm;
mod pir_db;
mod telemetry;

use anyhow::{bail, Result};
use block_cache::BlockCache;
use clap::Parser;
use code_resolver::CodeResolver;
use jsonrpsee::server::{RpcModule, Server};
use jsonrpsee::types::ErrorObjectOwned;
use morphogen_client::network::PirClient;
use serde_json::Value;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::time::sleep;
use tower::ServiceBuilder;
use tracing::{error, info, warn, Instrument};

#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
enum AdapterEnvironment {
    Dev,
    Test,
    Prod,
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about = "Morphogenesis RPC Adapter")]
struct Args {
    /// Port to listen on
    #[arg(short, long, default_value_t = 8545)]
    port: u16,

    /// Upstream Ethereum RPC URL
    #[arg(
        short,
        long,
        env = "UPSTREAM_RPC_URL",
        default_value = "https://ethereum-rpc.publicnode.com"
    )]
    upstream: String,

    /// PIR Server A URL
    #[arg(long, default_value = "http://localhost:3000")]
    pir_server_a: String,

    /// PIR Server B URL
    #[arg(long, default_value = "http://localhost:3001")]
    pir_server_b: String,

    /// Dictionary URL for CodeID resolution
    #[arg(
        long,
        env = "DICT_URL",
        default_value = "http://localhost:8080/mainnet_compact.dict"
    )]
    dict_url: String,

    /// CAS Base URL for bytecode fetching
    #[arg(long, env = "CAS_URL", default_value = "http://localhost:8080/cas")]
    cas_url: String,

    /// Print effective URL config and exit (internal diagnostics/test hook)
    #[arg(long, hide = true, default_value_t = false)]
    print_effective_config: bool,

    /// Required allowlist root for local file:// dictionary/CAS URLs
    #[arg(long)]
    file_url_root: Option<PathBuf>,

    /// Metadata refresh interval in seconds
    #[arg(long, default_value_t = 12)]
    refresh_interval: u64,

    /// Upstream request timeout in seconds
    #[arg(long, default_value_t = 15)]
    upstream_timeout: u64,

    /// Fall back to upstream RPC when PIR servers are unavailable
    #[arg(long, default_value_t = false)]
    fallback_to_upstream: bool,

    /// Deployment environment profile
    #[arg(long, value_enum, default_value_t = AdapterEnvironment::Prod)]
    environment: AdapterEnvironment,

    /// Explicit acknowledgement required to allow privacy-degrading fallback in prod
    #[arg(long, default_value_t = false)]
    allow_privacy_degraded_fallback: bool,

    /// Transaction relay URL for eth_sendRawTransaction.
    /// Defaults to Flashbots Protect so txs bypass the public mempool.
    #[arg(
        long,
        default_value = "https://rpc.flashbots.net/?hint=hash&originId=morphogenesis"
    )]
    tx_relay: String,

    /// Enable OpenTelemetry trace export (Datadog Agent OTLP gRPC compatible)
    #[arg(long, default_value_t = false)]
    otel_traces: bool,

    /// OTLP collector endpoint (Datadog Agent default: http://127.0.0.1:4317)
    #[arg(long, default_value = "http://127.0.0.1:4317")]
    otel_endpoint: String,

    /// Service name reported to APM
    #[arg(long, default_value = "morphogen-rpc-adapter")]
    otel_service_name: String,

    /// Deployment environment tag for traces
    #[arg(long, default_value = "e2e")]
    otel_env: String,

    /// Service version tag for traces
    #[arg(long, default_value = env!("CARGO_PKG_VERSION"))]
    otel_version: String,
}

#[cfg(test)]
impl Args {
    fn default_for_tests() -> Self {
        Self {
            port: 8545,
            upstream: "https://ethereum-rpc.publicnode.com".to_string(),
            pir_server_a: "http://localhost:3000".to_string(),
            pir_server_b: "http://localhost:3001".to_string(),
            dict_url: "http://localhost:8080/mainnet_compact.dict".to_string(),
            cas_url: "http://localhost:8080/cas".to_string(),
            print_effective_config: false,
            file_url_root: None,
            refresh_interval: 12,
            upstream_timeout: 15,
            fallback_to_upstream: false,
            environment: AdapterEnvironment::Prod,
            allow_privacy_degraded_fallback: false,
            tx_relay: "https://rpc.flashbots.net/?hint=hash&originId=morphogenesis".to_string(),
            otel_traces: false,
            otel_endpoint: "http://127.0.0.1:4317".to_string(),
            otel_service_name: "morphogen-rpc-adapter".to_string(),
            otel_env: "e2e".to_string(),
            otel_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

struct AdapterState {
    args: Args,
    http_client: reqwest::Client,
    pir_client: Arc<PirClient>,
    code_resolver: Arc<CodeResolver>,
    block_cache: Arc<RwLock<BlockCache>>,
    privacy_degraded_fallback_total: AtomicU64,
}

const PASSTHROUGH_METHODS: &[&str] = &[
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
const DROPPED_METHODS: &[(&str, &str)] = &[
    ("eth_getProof", "eth_getProof is disabled: it leaks account/storage interest to the RPC provider, defeating private state queries"),
    ("eth_sign", "eth_sign is disabled: signing should be done client-side by the wallet"),
    ("eth_signTransaction", "eth_signTransaction is disabled: signing should be done client-side by the wallet"),
];

/// Methods relayed to a privacy-preserving endpoint instead of the regular upstream.
/// eth_sendRawTransaction goes to Flashbots Protect to avoid public mempool exposure.
const RELAY_METHODS: &[&str] = &["eth_sendRawTransaction"];

fn validate_privacy_fallback_config(args: &Args) -> Result<()> {
    if args.fallback_to_upstream
        && args.environment == AdapterEnvironment::Prod
        && !args.allow_privacy_degraded_fallback
    {
        bail!("--fallback-to-upstream in prod requires --allow-privacy-degraded-fallback");
    }

    Ok(())
}

fn next_privacy_degraded_fallback_count(counter: &AtomicU64) -> u64 {
    counter.fetch_add(1, Ordering::Relaxed) + 1
}

fn record_privacy_degrading_fallback(state: &AdapterState, method: &str, reason: &str) {
    let total = next_privacy_degraded_fallback_count(&state.privacy_degraded_fallback_total);
    warn!(
        rpc.method = %method,
        privacy.degraded = true,
        privacy.fallback_reason = %reason,
        privacy.degraded_fallback_total = total,
        "Proxying private method to upstream (privacy degraded)"
    );
}

fn fail_closed_if_fallback_disabled(
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

fn has_nonempty_state_overrides(raw_params: &[Value]) -> Result<bool, ErrorObjectOwned> {
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

async fn handle_evm_method_with_overrides<ExecuteFn, ExecuteFuture, ExecuteOutput, MapSuccessFn>(
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

fn register_eth_estimate_gas_method(module: &mut RpcModule<AdapterState>) -> Result<()> {
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

fn register_eth_create_access_list_method(module: &mut RpcModule<AdapterState>) -> Result<()> {
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

fn filter_uses_tag(filter_obj: &Value, tag: &str) -> bool {
    ["fromBlock", "toBlock"].iter().any(|field| {
        filter_obj
            .get(field)
            .and_then(Value::as_str)
            .is_some_and(|value| value == tag)
    })
}

fn parse_block_number_quantity(number: &str, tag: &str) -> Result<u64, ErrorObjectOwned> {
    let hex = number.strip_prefix("0x").ok_or_else(|| {
        ErrorObjectOwned::owned(
            -32000,
            format!("Invalid {} block number format from upstream", tag),
            None::<()>,
        )
    })?;
    u64::from_str_radix(hex, 16).map_err(|_| {
        ErrorObjectOwned::owned(
            -32000,
            format!("Invalid {} block number value from upstream", tag),
            None::<()>,
        )
    })
}

async fn resolve_block_tag_height(
    client: &reqwest::Client,
    upstream_url: &str,
    tag: &'static str,
) -> Result<u64, ErrorObjectOwned> {
    let result = proxy_to_upstream(
        upstream_url,
        client,
        "eth_getBlockByNumber",
        serde_json::json!([tag, false]),
    )
    .await?;

    let number = result
        .get("number")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            ErrorObjectOwned::owned(
                -32000,
                format!("Missing {} block number from upstream", tag),
                None::<()>,
            )
        })?;

    parse_block_number_quantity(number, tag)
}

async fn resolve_filter_finality_heights(
    filter_obj: &Value,
    client: &reqwest::Client,
    upstream_url: &str,
) -> Result<(Option<u64>, Option<u64>), ErrorObjectOwned> {
    let safe_height = if filter_uses_tag(filter_obj, "safe") {
        Some(resolve_block_tag_height(client, upstream_url, "safe").await?)
    } else {
        None
    };

    let finalized_height = if filter_uses_tag(filter_obj, "finalized") {
        Some(resolve_block_tag_height(client, upstream_url, "finalized").await?)
    } else {
        None
    };

    Ok((safe_height, finalized_height))
}

fn effective_latest_for_filter(
    cache_latest: u64,
    safe_height: Option<u64>,
    finalized_height: Option<u64>,
) -> u64 {
    cache_latest
        .max(safe_height.unwrap_or(0))
        .max(finalized_height.unwrap_or(0))
}

async fn parse_log_filter_for_rpc(
    filter_obj: &Value,
    cache_latest: u64,
    client: &reqwest::Client,
    upstream_url: &str,
) -> Result<block_cache::LogFilter, ErrorObjectOwned> {
    let (safe_height, finalized_height) =
        resolve_filter_finality_heights(filter_obj, client, upstream_url).await?;
    let effective_latest = effective_latest_for_filter(cache_latest, safe_height, finalized_height);
    block_cache::parse_log_filter_object(
        filter_obj,
        effective_latest,
        safe_height,
        finalized_height,
    )
    .map_err(|e| ErrorObjectOwned::owned(-32602, e, None::<()>))
}

async fn handle_eth_get_logs(
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

async fn handle_eth_new_filter(
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

async fn handle_eth_get_filter_logs(
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

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    if args.print_effective_config {
        println!(
            "{}",
            serde_json::json!({
                "upstream": telemetry::sanitize_url_for_telemetry(&args.upstream),
                "dict_url": telemetry::sanitize_url_for_telemetry(&args.dict_url),
                "cas_url": telemetry::sanitize_url_for_telemetry(&args.cas_url),
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

async fn proxy_to_upstream(
    url: &str,
    client: &reqwest::Client,
    method: &str,
    params: Value,
) -> Result<Value, ErrorObjectOwned> {
    let sanitized_url = telemetry::sanitize_url_for_telemetry(url);
    let span = tracing::info_span!(
        "rpc.upstream",
        otel.kind = "client",
        otel.name = method,
        rpc.system = "ethereum-jsonrpc",
        rpc.method = %method,
        upstream.url = %sanitized_url
    );
    async move {
        info!("Proxying {} to upstream", method);

        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params
        });

        let response = client.post(url).json(&request).send().await.map_err(|e| {
            if e.is_timeout() {
                ErrorObjectOwned::owned(
                    -32000,
                    format!("Upstream timeout for {}", method),
                    None::<()>,
                )
            } else if e.is_connect() {
                ErrorObjectOwned::owned(
                    -32000,
                    format!("Upstream connection failed for {}", method),
                    None::<()>,
                )
            } else {
                upstream_request_failed_error(method)
            }
        })?;

        let json: Value = response
            .json()
            .await
            .map_err(|_| upstream_invalid_json_error(method))?;

        if let Some(error) = json.get("error") {
            let error_code = error.get("code").and_then(|c| c.as_i64()).unwrap_or(-32000) as i32;
            let error_message = error
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("Unknown error");
            warn!(
                rpc.method = %method,
                error.code = error_code,
                error.message = %error_message,
                "Upstream returned JSON-RPC error"
            );
            return Err(ErrorObjectOwned::owned(
                error_code,
                error_message.to_string(),
                error.get("data").cloned(),
            ));
        }

        Ok(json.get("result").cloned().unwrap_or(Value::Null))
    }
    .instrument(span)
    .await
}

fn upstream_request_failed_error(method: &str) -> ErrorObjectOwned {
    ErrorObjectOwned::owned(
        -32000,
        format!("Upstream request failed for {}", method),
        None::<()>,
    )
}

fn upstream_invalid_json_error(method: &str) -> ErrorObjectOwned {
    ErrorObjectOwned::owned(
        -32000,
        format!("Invalid JSON response for {}", method),
        None::<()>,
    )
}

#[cfg(test)]
mod tests {
    use super::{
        effective_latest_for_filter, fail_closed_if_fallback_disabled, handle_eth_get_logs,
        handle_eth_new_filter, has_nonempty_state_overrides, next_privacy_degraded_fallback_count,
        parse_log_filter_for_rpc, proxy_to_upstream, upstream_invalid_json_error,
        validate_privacy_fallback_config, AdapterEnvironment, Args, DROPPED_METHODS,
        PASSTHROUGH_METHODS, RELAY_METHODS,
    };
    use clap::Parser;
    use serde_json::{json, Value};
    use std::sync::atomic::AtomicU64;
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};
    use tokio::sync::RwLock;
    use tokio::task::JoinHandle;
    use tokio::time::sleep;

    #[test]
    fn args_parse_explicit_url_flags() {
        let args = Args::parse_from([
            "morphogen-rpc-adapter",
            "--upstream",
            "https://cli-upstream.example",
            "--dict-url",
            "https://cli-dict.example/mainnet_compact.dict",
            "--cas-url",
            "https://cli-dict.example/cas",
        ]);
        assert_eq!(args.upstream, "https://cli-upstream.example");
        assert_eq!(args.dict_url, "https://cli-dict.example/mainnet_compact.dict");
        assert_eq!(args.cas_url, "https://cli-dict.example/cas");
    }

    async fn read_http_request_body(socket: &mut TcpStream) -> Vec<u8> {
        let mut buffer = Vec::new();
        let mut chunk = [0_u8; 1024];
        let header_end = loop {
            let read = socket.read(&mut chunk).await.expect("read request bytes");
            assert!(read > 0, "request closed before headers were received");
            buffer.extend_from_slice(&chunk[..read]);
            if let Some(idx) = buffer.windows(4).position(|window| window == b"\r\n\r\n") {
                break idx + 4;
            }
        };

        let headers = std::str::from_utf8(&buffer[..header_end]).expect("request headers utf8");
        let content_length = headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                if name.eq_ignore_ascii_case("content-length") {
                    value.trim().parse::<usize>().ok()
                } else {
                    None
                }
            })
            .unwrap_or(0);

        let mut body = buffer[header_end..].to_vec();
        while body.len() < content_length {
            let read = socket
                .read(&mut chunk)
                .await
                .expect("read request body bytes");
            assert!(read > 0, "request closed before body was fully read");
            body.extend_from_slice(&chunk[..read]);
        }
        body.truncate(content_length);
        body
    }

    async fn spawn_mock_upstream(responses: Vec<Value>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind mock upstream listener");
        let addr = listener.local_addr().expect("listener addr");

        tokio::spawn(async move {
            for response in responses {
                let (mut socket, _) = listener.accept().await.expect("accept request");
                let mut buf = [0_u8; 4096];
                let _ = socket.read(&mut buf).await;

                let body = response.to_string();
                let response_text = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                socket
                    .write_all(response_text.as_bytes())
                    .await
                    .expect("write response");
            }
        });

        format!("http://{}", addr)
    }

    async fn spawn_mock_upstream_script(
        script: Vec<(&'static str, Value)>,
    ) -> (String, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind scripted mock upstream listener");
        let addr = listener.local_addr().expect("listener addr");

        let handle = tokio::spawn(async move {
            for (expected_method, response) in script {
                let (mut socket, _) = listener.accept().await.expect("accept request");
                let body = read_http_request_body(&mut socket).await;
                let req_json: Value =
                    serde_json::from_slice(&body).expect("valid json-rpc request");
                let method = req_json
                    .get("method")
                    .and_then(Value::as_str)
                    .unwrap_or("<missing>");
                assert_eq!(method, expected_method, "unexpected upstream method order");

                let body = response.to_string();
                let response_text = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                socket
                    .write_all(response_text.as_bytes())
                    .await
                    .expect("write response");
            }
        });

        (format!("http://{}", addr), handle)
    }

    fn make_test_state(upstream: String, fallback_to_upstream: bool) -> Arc<super::AdapterState> {
        let mut args = Args::default_for_tests();
        args.upstream = upstream;
        args.fallback_to_upstream = fallback_to_upstream;

        Arc::new(super::AdapterState {
            args,
            http_client: reqwest::Client::builder()
                .no_proxy()
                .build()
                .expect("build test http client"),
            pir_client: Arc::new(super::PirClient::new(
                "http://localhost:3000".to_string(),
                "http://localhost:3001".to_string(),
            )),
            code_resolver: Arc::new(super::CodeResolver::new(
                "http://localhost:8080/mainnet_compact.dict".to_string(),
                "http://localhost:8080/cas".to_string(),
            )),
            block_cache: Arc::new(RwLock::new(super::BlockCache::new())),
            privacy_degraded_fallback_total: AtomicU64::new(0),
        })
    }

    async fn start_filter_rpc_server(
        state: Arc<super::AdapterState>,
    ) -> (String, jsonrpsee::server::ServerHandle) {
        let server = super::Server::builder()
            .build(
                "127.0.0.1:0"
                    .parse::<std::net::SocketAddr>()
                    .expect("parse addr"),
            )
            .await
            .expect("build test rpc server");
        let addr = server.local_addr().expect("server local addr");
        let mut module = super::RpcModule::from_arc(state);

        module
            .register_async_method("eth_getLogs", |params, state, _| async move {
                let raw: Vec<Value> = params.parse()?;
                super::handle_eth_get_logs(raw, state.clone()).await
            })
            .expect("register eth_getLogs");
        module
            .register_async_method("eth_newFilter", |params, state, _| async move {
                let raw: Vec<Value> = params.parse()?;
                super::handle_eth_new_filter(raw, state.clone()).await
            })
            .expect("register eth_newFilter");
        module
            .register_async_method("eth_getFilterLogs", |params, state, _| async move {
                let (filter_id,): (String,) = params.parse()?;
                super::handle_eth_get_filter_logs(filter_id, state.clone()).await
            })
            .expect("register eth_getFilterLogs");

        let handle = server.start(module);
        (format!("http://{}", addr), handle)
    }

    async fn start_state_override_rpc_server(
        state: Arc<super::AdapterState>,
    ) -> (String, jsonrpsee::server::ServerHandle) {
        let server = super::Server::builder()
            .build(
                "127.0.0.1:0"
                    .parse::<std::net::SocketAddr>()
                    .expect("parse addr"),
            )
            .await
            .expect("build test rpc server");
        let addr = server.local_addr().expect("server local addr");
        let mut module = super::RpcModule::from_arc(state);

        super::register_eth_estimate_gas_method(&mut module).expect("register eth_estimateGas");
        super::register_eth_create_access_list_method(&mut module)
            .expect("register eth_createAccessList");

        let handle = server.start(module);
        (format!("http://{}", addr), handle)
    }

    async fn send_rpc_request(rpc_url: &str, method: &str, params: Value) -> Value {
        let client = reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("build client");
        client
            .post(rpc_url)
            .json(&json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": method,
                "params": params
            }))
            .send()
            .await
            .expect("send rpc request")
            .json::<Value>()
            .await
            .expect("parse rpc response")
    }

    #[test]
    fn test_passthrough_methods_exclude_private() {
        // Verify private methods are NOT in passthrough
        assert!(!PASSTHROUGH_METHODS.contains(&"eth_getStorageAt"));
        assert!(!PASSTHROUGH_METHODS.contains(&"eth_call"));
        assert!(!PASSTHROUGH_METHODS.contains(&"eth_estimateGas"));
        assert!(!PASSTHROUGH_METHODS.contains(&"eth_createAccessList"));
        assert!(!PASSTHROUGH_METHODS.contains(&"eth_getTransactionByHash"));
        assert!(!PASSTHROUGH_METHODS.contains(&"eth_getTransactionReceipt"));
        assert!(!PASSTHROUGH_METHODS.contains(&"eth_getLogs"));

        // Verify filter APIs are NOT in passthrough (now served locally)
        assert!(!PASSTHROUGH_METHODS.contains(&"eth_newFilter"));
        assert!(!PASSTHROUGH_METHODS.contains(&"eth_newBlockFilter"));
        assert!(!PASSTHROUGH_METHODS.contains(&"eth_newPendingTransactionFilter"));
        assert!(!PASSTHROUGH_METHODS.contains(&"eth_uninstallFilter"));
        assert!(!PASSTHROUGH_METHODS.contains(&"eth_getFilterChanges"));
        assert!(!PASSTHROUGH_METHODS.contains(&"eth_getFilterLogs"));

        // Verify dropped methods are NOT in passthrough
        assert!(!PASSTHROUGH_METHODS.contains(&"eth_getProof"));
        assert!(!PASSTHROUGH_METHODS.contains(&"eth_sign"));
        assert!(!PASSTHROUGH_METHODS.contains(&"eth_signTransaction"));

        // Verify relay methods are NOT in passthrough
        assert!(!PASSTHROUGH_METHODS.contains(&"eth_sendRawTransaction"));
    }

    #[test]
    fn test_relay_methods() {
        assert!(RELAY_METHODS.contains(&"eth_sendRawTransaction"));

        // No overlap with passthrough or dropped
        let dropped_names: Vec<&str> = DROPPED_METHODS.iter().map(|(name, _)| *name).collect();
        for method in RELAY_METHODS {
            assert!(
                !PASSTHROUGH_METHODS.contains(method),
                "{} in both relay and passthrough",
                method
            );
            assert!(
                !dropped_names.contains(method),
                "{} in both relay and dropped",
                method
            );
        }
    }

    #[test]
    fn test_dropped_methods() {
        let dropped_names: Vec<&str> = DROPPED_METHODS.iter().map(|(name, _)| *name).collect();

        // Privacy: leaks account/storage interest
        assert!(dropped_names.contains(&"eth_getProof"));

        // Security: remote signing
        assert!(dropped_names.contains(&"eth_sign"));
        assert!(dropped_names.contains(&"eth_signTransaction"));

        // No overlap with passthrough
        for name in &dropped_names {
            assert!(
                !PASSTHROUGH_METHODS.contains(name),
                "{} is in both dropped and passthrough",
                name
            );
        }
    }

    #[test]
    fn sanitize_url_strips_credentials_and_query_params() {
        // Standard URL - should return scheme + host
        assert_eq!(
            super::telemetry::sanitize_url_for_telemetry("https://api.example.com/path?key=secret"),
            "https://api.example.com"
        );

        // URL with credentials - should strip userinfo
        assert_eq!(
            super::telemetry::sanitize_url_for_telemetry("https://user:pass@api.example.com/path"),
            "https://api.example.com"
        );

        // URL with port - preserve host:port for better service differentiation
        assert_eq!(
            super::telemetry::sanitize_url_for_telemetry("http://localhost:8545/path"),
            "http://localhost:8545"
        );

        // Invalid URL - should return placeholder
        assert_eq!(
            super::telemetry::sanitize_url_for_telemetry("not-a-valid-url"),
            "<invalid-url>"
        );
    }

    #[tokio::test]
    async fn proxy_to_upstream_request_error_is_redacted() {
        let client = reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("build client");
        let err = proxy_to_upstream(
            "ftp://user:secret@example.com:8545",
            &client,
            "eth_getBalance",
            json!([]),
        )
        .await
        .expect_err("expected request error");

        assert_eq!(err.message(), "Upstream request failed for eth_getBalance");
    }

    #[tokio::test]
    async fn proxy_to_upstream_connect_error_is_redacted() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind temporary listener");
        let addr = listener.local_addr().expect("listener addr");
        drop(listener);

        let client = reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("build client");
        let url = format!("http://{}", addr);
        let err = proxy_to_upstream(&url, &client, "eth_getBalance", json!([]))
            .await
            .expect_err("expected connect error");

        assert_eq!(
            err.message(),
            "Upstream connection failed for eth_getBalance"
        );
    }

    #[tokio::test]
    async fn proxy_to_upstream_timeout_error_is_redacted() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind timeout listener");
        let addr = listener.local_addr().expect("listener addr");
        tokio::spawn(async move {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            let mut buf = [0_u8; 1024];
            let _ = socket.read(&mut buf).await;
            sleep(Duration::from_millis(250)).await;
        });

        let client = reqwest::Client::builder()
            .no_proxy()
            .timeout(Duration::from_millis(25))
            .build()
            .expect("build client");
        let url = format!("http://{}", addr);
        let err = proxy_to_upstream(&url, &client, "eth_getBalance", json!([]))
            .await
            .expect_err("expected timeout error");

        assert_eq!(err.message(), "Upstream timeout for eth_getBalance");
    }

    #[test]
    fn proxy_to_upstream_invalid_json_error_is_redacted() {
        let err = upstream_invalid_json_error("eth_getBalance");
        assert_eq!(err.message(), "Invalid JSON response for eth_getBalance");
    }

    #[test]
    fn privacy_fallback_defaults_to_fail_closed() {
        let args = Args::default_for_tests();
        assert!(!args.fallback_to_upstream);
        validate_privacy_fallback_config(&args).expect("default fail-closed config should pass");
    }

    #[test]
    fn privacy_fallback_prod_requires_explicit_override() {
        let mut args = Args::default_for_tests();
        args.fallback_to_upstream = true;
        args.environment = AdapterEnvironment::Prod;
        args.allow_privacy_degraded_fallback = false;

        let err = validate_privacy_fallback_config(&args)
            .expect_err("prod degraded fallback should require explicit override");
        assert!(err
            .to_string()
            .contains("--allow-privacy-degraded-fallback"));
    }

    #[test]
    fn privacy_fallback_prod_allows_override() {
        let mut args = Args::default_for_tests();
        args.fallback_to_upstream = true;
        args.environment = AdapterEnvironment::Prod;
        args.allow_privacy_degraded_fallback = true;

        validate_privacy_fallback_config(&args)
            .expect("prod degraded fallback with explicit override should pass");
    }

    #[test]
    fn privacy_fallback_non_prod_allows_without_override() {
        let mut args = Args::default_for_tests();
        args.fallback_to_upstream = true;
        args.environment = AdapterEnvironment::Dev;
        args.allow_privacy_degraded_fallback = false;

        validate_privacy_fallback_config(&args)
            .expect("non-prod degraded fallback should not require prod override");
    }

    #[test]
    fn privacy_fallback_counter_increments_monotonically() {
        let counter = AtomicU64::new(0);
        assert_eq!(next_privacy_degraded_fallback_count(&counter), 1);
        assert_eq!(next_privacy_degraded_fallback_count(&counter), 2);
        assert_eq!(next_privacy_degraded_fallback_count(&counter), 3);
    }

    #[test]
    fn fail_closed_gate_returns_error_when_fallback_disabled() {
        let err = fail_closed_if_fallback_disabled(false, -32000, "blocked")
            .expect_err("fail-closed gate should reject when fallback is disabled");
        assert_eq!(err.code(), -32000);
        assert_eq!(err.message(), "blocked");
    }

    #[test]
    fn fail_closed_gate_allows_when_fallback_enabled() {
        fail_closed_if_fallback_disabled(true, -32000, "unused")
            .expect("gate should allow when fallback is enabled");
    }

    #[test]
    fn state_override_gate_returns_invalid_params_when_fallback_disabled() {
        let err = fail_closed_if_fallback_disabled(false, -32602, "unsupported")
            .expect_err("state overrides gate should reject when fallback is disabled");
        assert_eq!(err.code(), -32602);
        assert_eq!(err.message(), "unsupported");
    }

    #[test]
    fn state_override_gate_allows_when_fallback_enabled() {
        fail_closed_if_fallback_disabled(true, -32602, "unused")
            .expect("state overrides gate should allow when fallback is enabled");
    }

    #[test]
    fn state_overrides_presence_is_deterministic() {
        assert!(!has_nonempty_state_overrides(&[]).expect("empty params"));
        assert!(!has_nonempty_state_overrides(&[json!({})]).expect("single param"));
        assert!(!has_nonempty_state_overrides(&[json!({}), json!("latest")]).expect("two params"));
        assert!(
            !has_nonempty_state_overrides(&[json!({}), json!("latest"), json!(null)])
                .expect("null overrides")
        );
        assert!(
            !has_nonempty_state_overrides(&[json!({}), json!("latest"), json!({})])
                .expect("empty object overrides")
        );
        assert!(has_nonempty_state_overrides(&[
            json!({}),
            json!("latest"),
            json!({"0xabc": {"balance": "0x1"}})
        ])
        .expect("non-empty overrides"));
    }

    #[test]
    fn state_overrides_reject_non_object_values() {
        let err = has_nonempty_state_overrides(&[json!({}), json!("latest"), json!(42)])
            .expect_err("non-object state overrides should fail");
        assert_eq!(err.code(), -32602);
    }

    #[tokio::test]
    async fn state_override_methods_reject_invalid_param_arity() {
        let state = make_test_state("http://127.0.0.1:9".to_string(), false);
        let (rpc_url, server_handle) = start_state_override_rpc_server(state).await;

        for method in ["eth_estimateGas", "eth_createAccessList"] {
            let test_cases = [
                (json!([]), "expected 1-3 params, got 0"),
                (
                    json!([{}, "latest", {"0xabc": {"balance": "0x1"}}, "extra"]),
                    "expected 1-3 params, got 4",
                ),
            ];
            for (params, expected_message) in test_cases {
                let response = send_rpc_request(&rpc_url, method, params).await;
                assert_eq!(response.get("result"), None);
                assert_eq!(
                    response
                        .get("error")
                        .and_then(|err| err.get("code"))
                        .and_then(Value::as_i64),
                    Some(-32602)
                );
                assert_eq!(
                    response
                        .get("error")
                        .and_then(|err| err.get("message"))
                        .and_then(Value::as_str),
                    Some(expected_message)
                );
            }
        }

        server_handle.stop().expect("stop test rpc server");
        server_handle.stopped().await;
    }

    #[tokio::test]
    async fn estimate_gas_state_overrides_return_invalid_params_when_fallback_disabled() {
        let state = make_test_state("http://127.0.0.1:9".to_string(), false);
        let (rpc_url, server_handle) = start_state_override_rpc_server(state).await;

        let response = send_rpc_request(
            &rpc_url,
            "eth_estimateGas",
            json!([{}, "latest", {"0xabc": {"balance": "0x1"}}]),
        )
        .await;

        assert_eq!(response.get("result"), None);
        assert_eq!(
            response
                .get("error")
                .and_then(|err| err.get("code"))
                .and_then(Value::as_i64),
            Some(-32602)
        );
        assert_eq!(
            response
                .get("error")
                .and_then(|err| err.get("message"))
                .and_then(Value::as_str),
            Some("state overrides not supported for local eth_estimateGas")
        );

        server_handle.stop().expect("stop test rpc server");
        server_handle.stopped().await;
    }

    #[tokio::test]
    async fn create_access_list_state_overrides_return_invalid_params_when_fallback_disabled() {
        let state = make_test_state("http://127.0.0.1:9".to_string(), false);
        let (rpc_url, server_handle) = start_state_override_rpc_server(state).await;

        let response = send_rpc_request(
            &rpc_url,
            "eth_createAccessList",
            json!([{}, "latest", {"0xabc": {"balance": "0x1"}}]),
        )
        .await;

        assert_eq!(response.get("result"), None);
        assert_eq!(
            response
                .get("error")
                .and_then(|err| err.get("code"))
                .and_then(Value::as_i64),
            Some(-32602)
        );
        assert_eq!(
            response
                .get("error")
                .and_then(|err| err.get("message"))
                .and_then(Value::as_str),
            Some("state overrides not supported for local eth_createAccessList")
        );

        server_handle.stop().expect("stop test rpc server");
        server_handle.stopped().await;
    }

    #[tokio::test]
    async fn state_overrides_proxy_upstream_when_fallback_enabled() {
        for method in ["eth_estimateGas", "eth_createAccessList"] {
            let (upstream, upstream_handle) = spawn_mock_upstream_script(vec![(
                method,
                json!({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": "0x1234"
                }),
            )])
            .await;
            let state = make_test_state(upstream, true);
            let (rpc_url, server_handle) = start_state_override_rpc_server(state).await;

            let params = json!([{}, "latest", {"0xabc": {"balance": "0x1"}}]);
            let response = send_rpc_request(&rpc_url, method, params).await;

            assert_eq!(response.get("result"), Some(&json!("0x1234")));
            assert!(response.get("error").is_none());

            server_handle.stop().expect("stop test rpc server");
            server_handle.stopped().await;

            tokio::time::timeout(Duration::from_secs(5), upstream_handle)
                .await
                .expect("upstream capture should complete")
                .expect("upstream capture task should succeed");
        }
    }

    #[test]
    fn effective_latest_includes_resolved_finality_heights() {
        assert_eq!(effective_latest_for_filter(100, None, None), 100);
        assert_eq!(effective_latest_for_filter(100, Some(120), None), 120);
        assert_eq!(effective_latest_for_filter(100, None, Some(130)), 130);
        assert_eq!(effective_latest_for_filter(100, Some(120), Some(130)), 130);
    }

    #[test]
    fn stale_cache_latest_does_not_invalidate_safe_default_range() {
        let filter_obj = json!({ "fromBlock": "safe" });
        let effective_latest = effective_latest_for_filter(100, Some(120), None);
        let filter = crate::block_cache::parse_log_filter_object(
            &filter_obj,
            effective_latest,
            Some(120),
            None,
        )
        .expect("safe range should remain valid when cache latest lags");
        assert_eq!(filter.from_block, 120);
        assert_eq!(filter.to_block, 120);
    }

    #[test]
    fn stale_cache_latest_does_not_invalidate_finalized_default_range() {
        let filter_obj = json!({ "fromBlock": "finalized" });
        let effective_latest = effective_latest_for_filter(100, None, Some(130));
        let filter = crate::block_cache::parse_log_filter_object(
            &filter_obj,
            effective_latest,
            None,
            Some(130),
        )
        .expect("finalized range should remain valid when cache latest lags");
        assert_eq!(filter.from_block, 130);
        assert_eq!(filter.to_block, 130);
    }

    #[test]
    fn explicit_latest_to_block_uses_effective_latest_when_safe_is_higher() {
        let filter_obj = json!({ "fromBlock": "safe", "toBlock": "latest" });
        let effective_latest = effective_latest_for_filter(100, Some(120), None);
        let filter = crate::block_cache::parse_log_filter_object(
            &filter_obj,
            effective_latest,
            Some(120),
            None,
        )
        .expect("explicit latest should resolve to effective latest");
        assert_eq!(filter.from_block, 120);
        assert_eq!(filter.to_block, 120);
    }

    #[tokio::test]
    async fn parse_log_filter_for_rpc_handles_stale_cache_for_safe_tag() {
        let upstream = spawn_mock_upstream(vec![json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": { "number": "0x78" } // 120
        })])
        .await;
        let client = reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("build client");
        let filter_obj = json!({ "fromBlock": "safe" });

        let filter = parse_log_filter_for_rpc(&filter_obj, 100, &client, &upstream)
            .await
            .expect("safe filter should parse with effective latest");

        assert_eq!(filter.from_block, 120);
        assert_eq!(filter.to_block, 120);
    }

    #[tokio::test]
    async fn parse_log_filter_for_rpc_handles_stale_cache_for_finalized_and_latest() {
        let upstream = spawn_mock_upstream(vec![json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": { "number": "0x82" } // 130
        })])
        .await;
        let client = reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("build client");
        let filter_obj = json!({ "fromBlock": "finalized", "toBlock": "latest" });

        let filter = parse_log_filter_for_rpc(&filter_obj, 100, &client, &upstream)
            .await
            .expect("finalized/latest filter should parse with effective latest");

        assert_eq!(filter.from_block, 130);
        assert_eq!(filter.to_block, 130);
    }

    #[tokio::test]
    async fn eth_get_logs_handler_accepts_stale_cache_safe_range_with_fallback() {
        let (upstream, upstream_handle) = spawn_mock_upstream_script(vec![
            (
                "eth_getBlockByNumber",
                json!({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": { "number": "0x78" }
                }),
            ),
            (
                "eth_getLogs",
                json!({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": []
                }),
            ),
        ])
        .await;
        let state = make_test_state(upstream, true);
        {
            let mut cache = state.block_cache.write().await;
            cache.insert_block(100, [0x01; 32], vec![], vec![]);
        }

        let result = handle_eth_get_logs(vec![json!({"fromBlock": "safe"})], state)
            .await
            .expect("handler should not reject stale-cache safe range");

        assert_eq!(result, Value::Array(vec![]));
        tokio::time::timeout(Duration::from_secs(5), upstream_handle)
            .await
            .expect("mock upstream script should complete")
            .expect("mock upstream task should succeed");
    }

    #[tokio::test]
    async fn eth_new_filter_handler_accepts_stale_cache_finalized_latest_range() {
        let (upstream, upstream_handle) = spawn_mock_upstream_script(vec![(
            "eth_getBlockByNumber",
            json!({
                "jsonrpc": "2.0",
                "id": 1,
                "result": { "number": "0x82" }
            }),
        )])
        .await;
        let state = make_test_state(upstream, false);
        {
            let mut cache = state.block_cache.write().await;
            cache.insert_block(100, [0x01; 32], vec![], vec![]);
        }

        let filter_id = handle_eth_new_filter(
            vec![json!({"fromBlock": "finalized", "toBlock": "latest"})],
            state.clone(),
        )
        .await
        .expect("handler should accept stale-cache finalized/latest filter")
        .as_str()
        .expect("filter id string")
        .to_string();

        let early_log = json!({"address": "0x1111", "topics": [], "data": "0x"});
        let finalized_log = json!({"address": "0x2222", "topics": [], "data": "0x"});
        {
            let mut cache = state.block_cache.write().await;
            cache.insert_block(
                110,
                [0x02; 32],
                vec![],
                vec![([0xAA; 32], json!({"logs": [early_log]}))],
            );
            cache.insert_block(
                130,
                [0x03; 32],
                vec![],
                vec![([0xBB; 32], json!({"logs": [finalized_log.clone()]}))],
            );
        }

        let logs = state
            .block_cache
            .write()
            .await
            .get_filter_logs(&filter_id)
            .expect("filter exists")
            .expect("log filter");
        assert_eq!(logs.len(), 1);
        assert_eq!(logs[0], finalized_log);
        tokio::time::timeout(Duration::from_secs(5), upstream_handle)
            .await
            .expect("mock upstream script should complete")
            .expect("mock upstream task should succeed");
    }

    #[tokio::test]
    async fn eth_get_logs_rpc_method_handles_stale_cache_safe_range() {
        let (upstream, upstream_handle) = spawn_mock_upstream_script(vec![
            (
                "eth_getBlockByNumber",
                json!({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": { "number": "0x78" }
                }),
            ),
            (
                "eth_getLogs",
                json!({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": []
                }),
            ),
        ])
        .await;
        let state = make_test_state(upstream, true);
        {
            let mut cache = state.block_cache.write().await;
            cache.insert_block(100, [0x01; 32], vec![], vec![]);
        }
        let (rpc_url, server_handle) = start_filter_rpc_server(state).await;
        let client = reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("build client");

        let response: Value = client
            .post(&rpc_url)
            .json(&json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "eth_getLogs",
                "params": [{"fromBlock": "safe"}]
            }))
            .send()
            .await
            .expect("send rpc request")
            .json()
            .await
            .expect("parse rpc response");
        assert_eq!(response.get("result"), Some(&json!([])));
        assert!(response.get("error").is_none());

        server_handle.stop().expect("stop test rpc server");
        server_handle.stopped().await;
        tokio::time::timeout(Duration::from_secs(5), upstream_handle)
            .await
            .expect("mock upstream script should complete")
            .expect("mock upstream task should succeed");
    }

    #[tokio::test]
    async fn eth_new_filter_rpc_method_handles_stale_cache_finalized_latest_range() {
        let (upstream, upstream_handle) = spawn_mock_upstream_script(vec![(
            "eth_getBlockByNumber",
            json!({
                "jsonrpc": "2.0",
                "id": 1,
                "result": { "number": "0x82" }
            }),
        )])
        .await;
        let state = make_test_state(upstream, false);
        {
            let mut cache = state.block_cache.write().await;
            cache.insert_block(100, [0x01; 32], vec![], vec![]);
        }
        let (rpc_url, server_handle) = start_filter_rpc_server(state.clone()).await;
        let client = reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("build client");

        let new_filter_response: Value = client
            .post(&rpc_url)
            .json(&json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "eth_newFilter",
                "params": [{"fromBlock": "finalized", "toBlock": "latest"}]
            }))
            .send()
            .await
            .expect("send eth_newFilter")
            .json()
            .await
            .expect("parse eth_newFilter response");
        let filter_id = new_filter_response
            .get("result")
            .and_then(Value::as_str)
            .expect("filter id result")
            .to_string();

        let finalized_log = json!({"address": "0x2222", "topics": [], "data": "0x"});
        {
            let mut cache = state.block_cache.write().await;
            cache.insert_block(
                110,
                [0x02; 32],
                vec![],
                vec![(
                    [0xAA; 32],
                    json!({"logs": [{"address": "0x1111", "topics": [], "data": "0x"}]}),
                )],
            );
            cache.insert_block(
                130,
                [0x03; 32],
                vec![],
                vec![([0xBB; 32], json!({"logs": [finalized_log.clone()]}))],
            );
        }

        let filter_logs_response: Value = client
            .post(&rpc_url)
            .json(&json!({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "eth_getFilterLogs",
                "params": [filter_id]
            }))
            .send()
            .await
            .expect("send eth_getFilterLogs")
            .json()
            .await
            .expect("parse eth_getFilterLogs response");
        let logs = filter_logs_response
            .get("result")
            .and_then(Value::as_array)
            .expect("logs array");
        assert_eq!(logs.len(), 1);
        assert_eq!(logs[0], finalized_log);

        server_handle.stop().expect("stop test rpc server");
        server_handle.stopped().await;
        tokio::time::timeout(Duration::from_secs(5), upstream_handle)
            .await
            .expect("mock upstream script should complete")
            .expect("mock upstream task should succeed");
    }
}
