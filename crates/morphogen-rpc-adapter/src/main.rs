mod code_resolver;
mod evm;
mod pir_db;

use anyhow::Result;
use clap::Parser;
use code_resolver::CodeResolver;
use jsonrpsee::server::{RpcModule, Server};
use jsonrpsee::types::ErrorObjectOwned;
use morphogen_client::network::PirClient;
use serde_json::Value;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{error, info, warn};

#[derive(Parser, Debug, Clone)]
#[command(author, version, about = "Morphogenesis RPC Adapter")]
struct Args {
    /// Port to listen on
    #[arg(short, long, default_value_t = 8545)]
    port: u16,

    /// Upstream Ethereum RPC URL
    #[arg(short, long, default_value = "https://ethereum-rpc.publicnode.com")]
    upstream: String,

    /// PIR Server A URL
    #[arg(long, default_value = "http://localhost:3000")]
    pir_server_a: String,

    /// PIR Server B URL
    #[arg(long, default_value = "http://localhost:3001")]
    pir_server_b: String,

    /// Dictionary URL for CodeID resolution
    #[arg(long, default_value = "http://localhost:8080/mainnet_compact.dict")]
    dict_url: String,

    /// CAS Base URL for bytecode fetching
    #[arg(long, default_value = "http://localhost:8080/cas")]
    cas_url: String,

    /// Metadata refresh interval in seconds
    #[arg(long, default_value_t = 12)]
    refresh_interval: u64,

    /// Upstream request timeout in seconds
    #[arg(long, default_value_t = 15)]
    upstream_timeout: u64,

    /// Fall back to upstream RPC when PIR servers are unavailable
    #[arg(long, default_value_t = false)]
    fallback_to_upstream: bool,
}

struct AdapterState {
    args: Args,
    http_client: reqwest::Client,
    pir_client: Arc<PirClient>,
    code_resolver: Arc<CodeResolver>,
}

const PASSTHROUGH_METHODS: &[&str] = &[
    "eth_blockNumber",
    "eth_chainId",
    "eth_gasPrice",
    "eth_estimateGas",
    "eth_sendRawTransaction",
    "net_version",
    "web3_clientVersion",
    // Wallet Essentials (History & Status)
    "eth_getTransactionByHash",
    "eth_getTransactionReceipt",
    "eth_getBlockByNumber",
    "eth_getBlockByHash",
    "eth_feeHistory",
    "eth_maxPriorityFeePerGas",
    // Warning: eth_getLogs leaks privacy to upstream
    "eth_getLogs",
    // Storage & State (eth_getStorageAt now private via PIR)
    "eth_getProof",
    // Account queries (read-only, safe to passthrough)
    "eth_accounts",
    // NOTE: eth_sign and eth_signTransaction are intentionally NOT included
    // These should be handled client-side by wallets to avoid remote signing risks
    // Filter APIs (for event monitoring)
    // TODO: Implement sticky routing for filter IDs if using load-balanced upstreams
    // Current implementation assumes single fixed upstream provider
    "eth_newFilter",
    "eth_newBlockFilter",
    "eth_newPendingTransactionFilter",
    "eth_uninstallFilter",
    "eth_getFilterChanges",
    "eth_getFilterLogs",
];

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let addr = SocketAddr::from(([127, 0, 0, 1], args.port));
    let server = Server::builder().build(addr).await?;

    let pir_client = Arc::new(PirClient::new(
        args.pir_server_a.clone(),
        args.pir_server_b.clone(),
    ));
    let code_resolver = Arc::new(CodeResolver::new(
        args.dict_url.clone(),
        args.cas_url.clone(),
    ));

    let http_client = reqwest::Client::builder()
        .timeout(Duration::from_secs(args.upstream_timeout))
        .connect_timeout(Duration::from_secs(5))
        .build()?;

    let state = Arc::new(AdapterState {
        args: args.clone(),
        http_client,
        pir_client,
        code_resolver,
    });

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
    module.register_async_method("eth_getBalance", |params, state, _| async move {
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
                    warn!("Falling back to upstream for eth_getBalance (privacy degraded)");
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
    })?;

    // Register eth_getTransactionCount (Private)
    module.register_async_method("eth_getTransactionCount", |params, state, _| async move {
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
                    warn!(
                        "Falling back to upstream for eth_getTransactionCount (privacy degraded)"
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
    })?;

    // Register eth_getCode (Private via CAS)
    module.register_async_method("eth_getCode", |params, state, _| async move {
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
                    warn!("Falling back to upstream for eth_getCode (privacy degraded)");
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
    })?;

    // Register eth_getStorageAt (Private via PIR)
    module.register_async_method("eth_getStorageAt", |params, state, _| async move {
        let (address_str, slot_str, block): (String, String, Value) = params.parse()?;
        let address_hex = address_str.strip_prefix("0x").unwrap_or(&address_str);
        let slot_hex = slot_str.strip_prefix("0x").unwrap_or(&slot_str);

        let mut address = [0u8; 20];
        hex::decode_to_slice(address_hex, &mut address).map_err(|e| {
            ErrorObjectOwned::owned(-32602, format!("Invalid address: {}", e), None::<()>)
        })?;

        let mut slot = [0u8; 32];
        // Pad slot to 32 bytes if shorter
        let slot_bytes = hex::decode(slot_hex).map_err(|e| {
            ErrorObjectOwned::owned(-32602, format!("Invalid slot: {}", e), None::<()>)
        })?;
        if slot_bytes.len() > 32 {
            return Err(ErrorObjectOwned::owned(
                -32602,
                "Slot too long (max 32 bytes)".to_string(),
                None::<()>,
            ));
        }
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
                    warn!("Falling back to upstream for eth_getStorageAt (privacy degraded)");
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
    })?;

    // Register eth_call (Private via local EVM execution)
    module.register_async_method("eth_call", |params, state, _| async move {
        // Block tag is optional; clients may send 1 or 2 params
        let raw: Vec<Value> = params.parse()?;
        let call_params = raw
            .first()
            .ok_or_else(|| ErrorObjectOwned::owned(-32602, "missing call object", None::<()>))?;
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
            Ok(output) => {
                Ok::<Value, ErrorObjectOwned>(Value::String(format!("0x{}", hex::encode(&output))))
            }
            Err(e) => {
                error!("Private eth_call failed: {}", e);
                if state.args.fallback_to_upstream {
                    warn!("Falling back to upstream for eth_call (privacy degraded)");
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
    })?;

    // Register passthrough methods
    for method in PASSTHROUGH_METHODS {
        let method_name = method.to_string();
        module.register_async_method(method, move |params, state, _| {
            let m = method_name.clone();
            async move {
                proxy_to_upstream(
                    &state.args.upstream,
                    &state.http_client,
                    &m,
                    params.parse()?,
                )
                .await
            }
        })?;
    }

    info!("Morphogenesis RPC Adapter listening on {}", addr);
    info!("Upstream RPC: {}", args.upstream);
    if args.fallback_to_upstream {
        warn!("Fallback to upstream enabled - privacy will be degraded when PIR is unavailable");
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
            ErrorObjectOwned::owned(-32000, e.to_string(), None::<()>)
        }
    })?;

    let json: Value = response
        .json()
        .await
        .map_err(|e| ErrorObjectOwned::owned(-32000, e.to_string(), None::<()>))?;

    if let Some(error) = json.get("error") {
        warn!("Upstream error for {}: {:?}", method, error);
        return Err(ErrorObjectOwned::owned(
            error.get("code").and_then(|c| c.as_i64()).unwrap_or(-32000) as i32,
            error
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("Unknown error")
                .to_string(),
            error.get("data").cloned(),
        ));
    }

    Ok(json.get("result").cloned().unwrap_or(Value::Null))
}

#[cfg(test)]
mod tests {
    use super::PASSTHROUGH_METHODS;

    #[test]
    fn test_passthrough_methods_include_filter_apis() {
        // Test against actual production allowlist (prevents regression)

        // Verify private methods are NOT in passthrough
        assert!(!PASSTHROUGH_METHODS.contains(&"eth_getStorageAt"));
        assert!(!PASSTHROUGH_METHODS.contains(&"eth_call"));

        // Verify eth_getProof is still passthrough
        assert!(PASSTHROUGH_METHODS.contains(&"eth_getProof"));

        // Verify filter APIs are included
        assert!(PASSTHROUGH_METHODS.contains(&"eth_newFilter"));
        assert!(PASSTHROUGH_METHODS.contains(&"eth_getFilterChanges"));

        // Verify signing methods are NOT included (security)
        assert!(!PASSTHROUGH_METHODS.contains(&"eth_sign"));
        assert!(!PASSTHROUGH_METHODS.contains(&"eth_signTransaction"));
    }
}
