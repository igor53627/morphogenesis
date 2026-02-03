mod code_resolver;

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
}

struct AdapterState {
    args: Args,
    http_client: reqwest::Client,
    pir_client: PirClient,
    code_resolver: CodeResolver,
}

const PASSTHROUGH_METHODS: &[&str] = &[
    "eth_blockNumber",
    "eth_chainId",
    "eth_gasPrice",
    "eth_estimateGas",
    "eth_sendRawTransaction",
    "eth_call", // Still passthrough for now until Phase 2
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
    // Storage & State (passthrough for now - TODO: implement private eth_getStorageAt)
    "eth_getStorageAt",
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

    let pir_client = PirClient::new(args.pir_server_a.clone(), args.pir_server_b.clone());
    let code_resolver = CodeResolver::new(args.dict_url.clone(), args.cas_url.clone());

    let state = Arc::new(AdapterState {
        args: args.clone(),
        http_client: reqwest::Client::new(),
        pir_client,
        code_resolver,
    });

    // Background task for metadata refresh
    let state_clone = state.clone();
    tokio::spawn(async move {
        loop {
            match state_clone.pir_client.update_metadata().await {
                Ok(m) => info!(
                    "Updated PIR metadata: epoch={}, block={}",
                    m.epoch_id, m.block_number
                ),
                Err(e) => warn!("Failed to update PIR metadata: {}", e),
            }
            sleep(Duration::from_secs(state_clone.args.refresh_interval)).await;
        }
    });

    let mut module = RpcModule::from_arc(state.clone());

    // Register eth_getBalance (Private)
    module.register_async_method("eth_getBalance", |params, state, _| async move {
        let (address_str, _block): (String, Value) = params.parse()?;
        let address_hex = address_str.strip_prefix("0x").unwrap_or(&address_str);
        let mut address = [0u8; 20];
        hex::decode_to_slice(address_hex, &mut address).map_err(|e| {
            ErrorObjectOwned::owned(-32602, format!("Invalid address: {}", e), None::<()>)
        })?;

        info!("Private eth_getBalance for 0x{}", address_hex);

        let account = state.pir_client.query_account(address).await.map_err(|e| {
            error!("PIR query failed: {}", e);
            ErrorObjectOwned::owned(-32000, "Internal PIR error".to_string(), None::<()>)
        })?;

        Ok::<Value, ErrorObjectOwned>(Value::String(format!("0x{:x}", account.balance)))
    })?;

    // Register eth_getTransactionCount (Private)
    module.register_async_method("eth_getTransactionCount", |params, state, _| async move {
        let (address_str, _block): (String, Value) = params.parse()?;
        let address_hex = address_str.strip_prefix("0x").unwrap_or(&address_str);
        let mut address = [0u8; 20];
        hex::decode_to_slice(address_hex, &mut address).map_err(|e| {
            ErrorObjectOwned::owned(-32602, format!("Invalid address: {}", e), None::<()>)
        })?;

        info!("Private eth_getTransactionCount for 0x{}", address_hex);

        let account = state.pir_client.query_account(address).await.map_err(|e| {
            error!("PIR query failed: {}", e);
            ErrorObjectOwned::owned(-32000, "Internal PIR error".to_string(), None::<()>)
        })?;

        Ok::<Value, ErrorObjectOwned>(Value::String(format!("0x{:x}", account.nonce)))
    })?;

    // Register eth_getCode (Private via CAS)
    module.register_async_method("eth_getCode", |params, state, _| async move {
        let (address_str, _block): (String, Value) = params.parse()?;
        let address_hex = address_str.strip_prefix("0x").unwrap_or(&address_str);
        let mut address = [0u8; 20];
        hex::decode_to_slice(address_hex, &mut address).map_err(|e| {
            ErrorObjectOwned::owned(-32602, format!("Invalid address: {}", e), None::<()>)
        })?;

        info!("Private eth_getCode for 0x{}", address_hex);

        // 1. PIR Query for Account Data
        let account = state.pir_client.query_account(address).await.map_err(|e| {
            error!("PIR query failed: {}", e);
            ErrorObjectOwned::owned(-32000, "Internal PIR error".to_string(), None::<()>)
        })?;

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

    let response = client
        .post(url)
        .json(&request)
        .send()
        .await
        .map_err(|e| ErrorObjectOwned::owned(-32000, e.to_string(), None::<()>))?;

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
    fn test_passthrough_methods_include_storage_and_filter_apis() {
        // Test against actual production allowlist (prevents regression)

        // Verify storage methods are included
        assert!(PASSTHROUGH_METHODS.contains(&"eth_getStorageAt"));
        assert!(PASSTHROUGH_METHODS.contains(&"eth_getProof"));

        // Verify filter APIs are included
        assert!(PASSTHROUGH_METHODS.contains(&"eth_newFilter"));
        assert!(PASSTHROUGH_METHODS.contains(&"eth_getFilterChanges"));

        // Verify signing methods are NOT included (security)
        assert!(!PASSTHROUGH_METHODS.contains(&"eth_sign"));
        assert!(!PASSTHROUGH_METHODS.contains(&"eth_signTransaction"));
    }
}
