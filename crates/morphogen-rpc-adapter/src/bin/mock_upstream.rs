use clap::Parser;
use jsonrpsee::server::{RpcModule, Server};
use jsonrpsee::types::ErrorObjectOwned;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

#[derive(Parser, Debug, Clone)]
#[command(author, version, about = "Deterministic mock upstream JSON-RPC server")]
struct Args {
    /// Port to listen on
    #[arg(short, long, default_value_t = 18545)]
    port: u16,
}

#[derive(Clone)]
struct MockState {
    chain: Arc<RwLock<ChainState>>,
}

#[derive(Clone)]
struct BlockData {
    hash: String,
    timestamp: u64,
    gas_limit: u64,
    transactions: Vec<Value>,
    receipts: Vec<Value>,
}

struct ChainState {
    latest: u64,
    blocks: BTreeMap<u64, BlockData>,
}

fn hex_u64(n: u64) -> String {
    format!("0x{:x}", n)
}

fn hash_hex(byte: u8) -> String {
    format!("0x{}", hex::encode([byte; 32]))
}

fn block_json(number: u64, block: &BlockData, full_tx: bool) -> Value {
    let transactions = if full_tx {
        block.transactions.clone()
    } else {
        block
            .transactions
            .iter()
            .filter_map(|tx| tx.get("hash").cloned())
            .collect::<Vec<_>>()
    };

    json!({
        "number": hex_u64(number),
        "hash": block.hash,
        "timestamp": hex_u64(block.timestamp),
        "gasLimit": hex_u64(block.gas_limit),
        "transactions": transactions,
    })
}

fn resolve_block_tag(tag: &str, latest: u64) -> Result<u64, ErrorObjectOwned> {
    if tag == "latest" || tag == "pending" || tag == "safe" || tag == "finalized" {
        return Ok(latest);
    }
    if tag == "earliest" {
        return Ok(0);
    }
    let hex = tag.strip_prefix("0x").unwrap_or(tag);
    let number = u64::from_str_radix(hex, 16).map_err(|e| {
        ErrorObjectOwned::owned(
            -32602,
            format!("invalid block tag '{}': {}", tag, e),
            None::<()>,
        )
    })?;
    Ok(number)
}

fn extract_logs_in_range(chain: &ChainState, from: u64, to: u64, filter: &Value) -> Vec<Value> {
    let address_filter = filter
        .get("address")
        .and_then(|v| v.as_str())
        .map(|s| s.to_lowercase());

    let topic0_filter = filter
        .get("topics")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.first())
        .and_then(|v| v.as_str())
        .map(|s| s.to_lowercase());

    let mut out = Vec::new();
    for (_, block) in chain.blocks.range(from..=to) {
        for receipt in &block.receipts {
            if let Some(logs) = receipt.get("logs").and_then(|v| v.as_array()) {
                for log in logs {
                    if let Some(addr) = &address_filter {
                        let log_addr = log
                            .get("address")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_lowercase();
                        if &log_addr != addr {
                            continue;
                        }
                    }
                    if let Some(t0) = &topic0_filter {
                        let actual = log
                            .get("topics")
                            .and_then(|v| v.as_array())
                            .and_then(|arr| arr.first())
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_lowercase();
                        if &actual != t0 {
                            continue;
                        }
                    }
                    out.push(log.clone());
                }
            }
        }
    }
    out
}

fn seed_chain() -> ChainState {
    let block_number = 1u64;
    let block_hash = hash_hex(0x11);
    let tx_hash = hash_hex(0x22);
    let log_topic = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef";
    let test_addr = "0x000000000000000000000000000000000000031c";

    let tx = json!({
        "hash": tx_hash,
        "from": test_addr,
        "to": test_addr,
        "value": "0x0",
        "input": "0x",
        "blockNumber": hex_u64(block_number),
        "transactionIndex": "0x0",
    });

    let log = json!({
        "address": test_addr,
        "topics": [log_topic],
        "data": "0x",
        "blockNumber": hex_u64(block_number),
        "transactionHash": tx_hash,
        "transactionIndex": "0x0",
        "blockHash": block_hash,
        "logIndex": "0x0",
        "removed": false,
    });

    let receipt = json!({
        "transactionHash": tx_hash,
        "blockHash": block_hash,
        "blockNumber": hex_u64(block_number),
        "transactionIndex": "0x0",
        "status": "0x1",
        "logs": [log],
    });

    let mut blocks = BTreeMap::new();
    blocks.insert(
        block_number,
        BlockData {
            hash: block_hash,
            timestamp: 1_700_000_000,
            gas_limit: 30_000_000,
            transactions: vec![tx],
            receipts: vec![receipt],
        },
    );

    ChainState { latest: 1, blocks }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let state = Arc::new(MockState {
        chain: Arc::new(RwLock::new(seed_chain())),
    });
    let mut module = RpcModule::from_arc(state);

    module.register_async_method("eth_chainId", |_params, _state, _| async move {
        Ok::<Value, ErrorObjectOwned>(Value::String("0x1".to_string()))
    })?;

    module.register_async_method("net_version", |_params, _state, _| async move {
        Ok::<Value, ErrorObjectOwned>(Value::String("1".to_string()))
    })?;

    module.register_async_method("eth_blockNumber", |_params, state, _| async move {
        let latest = state.chain.read().await.latest;
        Ok::<Value, ErrorObjectOwned>(Value::String(hex_u64(latest)))
    })?;

    module.register_async_method("eth_getBlockByNumber", |params, state, _| async move {
        let (tag, full): (String, bool) = params.parse()?;
        let chain = state.chain.read().await;
        let number = resolve_block_tag(&tag, chain.latest)?;
        let out = match chain.blocks.get(&number) {
            Some(block) => block_json(number, block, full),
            None => Value::Null,
        };
        Ok::<Value, ErrorObjectOwned>(out)
    })?;

    module.register_async_method("eth_getBlockReceipts", |params, state, _| async move {
        let (tag,): (String,) = params.parse()?;
        let chain = state.chain.read().await;
        let number = resolve_block_tag(&tag, chain.latest)?;
        let out = match chain.blocks.get(&number) {
            Some(block) => Value::Array(block.receipts.clone()),
            None => Value::Null,
        };
        Ok::<Value, ErrorObjectOwned>(out)
    })?;

    module.register_async_method("eth_getTransactionReceipt", |params, state, _| async move {
        let (hash,): (String,) = params.parse()?;
        let hash_lc = hash.to_lowercase();
        let chain = state.chain.read().await;
        for block in chain.blocks.values() {
            for receipt in &block.receipts {
                let tx_hash = receipt
                    .get("transactionHash")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_lowercase();
                if tx_hash == hash_lc {
                    return Ok::<Value, ErrorObjectOwned>(receipt.clone());
                }
            }
        }
        Ok::<Value, ErrorObjectOwned>(Value::Null)
    })?;

    module.register_async_method("eth_getLogs", |params, state, _| async move {
        let raw: Vec<Value> = params.parse()?;
        if raw.len() != 1 {
            return Err(ErrorObjectOwned::owned(
                -32602,
                format!("expected 1 param, got {}", raw.len()),
                None::<()>,
            ));
        }
        let filter = &raw[0];

        let chain = state.chain.read().await;
        let latest = chain.latest;
        let from = filter
            .get("fromBlock")
            .and_then(|v| v.as_str())
            .map(|s| resolve_block_tag(s, latest))
            .transpose()?
            .unwrap_or(latest);
        let to = filter
            .get("toBlock")
            .and_then(|v| v.as_str())
            .map(|s| resolve_block_tag(s, latest))
            .transpose()?
            .unwrap_or(latest);

        if from > to {
            return Err(ErrorObjectOwned::owned(
                -32602,
                format!("invalid range: fromBlock {} > toBlock {}", from, to),
                None::<()>,
            ));
        }

        let logs = extract_logs_in_range(&chain, from, to, filter);
        Ok::<Value, ErrorObjectOwned>(Value::Array(logs))
    })?;

    module.register_async_method("evm_mine", |params, state, _| async move {
        let raw: Vec<Value> = params.parse()?;
        let count = raw.first().and_then(|v| v.as_u64()).unwrap_or(1);

        let mut chain = state.chain.write().await;
        for _ in 0..count {
            let next = chain.latest + 1;
            let hash = hash_hex((next as u8).wrapping_add(0x30));
            let timestamp = chain
                .blocks
                .get(&chain.latest)
                .map(|b| b.timestamp + 12)
                .unwrap_or(1_700_000_000 + (next * 12));
            chain.blocks.insert(
                next,
                BlockData {
                    hash,
                    timestamp,
                    gas_limit: 30_000_000,
                    transactions: vec![],
                    receipts: vec![],
                },
            );
            chain.latest = next;
        }

        Ok::<Value, ErrorObjectOwned>(Value::String("0x0".to_string()))
    })?;

    let addr = SocketAddr::from(([127, 0, 0, 1], args.port));
    let server = Server::builder().build(addr).await?;
    info!("Mock upstream listening on {}", addr);
    let handle = server.start(module);
    handle.stopped().await;
    Ok(())
}
