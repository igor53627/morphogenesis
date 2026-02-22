use crate::code_resolver::CodeResolver;
use crate::pir_db::{AccessListCollector, PirDatabase, SharedAccessListCollector};
use alloy_primitives::{Address, Bytes, B256, U256};
use morphogen_client::network::PirClient;
use revm::{
    context::TxEnv,
    context_interface::result::{ExecutionResult, Output},
    context_interface::transaction::{
        AccessList as TxAccessList, AccessListItem as TxAccessListItem,
    },
    primitives::TxKind,
    Context, ExecuteEvm, MainBuilder, MainContext,
};
use serde_json::Value;
use std::sync::{Arc, Mutex};
use tokio::runtime::Handle;
use tracing::{info, warn};

/// Error types for eth_call, distinguishing input validation from internal failures.
#[derive(Debug)]
pub enum EthCallError {
    /// Invalid input parameters (maps to JSON-RPC -32602)
    InvalidParams(String),
    /// Internal execution failure (maps to JSON-RPC -32000)
    Internal(String),
}

/// Block environment fetched from upstream for accurate EVM execution.
struct BlockInfo {
    number: u64,
    timestamp: u64,
    gas_limit: u64,
    /// True when block exposes blob gas fields (Cancun-era behavior).
    is_cancun_or_later: bool,
}

/// Known named block tags per Ethereum JSON-RPC specification.
const NAMED_BLOCK_TAGS: &[&str] = &["latest", "earliest", "pending", "safe", "finalized"];

/// Validate block tag: must be a named tag, hex quantity ("0x..."), or null.
/// Rejects EIP-1898 object forms since PIR state is epoch-based.
fn validate_block_tag(block_tag: &Value) -> Result<String, String> {
    if block_tag.is_null() {
        return Ok("latest".to_string());
    }
    let s = block_tag.as_str().ok_or_else(|| {
        if block_tag.is_object() {
            "EIP-1898 object form not supported".to_string()
        } else {
            "block parameter must be a string or null".to_string()
        }
    })?;
    if NAMED_BLOCK_TAGS.contains(&s) {
        return Ok(s.to_string());
    }
    // Must be a hex quantity: "0x" followed by one or more hex digits
    let hex = s.strip_prefix("0x").ok_or_else(|| {
        format!(
            "invalid block tag '{}': expected named tag or hex quantity",
            s
        )
    })?;
    if hex.is_empty() || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(format!(
            "invalid block tag '{}': expected named tag or hex quantity",
            s
        ));
    }
    Ok(s.to_string())
}

/// Fetch block header fields from upstream for the EVM block env.
/// `block_tag` must be a pre-validated tag string (from `validate_block_tag`).
fn parse_u64_env(var_name: &str, raw: &str) -> Option<u64> {
    let parsed = if let Some(hex) = raw.strip_prefix("0x") {
        u64::from_str_radix(hex, 16)
    } else {
        raw.parse::<u64>()
    };

    match parsed {
        Ok(value) => Some(value),
        Err(e) => {
            warn!(
                env_var = var_name,
                env_value = raw,
                error = %e,
                "Invalid u64 env var; ignoring value"
            );
            None
        }
    }
}

fn configured_cancun_block() -> Option<u64> {
    std::env::var("MORPHOGEN_CANCUN_BLOCK")
        .ok()
        .and_then(|v| parse_u64_env("MORPHOGEN_CANCUN_BLOCK", &v))
}

fn infer_is_cancun_or_later(
    block_result: &Value,
    block_number: u64,
    fallback_cancun_block: Option<u64>,
) -> bool {
    let has_blob_fields = block_result
        .get("blobGasUsed")
        .is_some_and(|v| !v.is_null())
        || block_result
            .get("excessBlobGas")
            .is_some_and(|v| !v.is_null());
    if has_blob_fields {
        return true;
    }

    fallback_cancun_block.is_some_and(|fork_block| block_number >= fork_block)
}

async fn fetch_block_info(
    client: &reqwest::Client,
    upstream_url: &str,
    block_tag: &str,
) -> Result<BlockInfo, String> {
    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_getBlockByNumber",
        "params": [block_tag, false]
    });

    let resp = client
        .post(upstream_url)
        .json(&request)
        .send()
        .await
        .map_err(|e| format!("Block info fetch: {}", e))?;

    if !resp.status().is_success() {
        return Err(format!("Block info HTTP {}", resp.status()));
    }

    let json: Value = resp
        .json()
        .await
        .map_err(|e| format!("Block info parse: {}", e))?;

    if let Some(err) = json.get("error") {
        let msg = err
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("unknown");
        return Err(format!("Block info RPC error: {}", msg));
    }

    let result = json
        .get("result")
        .filter(|v| !v.is_null())
        .ok_or("No block result from upstream")?;

    let parse_hex_u64 = |field: &str| -> Result<u64, String> {
        let s = result
            .get(field)
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("missing block field '{}'", field))?;
        let hex = s.strip_prefix("0x").unwrap_or(s);
        u64::from_str_radix(hex, 16).map_err(|e| format!("invalid block field '{}': {}", field, e))
    };

    let number = parse_hex_u64("number")?;
    let timestamp = parse_hex_u64("timestamp")?;
    let gas_limit = parse_hex_u64("gasLimit")?;

    Ok(BlockInfo {
        number,
        timestamp,
        gas_limit,
        is_cancun_or_later: infer_is_cancun_or_later(result, number, configured_cancun_block()),
    })
}

/// Parsed call parameters for EVM execution.
struct CallParams {
    from: Address,
    to: Option<Address>,
    data: Bytes,
    value: U256,
    gas: u64,
    access_list: TxAccessList,
}

/// Parse and validate call parameters from JSON-RPC input.
fn parse_call_params(call_params: &Value) -> Result<CallParams, EthCallError> {
    let from = parse_address_strict(call_params.get("from"))
        .map_err(|e| EthCallError::InvalidParams(format!("invalid 'from': {}", e)))?
        .unwrap_or_default();
    let to = parse_address_strict(call_params.get("to"))
        .map_err(|e| EthCallError::InvalidParams(format!("invalid 'to': {}", e)))?;
    let data = parse_bytes_strict(call_params.get("data").or(call_params.get("input")))
        .map_err(|e| EthCallError::InvalidParams(format!("invalid 'data': {}", e)))?;
    let value = parse_u256_strict(call_params.get("value"))
        .map_err(|e| EthCallError::InvalidParams(format!("invalid 'value': {}", e)))?;
    let gas = parse_u64_strict(call_params.get("gas"))
        .map_err(|e| EthCallError::InvalidParams(format!("invalid 'gas': {}", e)))?
        .unwrap_or(30_000_000);
    let access_list = parse_access_list_strict(call_params.get("accessList"))
        .map_err(|e| EthCallError::InvalidParams(format!("invalid 'accessList': {}", e)))?;
    Ok(CallParams {
        from,
        to,
        data,
        value,
        gas,
        access_list,
    })
}

/// Maximum number of access-list entries to prefetch.
/// Caps memory/network cost from attacker-controlled input.
const MAX_PREFETCH_ACCOUNTS: usize = 64;
const MAX_PREFETCH_STORAGE: usize = 256;
/// Safety cap for user-provided access-list entries in request payloads.
const MAX_ACCESS_LIST_ENTRIES: usize = 1024;
/// Safety cap for total user-provided access-list storage-key items.
const MAX_ACCESS_LIST_STORAGE_KEYS: usize = 4096;

/// Parse a fixed-width hex value into an array (`0x` prefix optional).
fn parse_prefixed_hex_array<const N: usize>(raw: &str) -> Option<[u8; N]> {
    let hex = raw.strip_prefix("0x").unwrap_or(raw);
    if hex.len() != N * 2 || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    let mut out = [0u8; N];
    hex::decode_to_slice(hex, &mut out).ok()?;
    Some(out)
}

fn access_list_storage_keys(entry: &Value) -> Option<&[Value]> {
    entry
        .get("storageKeys")
        .or_else(|| entry.get("storage_keys"))
        .and_then(Value::as_array)
        .map(Vec::as_slice)
}

fn parse_access_list_strict(val: Option<&Value>) -> Result<TxAccessList, String> {
    let Some(v) = val else {
        return Ok(TxAccessList::default());
    };
    if v.is_null() {
        return Ok(TxAccessList::default());
    }

    let entries = v
        .as_array()
        .ok_or("expected accessList to be an array".to_string())?;
    if entries.len() > MAX_ACCESS_LIST_ENTRIES {
        return Err(format!(
            "accessList exceeds entry limit: got {}, max {}",
            entries.len(),
            MAX_ACCESS_LIST_ENTRIES
        ));
    }
    let mut items = Vec::with_capacity(entries.len());
    let mut total_storage_keys = 0usize;

    for (entry_idx, entry) in entries.iter().enumerate() {
        let address = entry
            .get("address")
            .and_then(Value::as_str)
            .and_then(parse_prefixed_hex_array::<20>)
            .map(|bytes| Address::from_slice(&bytes))
            .ok_or_else(|| {
                format!(
                    "accessList[{entry_idx}].address must be a 20-byte hex string (0x-prefixed optional)"
                )
            })?;

        let storage_vals = access_list_storage_keys(entry)
            .ok_or_else(|| format!("accessList[{entry_idx}].storageKeys must be an array"))?;
        total_storage_keys = total_storage_keys
            .checked_add(storage_vals.len())
            .ok_or_else(|| "accessList storage key count overflow".to_string())?;
        if total_storage_keys > MAX_ACCESS_LIST_STORAGE_KEYS {
            return Err(format!(
                "accessList exceeds storage key limit: got {}, max {}",
                total_storage_keys, MAX_ACCESS_LIST_STORAGE_KEYS
            ));
        }
        let mut storage_keys = Vec::with_capacity(storage_vals.len());

        for (key_idx, key) in storage_vals.iter().enumerate() {
            let key_str = key.as_str().ok_or_else(|| {
                format!("accessList[{entry_idx}].storageKeys[{key_idx}] must be a hex string")
            })?;
            let slot = parse_prefixed_hex_array::<32>(key_str).ok_or_else(|| {
                format!(
                    "accessList[{entry_idx}].storageKeys[{key_idx}] must be a 32-byte hex string (0x-prefixed optional)"
                )
            })?;
            storage_keys.push(B256::from(slot));
        }

        items.push(TxAccessListItem {
            address,
            storage_keys,
        });
    }

    Ok(TxAccessList(items))
}

/// Parse and batch-prefetch all accounts and storage slots from an EIP-2930 access list.
/// This populates the PirClient's internal cache so subsequent revm Database calls
/// hit warm cache instead of making individual network round-trips.
/// Deduplicates entries and enforces size caps to prevent abuse.
async fn prefetch_access_list(pir_client: &Arc<PirClient>, entries: &[Value]) {
    use std::collections::HashSet;

    let mut seen_addrs: HashSet<[u8; 20]> = HashSet::new();
    let mut addresses: Vec<[u8; 20]> = Vec::new();
    let mut seen_storage: HashSet<([u8; 20], [u8; 32])> = HashSet::new();
    let mut storage_queries: Vec<([u8; 20], [u8; 32])> = Vec::new();

    for entry in entries {
        if addresses.len() >= MAX_PREFETCH_ACCOUNTS && storage_queries.len() >= MAX_PREFETCH_STORAGE
        {
            break;
        }

        let addr = match entry
            .get("address")
            .and_then(|v| v.as_str())
            .and_then(parse_prefixed_hex_array::<20>)
        {
            Some(addr) => addr,
            None => continue,
        };

        if addresses.len() < MAX_PREFETCH_ACCOUNTS && seen_addrs.insert(addr) {
            addresses.push(addr);
        }

        if let Some(keys) = access_list_storage_keys(entry) {
            for key in keys {
                if storage_queries.len() >= MAX_PREFETCH_STORAGE {
                    break;
                }
                if let Some(slot) = key.as_str().and_then(parse_prefixed_hex_array::<32>) {
                    if seen_storage.insert((addr, slot)) {
                        storage_queries.push((addr, slot));
                    }
                }
            }
        }
    }

    // Fire both batch queries concurrently
    let accounts_fut = async {
        if !addresses.is_empty() {
            if let Err(e) = pir_client.query_accounts_batch(&addresses).await {
                info!("Access list account prefetch failed (non-fatal): {}", e);
            }
        }
    };

    let storage_fut = async {
        if !storage_queries.is_empty() {
            if let Err(e) = pir_client.query_storages_batch(&storage_queries).await {
                info!("Access list storage prefetch failed (non-fatal): {}", e);
            }
        }
    };

    tokio::join!(accounts_fut, storage_fut);
}

/// Seed access-list collector with user-provided EIP-2930 list.
/// This preserves explicit caller hints in the final eth_createAccessList response.
/// Returns true when input had to be truncated by safety limits.
fn seed_access_list_collector(call_params: &Value, collector: &SharedAccessListCollector) -> bool {
    use std::collections::HashSet;

    let Some(entries) = call_params.get("accessList").and_then(|v| v.as_array()) else {
        return false;
    };

    let mut guard = match collector.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };

    let mut truncated = entries.len() > MAX_ACCESS_LIST_ENTRIES;
    let mut seen_accounts: HashSet<[u8; 20]> = HashSet::new();
    let mut seen_storage: HashSet<([u8; 20], [u8; 32])> = HashSet::new();
    let mut scanned_storage_items = 0usize;

    for entry in entries.iter().take(MAX_ACCESS_LIST_ENTRIES) {
        let Some(addr) = entry
            .get("address")
            .and_then(|v| v.as_str())
            .and_then(parse_prefixed_hex_array::<20>)
        else {
            continue;
        };

        if seen_accounts.insert(addr) {
            guard.record_account(addr);
        }

        if let Some(keys) = access_list_storage_keys(entry) {
            for key in keys {
                if scanned_storage_items >= MAX_ACCESS_LIST_STORAGE_KEYS {
                    truncated = true;
                    break;
                }
                scanned_storage_items += 1;
                if let Some(slot) = key.as_str().and_then(parse_prefixed_hex_array::<32>) {
                    if seen_storage.insert((addr, slot)) {
                        guard.record_storage(addr, slot);
                    }
                }
            }
        }
    }

    truncated
}

/// Run EVM execution and return the raw ExecutionResult.
/// Sets basefee=0 for simulation so calls with gas_price=0 don't fail
/// (matches Geth's behavior for eth_call/eth_estimateGas).
async fn run_evm(
    pir_client: Arc<PirClient>,
    code_resolver: Arc<CodeResolver>,
    upstream_client: reqwest::Client,
    upstream_url: String,
    call_params: &Value,
    block_tag: &Value,
    access_list_collector: Option<SharedAccessListCollector>,
) -> Result<(ExecutionResult, CallParams, bool), EthCallError> {
    let parsed = parse_call_params(call_params)?;

    let validated_tag = validate_block_tag(block_tag).map_err(EthCallError::InvalidParams)?;

    let block_info = fetch_block_info(&upstream_client, &upstream_url, &validated_tag)
        .await
        .map_err(EthCallError::Internal)?;

    // Access-list prefetch: batch-fetch all listed accounts and storage slots
    // to populate the PirClient cache before EVM execution begins.
    if let Some(access_list) = call_params.get("accessList") {
        if let Some(entries) = access_list.as_array() {
            prefetch_access_list(&pir_client, entries).await;
        }
    }

    let from = parsed.from;
    let to = parsed.to;
    let data = parsed.data.clone();
    let value = parsed.value;
    let gas = parsed.gas;
    let access_list = parsed.access_list.clone();

    let handle = Handle::current();

    let is_cancun_or_later = block_info.is_cancun_or_later;
    let result = tokio::task::spawn_blocking(move || {
        let pir_db = match access_list_collector {
            Some(collector) => PirDatabase::new_with_access_list_collector(
                pir_client,
                code_resolver,
                upstream_client,
                upstream_url,
                handle,
                collector,
            ),
            None => PirDatabase::new(
                pir_client,
                code_resolver,
                upstream_client,
                upstream_url,
                handle,
            ),
        };

        let tx = TxEnv::builder()
            .caller(from)
            .kind(match to {
                Some(addr) => TxKind::Call(addr),
                None => TxKind::Create,
            })
            .data(data)
            .value(value)
            .gas_limit(gas)
            .access_list(access_list)
            .build()
            .map_err(|e| EthCallError::Internal(format!("Failed to build tx: {:?}", e)))?;

        let mut evm = Context::mainnet()
            .with_db(pir_db)
            .modify_block_chained(|b| {
                b.number = U256::from(block_info.number);
                b.timestamp = U256::from(block_info.timestamp);
                b.gas_limit = block_info.gas_limit;
                // Set basefee=0 for simulation so calls with gas_price=0 don't fail
                // with GasPriceLessThanBasefee. Matches Geth's eth_call behavior.
                b.basefee = 0;
            })
            .build_mainnet();

        let result = evm
            .transact(tx)
            .map_err(|e| EthCallError::Internal(format!("EVM execution failed: {:?}", e)))?;

        Ok(result.result)
    })
    .await
    .map_err(|e| EthCallError::Internal(format!("spawn_blocking: {}", e)))??;

    Ok((result, parsed, is_cancun_or_later))
}

/// Execute an eth_call privately using revm with PIR-backed state.
/// All account, storage, and code lookups go through PIR so the
/// server never learns which addresses/slots are being accessed.
pub async fn execute_eth_call(
    pir_client: Arc<PirClient>,
    code_resolver: Arc<CodeResolver>,
    upstream_client: reqwest::Client,
    upstream_url: String,
    call_params: &Value,
    block_tag: &Value,
) -> Result<Bytes, EthCallError> {
    info!("Executing private eth_call via revm");

    let (result, _, _) = run_evm(
        pir_client,
        code_resolver,
        upstream_client,
        upstream_url,
        call_params,
        block_tag,
        None,
    )
    .await?;

    match result {
        ExecutionResult::Success { output, .. } => match output {
            Output::Call(data) => Ok(data),
            Output::Create(data, _) => Ok(data),
        },
        ExecutionResult::Revert { output, .. } => Err(EthCallError::Internal(format!(
            "execution reverted: 0x{}",
            hex::encode(&output)
        ))),
        ExecutionResult::Halt { reason, .. } => Err(EthCallError::Internal(format!(
            "execution halted: {:?}",
            reason
        ))),
    }
}

/// Execute an eth_estimateGas privately using revm with PIR-backed state.
/// Returns gas_used with a 20% safety margin (standard practice, matches Geth).
pub async fn execute_eth_estimate_gas(
    pir_client: Arc<PirClient>,
    code_resolver: Arc<CodeResolver>,
    upstream_client: reqwest::Client,
    upstream_url: String,
    call_params: &Value,
    block_tag: &Value,
) -> Result<u64, EthCallError> {
    info!("Executing private eth_estimateGas via revm");

    let (result, _, _) = run_evm(
        pir_client,
        code_resolver,
        upstream_client,
        upstream_url,
        call_params,
        block_tag,
        None,
    )
    .await?;

    match result {
        ExecutionResult::Success { gas_used, .. } => {
            // 20% safety margin, standard practice
            Ok(gas_used * 120 / 100)
        }
        ExecutionResult::Revert { output, .. } => Err(EthCallError::Internal(format!(
            "execution reverted: 0x{}",
            hex::encode(&output)
        ))),
        ExecutionResult::Halt { reason, .. } => Err(EthCallError::Internal(format!(
            "execution halted: {:?}",
            reason
        ))),
    }
}

fn warm_addresses_for_call(call: &CallParams, is_cancun_or_later: bool) -> Vec<[u8; 20]> {
    let mut warmed = Vec::with_capacity(12);
    warmed.push(call.from.into_array());
    if let Some(to) = call.to {
        warmed.push(to.into_array());
    }
    // Precompiles 0x1..0x9 are always warm (EIP-2929); 0x0a is warm from Cancun.
    let max_precompile = if is_cancun_or_later { 10 } else { 9 };
    for idx in 1u8..=max_precompile {
        let mut addr = [0u8; 20];
        addr[19] = idx;
        warmed.push(addr);
    }
    warmed.sort_unstable();
    warmed.dedup();
    warmed
}

fn gas_used_from_result(result: &ExecutionResult) -> u64 {
    match result {
        ExecutionResult::Success { gas_used, .. } => *gas_used,
        ExecutionResult::Revert { gas_used, .. } => *gas_used,
        ExecutionResult::Halt { gas_used, .. } => *gas_used,
    }
}

fn collected_access_list(
    collector: &SharedAccessListCollector,
    call: &CallParams,
    is_cancun_or_later: bool,
) -> Value {
    let mut guard = match collector.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    for addr in warm_addresses_for_call(call, is_cancun_or_later) {
        guard.remove_account_if_empty(&addr);
    }
    guard.to_rpc_entries()
}

fn with_access_list_override(call_params: &Value, access_list: Value) -> Option<Value> {
    let mut object = call_params.as_object()?.clone();
    object.insert("accessList".to_string(), access_list);
    Some(Value::Object(object))
}

fn create_access_list_result_from_execution(
    result: &ExecutionResult,
    access_list: Value,
    gas_used: u64,
) -> Value {
    let mut object = serde_json::Map::new();
    object.insert("accessList".to_string(), access_list);
    object.insert(
        "gasUsed".to_string(),
        Value::String(format!("0x{:x}", gas_used)),
    );

    let maybe_error = match result {
        ExecutionResult::Success { .. } => None,
        ExecutionResult::Revert { output, .. } => {
            Some(format!("execution reverted: 0x{}", hex::encode(output)))
        }
        ExecutionResult::Halt { reason, .. } => Some(format!("execution halted: {:?}", reason)),
    };

    if let Some(error) = maybe_error {
        object.insert("error".to_string(), Value::String(error));
    }

    Value::Object(object)
}

/// Execute eth_createAccessList privately using local EVM execution.
/// Returns a JSON-RPC shaped object containing `accessList` and `gasUsed`.
pub async fn execute_eth_create_access_list(
    pir_client: Arc<PirClient>,
    code_resolver: Arc<CodeResolver>,
    upstream_client: reqwest::Client,
    upstream_url: String,
    call_params: &Value,
    block_tag: &Value,
) -> Result<Value, EthCallError> {
    info!("Executing private eth_createAccessList via revm");

    let collector = Arc::new(Mutex::new(AccessListCollector::default()));
    let seed_truncated = seed_access_list_collector(call_params, &collector);
    if seed_truncated {
        return Err(EthCallError::InvalidParams(format!(
            "accessList exceeds safety limits (max entries={}, max storage items={})",
            MAX_ACCESS_LIST_ENTRIES, MAX_ACCESS_LIST_STORAGE_KEYS
        )));
    }

    let (result, parsed_call, is_cancun_or_later) = run_evm(
        Arc::clone(&pir_client),
        Arc::clone(&code_resolver),
        upstream_client.clone(),
        upstream_url.clone(),
        call_params,
        block_tag,
        Some(Arc::clone(&collector)),
    )
    .await?;

    let access_list = collected_access_list(&collector, &parsed_call, is_cancun_or_later);
    let mut gas_used = gas_used_from_result(&result);
    if let Some(replay_call_params) = with_access_list_override(call_params, access_list.clone()) {
        if let Ok((replayed_result, _, _)) = run_evm(
            Arc::clone(&pir_client),
            Arc::clone(&code_resolver),
            upstream_client.clone(),
            upstream_url.clone(),
            &replay_call_params,
            block_tag,
            None,
        )
        .await
        {
            gas_used = gas_used_from_result(&replayed_result);
        } else {
            warn!("eth_createAccessList gasUsed replay failed; using initial execution gasUsed");
        }
    }

    Ok(create_access_list_result_from_execution(
        &result,
        access_list,
        gas_used,
    ))
}

/// Parse an address field. Returns Ok(None) if absent, Err if present but invalid.
fn parse_address_strict(val: Option<&Value>) -> Result<Option<Address>, String> {
    let Some(v) = val else { return Ok(None) };
    if v.is_null() {
        return Ok(None);
    }
    let s = v.as_str().ok_or("expected hex string")?;
    let hex = s.strip_prefix("0x").unwrap_or(s);
    let bytes = hex::decode(hex).map_err(|e| e.to_string())?;
    if bytes.len() != 20 {
        return Err(format!("expected 20 bytes, got {}", bytes.len()));
    }
    Ok(Some(Address::from_slice(&bytes)))
}

/// Parse a bytes field. Returns Ok(empty) if absent, Err if present but invalid hex.
fn parse_bytes_strict(val: Option<&Value>) -> Result<Bytes, String> {
    let Some(v) = val else {
        return Ok(Bytes::new());
    };
    if v.is_null() {
        return Ok(Bytes::new());
    }
    let s = v.as_str().ok_or("expected hex string")?;
    let hex = s.strip_prefix("0x").unwrap_or(s);
    if hex.is_empty() {
        return Ok(Bytes::new());
    }
    hex::decode(hex).map(Bytes::from).map_err(|e| e.to_string())
}

/// Parse a U256 field. Returns Ok(ZERO) if absent, Err if present but invalid.
fn parse_u256_strict(val: Option<&Value>) -> Result<U256, String> {
    let Some(v) = val else {
        return Ok(U256::ZERO);
    };
    if v.is_null() {
        return Ok(U256::ZERO);
    }
    let s = v.as_str().ok_or("expected hex string")?;
    let hex = s.strip_prefix("0x").unwrap_or(s);
    if hex.is_empty() {
        return Ok(U256::ZERO);
    }
    U256::from_str_radix(hex, 16).map_err(|e| e.to_string())
}

/// Parse a u64 field. Returns Ok(None) if absent, Err if present but invalid.
/// Follows Ethereum JSON-RPC hex quantity rules: requires "0x" prefix, "0x0" for zero.
fn parse_u64_strict(val: Option<&Value>) -> Result<Option<u64>, String> {
    let Some(v) = val else { return Ok(None) };
    if v.is_null() {
        return Ok(None);
    }
    let s = v.as_str().ok_or("expected hex string")?;
    let hex = s
        .strip_prefix("0x")
        .ok_or_else(|| format!("expected 0x-prefixed hex quantity, got '{}'", s))?;
    if hex.is_empty() {
        return Err("invalid hex quantity: no digits after prefix".to_string());
    }
    // Reject leading zeros per JSON-RPC quantity encoding (except "0x0")
    if hex.len() > 1 && hex.starts_with('0') {
        return Err(format!(
            "invalid hex quantity '{}': leading zeros not allowed",
            s
        ));
    }
    u64::from_str_radix(hex, 16)
        .map(Some)
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use revm::context_interface::result::{HaltReason, SuccessReason};
    use serde_json::json;

    #[test]
    fn parse_address_with_0x_prefix() {
        let val = json!("0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045");
        let addr = parse_address_strict(Some(&val)).unwrap().unwrap();
        let expected: [u8; 20] = hex::decode("d8dA6BF26964aF9D7eEd9e03E53415D37aA96045")
            .unwrap()
            .try_into()
            .unwrap();
        assert_eq!(addr.into_array(), expected);
    }

    #[test]
    fn parse_address_without_prefix() {
        let val = json!("d8dA6BF26964aF9D7eEd9e03E53415D37aA96045");
        assert!(parse_address_strict(Some(&val)).unwrap().is_some());
    }

    #[test]
    fn parse_address_none_on_absent() {
        assert_eq!(parse_address_strict(None).unwrap(), None);
        assert_eq!(parse_address_strict(Some(&json!(null))).unwrap(), None);
    }

    #[test]
    fn parse_address_err_on_invalid() {
        assert!(parse_address_strict(Some(&json!("0xinvalid"))).is_err());
        assert!(parse_address_strict(Some(&json!("0xaabb"))).is_err()); // too short
        assert!(parse_address_strict(Some(&json!(42))).is_err()); // not a string
    }

    #[test]
    fn parse_bytes_with_data() {
        let val = json!("0x60806040");
        let bytes = parse_bytes_strict(Some(&val)).unwrap();
        assert_eq!(bytes.as_ref(), &[0x60, 0x80, 0x60, 0x40]);
    }

    #[test]
    fn parse_bytes_empty_on_none() {
        assert!(parse_bytes_strict(None).unwrap().is_empty());
        assert!(parse_bytes_strict(Some(&json!(null))).unwrap().is_empty());
    }

    #[test]
    fn parse_bytes_err_on_invalid_hex() {
        assert!(parse_bytes_strict(Some(&json!("0xZZZZ"))).is_err());
    }

    #[test]
    fn parse_u256_hex_value() {
        let val = json!("0xde0b6b3a7640000"); // 1 ETH in wei
        let result = parse_u256_strict(Some(&val)).unwrap();
        assert_eq!(result, U256::from(1_000_000_000_000_000_000u64));
    }

    #[test]
    fn parse_u256_zero_on_none() {
        assert_eq!(parse_u256_strict(None).unwrap(), U256::ZERO);
    }

    #[test]
    fn parse_u256_err_on_invalid() {
        assert!(parse_u256_strict(Some(&json!("not_hex"))).is_err());
    }

    #[test]
    fn parse_u64_gas_value() {
        let val = json!("0x1c9c380"); // 30_000_000
        assert_eq!(parse_u64_strict(Some(&val)).unwrap(), Some(30_000_000));
    }

    #[test]
    fn parse_u64_none_on_missing() {
        assert_eq!(parse_u64_strict(None).unwrap(), None);
    }

    #[test]
    fn parse_u64_err_on_invalid() {
        assert!(parse_u64_strict(Some(&json!("xyz"))).is_err());
    }

    #[test]
    fn parse_u64_bare_0x_is_error() {
        assert!(parse_u64_strict(Some(&json!("0x"))).is_err());
    }

    #[test]
    fn parse_u64_0x0_is_zero() {
        assert_eq!(parse_u64_strict(Some(&json!("0x0"))).unwrap(), Some(0));
    }

    #[test]
    fn validate_block_tag_named_tags() {
        for tag in &["latest", "earliest", "pending", "safe", "finalized"] {
            assert_eq!(validate_block_tag(&json!(tag)).unwrap(), *tag);
        }
    }

    #[test]
    fn validate_block_tag_hex_quantity() {
        assert_eq!(validate_block_tag(&json!("0x1")).unwrap(), "0x1");
        assert_eq!(validate_block_tag(&json!("0xff")).unwrap(), "0xff");
    }

    #[test]
    fn validate_block_tag_null_defaults_latest() {
        assert_eq!(validate_block_tag(&json!(null)).unwrap(), "latest");
    }

    #[test]
    fn validate_block_tag_rejects_object() {
        assert!(validate_block_tag(&json!({"blockNumber": "0x1"})).is_err());
    }

    #[test]
    fn validate_block_tag_rejects_invalid_string() {
        assert!(validate_block_tag(&json!("foo")).is_err());
        assert!(validate_block_tag(&json!("0x")).is_err()); // no digits
        assert!(validate_block_tag(&json!("0xZZ")).is_err()); // not hex
    }

    #[test]
    fn validate_block_tag_rejects_non_string_types() {
        assert!(validate_block_tag(&json!(1)).is_err());
        assert!(validate_block_tag(&json!(true)).is_err());
        assert!(validate_block_tag(&json!([1, 2])).is_err());
    }

    #[test]
    fn infer_is_cancun_or_later_detects_blob_fields() {
        let block = json!({
            "blobGasUsed": "0x0"
        });
        assert!(infer_is_cancun_or_later(&block, 10, None));
    }

    #[test]
    fn infer_is_cancun_or_later_is_false_without_blob_fields_or_fallback() {
        let block = json!({
            "blobGasUsed": null,
            "excessBlobGas": null
        });
        assert!(!infer_is_cancun_or_later(&block, 10, None));
    }

    #[test]
    fn infer_is_cancun_or_later_uses_fallback_schedule() {
        let block = json!({});
        assert!(!infer_is_cancun_or_later(&block, 99, Some(100)));
        assert!(infer_is_cancun_or_later(&block, 100, Some(100)));
    }

    #[test]
    fn parse_u64_env_accepts_decimal_and_hex() {
        assert_eq!(parse_u64_env("TEST_VAR", "42"), Some(42));
        assert_eq!(parse_u64_env("TEST_VAR", "0x2a"), Some(42));
    }

    #[test]
    fn parse_u64_env_rejects_invalid_values() {
        assert_eq!(parse_u64_env("TEST_VAR", "not-a-number"), None);
        assert_eq!(parse_u64_env("TEST_VAR", "0xzz"), None);
    }

    #[test]
    fn parse_u64_rejects_no_prefix() {
        assert!(parse_u64_strict(Some(&json!("10"))).is_err());
    }

    #[test]
    fn parse_u64_rejects_leading_zeros() {
        assert!(parse_u64_strict(Some(&json!("0x00"))).is_err());
        assert!(parse_u64_strict(Some(&json!("0x01"))).is_err());
    }

    #[test]
    fn parse_access_list_strict_parses_entries() {
        let address = format!("0x{}", "11".repeat(20));
        let slot = format!("0x{}", "22".repeat(32));
        let input = json!([
            {
                "address": address,
                "storageKeys": [slot]
            }
        ]);

        let parsed = parse_access_list_strict(Some(&input)).expect("access list parses");
        assert_eq!(parsed.0.len(), 1);
        assert_eq!(parsed.0[0].storage_keys.len(), 1);
    }

    #[test]
    fn parse_access_list_strict_rejects_invalid_slot_width() {
        let address = format!("0x{}", "11".repeat(20));
        let input = json!([
            {
                "address": address,
                "storageKeys": ["0x1234"]
            }
        ]);

        assert!(parse_access_list_strict(Some(&input)).is_err());
    }

    #[test]
    fn parse_access_list_strict_rejects_too_many_entries() {
        let address = format!("0x{}", "11".repeat(20));
        let entries: Vec<Value> = (0..(MAX_ACCESS_LIST_ENTRIES + 1))
            .map(|_| {
                json!({
                    "address": address,
                    "storageKeys": []
                })
            })
            .collect();
        let input = Value::Array(entries);

        assert!(parse_access_list_strict(Some(&input)).is_err());
    }

    #[test]
    fn parse_access_list_strict_rejects_too_many_storage_keys() {
        let address = format!("0x{}", "11".repeat(20));
        let storage_keys: Vec<Value> = (0..(MAX_ACCESS_LIST_STORAGE_KEYS + 1))
            .map(|i| {
                let mut slot = [0u8; 32];
                slot[24..32].copy_from_slice(&(i as u64).to_be_bytes());
                Value::String(format!("0x{}", hex::encode(slot)))
            })
            .collect();
        let input = json!([
            {
                "address": address,
                "storageKeys": storage_keys
            }
        ]);

        assert!(parse_access_list_strict(Some(&input)).is_err());
    }

    #[test]
    fn seed_access_list_collector_dedups_and_validates_lengths() {
        let collector = Arc::new(Mutex::new(AccessListCollector::default()));
        let address = format!("0x{}", "11".repeat(20));
        let slot = format!("0x{}", "22".repeat(32));
        let params = json!({
            "accessList": [
                {
                    "address": address,
                    "storageKeys": [slot, slot, "0x1234", "0xzz"]
                },
                {
                    "address": "0x1234",
                    "storageKeys": ["0x4444"]
                }
            ]
        });

        assert!(!seed_access_list_collector(&params, &collector));
        let entries = collector.lock().unwrap().to_rpc_entries();

        assert_eq!(
            entries,
            json!([{
                "address": format!("0x{}", "11".repeat(20)),
                "storageKeys": [format!("0x{}", "22".repeat(32))]
            }])
        );
    }

    #[test]
    fn seed_access_list_collector_reports_truncation_when_entry_cap_hit() {
        let collector = Arc::new(Mutex::new(AccessListCollector::default()));
        let address = format!("0x{}", "11".repeat(20));
        let entries: Vec<Value> = (0..(MAX_ACCESS_LIST_ENTRIES + 1))
            .map(|_| {
                json!({
                    "address": address,
                    "storageKeys": []
                })
            })
            .collect();
        let params = json!({ "accessList": entries });

        assert!(seed_access_list_collector(&params, &collector));
        let seeded = collector.lock().unwrap().to_rpc_entries();
        assert_eq!(seeded.as_array().unwrap().len(), 1);
    }

    #[test]
    fn seed_access_list_collector_reports_truncation_when_storage_cap_hit() {
        let collector = Arc::new(Mutex::new(AccessListCollector::default()));
        let address = format!("0x{}", "11".repeat(20));
        let storage_keys: Vec<Value> = (0..(MAX_ACCESS_LIST_STORAGE_KEYS + 1))
            .map(|i| {
                let mut slot = [0u8; 32];
                slot[24..32].copy_from_slice(&(i as u64).to_be_bytes());
                Value::String(format!("0x{}", hex::encode(slot)))
            })
            .collect();
        let params = json!({
            "accessList": [{
                "address": address,
                "storageKeys": storage_keys
            }]
        });

        assert!(seed_access_list_collector(&params, &collector));
        let seeded = collector.lock().unwrap().to_rpc_entries();
        assert_eq!(seeded.as_array().unwrap().len(), 1);
        let seeded_keys = seeded
            .as_array()
            .and_then(|entries| entries.first())
            .and_then(|entry| entry.get("storageKeys"))
            .and_then(Value::as_array)
            .expect("seeded entry has storageKeys array");
        assert_eq!(seeded_keys.len(), MAX_ACCESS_LIST_STORAGE_KEYS);
    }

    #[test]
    fn seed_access_list_collector_supports_storage_keys_alias() {
        let collector = Arc::new(Mutex::new(AccessListCollector::default()));
        let address = format!("0x{}", "11".repeat(20));
        let slot = format!("0x{}", "22".repeat(32));
        let params = json!({
            "accessList": [{
                "address": address,
                "storage_keys": [slot]
            }]
        });

        assert!(!seed_access_list_collector(&params, &collector));
        let entries = collector.lock().unwrap().to_rpc_entries();
        assert_eq!(
            entries,
            json!([{
                "address": format!("0x{}", "11".repeat(20)),
                "storageKeys": [format!("0x{}", "22".repeat(32))]
            }])
        );
    }

    #[test]
    fn seed_access_list_collector_truncates_invalid_heavy_storage_keys() {
        let collector = Arc::new(Mutex::new(AccessListCollector::default()));
        let address = format!("0x{}", "11".repeat(20));
        let storage_keys: Vec<Value> = (0..(MAX_ACCESS_LIST_STORAGE_KEYS + 1))
            .map(|_| Value::String("0x1234".to_string()))
            .collect();
        let params = json!({
            "accessList": [{
                "address": address,
                "storageKeys": storage_keys
            }]
        });

        assert!(seed_access_list_collector(&params, &collector));
        let entries = collector.lock().unwrap().to_rpc_entries();
        assert_eq!(
            entries,
            json!([{
                "address": format!("0x{}", "11".repeat(20)),
                "storageKeys": []
            }])
        );
    }

    #[test]
    fn create_access_list_result_success_excludes_warm_accounts() {
        let from = [0xaa_u8; 20];
        let to = [0xbb_u8; 20];
        let keep = [0xcc_u8; 20];
        let mut precompile = [0_u8; 20];
        precompile[19] = 1;
        let mut precompile_ten = [0_u8; 20];
        precompile_ten[19] = 10;
        let keep_slot = [0x33_u8; 32];
        let warm_slot = [0x44_u8; 32];
        let collector = Arc::new(Mutex::new(AccessListCollector::default()));
        {
            let mut guard = collector.lock().unwrap();
            guard.record_account(from);
            guard.record_account(to);
            guard.record_account(precompile);
            guard.record_account(precompile_ten);
            guard.record_storage(from, warm_slot);
            guard.record_storage(keep, keep_slot);
        }
        let call = CallParams {
            from: Address::from_slice(&from),
            to: Some(Address::from_slice(&to)),
            data: Bytes::new(),
            value: U256::ZERO,
            gas: 21_000,
            access_list: TxAccessList::default(),
        };
        let result = ExecutionResult::Success {
            reason: SuccessReason::Return,
            gas_used: 21_000,
            gas_refunded: 0,
            logs: vec![],
            output: Output::Call(Bytes::new()),
        };

        let access_list = collected_access_list(&collector, &call, true);
        let response = create_access_list_result_from_execution(&result, access_list, 21_000);
        let object = response.as_object().unwrap();
        assert_eq!(object.get("gasUsed"), Some(&Value::String("0x5208".into())));
        assert!(object.get("error").is_none());
        assert_eq!(
            object.get("accessList"),
            Some(&json!([
                {
                    "address": format!("0x{}", hex::encode(from)),
                    "storageKeys": [format!("0x{}", hex::encode(warm_slot))]
                },
                {
                    "address": format!("0x{}", hex::encode(keep)),
                    "storageKeys": [format!("0x{}", hex::encode(keep_slot))]
                }
            ]))
        );
    }

    #[test]
    fn create_access_list_result_keeps_precompile_ten_before_cancun() {
        let from = [0xaa_u8; 20];
        let mut precompile_ten = [0_u8; 20];
        precompile_ten[19] = 10;
        let collector = Arc::new(Mutex::new(AccessListCollector::default()));
        {
            let mut guard = collector.lock().unwrap();
            guard.record_account(precompile_ten);
        }
        let call = CallParams {
            from: Address::from_slice(&from),
            to: None,
            data: Bytes::new(),
            value: U256::ZERO,
            gas: 21_000,
            access_list: TxAccessList::default(),
        };

        let access_list = collected_access_list(&collector, &call, false);
        assert_eq!(
            access_list,
            json!([{
                "address": format!("0x{}", hex::encode(precompile_ten)),
                "storageKeys": []
            }])
        );
    }

    #[test]
    fn create_access_list_result_includes_error_for_revert_and_halt() {
        let access_list = json!([]);

        let revert = ExecutionResult::Revert {
            gas_used: 123,
            output: Bytes::from(vec![0xde, 0xad]),
        };
        let revert_response =
            create_access_list_result_from_execution(&revert, access_list.clone(), 123);
        let revert_obj = revert_response.as_object().unwrap();
        assert_eq!(
            revert_obj.get("gasUsed"),
            Some(&Value::String("0x7b".into()))
        );
        assert_eq!(
            revert_obj.get("error"),
            Some(&Value::String("execution reverted: 0xdead".into()))
        );

        let halt = ExecutionResult::Halt {
            reason: HaltReason::CallTooDeep,
            gas_used: 456,
        };
        let halt_response = create_access_list_result_from_execution(&halt, access_list, 456);
        let halt_obj = halt_response.as_object().unwrap();
        assert_eq!(
            halt_obj.get("gasUsed"),
            Some(&Value::String("0x1c8".into()))
        );
        let halt_error = halt_obj
            .get("error")
            .and_then(Value::as_str)
            .expect("halt error present");
        assert!(halt_error.contains("execution halted: CallTooDeep"));
    }
}
