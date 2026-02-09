use crate::code_resolver::CodeResolver;
use crate::pir_db::PirDatabase;
use alloy_primitives::{Address, Bytes, U256};
use morphogen_client::network::PirClient;
use revm::{
    context::TxEnv,
    context_interface::result::{ExecutionResult, Output},
    primitives::TxKind,
    Context, ExecuteEvm, MainBuilder, MainContext,
};
use serde_json::Value;
use std::sync::Arc;
use tokio::runtime::Handle;
use tracing::info;

/// Execute an eth_call privately using revm with PIR-backed state.
/// All account, storage, and code lookups go through PIR so the
/// server never learns which addresses/slots are being accessed.
pub async fn execute_eth_call(
    pir_client: Arc<PirClient>,
    code_resolver: Arc<CodeResolver>,
    upstream_client: reqwest::Client,
    upstream_url: String,
    call_params: &Value,
) -> Result<Bytes, String> {
    // Parse call parameters before moving into blocking task
    let from = parse_address(call_params.get("from")).unwrap_or_default();
    let to = parse_address(call_params.get("to"));
    let data = parse_bytes(call_params.get("data").or(call_params.get("input")));
    let value = parse_u256(call_params.get("value"));
    let gas = parse_u64(call_params.get("gas")).unwrap_or(30_000_000);

    info!(
        %from,
        to = ?to,
        data_len = data.len(),
        "Executing private eth_call via revm"
    );

    let handle = Handle::current();

    tokio::task::spawn_blocking(move || {
        let pir_db = PirDatabase::new(
            pir_client,
            code_resolver,
            upstream_client,
            upstream_url,
            handle,
        );

        let tx = TxEnv::builder()
            .caller(from)
            .kind(match to {
                Some(addr) => TxKind::Call(addr),
                None => TxKind::Create,
            })
            .data(data)
            .value(value)
            .gas_limit(gas)
            .build()
            .map_err(|e| format!("Failed to build tx: {:?}", e))?;

        let mut evm = Context::mainnet()
            .with_db(pir_db)
            .modify_block_chained(|b| {
                b.basefee = 0; // eth_call doesn't charge fees
            })
            .build_mainnet();

        let result = evm
            .transact(tx)
            .map_err(|e| format!("EVM execution failed: {:?}", e))?;

        match result.result {
            ExecutionResult::Success { output, .. } => match output {
                Output::Call(data) => Ok(data),
                Output::Create(data, _) => Ok(data),
            },
            ExecutionResult::Revert { output, .. } => {
                Err(format!("execution reverted: 0x{}", hex::encode(&output)))
            }
            ExecutionResult::Halt { reason, .. } => Err(format!("execution halted: {:?}", reason)),
        }
    })
    .await
    .map_err(|e| format!("spawn_blocking: {}", e))?
}

fn parse_address(val: Option<&Value>) -> Option<Address> {
    let s = val?.as_str()?;
    let hex = s.strip_prefix("0x").unwrap_or(s);
    let bytes = hex::decode(hex).ok()?;
    if bytes.len() != 20 {
        return None;
    }
    Some(Address::from_slice(&bytes))
}

fn parse_bytes(val: Option<&Value>) -> Bytes {
    let Some(s) = val.and_then(|v| v.as_str()) else {
        return Bytes::new();
    };
    let hex = s.strip_prefix("0x").unwrap_or(s);
    hex::decode(hex).map(Bytes::from).unwrap_or_default()
}

fn parse_u256(val: Option<&Value>) -> U256 {
    let Some(s) = val.and_then(|v| v.as_str()) else {
        return U256::ZERO;
    };
    let hex = s.strip_prefix("0x").unwrap_or(s);
    U256::from_str_radix(hex, 16).unwrap_or(U256::ZERO)
}

fn parse_u64(val: Option<&Value>) -> Option<u64> {
    let s = val?.as_str()?;
    let hex = s.strip_prefix("0x").unwrap_or(s);
    u64::from_str_radix(hex, 16).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_address_with_0x_prefix() {
        let val = json!("0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045");
        let addr = parse_address(Some(&val)).unwrap();
        assert_eq!(
            format!("{:?}", addr).to_lowercase(),
            "0xd8da6bf26964af9d7eed9e03e53415d37aa96045"
        );
    }

    #[test]
    fn parse_address_without_prefix() {
        let val = json!("d8dA6BF26964aF9D7eEd9e03E53415D37aA96045");
        assert!(parse_address(Some(&val)).is_some());
    }

    #[test]
    fn parse_address_none_on_invalid() {
        assert!(parse_address(None).is_none());
        assert!(parse_address(Some(&json!("0xinvalid"))).is_none());
        assert!(parse_address(Some(&json!("0xaabb"))).is_none()); // too short
        assert!(parse_address(Some(&json!(42))).is_none()); // not a string
    }

    #[test]
    fn parse_bytes_with_data() {
        let val = json!("0x60806040");
        let bytes = parse_bytes(Some(&val));
        assert_eq!(bytes.as_ref(), &[0x60, 0x80, 0x60, 0x40]);
    }

    #[test]
    fn parse_bytes_empty_on_none() {
        assert!(parse_bytes(None).is_empty());
        assert!(parse_bytes(Some(&json!(null))).is_empty());
    }

    #[test]
    fn parse_u256_hex_value() {
        let val = json!("0xde0b6b3a7640000"); // 1 ETH in wei
        let result = parse_u256(Some(&val));
        assert_eq!(result, U256::from(1_000_000_000_000_000_000u64));
    }

    #[test]
    fn parse_u256_zero_on_none() {
        assert_eq!(parse_u256(None), U256::ZERO);
    }

    #[test]
    fn parse_u64_gas_value() {
        let val = json!("0x1c9c380"); // 30_000_000
        assert_eq!(parse_u64(Some(&val)), Some(30_000_000));
    }

    #[test]
    fn parse_u64_none_on_missing() {
        assert_eq!(parse_u64(None), None);
    }
}
