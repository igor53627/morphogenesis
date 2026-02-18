use async_trait::async_trait;
use morphogen_client::network::PirClient;
use morphogen_client::CodeResolver;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;
use wasm_bindgen::prelude::*;

const SAFE_READ_ONLY_PASSTHROUGH_METHODS: &[&str] = &[
    "eth_chainId",
    "eth_blockNumber",
    "eth_gasPrice",
    "eth_call",
    "eth_estimateGas",
    "eth_feeHistory",
    "eth_maxPriorityFeePerGas",
    "eth_blobBaseFee",
    "eth_syncing",
    "eth_coinbase",
    "eth_mining",
    "eth_hashrate",
    "eth_getBlockByHash",
    "eth_getBlockByNumber",
    "eth_getBlockTransactionCountByHash",
    "eth_getBlockTransactionCountByNumber",
    "eth_getUncleByBlockHashAndIndex",
    "eth_getUncleByBlockNumberAndIndex",
    "eth_getUncleCountByBlockHash",
    "eth_getUncleCountByBlockNumber",
    "eth_getTransactionByHash",
    "eth_getTransactionByBlockHashAndIndex",
    "eth_getTransactionByBlockNumberAndIndex",
    "eth_getTransactionReceipt",
    "eth_getLogs",
    "net_version",
    "net_listening",
    "net_peerCount",
    "web3_clientVersion",
    "web3_sha3",
];

const UNSAFE_METHODS: &[&str] = &[
    "eth_sendRawTransaction",
    "eth_sendTransaction",
    "eth_sign",
    "eth_signTransaction",
    "personal_sign",
    "eth_accounts",
    "eth_requestAccounts",
    "eth_subscribe",
    "eth_unsubscribe",
    "eth_newFilter",
    "eth_newBlockFilter",
    "eth_newPendingTransactionFilter",
    "eth_uninstallFilter",
    "eth_getFilterChanges",
    "eth_getFilterLogs",
    "eth_submitWork",
    "eth_submitHashrate",
];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GatewayError(JsonRpcError);

impl GatewayError {
    pub fn invalid_params(message: impl Into<String>) -> Self {
        Self(JsonRpcError {
            code: -32602,
            message: message.into(),
            data: None,
        })
    }

    pub fn method_not_found(method: &str) -> Self {
        Self(JsonRpcError {
            code: -32601,
            message: format!("Unsupported method: {method}"),
            data: None,
        })
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self(JsonRpcError {
            code: -32000,
            message: message.into(),
            data: None,
        })
    }

    pub fn json_rpc(code: i64, message: impl Into<String>, data: Option<Value>) -> Self {
        Self(JsonRpcError {
            code,
            message: message.into(),
            data,
        })
    }

    pub fn into_json(self) -> JsonRpcError {
        self.0
    }
}

#[async_trait(?Send)]
pub trait PrivateApi {
    async fn get_balance(&self, address: [u8; 20]) -> Result<u128, GatewayError>;
    async fn get_transaction_count(&self, address: [u8; 20]) -> Result<u64, GatewayError>;
    async fn get_storage_at(
        &self,
        address: [u8; 20],
        slot: [u8; 32],
    ) -> Result<[u8; 32], GatewayError>;
    async fn get_code(&self, address: [u8; 20]) -> Result<Vec<u8>, GatewayError>;
}

#[async_trait(?Send)]
pub trait UpstreamApi {
    async fn request(&self, method: &str, params: Value) -> Result<Value, GatewayError>;
}

pub struct GatewayCore<P, U> {
    private_api: P,
    upstream_api: U,
}

impl<P, U> GatewayCore<P, U> {
    pub fn new(private_api: P, upstream_api: U) -> Self {
        Self {
            private_api,
            upstream_api,
        }
    }
}

impl<P, U> GatewayCore<P, U>
where
    P: PrivateApi,
    U: UpstreamApi,
{
    pub async fn request_json(&self, method: &str, params: Value) -> Result<Value, JsonRpcError> {
        self.request_inner(method, params)
            .await
            .map_err(GatewayError::into_json)
    }

    async fn request_inner(&self, method: &str, params: Value) -> Result<Value, GatewayError> {
        match method {
            "eth_getBalance" => {
                let args = expect_params_array(method, &params, 1)?;
                let address = parse_address(&args[0])?;
                validate_private_block_param(method, args.get(1))?;
                let balance = self.private_api.get_balance(address).await?;
                Ok(Value::String(format!("0x{:x}", balance)))
            }
            "eth_getTransactionCount" => {
                let args = expect_params_array(method, &params, 1)?;
                let address = parse_address(&args[0])?;
                validate_private_block_param(method, args.get(1))?;
                let nonce = self.private_api.get_transaction_count(address).await?;
                Ok(Value::String(format!("0x{:x}", nonce)))
            }
            "eth_getStorageAt" => {
                let args = expect_params_array(method, &params, 2)?;
                let address = parse_address(&args[0])?;
                let slot = parse_storage_slot(&args[1])?;
                validate_private_block_param(method, args.get(2))?;
                let value = self.private_api.get_storage_at(address, slot).await?;
                Ok(Value::String(format!("0x{}", hex::encode(value))))
            }
            "eth_getCode" => {
                let args = expect_params_array(method, &params, 1)?;
                let address = parse_address(&args[0])?;
                validate_private_block_param(method, args.get(1))?;
                let code = self.private_api.get_code(address).await?;
                Ok(Value::String(format!("0x{}", hex::encode(code))))
            }
            _ if is_safe_read_only_passthrough(method) => {
                self.upstream_api.request(method, params).await
            }
            _ => Err(GatewayError::method_not_found(method)),
        }
    }
}

fn expect_params_array<'a>(
    method: &str,
    params: &'a Value,
    min_len: usize,
) -> Result<&'a [Value], GatewayError> {
    let values = params
        .as_array()
        .ok_or_else(|| GatewayError::invalid_params(format!("{method} expects array params")))?;

    if values.len() < min_len {
        return Err(GatewayError::invalid_params(format!(
            "{method} expects at least {min_len} parameter(s)"
        )));
    }

    Ok(values.as_slice())
}

fn parse_address(value: &Value) -> Result<[u8; 20], GatewayError> {
    let address_str = value
        .as_str()
        .ok_or_else(|| GatewayError::invalid_params("address must be a hex string"))?;
    let address_hex = address_str.strip_prefix("0x").unwrap_or(address_str);

    if address_hex.len() != 40 {
        return Err(GatewayError::invalid_params(
            "address must be exactly 20 bytes",
        ));
    }

    let mut address = [0u8; 20];
    hex::decode_to_slice(address_hex, &mut address)
        .map_err(|e| GatewayError::invalid_params(format!("invalid address: {e}")))?;

    Ok(address)
}

fn parse_storage_slot(value: &Value) -> Result<[u8; 32], GatewayError> {
    let slot_str = value
        .as_str()
        .ok_or_else(|| GatewayError::invalid_params("slot must be a hex string"))?;
    let slot_hex = slot_str.strip_prefix("0x").unwrap_or(slot_str);

    if slot_hex.len() > 64 {
        return Err(GatewayError::invalid_params("slot too long (max 32 bytes)"));
    }

    let normalized_hex = if slot_hex.len() % 2 == 0 {
        slot_hex.to_owned()
    } else {
        format!("0{slot_hex}")
    };

    let slot_bytes = hex::decode(&normalized_hex)
        .map_err(|e| GatewayError::invalid_params(format!("invalid slot: {e}")))?;

    let mut slot = [0u8; 32];
    let offset = 32 - slot_bytes.len();
    slot[offset..].copy_from_slice(&slot_bytes);
    Ok(slot)
}

fn validate_private_block_param(
    method: &str,
    block_param: Option<&Value>,
) -> Result<(), GatewayError> {
    let Some(block_param) = block_param else {
        return Ok(());
    };

    if block_param.is_null() {
        return Ok(());
    }

    let tag = block_param.as_str().ok_or_else(|| {
        GatewayError::invalid_params(format!("{method} block parameter must be a string tag"))
    })?;

    if tag.eq_ignore_ascii_case("latest") {
        return Ok(());
    }

    Err(GatewayError::invalid_params(format!(
        "{method} currently supports only \"latest\" block tag"
    )))
}

pub fn is_safe_read_only_passthrough(method: &str) -> bool {
    if UNSAFE_METHODS.contains(&method) {
        return false;
    }

    SAFE_READ_ONLY_PASSTHROUGH_METHODS.contains(&method)
}

pub struct PirPrivateApi {
    pir_client: Arc<PirClient>,
    code_resolver: Arc<CodeResolver>,
}

impl PirPrivateApi {
    pub fn new(
        pir_server_a: String,
        pir_server_b: String,
        dict_url: String,
        cas_url: String,
    ) -> Self {
        Self {
            pir_client: Arc::new(PirClient::new(pir_server_a, pir_server_b)),
            code_resolver: Arc::new(CodeResolver::new(dict_url, cas_url)),
        }
    }
}

#[async_trait(?Send)]
impl PrivateApi for PirPrivateApi {
    async fn get_balance(&self, address: [u8; 20]) -> Result<u128, GatewayError> {
        self.pir_client
            .query_account(address)
            .await
            .map(|account| account.balance)
            .map_err(|e| GatewayError::internal(format!("private balance query failed: {e}")))
    }

    async fn get_transaction_count(&self, address: [u8; 20]) -> Result<u64, GatewayError> {
        self.pir_client
            .query_account(address)
            .await
            .map(|account| account.nonce)
            .map_err(|e| GatewayError::internal(format!("private nonce query failed: {e}")))
    }

    async fn get_storage_at(
        &self,
        address: [u8; 20],
        slot: [u8; 32],
    ) -> Result<[u8; 32], GatewayError> {
        self.pir_client
            .query_storage(address, slot)
            .await
            .map(|storage| storage.value)
            .map_err(|e| GatewayError::internal(format!("private storage query failed: {e}")))
    }

    async fn get_code(&self, address: [u8; 20]) -> Result<Vec<u8>, GatewayError> {
        let account = self
            .pir_client
            .query_account(address)
            .await
            .map_err(|e| GatewayError::internal(format!("private code query failed: {e}")))?;

        let code_hash = if let Some(code_hash) = account.code_hash {
            Some(code_hash)
        } else if let Some(code_id) = account.code_id {
            Some(
                self.code_resolver
                    .resolve_code_hash(code_id)
                    .await
                    .map_err(|e| {
                        GatewayError::internal(format!("code hash resolution failed: {e}"))
                    })?,
            )
        } else {
            None
        };

        match code_hash {
            Some(hash) => self
                .code_resolver
                .fetch_bytecode(hash)
                .await
                .map_err(|e| GatewayError::internal(format!("bytecode fetch failed: {e}"))),
            None => Ok(Vec::new()),
        }
    }
}

pub struct HttpUpstreamApi {
    upstream_url: String,
    http_client: reqwest::Client,
}

impl HttpUpstreamApi {
    pub fn new(upstream_url: String, request_timeout_ms: u64) -> Result<Self, GatewayError> {
        let builder = reqwest::Client::builder();
        #[cfg(not(target_arch = "wasm32"))]
        let builder = builder
            .timeout(Duration::from_millis(request_timeout_ms.max(1)))
            .connect_timeout(Duration::from_secs(5));
        #[cfg(target_arch = "wasm32")]
        let _ = request_timeout_ms;

        let http_client = builder
            .build()
            .map_err(|e| GatewayError::internal(format!("failed to build upstream client: {e}")))?;

        Ok(Self {
            upstream_url,
            http_client,
        })
    }
}

#[async_trait(?Send)]
impl UpstreamApi for HttpUpstreamApi {
    async fn request(&self, method: &str, params: Value) -> Result<Value, GatewayError> {
        let payload = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params,
        });

        let response = self
            .http_client
            .post(&self.upstream_url)
            .json(&payload)
            .send()
            .await
            .map_err(|e| upstream_request_error(method, &e))?;

        let status = response.status();
        if !status.is_success() {
            if let Ok(response_json) = response.json::<Value>().await {
                if let Some(error) = extract_json_rpc_error(&response_json) {
                    return Err(error);
                }
            }
            return Err(GatewayError::internal(format!(
                "Upstream HTTP status {status} for {method}"
            )));
        }

        let response_json: Value = response
            .json::<Value>()
            .await
            .map_err(|_| GatewayError::internal(format!("Invalid JSON response for {method}")))?;

        if let Some(error) = extract_json_rpc_error(&response_json) {
            return Err(error);
        }

        Ok(response_json.get("result").cloned().unwrap_or(Value::Null))
    }
}

fn upstream_request_error(method: &str, error: &reqwest::Error) -> GatewayError {
    if error.is_timeout() {
        return GatewayError::internal(format!("Upstream timeout for {method}"));
    }

    #[cfg(not(target_arch = "wasm32"))]
    if error.is_connect() {
        return GatewayError::internal(format!("Upstream connection failed for {method}"));
    }

    GatewayError::internal(format!("Upstream request failed for {method}"))
}

fn extract_json_rpc_error(response_json: &Value) -> Option<GatewayError> {
    let error = response_json.get("error")?.as_object()?;
    let code = error.get("code").and_then(Value::as_i64).unwrap_or(-32000);
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("Upstream error")
        .to_owned();
    let data = error.get("data").cloned();
    Some(GatewayError::json_rpc(code, message, data))
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GatewayConfig {
    pub upstream_url: String,
    pub pir_server_a: String,
    pub pir_server_b: String,
    pub dict_url: String,
    pub cas_url: String,
    #[serde(default = "default_request_timeout_ms")]
    pub request_timeout_ms: u64,
}

fn default_request_timeout_ms() -> u64 {
    15_000
}

#[derive(Debug, Deserialize)]
struct Eip1193Request {
    method: String,
    #[serde(default = "default_params")]
    params: Value,
}

fn default_params() -> Value {
    Value::Array(Vec::new())
}

#[wasm_bindgen]
pub struct WasmGateway {
    inner: Arc<GatewayCore<PirPrivateApi, HttpUpstreamApi>>,
}

#[wasm_bindgen]
impl WasmGateway {
    #[wasm_bindgen(constructor)]
    pub fn new(config: JsValue) -> Result<WasmGateway, JsValue> {
        let parsed_config: GatewayConfig = serde_wasm_bindgen::from_value(config).map_err(|e| {
            to_js_error(GatewayError::invalid_params(format!("invalid config: {e}")))
        })?;

        let private_api = PirPrivateApi::new(
            parsed_config.pir_server_a,
            parsed_config.pir_server_b,
            parsed_config.dict_url,
            parsed_config.cas_url,
        );
        let upstream_api =
            HttpUpstreamApi::new(parsed_config.upstream_url, parsed_config.request_timeout_ms)
                .map_err(to_js_error)?;

        Ok(Self {
            inner: Arc::new(GatewayCore::new(private_api, upstream_api)),
        })
    }

    #[wasm_bindgen(js_name = request)]
    pub async fn request(&self, payload: JsValue) -> Result<JsValue, JsValue> {
        let request: Eip1193Request = serde_wasm_bindgen::from_value(payload).map_err(|e| {
            to_js_error(GatewayError::invalid_params(format!(
                "invalid request: {e}"
            )))
        })?;

        let result = self
            .inner
            .request_json(&request.method, request.params)
            .await
            .map_err(to_js_error_from_json)?;

        serde_wasm_bindgen::to_value(&result).map_err(|e| {
            to_js_error(GatewayError::internal(format!(
                "response encoding failed: {e}"
            )))
        })
    }
}

fn to_js_error(error: GatewayError) -> JsValue {
    to_js_error_from_json(error.into_json())
}

fn to_js_error_from_json(error: JsonRpcError) -> JsValue {
    match serde_wasm_bindgen::to_value(&error) {
        Ok(value) => value,
        Err(_) => JsValue::from_str(&format!("JSON-RPC error {}: {}", error.code, error.message)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::rc::Rc;

    #[cfg(not(target_arch = "wasm32"))]
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    #[cfg(not(target_arch = "wasm32"))]
    use tokio::net::TcpListener;

    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test;

    #[derive(Clone, Debug, Default)]
    struct CallLog {
        private: Rc<RefCell<Vec<String>>>,
        upstream: Rc<RefCell<Vec<String>>>,
    }

    #[derive(Clone)]
    struct MockPrivateApi {
        log: CallLog,
        fail_method: Option<String>,
    }

    impl MockPrivateApi {
        fn should_fail(&self, method: &str) -> Result<(), GatewayError> {
            if self.fail_method.as_deref() == Some(method) {
                return Err(GatewayError::internal(format!("{method} private failure")));
            }
            Ok(())
        }
    }

    #[async_trait(?Send)]
    impl PrivateApi for MockPrivateApi {
        async fn get_balance(&self, _address: [u8; 20]) -> Result<u128, GatewayError> {
            self.log
                .private
                .borrow_mut()
                .push("eth_getBalance".to_owned());
            self.should_fail("eth_getBalance")?;
            Ok(0x42)
        }

        async fn get_transaction_count(&self, _address: [u8; 20]) -> Result<u64, GatewayError> {
            self.log
                .private
                .borrow_mut()
                .push("eth_getTransactionCount".to_owned());
            self.should_fail("eth_getTransactionCount")?;
            Ok(0x7)
        }

        async fn get_storage_at(
            &self,
            _address: [u8; 20],
            _slot: [u8; 32],
        ) -> Result<[u8; 32], GatewayError> {
            self.log
                .private
                .borrow_mut()
                .push("eth_getStorageAt".to_owned());
            self.should_fail("eth_getStorageAt")?;
            Ok([0xAA; 32])
        }

        async fn get_code(&self, _address: [u8; 20]) -> Result<Vec<u8>, GatewayError> {
            self.log.private.borrow_mut().push("eth_getCode".to_owned());
            self.should_fail("eth_getCode")?;
            Ok(vec![0x60, 0x00])
        }
    }

    #[derive(Clone)]
    struct MockUpstreamApi {
        log: CallLog,
        values: HashMap<String, Value>,
    }

    #[async_trait(?Send)]
    impl UpstreamApi for MockUpstreamApi {
        async fn request(&self, method: &str, _params: Value) -> Result<Value, GatewayError> {
            self.log.upstream.borrow_mut().push(method.to_owned());
            Ok(self.values.get(method).cloned().unwrap_or(Value::Null))
        }
    }

    fn build_gateway(
        fail_private_method: Option<&str>,
    ) -> (GatewayCore<MockPrivateApi, MockUpstreamApi>, CallLog) {
        let log = CallLog::default();

        let private_api = MockPrivateApi {
            log: log.clone(),
            fail_method: fail_private_method.map(str::to_owned),
        };
        let upstream_api = MockUpstreamApi {
            log: log.clone(),
            values: HashMap::from([
                ("eth_chainId".to_owned(), Value::String("0x1".to_owned())),
                (
                    "eth_getBlockByHash".to_owned(),
                    Value::String("ok".to_owned()),
                ),
            ]),
        };

        (GatewayCore::new(private_api, upstream_api), log)
    }

    fn sample_address() -> &'static str {
        "0x1111111111111111111111111111111111111111"
    }

    #[cfg(not(target_arch = "wasm32"))]
    async fn spawn_upstream_server(status_line: &str, body: &str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind listener");
        let addr = listener.local_addr().expect("listener addr");
        let status = status_line.to_owned();
        let response_body = body.to_owned();
        tokio::spawn(async move {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            let mut buf = [0u8; 2048];
            let _ = socket.read(&mut buf).await;
            let response = format!(
                "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            let _ = socket.write_all(response.as_bytes()).await;
        });
        format!("http://{}", addr)
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    async fn private_balance_routes_to_private_and_skips_upstream() {
        let (gateway, log) = build_gateway(None);
        let params = serde_json::json!([sample_address(), "latest"]);

        let result = gateway
            .request_json("eth_getBalance", params)
            .await
            .expect("balance request should succeed");

        assert_eq!(result, Value::String("0x42".to_owned()));
        assert_eq!(log.private.borrow().as_slice(), ["eth_getBalance"]);
        assert!(log.upstream.borrow().is_empty());
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    async fn private_get_transaction_count_routes_to_private() {
        let (gateway, log) = build_gateway(None);
        let params = serde_json::json!([sample_address(), "latest"]);

        let result = gateway
            .request_json("eth_getTransactionCount", params)
            .await
            .expect("nonce request should succeed");

        assert_eq!(result, Value::String("0x7".to_owned()));
        assert_eq!(log.private.borrow().as_slice(), ["eth_getTransactionCount"]);
        assert!(log.upstream.borrow().is_empty());
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    async fn private_get_storage_at_routes_to_private() {
        let (gateway, log) = build_gateway(None);
        let params = serde_json::json!([sample_address(), "0x01", "latest"]);

        let result = gateway
            .request_json("eth_getStorageAt", params)
            .await
            .expect("storage request should succeed");

        assert_eq!(result, Value::String(format!("0x{}", "aa".repeat(32))));
        assert_eq!(log.private.borrow().as_slice(), ["eth_getStorageAt"]);
        assert!(log.upstream.borrow().is_empty());
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    async fn private_get_code_routes_to_private() {
        let (gateway, log) = build_gateway(None);
        let params = serde_json::json!([sample_address(), "latest"]);

        let result = gateway
            .request_json("eth_getCode", params)
            .await
            .expect("code request should succeed");

        assert_eq!(result, Value::String("0x6000".to_owned()));
        assert_eq!(log.private.borrow().as_slice(), ["eth_getCode"]);
        assert!(log.upstream.borrow().is_empty());
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    async fn chain_id_passthrough_routes_upstream() {
        let (gateway, log) = build_gateway(None);
        let result = gateway
            .request_json("eth_chainId", serde_json::json!([]))
            .await
            .expect("chain id should passthrough");

        assert_eq!(result, Value::String("0x1".to_owned()));
        assert!(log.private.borrow().is_empty());
        assert_eq!(log.upstream.borrow().as_slice(), ["eth_chainId"]);
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    async fn safe_read_only_fallback_routes_upstream() {
        let (gateway, log) = build_gateway(None);
        let result = gateway
            .request_json("eth_getBlockByHash", serde_json::json!(["0x123", false]))
            .await
            .expect("safe fallback should passthrough");

        assert_eq!(result, Value::String("ok".to_owned()));
        assert!(log.private.borrow().is_empty());
        assert_eq!(log.upstream.borrow().as_slice(), ["eth_getBlockByHash"]);
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    async fn unsafe_write_method_is_rejected() {
        let (gateway, log) = build_gateway(None);
        let err = gateway
            .request_json("eth_sendRawTransaction", serde_json::json!(["0xdeadbeef"]))
            .await
            .expect_err("write method should be rejected");

        assert_eq!(err.code, -32601);
        assert!(log.private.borrow().is_empty());
        assert!(log.upstream.borrow().is_empty());
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    async fn unsafe_submit_methods_are_rejected() {
        let (gateway, log) = build_gateway(None);

        let submit_work = gateway
            .request_json("eth_submitWork", serde_json::json!(["0x1", "0x2", "0x3"]))
            .await
            .expect_err("eth_submitWork should be rejected");
        let submit_hashrate = gateway
            .request_json(
                "eth_submitHashrate",
                serde_json::json!(["0x1", "0x1234567890abcdef"]),
            )
            .await
            .expect_err("eth_submitHashrate should be rejected");

        assert_eq!(submit_work.code, -32601);
        assert_eq!(submit_hashrate.code, -32601);
        assert!(log.private.borrow().is_empty());
        assert!(log.upstream.borrow().is_empty());
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    async fn account_methods_are_not_passthrough() {
        let (gateway, log) = build_gateway(None);
        let err = gateway
            .request_json("eth_accounts", serde_json::json!([]))
            .await
            .expect_err("eth_accounts should not route through gateway");

        assert_eq!(err.code, -32601);
        assert!(log.private.borrow().is_empty());
        assert!(log.upstream.borrow().is_empty());
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    async fn non_latest_private_block_tag_is_rejected() {
        let (gateway, log) = build_gateway(None);
        let err = gateway
            .request_json(
                "eth_getBalance",
                serde_json::json!([sample_address(), "0x10"]),
            )
            .await
            .expect_err("historical block tags are not currently supported");

        assert_eq!(err.code, -32602);
        assert!(err.message.contains("only \"latest\""));
        assert!(log.private.borrow().is_empty());
        assert!(log.upstream.borrow().is_empty());
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    async fn non_latest_storage_block_tag_is_rejected() {
        let (gateway, log) = build_gateway(None);
        let err = gateway
            .request_json(
                "eth_getStorageAt",
                serde_json::json!([sample_address(), "0x01", "earliest"]),
            )
            .await
            .expect_err("historical storage block tags are not currently supported");

        assert_eq!(err.code, -32602);
        assert!(err.message.contains("only \"latest\""));
        assert!(log.private.borrow().is_empty());
        assert!(log.upstream.borrow().is_empty());
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    async fn invalid_address_maps_to_invalid_params_error() {
        let (gateway, _) = build_gateway(None);
        let err = gateway
            .request_json("eth_getBalance", serde_json::json!(["0x1234", "latest"]))
            .await
            .expect_err("invalid address should fail");

        assert_eq!(err.code, -32602);
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    async fn private_errors_map_to_json_rpc_error_object() {
        let (gateway, log) = build_gateway(Some("eth_getCode"));
        let err = gateway
            .request_json(
                "eth_getCode",
                serde_json::json!([sample_address(), "latest"]),
            )
            .await
            .expect_err("private failure should map");

        assert_eq!(err.code, -32000);
        assert!(err.message.contains("eth_getCode private failure"));
        assert_eq!(log.private.borrow().as_slice(), ["eth_getCode"]);
        assert!(log.upstream.borrow().is_empty());
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    async fn wasm_harness_runs_private_and_passthrough_paths() {
        let (gateway, log) = build_gateway(None);

        let balance = gateway
            .request_json(
                "eth_getBalance",
                serde_json::json!([sample_address(), "latest"]),
            )
            .await
            .expect("private balance should succeed");
        let chain_id = gateway
            .request_json("eth_chainId", serde_json::json!([]))
            .await
            .expect("passthrough chain id should succeed");

        assert_eq!(balance, Value::String("0x42".to_owned()));
        assert_eq!(chain_id, Value::String("0x1".to_owned()));
        assert_eq!(log.private.borrow().as_slice(), ["eth_getBalance"]);
        assert_eq!(log.upstream.borrow().as_slice(), ["eth_chainId"]);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    async fn upstream_error_null_returns_result() {
        let url = spawn_upstream_server(
            "200 OK",
            r#"{"jsonrpc":"2.0","id":1,"result":"0x1","error":null}"#,
        )
        .await;
        let api = HttpUpstreamApi::new(url, 1000).expect("build upstream api");

        let result = api
            .request("eth_chainId", serde_json::json!([]))
            .await
            .expect("error:null should not be treated as failure");

        assert_eq!(result, Value::String("0x1".to_owned()));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    async fn upstream_non_success_status_reports_http_status() {
        let url = spawn_upstream_server("500 Internal Server Error", r#"{"error":"boom"}"#).await;
        let api = HttpUpstreamApi::new(url, 1000).expect("build upstream api");

        let err = api
            .request("eth_chainId", serde_json::json!([]))
            .await
            .expect_err("non-2xx upstream status should fail");
        let err = err.into_json();

        assert_eq!(err.code, -32000);
        assert!(err.message.contains("500 Internal Server Error"));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    async fn upstream_non_success_status_with_json_rpc_error_is_forwarded() {
        let url = spawn_upstream_server(
            "500 Internal Server Error",
            r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32042,"message":"rate limited","data":{"retryAfter":1}}}"#,
        )
        .await;
        let api = HttpUpstreamApi::new(url, 1000).expect("build upstream api");

        let err = api
            .request("eth_chainId", serde_json::json!([]))
            .await
            .expect_err("json-rpc error body should be forwarded even on non-2xx");
        let err = err.into_json();

        assert_eq!(err.code, -32042);
        assert_eq!(err.message, "rate limited");
        assert_eq!(err.data, Some(serde_json::json!({ "retryAfter": 1 })));
    }
}
