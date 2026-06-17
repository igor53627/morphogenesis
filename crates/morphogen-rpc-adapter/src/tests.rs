use super::config::AdapterEnvironment;
use super::filters::{effective_latest_for_filter, parse_log_filter_for_rpc};
use super::methods::has_nonempty_state_overrides;
use super::proxy::upstream_invalid_json_error;
use super::state::next_privacy_degraded_fallback_count;
use super::{
    fail_closed_if_fallback_disabled, handle_eth_get_logs, handle_eth_new_filter,
    proxy_to_upstream, validate_privacy_fallback_config, Args, DROPPED_METHODS,
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
    assert_eq!(
        args.dict_url,
        "https://cli-dict.example/mainnet_compact.dict"
    );
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
            let req_json: Value = serde_json::from_slice(&body).expect("valid json-rpc request");
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
    let filter =
        crate::block_cache::parse_log_filter_object(&filter_obj, effective_latest, Some(120), None)
            .expect("safe range should remain valid when cache latest lags");
    assert_eq!(filter.from_block, 120);
    assert_eq!(filter.to_block, 120);
}

#[test]
fn stale_cache_latest_does_not_invalidate_finalized_default_range() {
    let filter_obj = json!({ "fromBlock": "finalized" });
    let effective_latest = effective_latest_for_filter(100, None, Some(130));
    let filter =
        crate::block_cache::parse_log_filter_object(&filter_obj, effective_latest, None, Some(130))
            .expect("finalized range should remain valid when cache latest lags");
    assert_eq!(filter.from_block, 130);
    assert_eq!(filter.to_block, 130);
}

#[test]
fn explicit_latest_to_block_uses_effective_latest_when_safe_is_higher() {
    let filter_obj = json!({ "fromBlock": "safe", "toBlock": "latest" });
    let effective_latest = effective_latest_for_filter(100, Some(120), None);
    let filter =
        crate::block_cache::parse_log_filter_object(&filter_obj, effective_latest, Some(120), None)
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
