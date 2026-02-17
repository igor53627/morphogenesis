#[path = "../telemetry.rs"]
#[allow(dead_code)]
mod telemetry;

use anyhow::{bail, Context, Result};
use clap::Parser;
use opentelemetry::global;
use opentelemetry::propagation::Injector;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde_json::Value;
use tracing::{info, warn, Instrument};
use tracing_opentelemetry::OpenTelemetrySpanExt;

#[derive(Parser, Debug, Clone)]
#[command(
    author,
    version,
    about = "OpenTelemetry-instrumented E2E client for Morphogenesis RPC Adapter"
)]
struct Args {
    /// RPC URL of the local adapter
    #[arg(long, default_value = "http://127.0.0.1:8545")]
    rpc_url: String,

    /// Account address used by the E2E fixture
    #[arg(long, default_value = "0x000000000000000000000000000000000000031c")]
    address: String,

    /// Enable OpenTelemetry trace export
    #[arg(long, default_value_t = false)]
    otel_traces: bool,

    /// OTLP collector endpoint (default: http://127.0.0.1:4317)
    #[arg(long, default_value = "http://127.0.0.1:4317")]
    otel_endpoint: String,

    /// Service name reported to APM
    #[arg(long, default_value = "morphogen-e2e-client")]
    otel_service_name: String,

    /// Deployment environment tag for traces
    #[arg(long, default_value = "e2e")]
    otel_env: String,

    /// Service version tag for traces
    #[arg(long, default_value = "local")]
    otel_version: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let _telemetry_guard = telemetry::init_tracing(telemetry::OtelSettings {
        enabled: args.otel_traces,
        endpoint: args.otel_endpoint.clone(),
        service_name: args.otel_service_name.clone(),
        environment: args.otel_env.clone(),
        service_version: args.otel_version.clone(),
    })?;

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .context("failed to build HTTP client")?;

    let balance = rpc_call(
        &client,
        &args.rpc_url,
        "eth_getBalance",
        serde_json::json!([args.address, "latest"]),
    )
    .await?;
    let nonce = rpc_call(
        &client,
        &args.rpc_url,
        "eth_getTransactionCount",
        serde_json::json!([args.address, "latest"]),
    )
    .await?;
    let code = rpc_call(
        &client,
        &args.rpc_url,
        "eth_getCode",
        serde_json::json!([args.address, "latest"]),
    )
    .await?;

    let balance_wei = parse_hex_quantity_u128(&balance).context("eth_getBalance parse failure")?;
    let nonce_value =
        parse_hex_quantity_u64(&nonce).context("eth_getTransactionCount parse failure")?;
    let code_hex = code
        .as_str()
        .context("eth_getCode result should be a string hex value")?;

    info!(%balance_wei, "client observed balance");
    info!(%nonce_value, "client observed nonce");
    info!(code = %code_hex, "client observed code");

    if balance_wei != 100_000_000_000_000_000_000u128 {
        bail!(
            "balance mismatch: expected 100000000000000000000, got {}",
            balance_wei
        );
    }
    if nonce_value != 123 {
        bail!("nonce mismatch: expected 123, got {}", nonce_value);
    }
    if !code_hex.eq_ignore_ascii_case("0x60016001") {
        bail!("code mismatch: expected 0x60016001, got {}", code_hex);
    }

    println!("Balance: {}", balance_wei);
    println!("Nonce: {}", nonce_value);
    println!("Code: {}", code_hex);
    println!("ALL TESTS PASSED.");
    Ok(())
}

async fn rpc_call(
    client: &reqwest::Client,
    rpc_url: &str,
    method: &str,
    params: Value,
) -> Result<Value> {
    let sanitized_rpc_url = telemetry::sanitize_url_for_telemetry(rpc_url);
    let request_span = tracing::info_span!(
        "rpc.client.request",
        otel.kind = "client",
        otel.name = method,
        rpc.system = "ethereum-jsonrpc",
        rpc.method = %method,
        peer.service = "morphogen-rpc-adapter",
        server.address = %sanitized_rpc_url
    );

    async move {
        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params
        });

        let mut headers = HeaderMap::new();
        inject_current_context(&mut headers);

        let response = client
            .post(rpc_url)
            .headers(headers)
            .json(&request)
            .send()
            .await
            .with_context(|| format!("request to {} failed", method))?;

        let status = response.status();
        let json: Value = response
            .json()
            .await
            .with_context(|| format!("invalid JSON response for {}", method))?;

        if !status.is_success() {
            bail!("{} returned HTTP {}: {}", method, status, json);
        }

        if let Some(error) = json.get("error") {
            bail!("{} returned JSON-RPC error: {}", method, error);
        }

        let result = json.get("result").cloned().unwrap_or(Value::Null);
        info!(rpc_method = %method, rpc_result = %result, "client RPC completed");
        Ok(result)
    }
    .instrument(request_span)
    .await
}

fn inject_current_context(headers: &mut HeaderMap) {
    let context = tracing::Span::current().context();
    global::get_text_map_propagator(|propagator| {
        propagator.inject_context(&context, &mut HeaderInjector(headers));
    });
}

fn parse_hex_quantity_u128(value: &Value) -> Result<u128> {
    let raw = value.as_str().context("result is not a string")?;
    let hex = raw
        .strip_prefix("0x")
        .with_context(|| format!("quantity should start with 0x, got {}", raw))?;
    if hex.is_empty() {
        bail!("empty hex quantity");
    }
    u128::from_str_radix(hex, 16).with_context(|| format!("invalid hex quantity: {}", raw))
}

fn parse_hex_quantity_u64(value: &Value) -> Result<u64> {
    let raw = value.as_str().context("result is not a string")?;
    let hex = raw
        .strip_prefix("0x")
        .with_context(|| format!("quantity should start with 0x, got {}", raw))?;
    if hex.is_empty() {
        bail!("empty hex quantity");
    }
    u64::from_str_radix(hex, 16).with_context(|| format!("invalid hex quantity: {}", raw))
}

struct HeaderInjector<'a>(&'a mut HeaderMap);

impl Injector for HeaderInjector<'_> {
    fn set(&mut self, key: &str, value: String) {
        let name = match HeaderName::from_bytes(key.as_bytes()) {
            Ok(name) => name,
            Err(error) => {
                warn!(%key, %error, "invalid header name for trace context");
                return;
            }
        };
        let value = match HeaderValue::from_str(&value) {
            Ok(value) => value,
            Err(error) => {
                warn!(
                    %key,
                    value_len = value.len(),
                    %error,
                    "invalid header value for trace context"
                );
                return;
            }
        };
        self.0.insert(name, value);
    }
}
