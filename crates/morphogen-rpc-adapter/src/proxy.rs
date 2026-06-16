//! Upstream JSON-RPC forwarding and error redaction.
//!
//! Extracted from the crate root in TASK-54.2. These helpers are the single
//! seam through which passthrough/relay methods reach the upstream provider.
//! Error messages are intentionally generic (no URL/body echo) to avoid
//! leaking credentials or PII; URL values are redacted via
//! [`crate::telemetry::sanitize_url_for_telemetry`] before any tracing.
//!
//! Visibility: all items are `pub(crate)` — intentional seams per the
//! TASK-54 design constraints (no broad `pub` of internals).

use anyhow::Result;
use jsonrpsee::types::ErrorObjectOwned;
use serde_json::Value;
use tracing::{info, warn, Instrument};

/// Forward a JSON-RPC `method`/`params` call to the upstream `url` using the
/// shared `client`, returning the upstream `result` on success.
///
/// Upstream errors are converted to [`ErrorObjectOwned`] with the upstream
/// code/message intact; transport failures return generic `-32000` messages
/// that never echo the URL or request body (redaction invariant).
pub(crate) async fn proxy_to_upstream(
    url: &str,
    client: &reqwest::Client,
    method: &str,
    params: Value,
) -> Result<Value, ErrorObjectOwned> {
    let sanitized_url = crate::telemetry::sanitize_url_for_telemetry(url);
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

/// Generic `-32000` error used when an upstream HTTP request itself fails
/// (covers network/transport errors other than timeout/connect). The message
/// is intentionally generic and never echoes the URL or body.
pub(crate) fn upstream_request_failed_error(method: &str) -> ErrorObjectOwned {
    ErrorObjectOwned::owned(
        -32000,
        format!("Upstream request failed for {}", method),
        None::<()>,
    )
}

/// Generic `-32000` error used when the upstream response cannot be parsed as
/// JSON. The message is intentionally generic and never echoes the body.
pub(crate) fn upstream_invalid_json_error(method: &str) -> ErrorObjectOwned {
    ErrorObjectOwned::owned(
        -32000,
        format!("Invalid JSON response for {}", method),
        None::<()>,
    )
}
