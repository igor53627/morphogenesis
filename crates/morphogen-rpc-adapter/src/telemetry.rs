use anyhow::{bail, Context, Result};
use http::{Extensions, HeaderMap};
use opentelemetry::propagation::Extractor;
use opentelemetry::trace::TraceContextExt;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry::KeyValue;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::SdkTracerProvider;
use opentelemetry_sdk::Resource;
use std::task::{Context as TaskContext, Poll};
use tower::{Layer, Service};
use tracing_opentelemetry::OpenTelemetrySpanExt;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

#[derive(Debug, Clone)]
pub struct OtelSettings {
    pub enabled: bool,
    pub endpoint: String,
    pub service_name: String,
    pub environment: String,
    pub service_version: String,
}

pub struct TelemetryGuard {
    tracer_provider: Option<SdkTracerProvider>,
}

#[derive(Debug, Clone, Default)]
pub struct TraceContextLayer;

#[derive(Debug, Clone)]
pub struct TraceContextService<S> {
    inner: S,
}

impl<S> Layer<S> for TraceContextLayer {
    type Service = TraceContextService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        TraceContextService { inner }
    }
}

impl<S, B> Service<http::Request<B>> for TraceContextService<S>
where
    S: Service<http::Request<B>>,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = S::Future;

    fn poll_ready(&mut self, cx: &mut TaskContext<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, mut request: http::Request<B>) -> Self::Future {
        let headers = request.headers().clone();
        capture_trace_context(&headers, request.extensions_mut());
        self.inner.call(request)
    }
}

#[derive(Debug, Clone, Default)]
pub struct TraceContextHeaders {
    pub traceparent: Option<String>,
    pub tracestate: Option<String>,
    pub baggage: Option<String>,
}

impl Drop for TelemetryGuard {
    fn drop(&mut self) {
        if let Some(provider) = &self.tracer_provider {
            if let Err(err) = provider.shutdown() {
                eprintln!("failed to shutdown tracer provider: {err}");
            }
        }
    }
}

impl TelemetryGuard {
    fn disabled() -> Self {
        Self {
            tracer_provider: None,
        }
    }
}

pub fn capture_trace_context(headers: &HeaderMap, extensions: &mut Extensions) {
    let ctx = TraceContextHeaders {
        traceparent: headers
            .get("traceparent")
            .and_then(|value| value.to_str().ok())
            .map(ToOwned::to_owned),
        tracestate: headers
            .get("tracestate")
            .and_then(|value| value.to_str().ok())
            .map(ToOwned::to_owned),
        baggage: headers
            .get("baggage")
            .and_then(|value| value.to_str().ok())
            .map(ToOwned::to_owned),
    };
    extensions.insert(ctx);
}

pub fn rpc_server_span(method: &'static str, extensions: &Extensions) -> tracing::Span {
    let span = tracing::info_span!(
        "rpc.request",
        otel.kind = "server",
        otel.name = method,
        rpc.system = "ethereum-jsonrpc",
        rpc.method = method
    );
    if let Some(parent) = extract_remote_parent_context(extensions) {
        let _ = span.set_parent(parent);
    }
    span
}

fn extract_remote_parent_context(extensions: &Extensions) -> Option<opentelemetry::Context> {
    let headers = extensions.get::<TraceContextHeaders>()?;
    headers.traceparent.as_ref()?;

    let extracted = opentelemetry::global::get_text_map_propagator(|propagator| {
        propagator.extract(&TraceHeadersExtractor { headers })
    });
    if extracted.span().span_context().is_valid() {
        Some(extracted)
    } else {
        None
    }
}

struct TraceHeadersExtractor<'a> {
    headers: &'a TraceContextHeaders,
}

impl Extractor for TraceHeadersExtractor<'_> {
    fn get(&self, key: &str) -> Option<&str> {
        match key {
            "traceparent" => self.headers.traceparent.as_deref(),
            "tracestate" => self.headers.tracestate.as_deref(),
            "baggage" => self.headers.baggage.as_deref(),
            _ => None,
        }
    }

    fn keys(&self) -> Vec<&str> {
        let mut keys = Vec::with_capacity(3);
        if self.headers.traceparent.is_some() {
            keys.push("traceparent");
        }
        if self.headers.tracestate.is_some() {
            keys.push("tracestate");
        }
        if self.headers.baggage.is_some() {
            keys.push("baggage");
        }
        keys
    }
}

#[cfg(test)]
fn has_remote_parent(extensions: &Extensions) -> bool {
    extract_remote_parent_context(extensions).is_some()
}

/// Sanitize URL for telemetry by keeping scheme + host (+ port when present).
/// Credentials, path, query, and fragment are always removed.
pub fn sanitize_url_for_telemetry(url: &str) -> String {
    match reqwest::Url::parse(url) {
        Ok(parsed) => {
            let host = parsed.host_str().unwrap_or("unknown");
            let host = if host.contains(':') && !host.starts_with('[') && !host.ends_with(']') {
                format!("[{host}]")
            } else {
                host.to_string()
            };

            if let Some(port) = parsed.port() {
                format!("{}://{}:{}", parsed.scheme(), host, port)
            } else {
                format!("{}://{}", parsed.scheme(), host)
            }
        }
        Err(_) => "<invalid-url>".to_string(),
    }
}

pub fn normalize_otlp_endpoint(raw: &str) -> Result<String> {
    let candidate = if raw.contains("://") {
        raw.to_string()
    } else {
        format!("http://{raw}")
    };

    let parsed = reqwest::Url::parse(&candidate)
        .with_context(|| format!("invalid OTLP endpoint: {candidate}"))?;
    match parsed.scheme() {
        "http" | "https" => {}
        scheme => bail!("unsupported OTLP endpoint scheme: {scheme}"),
    }
    if parsed.host_str().is_none() {
        bail!("OTLP endpoint must include host");
    }

    Ok(parsed.as_str().trim_end_matches('/').to_string())
}

#[cfg(test)]
pub fn default_otel_settings() -> OtelSettings {
    OtelSettings {
        enabled: false,
        endpoint: "http://127.0.0.1:4317".to_string(),
        service_name: "morphogen-rpc-adapter".to_string(),
        environment: "e2e".to_string(),
        service_version: env!("CARGO_PKG_VERSION").to_string(),
    }
}

/// Default EnvFilter for tracing when RUST_LOG is not set.
/// Includes both the RPC adapter and E2E client targets.
pub const DEFAULT_ENV_FILTER: &str = "morphogen_rpc_adapter=info,morphogen_e2e_client=info";

pub fn init_tracing(otel: OtelSettings) -> Result<TelemetryGuard> {
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| DEFAULT_ENV_FILTER.into());
    let fmt_layer = tracing_subscriber::fmt::layer();

    if !otel.enabled {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt_layer)
            .try_init()
            .context("failed to initialize tracing subscriber")?;
        return Ok(TelemetryGuard::disabled());
    }

    let endpoint = normalize_otlp_endpoint(&otel.endpoint)?;
    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint)
        .build()
        .context("failed to build OTLP span exporter")?;

    let resource = Resource::builder()
        .with_service_name(otel.service_name.clone())
        .with_attributes(vec![
            KeyValue::new("deployment.environment.name", otel.environment),
            KeyValue::new("service.version", otel.service_version),
        ])
        .build();

    let tracer_provider = SdkTracerProvider::builder()
        .with_resource(resource)
        .with_batch_exporter(exporter)
        .build();
    let tracer = tracer_provider.tracer("morphogen-rpc-adapter");

    opentelemetry::global::set_text_map_propagator(TraceContextPropagator::new());
    opentelemetry::global::set_tracer_provider(tracer_provider.clone());

    tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer)
        .with(tracing_opentelemetry::layer().with_tracer(tracer))
        .try_init()
        .context("failed to initialize tracing + opentelemetry subscriber")?;

    Ok(TelemetryGuard {
        tracer_provider: Some(tracer_provider),
    })
}

#[cfg(test)]
mod tests {
    use super::{
        capture_trace_context, default_otel_settings, has_remote_parent, normalize_otlp_endpoint,
        sanitize_url_for_telemetry,
    };
    use http::{Extensions, HeaderMap, HeaderValue};
    use opentelemetry::global;
    use opentelemetry_sdk::propagation::TraceContextPropagator;

    #[test]
    fn normalize_endpoint_accepts_explicit_http_scheme() {
        let endpoint = normalize_otlp_endpoint("http://127.0.0.1:4317").expect("valid endpoint");
        assert_eq!(endpoint, "http://127.0.0.1:4317");
    }

    #[test]
    fn normalize_endpoint_rejects_unsupported_scheme() {
        let err = normalize_otlp_endpoint("ftp://collector.local:4317")
            .expect_err("unsupported scheme should fail");
        assert!(
            err.to_string().contains("unsupported OTLP endpoint scheme"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn default_settings_have_otlp_compatible_defaults() {
        let settings = default_otel_settings();
        assert!(!settings.enabled);
        assert_eq!(settings.endpoint, "http://127.0.0.1:4317");
        assert_eq!(settings.service_name, "morphogen-rpc-adapter");
        assert_eq!(settings.environment, "e2e");
        assert_eq!(settings.service_version, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn capture_trace_context_records_trace_headers_into_extensions() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "traceparent",
            HeaderValue::from_static("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"),
        );
        headers.insert("tracestate", HeaderValue::from_static("dd=s:1;t.dm:-0"));
        headers.insert("baggage", HeaderValue::from_static("k=v"));

        let mut extensions = Extensions::new();
        capture_trace_context(&headers, &mut extensions);

        let stored = extensions
            .get::<super::TraceContextHeaders>()
            .expect("trace headers should be present");
        assert_eq!(
            stored.traceparent.as_deref(),
            Some("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01")
        );
        assert_eq!(stored.tracestate.as_deref(), Some("dd=s:1;t.dm:-0"));
        assert_eq!(stored.baggage.as_deref(), Some("k=v"));
    }

    #[test]
    fn valid_remote_parent_is_detected_from_extensions() {
        global::set_text_map_propagator(TraceContextPropagator::new());
        let mut headers = HeaderMap::new();
        headers.insert(
            "traceparent",
            HeaderValue::from_static("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"),
        );

        let mut extensions = Extensions::new();
        capture_trace_context(&headers, &mut extensions);
        assert!(has_remote_parent(&extensions));
    }

    #[test]
    fn missing_remote_parent_is_not_detected() {
        global::set_text_map_propagator(TraceContextPropagator::new());
        let mut extensions = Extensions::new();
        let headers = HeaderMap::new();
        capture_trace_context(&headers, &mut extensions);
        assert!(!has_remote_parent(&extensions));
    }

    #[test]
    fn default_env_filter_includes_both_binaries() {
        // Verify the default filter string includes both adapter and e2e-client targets
        // This ensures spans from both binaries are emitted by default
        // Use the actual constant from production code to prevent drift
        let default_filter = super::DEFAULT_ENV_FILTER;

        // Parse the filter to ensure it's valid - this will panic if the filter is invalid
        let _filter = tracing_subscriber::EnvFilter::new(default_filter);

        // Verify the filter string contains both targets
        assert!(default_filter.contains("morphogen_rpc_adapter=info"));
        assert!(default_filter.contains("morphogen_e2e_client=info"));
    }

    #[test]
    fn sanitize_url_preserves_host_and_port_without_credentials() {
        assert_eq!(
            sanitize_url_for_telemetry("https://user:pass@api.example.com:8443/path?key=secret"),
            "https://api.example.com:8443"
        );
        assert_eq!(
            sanitize_url_for_telemetry("http://127.0.0.1:8545/anything"),
            "http://127.0.0.1:8545"
        );
        assert_eq!(
            sanitize_url_for_telemetry("not-a-valid-url"),
            "<invalid-url>"
        );
    }
}
