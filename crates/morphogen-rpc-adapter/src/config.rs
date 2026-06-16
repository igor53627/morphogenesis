//! CLI / env configuration and the privacy-fallback validator.
//!
//! Extracted from the crate root in TASK-54.3. This module owns the `clap`
//! definition (`Args`), the deployment-environment profile
//! (`AdapterEnvironment`), and [`validate_privacy_fallback_config`], which
//! enforces the fail-closed-by-default privacy invariant (TASK-37): in prod,
//! `--fallback-to-upstream` requires the explicit `--allow-privacy-degraded-fallback`
//! acknowledgement.
//!
//! Visibility: all items are `pub(crate)` — intentional seams per the
//! TASK-54 design constraints (no broad `pub` of internals).

use anyhow::{bail, Result};
use clap::Parser;
use std::path::PathBuf;

#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AdapterEnvironment {
    Dev,
    Test,
    Prod,
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about = "Morphogenesis RPC Adapter")]
pub(crate) struct Args {
    /// Port to listen on
    #[arg(short, long, default_value_t = 8545)]
    pub(crate) port: u16,

    /// Upstream Ethereum RPC URL
    #[arg(
        short,
        long,
        env = "UPSTREAM_RPC_URL",
        default_value = "https://ethereum-rpc.publicnode.com"
    )]
    pub(crate) upstream: String,

    /// PIR Server A URL
    #[arg(long, default_value = "http://localhost:3000")]
    pub(crate) pir_server_a: String,

    /// PIR Server B URL
    #[arg(long, default_value = "http://localhost:3001")]
    pub(crate) pir_server_b: String,

    /// Dictionary URL for CodeID resolution
    #[arg(
        long,
        env = "DICT_URL",
        default_value = "http://localhost:8080/mainnet_compact.dict"
    )]
    pub(crate) dict_url: String,

    /// CAS Base URL for bytecode fetching
    #[arg(long, env = "CAS_URL", default_value = "http://localhost:8080/cas")]
    pub(crate) cas_url: String,

    /// Print effective URL config and exit (internal diagnostics/test hook)
    #[arg(long, hide = true, default_value_t = false)]
    pub(crate) print_effective_config: bool,

    /// Required allowlist root for local file:// dictionary/CAS URLs
    #[arg(long)]
    pub(crate) file_url_root: Option<PathBuf>,

    /// Metadata refresh interval in seconds
    #[arg(long, default_value_t = 12)]
    pub(crate) refresh_interval: u64,

    /// Upstream request timeout in seconds
    #[arg(long, default_value_t = 15)]
    pub(crate) upstream_timeout: u64,

    /// Fall back to upstream RPC when PIR servers are unavailable
    #[arg(long, default_value_t = false)]
    pub(crate) fallback_to_upstream: bool,

    /// Deployment environment profile
    #[arg(long, value_enum, default_value_t = AdapterEnvironment::Prod)]
    pub(crate) environment: AdapterEnvironment,

    /// Explicit acknowledgement required to allow privacy-degrading fallback in prod
    #[arg(long, default_value_t = false)]
    pub(crate) allow_privacy_degraded_fallback: bool,

    /// Transaction relay URL for eth_sendRawTransaction.
    /// Defaults to Flashbots Protect so txs bypass the public mempool.
    #[arg(
        long,
        default_value = "https://rpc.flashbots.net/?hint=hash&originId=morphogenesis"
    )]
    pub(crate) tx_relay: String,

    /// Enable OpenTelemetry trace export (Datadog Agent OTLP gRPC compatible)
    #[arg(long, default_value_t = false)]
    pub(crate) otel_traces: bool,

    /// OTLP collector endpoint (Datadog Agent default: http://127.0.0.1:4317)
    #[arg(long, default_value = "http://127.0.0.1:4317")]
    pub(crate) otel_endpoint: String,

    /// Service name reported to APM
    #[arg(long, default_value = "morphogen-rpc-adapter")]
    pub(crate) otel_service_name: String,

    /// Deployment environment tag for traces
    #[arg(long, default_value = "e2e")]
    pub(crate) otel_env: String,

    /// Service version tag for traces
    #[arg(long, default_value = env!("CARGO_PKG_VERSION"))]
    pub(crate) otel_version: String,
}

#[cfg(test)]
impl Args {
    pub(crate) fn default_for_tests() -> Self {
        Self {
            port: 8545,
            upstream: "https://ethereum-rpc.publicnode.com".to_string(),
            pir_server_a: "http://localhost:3000".to_string(),
            pir_server_b: "http://localhost:3001".to_string(),
            dict_url: "http://localhost:8080/mainnet_compact.dict".to_string(),
            cas_url: "http://localhost:8080/cas".to_string(),
            print_effective_config: false,
            file_url_root: None,
            refresh_interval: 12,
            upstream_timeout: 15,
            fallback_to_upstream: false,
            environment: AdapterEnvironment::Prod,
            allow_privacy_degraded_fallback: false,
            tx_relay: "https://rpc.flashbots.net/?hint=hash&originId=morphogenesis".to_string(),
            otel_traces: false,
            otel_endpoint: "http://127.0.0.1:4317".to_string(),
            otel_service_name: "morphogen-rpc-adapter".to_string(),
            otel_env: "e2e".to_string(),
            otel_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

/// Enforce the fail-closed privacy-fallback policy (TASK-37).
///
/// In `Prod`, `--fallback-to-upstream` is rejected unless the operator also
/// passes `--allow-privacy-degraded-fallback`, because falling back to a
/// regular upstream RPC reveals the queried address to the provider and
/// degrades the PIR privacy guarantee.
pub(crate) fn validate_privacy_fallback_config(args: &Args) -> Result<()> {
    if args.fallback_to_upstream
        && args.environment == AdapterEnvironment::Prod
        && !args.allow_privacy_degraded_fallback
    {
        bail!("--fallback-to-upstream in prod requires --allow-privacy-degraded-fallback");
    }

    Ok(())
}
