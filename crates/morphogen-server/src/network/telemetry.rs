use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

pub fn init_tracing() {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "morphogen_server=info,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer()) // Default text for now, JSON if needed
        .init();
}

pub fn init_metrics() -> PrometheusHandle {
    let builder = PrometheusBuilder::new();
    builder.install_recorder().expect("failed to install Prometheus recorder")
}
