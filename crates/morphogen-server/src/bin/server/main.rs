#![cfg(feature = "network")]

//! Production server binary.

// Submodule declarations MUST precede any item that references them
// (load_file_config / init_gpu_resources / run() reference config types).
// Extracted to sibling files in TASK-55.2 (config_helpers) and TASK-55.3 (config).
mod config;
mod config_helpers;
#[allow(unused_imports)]
use config::*;
#[allow(unused_imports)]
use config_helpers::*;

use clap::{ArgAction, Parser};
use morphogen_core::{DeltaBuffer, EpochSnapshot, GlobalState};
use morphogen_server::{
    epoch::{self, EpochManager},
    network::{
        create_router_with_concurrency, telemetry, AppState, EpochMetadata, PagePirConfig,
        MAX_CONCURRENT_SCANS,
    },
    Environment, ServerConfig,
};
use morphogen_storage::ChunkedMatrix;
use serde::Deserialize;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io::{BufReader, Read};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::watch;

#[cfg(feature = "cuda")]
use morphogen_gpu_dpf::kernel::GpuScanner;
#[cfg(feature = "cuda")]
use morphogen_gpu_dpf::storage::GpuPageMatrix;
#[cfg(feature = "cuda")]
use std::sync::Mutex;

const DEFAULT_BIND_ADDR: &str = "127.0.0.1:3000";
const DEFAULT_MERGE_INTERVAL_MS: u64 = 1_000;
const DEFAULT_MATRIX_SEED: u64 = 42;
const DEFAULT_PAGE_DOMAIN_BITS: usize = 25;
const DEFAULT_PAGE_ROWS_PER_PAGE: usize = 1;
const DEFAULT_SEEDS: [u64; 3] = [0x1234, 0x5678, 0x9ABC];
const CTRL_C_INITIAL_RETRY_DELAY_MS: u64 = 250;
const CTRL_C_MAX_RETRY_DELAY_MS: u64 = 5_000;
const CTRL_C_MAX_FAILURES_BEFORE_FORCED_SHUTDOWN: u32 = 12;
const DEFAULT_ADMIN_SNAPSHOT_MAX_BYTES: u64 = 1_073_741_824;
pub const DEFAULT_ADMIN_MTLS_SUBJECT_HEADER: &str = "x-mtls-subject";

/// Type alias for stub GPU scanner (non-CUDA builds).
type StubGpuScanner = Option<Arc<()>>;
/// Type alias for stub GPU matrix (non-CUDA builds).
type StubGpuMatrix = Option<Arc<std::sync::Mutex<Option<()>>>>;
type ShutdownFuture = Pin<Box<dyn std::future::Future<Output = ()> + Send>>;

#[cfg(feature = "cuda")]
fn init_gpu_resources(
    runtime: &RuntimeConfig,
    matrix: &ChunkedMatrix,
) -> Result<
    (
        Option<Arc<GpuScanner>>,
        Option<Arc<Mutex<Option<GpuPageMatrix>>>>,
    ),
    StartupError,
> {
    tracing::info!(
        device = runtime.gpu_device,
        preload = runtime.gpu_preload,
        "initializing CUDA resources"
    );

    let scanner = Arc::new(
        GpuScanner::new(runtime.gpu_device)
            .map_err(|e| StartupError::new(format!("failed to initialize GPU scanner: {}", e)))?,
    );

    let gpu_matrix = if runtime.gpu_preload {
        GpuPageMatrix::from_chunked_matrix(scanner.device.clone(), matrix).map_err(|e| {
            StartupError::new(format!(
                "failed to preload GPU matrix from CPU matrix: {}",
                e
            ))
        })?
    } else {
        let num_pages = matrix.total_size_bytes() / runtime.row_size_bytes;
        GpuPageMatrix::alloc_empty(scanner.device.clone(), num_pages)
            .map_err(|e| StartupError::new(format!("failed to allocate empty GPU matrix: {}", e)))?
    };

    Ok((Some(scanner), Some(Arc::new(Mutex::new(Some(gpu_matrix))))))
}

fn load_file_config(path: &Path) -> Result<FileConfig, StartupError> {
    let raw = std::fs::read_to_string(path).map_err(|e| {
        StartupError::new(format!(
            "failed to read config file '{}': {}",
            path.display(),
            e
        ))
    })?;

    serde_json::from_str::<FileConfig>(&raw).map_err(|e| {
        StartupError::new(format!(
            "failed to parse config file '{}' as JSON: {}",
            path.display(),
            e
        ))
    })
}

fn load_matrix_from_file(
    path: &Path,
    row_size_bytes: usize,
    chunk_size_bytes: usize,
) -> Result<ChunkedMatrix, StartupError> {
    let file = File::open(path).map_err(|e| {
        StartupError::new(format!(
            "failed to open matrix file '{}': {}",
            path.display(),
            e
        ))
    })?;
    let metadata = file.metadata().map_err(|e| {
        StartupError::new(format!(
            "failed to stat matrix file '{}': {}",
            path.display(),
            e
        ))
    })?;

    let total_size_bytes = usize::try_from(metadata.len()).map_err(|_| {
        StartupError::new(format!(
            "matrix file '{}' is too large for this platform",
            path.display()
        ))
    })?;

    if total_size_bytes == 0 {
        return Err(StartupError::new(format!(
            "matrix file '{}' is empty",
            path.display()
        )));
    }

    if !total_size_bytes.is_multiple_of(row_size_bytes) {
        return Err(StartupError::new(format!(
            "matrix file size ({}) must be divisible by row_size_bytes ({})",
            total_size_bytes, row_size_bytes
        )));
    }

    let mut matrix = ChunkedMatrix::new(total_size_bytes, chunk_size_bytes);
    let mut reader = BufReader::new(file);
    let mut row_buf = vec![0u8; row_size_bytes];
    let num_rows = total_size_bytes / row_size_bytes;

    for row_idx in 0..num_rows {
        reader.read_exact(&mut row_buf).map_err(|e| {
            StartupError::new(format!(
                "failed reading row {} from matrix file '{}': {}",
                row_idx,
                path.display(),
                e
            ))
        })?;
        matrix.write_row(row_idx, row_size_bytes, &row_buf);
    }

    Ok(matrix)
}

fn build_matrix(runtime: &RuntimeConfig) -> Result<ChunkedMatrix, StartupError> {
    if let Some(path) = runtime.matrix_file.as_ref() {
        load_matrix_from_file(path, runtime.row_size_bytes, runtime.chunk_size_bytes)
    } else {
        let matrix_size_bytes = runtime.matrix_size_bytes.ok_or_else(|| {
            StartupError::new("matrix_size_bytes is required for synthetic matrix")
        })?;
        let mut matrix = ChunkedMatrix::new(matrix_size_bytes, runtime.chunk_size_bytes);
        matrix.fill_with_pattern(runtime.matrix_seed);
        Ok(matrix)
    }
}

fn validate_server_config(
    runtime: &RuntimeConfig,
    matrix_size_bytes: usize,
) -> Result<(), StartupError> {
    let config = ServerConfig {
        environment: runtime.environment,
        row_size_bytes: runtime.row_size_bytes,
        chunk_size_bytes: runtime.chunk_size_bytes,
        matrix_size_bytes,
        bench_fill_seed: Some(runtime.matrix_seed),
    };

    config
        .validate()
        .map_err(|e| StartupError::new(format!("invalid runtime configuration: {}", e)))
}

fn build_page_config(
    cfg: Option<PagePirRuntimeConfig>,
    num_rows: usize,
) -> Result<Option<PagePirConfig>, StartupError> {
    let Some(cfg) = cfg else {
        return Ok(None);
    };

    let num_pages = 1usize
        .checked_shl(cfg.domain_bits as u32)
        .ok_or_else(|| StartupError::new("page_domain_bits is too large"))?;
    let row_capacity = num_pages
        .checked_mul(cfg.rows_per_page)
        .ok_or_else(|| StartupError::new("page_domain_bits * page_rows_per_page overflowed"))?;

    if num_rows > row_capacity {
        return Err(StartupError::new(format!(
            "page config capacity ({}) is smaller than matrix rows ({})",
            row_capacity, num_rows
        )));
    }

    Ok(Some(PagePirConfig {
        domain_bits: cfg.domain_bits,
        rows_per_page: cfg.rows_per_page,
        prg_keys: cfg.prg_keys,
    }))
}

fn should_log_ctrl_c_failure(failures: u32) -> bool {
    failures <= 3 || failures.is_multiple_of(10)
}

fn should_force_shutdown_after_ctrl_c_failures(failures: u32) -> bool {
    failures >= CTRL_C_MAX_FAILURES_BEFORE_FORCED_SHUTDOWN
}

#[cfg(unix)]
fn ctrl_c_failures_force_shutdown() -> bool {
    false
}

#[cfg(not(unix))]
fn ctrl_c_failures_force_shutdown() -> bool {
    true
}

fn next_ctrl_c_retry_delay_ms(current_delay_ms: u64) -> u64 {
    (current_delay_ms * 2).min(CTRL_C_MAX_RETRY_DELAY_MS)
}

#[cfg(unix)]
struct ShutdownWaiters {
    ctrl_c: ShutdownFuture,
    sigterm: tokio::signal::unix::Signal,
}

#[cfg(not(unix))]
struct ShutdownWaiters {
    ctrl_c: ShutdownFuture,
}

async fn wait_for_ctrl_c_signal_with<WaitOnce, WaitFuture, SleepFn, SleepFuture>(
    mut wait_once: WaitOnce,
    mut sleep_fn: SleepFn,
) where
    WaitOnce: FnMut() -> WaitFuture,
    WaitFuture: std::future::Future<Output = std::io::Result<()>>,
    SleepFn: FnMut(Duration) -> SleepFuture,
    SleepFuture: std::future::Future<Output = ()>,
{
    let mut failures = 0u32;
    let mut retry_delay_ms = CTRL_C_INITIAL_RETRY_DELAY_MS;

    loop {
        match wait_once().await {
            Ok(()) => break,
            Err(err) => {
                failures = failures.saturating_add(1);
                if should_log_ctrl_c_failure(failures) {
                    tracing::error!(failures, "ctrl+c signal stream failed: {}", err);
                }

                if should_force_shutdown_after_ctrl_c_failures(failures) {
                    if ctrl_c_failures_force_shutdown() {
                        tracing::error!(
                            failures,
                            "ctrl+c signal stream failed repeatedly; forcing shutdown"
                        );
                        break;
                    }

                    tracing::error!(
                        failures,
                        "ctrl+c signal stream failed repeatedly; disabling ctrl+c waiter and relying on SIGTERM"
                    );
                    std::future::pending::<()>().await;
                }

                sleep_fn(Duration::from_millis(retry_delay_ms)).await;
                retry_delay_ms = next_ctrl_c_retry_delay_ms(retry_delay_ms);
            }
        }
    }
}

async fn wait_for_ctrl_c_signal() {
    wait_for_ctrl_c_signal_with(tokio::signal::ctrl_c, tokio::time::sleep).await;
}

#[cfg(unix)]
fn install_shutdown_waiters() -> Result<ShutdownWaiters, StartupError> {
    let sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .map_err(|err| StartupError::new(format!("failed to install SIGTERM handler: {}", err)))?;
    Ok(ShutdownWaiters {
        ctrl_c: Box::pin(wait_for_ctrl_c_signal()),
        sigterm,
    })
}

#[cfg(not(unix))]
fn install_shutdown_waiters() -> Result<ShutdownWaiters, StartupError> {
    Ok(ShutdownWaiters {
        ctrl_c: Box::pin(wait_for_ctrl_c_signal()),
    })
}

#[cfg(unix)]
async fn drive_shutdown_from_futures<F1, F2>(
    ctrl_c: F1,
    sigterm: F2,
    shutdown_tx: watch::Sender<bool>,
) where
    F1: std::future::Future<Output = ()>,
    F2: std::future::Future<Output = ()>,
{
    tokio::select! {
        _ = ctrl_c => {}
        _ = sigterm => {}
    }
    tracing::info!("shutdown signal received");
    let _ = shutdown_tx.send(true);
}

#[cfg(not(unix))]
async fn drive_shutdown_from_futures<F1>(ctrl_c: F1, shutdown_tx: watch::Sender<bool>)
where
    F1: std::future::Future<Output = ()>,
{
    ctrl_c.await;
    tracing::info!("shutdown signal received");
    let _ = shutdown_tx.send(true);
}

#[cfg(unix)]
async fn drive_shutdown_signal(mut waiters: ShutdownWaiters, shutdown_tx: watch::Sender<bool>) {
    drive_shutdown_from_futures(
        waiters.ctrl_c.as_mut(),
        async move {
            let _ = waiters.sigterm.recv().await;
        },
        shutdown_tx,
    )
    .await;
}

#[cfg(not(unix))]
async fn drive_shutdown_signal(waiters: ShutdownWaiters, shutdown_tx: watch::Sender<bool>) {
    drive_shutdown_from_futures(waiters.ctrl_c, shutdown_tx).await;
}

#[tokio::main]
async fn main() {
    if let Err(err) = run().await {
        eprintln!("morphogen-server startup failed: {}", err);
        std::process::exit(1);
    }
}

async fn run() -> Result<(), StartupError> {
    telemetry::init_tracing();
    let metrics_handle = telemetry::init_metrics();

    let cli = CliArgs::parse();
    let env = EnvConfig::from_process_env()?;
    let config_path = cli.config.clone().or_else(|| env.config.clone());

    let file = match config_path {
        Some(path) => Some(load_file_config(&path)?),
        None => None,
    };

    let runtime = RuntimeConfig::resolve(cli, env, file)?;
    // env_var() trims and drops empty values, so alias fallback is safe.
    let admin_snapshot_token = env_var("MORPHOGEN_ADMIN_BEARER_TOKEN")
        .or_else(|| env_var("MORPHOGEN_ADMIN_SNAPSHOT_TOKEN"));
    let admin_mtls_subject_header =
        parse_admin_mtls_subject_header(env_var("MORPHOGEN_ADMIN_MTLS_SUBJECT_HEADER"))?;
    let admin_mtls_allowed_subjects = env_var("MORPHOGEN_ADMIN_MTLS_ALLOWED_SUBJECTS")
        .map(|raw| {
            raw.split(',')
                .map(|subject| subject.trim().to_string())
                .filter(|subject| !subject.is_empty())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let admin_mtls_trust_proxy_headers =
        parse_env_bool("MORPHOGEN_ADMIN_TRUST_PROXY_HEADERS")?.unwrap_or(false);
    validate_admin_mtls_proxy_trust(&admin_mtls_allowed_subjects, admin_mtls_trust_proxy_headers)?;
    let admin_snapshot_allow_local_paths =
        parse_env_bool("MORPHOGEN_ADMIN_ALLOW_LOCAL_SNAPSHOT")?.unwrap_or(false);
    let admin_snapshot_allowed_hosts = env_var("MORPHOGEN_ADMIN_SNAPSHOT_ALLOWED_HOSTS")
        .map(|raw| {
            raw.split(',')
                .map(|host| host.trim().to_ascii_lowercase())
                .filter(|host| !host.is_empty())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let admin_snapshot_max_bytes_u64 = parse_env_u64("MORPHOGEN_ADMIN_SNAPSHOT_MAX_BYTES")?
        .unwrap_or(DEFAULT_ADMIN_SNAPSHOT_MAX_BYTES);
    let admin_snapshot_max_bytes = usize::try_from(admin_snapshot_max_bytes_u64).map_err(|_| {
        StartupError::new(format!(
            "MORPHOGEN_ADMIN_SNAPSHOT_MAX_BYTES={} exceeds usize::MAX",
            admin_snapshot_max_bytes_u64
        ))
    })?;

    let matrix = Arc::new(build_matrix(&runtime)?);
    validate_server_config(&runtime, matrix.total_size_bytes())?;

    let num_rows = matrix.total_size_bytes() / runtime.row_size_bytes;
    let page_config = build_page_config(runtime.page_config.clone(), num_rows)?;

    #[cfg(feature = "cuda")]
    let (gpu_scanner, gpu_matrix) = init_gpu_resources(&runtime, matrix.as_ref())?;

    #[cfg(not(feature = "cuda"))]
    let (_gpu_scanner, _gpu_matrix): (StubGpuScanner, StubGpuMatrix) = (None, None);

    let snapshot = EpochSnapshot {
        epoch_id: 0,
        matrix,
    };
    let pending = Arc::new(DeltaBuffer::new_with_epoch(runtime.row_size_bytes, 0));
    let global = Arc::new(GlobalState::new(Arc::new(snapshot), pending));

    let (epoch_tx, epoch_rx) = watch::channel(EpochMetadata {
        epoch_id: 0,
        num_rows,
        seeds: runtime.seeds,
        block_number: runtime.block_number,
        state_root: runtime.state_root,
    });

    #[cfg(feature = "cuda")]
    let shared_gpu_matrix = gpu_matrix
        .clone()
        .unwrap_or_else(|| Arc::new(Mutex::new(None)));

    #[cfg(feature = "cuda")]
    let epoch_manager = Arc::new(
        EpochManager::new_with_gpu_matrix(
            global.clone(),
            runtime.row_size_bytes,
            shared_gpu_matrix.clone(),
        )
        .map_err(|e| StartupError::new(format!("failed to initialize epoch manager: {}", e)))?,
    );

    #[cfg(not(feature = "cuda"))]
    let epoch_manager = Arc::new(
        EpochManager::new(global.clone(), runtime.row_size_bytes)
            .map_err(|e| StartupError::new(format!("failed to initialize epoch manager: {}", e)))?,
    );

    #[cfg(feature = "cuda")]
    let gpu_matrix_state = if gpu_matrix.is_some() {
        Some(shared_gpu_matrix)
    } else {
        None
    };

    let state = Arc::new(AppState {
        global: global.clone(),
        epoch_manager: epoch_manager.clone(),
        epoch_tx,
        snapshot_rotation_lock: Arc::new(tokio::sync::Mutex::new(())),
        admin_snapshot_token,
        admin_mtls_subject_header,
        admin_mtls_allowed_subjects,
        admin_mtls_trust_proxy_headers,
        admin_snapshot_allow_local_paths,
        admin_snapshot_allowed_hosts,
        admin_snapshot_max_bytes,
        row_size_bytes: runtime.row_size_bytes,
        num_rows,
        seeds: runtime.seeds,
        block_number: runtime.block_number,
        state_root: runtime.state_root,
        epoch_rx,
        page_config,
        #[cfg(feature = "cuda")]
        gpu_scanner,
        #[cfg(feature = "cuda")]
        gpu_matrix: gpu_matrix_state,
    });

    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let shutdown_waiters = install_shutdown_waiters()?;
    let merge_worker = tokio::spawn(epoch::spawn_merge_worker(
        epoch_manager,
        runtime.merge_interval,
        shutdown_rx,
    ));

    let app =
        create_router_with_concurrency(state, runtime.max_concurrent_scans, Some(metrics_handle));

    let listener = tokio::net::TcpListener::bind(runtime.bind_addr)
        .await
        .map_err(|e| StartupError::new(format!("failed to bind {}: {}", runtime.bind_addr, e)))?;

    if !runtime.bind_addr.ip().is_loopback() {
        tracing::warn!(
            bind_addr = %runtime.bind_addr,
            "non-loopback bind configured; deploy behind TLS termination and restrict /admin/* with network ACLs"
        );
    }
    if runtime.environment == Environment::Prod && runtime.bind_addr_is_default {
        tracing::warn!(
            bind_addr = %runtime.bind_addr,
            "using default loopback bind in prod; set MORPHOGEN_SERVER_BIND_ADDR explicitly if remote access is required"
        );
    }

    tracing::info!(
        bind_addr = %runtime.bind_addr,
        max_concurrent_scans = runtime.max_concurrent_scans,
        merge_interval_ms = runtime.merge_interval.as_millis(),
        "morphogen-server listening"
    );

    let shutdown_tx_for_signal = shutdown_tx.clone();
    let graceful_shutdown = async move {
        drive_shutdown_signal(shutdown_waiters, shutdown_tx_for_signal).await;
    };

    let serve_result = axum::serve(listener, app)
        .with_graceful_shutdown(graceful_shutdown)
        .await;

    let _ = shutdown_tx.send(true);
    if let Err(err) = merge_worker.await {
        return Err(StartupError::new(format!(
            "merge worker join failed during shutdown: {}",
            err
        )));
    }

    serve_result.map_err(|e| StartupError::new(format!("server runtime failed: {}", e)))
}

#[cfg(test)]
mod tests;
