//! Production server binary.

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
const DEFAULT_ADMIN_MTLS_SUBJECT_HEADER: &str = "x-mtls-subject";

/// Type alias for stub GPU scanner (non-CUDA builds).
type StubGpuScanner = Option<Arc<()>>;
/// Type alias for stub GPU matrix (non-CUDA builds).
type StubGpuMatrix = Option<Arc<std::sync::Mutex<Option<()>>>>;
type ShutdownFuture = Pin<Box<dyn std::future::Future<Output = ()> + Send>>;

#[derive(Debug, Clone)]
struct StartupError {
    message: String,
}

impl StartupError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for StartupError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl Error for StartupError {}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about = "Morphogenesis PIR server")]
struct CliArgs {
    /// Optional JSON config file path.
    #[arg(long)]
    config: Option<PathBuf>,

    /// Bind address, for example 127.0.0.1:3000.
    #[arg(long)]
    bind_addr: Option<String>,

    /// Environment profile: dev, test, or prod.
    #[arg(long)]
    environment: Option<String>,

    /// Row size in bytes.
    #[arg(long)]
    row_size_bytes: Option<usize>,

    /// Chunk size in bytes.
    #[arg(long)]
    chunk_size_bytes: Option<usize>,

    /// Synthetic matrix size in bytes.
    #[arg(long)]
    matrix_size_bytes: Option<usize>,

    /// Path to a matrix file to load at startup.
    #[arg(long)]
    matrix_file: Option<PathBuf>,

    /// Seed used for synthetic matrix generation.
    #[arg(long)]
    matrix_seed: Option<u64>,

    /// Allow synthetic matrix generation when no matrix file is provided.
    #[arg(long, action = ArgAction::SetTrue)]
    allow_synthetic_matrix: bool,

    /// Forbid synthetic matrix generation even if configured in file/env.
    #[arg(long = "no-allow-synthetic-matrix", action = ArgAction::SetTrue, conflicts_with = "allow_synthetic_matrix")]
    no_allow_synthetic_matrix: bool,

    /// Merge worker interval in milliseconds.
    #[arg(long)]
    merge_interval_ms: Option<u64>,

    /// Maximum concurrent scan requests.
    #[arg(long)]
    max_concurrent_scans: Option<usize>,

    /// Comma-separated epoch seeds (three u64 values).
    #[arg(long)]
    seeds: Option<String>,

    /// Initial block number reported by metadata endpoint.
    #[arg(long)]
    block_number: Option<u64>,

    /// Initial state root (32-byte hex string).
    #[arg(long)]
    state_root: Option<String>,

    /// Disable page PIR metadata and endpoints.
    #[arg(long, action = ArgAction::SetTrue)]
    disable_page_pir: bool,

    /// Force-enable page PIR metadata and endpoints.
    #[arg(long, action = ArgAction::SetTrue, conflicts_with = "disable_page_pir")]
    enable_page_pir: bool,

    /// Page PIR domain bits.
    #[arg(long)]
    page_domain_bits: Option<usize>,

    /// Rows per page in page PIR metadata.
    #[arg(long)]
    page_rows_per_page: Option<usize>,

    /// First 16-byte PRG key as hex.
    #[arg(long)]
    page_prg_key_0: Option<String>,

    /// Second 16-byte PRG key as hex.
    #[arg(long)]
    page_prg_key_1: Option<String>,

    /// CUDA device ordinal.
    #[cfg(feature = "cuda")]
    #[arg(long)]
    gpu_device: Option<usize>,

    /// If true, preload GPU matrix from CPU matrix at startup.
    #[cfg(feature = "cuda")]
    #[arg(long, action = ArgAction::SetTrue)]
    gpu_preload: bool,

    /// Disable GPU matrix preload even when enabled in file/env.
    #[cfg(feature = "cuda")]
    #[arg(long = "no-gpu-preload", action = ArgAction::SetTrue, conflicts_with = "gpu_preload")]
    no_gpu_preload: bool,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct FileConfig {
    bind_addr: Option<String>,
    environment: Option<String>,
    row_size_bytes: Option<usize>,
    chunk_size_bytes: Option<usize>,
    matrix_size_bytes: Option<usize>,
    matrix_file: Option<PathBuf>,
    matrix_seed: Option<u64>,
    allow_synthetic_matrix: Option<bool>,
    merge_interval_ms: Option<u64>,
    max_concurrent_scans: Option<usize>,
    seeds: Option<String>,
    block_number: Option<u64>,
    state_root: Option<String>,
    disable_page_pir: Option<bool>,
    page_domain_bits: Option<usize>,
    page_rows_per_page: Option<usize>,
    page_prg_key_0: Option<String>,
    page_prg_key_1: Option<String>,
    #[cfg(feature = "cuda")]
    gpu_device: Option<usize>,
    #[cfg(feature = "cuda")]
    gpu_preload: Option<bool>,
}

#[derive(Debug, Clone, Default)]
struct EnvConfig {
    config: Option<PathBuf>,
    bind_addr: Option<String>,
    environment: Option<String>,
    row_size_bytes: Option<usize>,
    chunk_size_bytes: Option<usize>,
    matrix_size_bytes: Option<usize>,
    matrix_file: Option<PathBuf>,
    matrix_seed: Option<u64>,
    allow_synthetic_matrix: Option<bool>,
    merge_interval_ms: Option<u64>,
    max_concurrent_scans: Option<usize>,
    seeds: Option<String>,
    block_number: Option<u64>,
    state_root: Option<String>,
    disable_page_pir: Option<bool>,
    page_domain_bits: Option<usize>,
    page_rows_per_page: Option<usize>,
    page_prg_key_0: Option<String>,
    page_prg_key_1: Option<String>,
    #[cfg(feature = "cuda")]
    gpu_device: Option<usize>,
    #[cfg(feature = "cuda")]
    gpu_preload: Option<bool>,
}

impl EnvConfig {
    fn from_process_env() -> Result<Self, StartupError> {
        Ok(Self {
            config: env_var("MORPHOGEN_SERVER_CONFIG").map(PathBuf::from),
            bind_addr: env_var("MORPHOGEN_SERVER_BIND_ADDR"),
            environment: env_var("MORPHOGEN_SERVER_ENV"),
            row_size_bytes: parse_env_usize("MORPHOGEN_SERVER_ROW_SIZE_BYTES")?,
            chunk_size_bytes: parse_env_usize("MORPHOGEN_SERVER_CHUNK_SIZE_BYTES")?,
            matrix_size_bytes: parse_env_usize("MORPHOGEN_SERVER_MATRIX_SIZE_BYTES")?,
            matrix_file: env_var("MORPHOGEN_SERVER_MATRIX_FILE").map(PathBuf::from),
            matrix_seed: parse_env_u64("MORPHOGEN_SERVER_MATRIX_SEED")?,
            allow_synthetic_matrix: parse_env_bool("MORPHOGEN_SERVER_ALLOW_SYNTHETIC_MATRIX")?,
            merge_interval_ms: parse_env_u64("MORPHOGEN_SERVER_MERGE_INTERVAL_MS")?,
            max_concurrent_scans: parse_env_usize_any(&[
                "MORPHOGEN_SERVER_A_MAX_CONCURRENT_SCANS",
                "MORPHOGEN_SERVER_B_MAX_CONCURRENT_SCANS",
                "MORPHOGEN_SERVER_MAX_CONCURRENT_SCANS",
            ])?,
            seeds: env_var("MORPHOGEN_SERVER_SEEDS"),
            block_number: parse_env_u64("MORPHOGEN_SERVER_BLOCK_NUMBER")?,
            state_root: env_var("MORPHOGEN_SERVER_STATE_ROOT"),
            disable_page_pir: parse_env_bool("MORPHOGEN_SERVER_DISABLE_PAGE_PIR")?,
            page_domain_bits: parse_env_usize("MORPHOGEN_SERVER_PAGE_DOMAIN_BITS")?,
            page_rows_per_page: parse_env_usize("MORPHOGEN_SERVER_PAGE_ROWS_PER_PAGE")?,
            page_prg_key_0: env_var("MORPHOGEN_SERVER_PAGE_PRG_KEY_0"),
            page_prg_key_1: env_var("MORPHOGEN_SERVER_PAGE_PRG_KEY_1"),
            #[cfg(feature = "cuda")]
            gpu_device: parse_env_usize("MORPHOGEN_SERVER_GPU_DEVICE")?,
            #[cfg(feature = "cuda")]
            gpu_preload: parse_env_bool("MORPHOGEN_SERVER_GPU_PRELOAD")?,
        })
    }

    #[cfg(test)]
    fn default_for_tests() -> Self {
        Self::default()
    }
}

#[derive(Debug, Clone)]
struct PagePirRuntimeConfig {
    domain_bits: usize,
    rows_per_page: usize,
    prg_keys: [[u8; 16]; 2],
}

#[derive(Debug, Clone)]
struct RuntimeConfig {
    bind_addr: SocketAddr,
    bind_addr_is_default: bool,
    environment: Environment,
    row_size_bytes: usize,
    chunk_size_bytes: usize,
    matrix_size_bytes: Option<usize>,
    matrix_file: Option<PathBuf>,
    matrix_seed: u64,
    merge_interval: Duration,
    max_concurrent_scans: usize,
    seeds: [u64; 3],
    block_number: u64,
    state_root: [u8; 32],
    page_config: Option<PagePirRuntimeConfig>,
    #[cfg(feature = "cuda")]
    gpu_device: usize,
    #[cfg(feature = "cuda")]
    gpu_preload: bool,
}

impl RuntimeConfig {
    fn resolve(
        cli: CliArgs,
        env: EnvConfig,
        file: Option<FileConfig>,
    ) -> Result<Self, StartupError> {
        let file = file.unwrap_or_default();
        let cli_allow_synthetic_matrix = cli.allow_synthetic_matrix_override();
        let cli_disable_page_pir = cli.disable_page_pir_override();
        #[cfg(feature = "cuda")]
        let cli_gpu_preload = cli.gpu_preload_override();

        let env_name = pick3(cli.environment, env.environment, file.environment)
            .unwrap_or_else(|| "prod".to_string());
        let environment = parse_environment(&env_name)?;
        let defaults = ServerConfig::for_env(environment);

        let bind_addr_raw = pick3(cli.bind_addr, env.bind_addr, file.bind_addr);
        let bind_addr_is_default = bind_addr_raw.is_none();
        let bind_addr_raw = bind_addr_raw.unwrap_or_else(|| DEFAULT_BIND_ADDR.to_string());
        let bind_addr = bind_addr_raw.parse::<SocketAddr>().map_err(|e| {
            StartupError::new(format!("invalid bind_addr '{}': {}", bind_addr_raw, e))
        })?;

        let row_size_bytes = pick3(cli.row_size_bytes, env.row_size_bytes, file.row_size_bytes)
            .unwrap_or(defaults.row_size_bytes);
        let chunk_size_bytes = pick3(
            cli.chunk_size_bytes,
            env.chunk_size_bytes,
            file.chunk_size_bytes,
        )
        .unwrap_or(defaults.chunk_size_bytes);

        if row_size_bytes == 0 {
            return Err(StartupError::new("row_size_bytes must be > 0"));
        }
        if chunk_size_bytes == 0 {
            return Err(StartupError::new("chunk_size_bytes must be > 0"));
        }
        if !chunk_size_bytes.is_multiple_of(row_size_bytes) {
            return Err(StartupError::new(format!(
                "chunk_size_bytes ({}) must be divisible by row_size_bytes ({})",
                chunk_size_bytes, row_size_bytes
            )));
        }

        let matrix_file = pick3(cli.matrix_file, env.matrix_file, file.matrix_file);
        let allow_synthetic_matrix = pick3(
            cli_allow_synthetic_matrix,
            env.allow_synthetic_matrix,
            file.allow_synthetic_matrix,
        )
        .unwrap_or(false);

        let mut matrix_size_bytes = pick3(
            cli.matrix_size_bytes,
            env.matrix_size_bytes,
            file.matrix_size_bytes,
        );

        if matrix_file.is_some() && matrix_size_bytes.is_some() {
            return Err(StartupError::new(
                "matrix_size_bytes cannot be set when matrix_file is provided",
            ));
        }

        if matrix_file.is_none() && !allow_synthetic_matrix {
            return Err(StartupError::new(
                "matrix_file is required unless allow_synthetic_matrix=true",
            ));
        }

        if matrix_file.is_none() {
            matrix_size_bytes = Some(matrix_size_bytes.unwrap_or(defaults.matrix_size_bytes));
        }

        let matrix_seed = pick3(cli.matrix_seed, env.matrix_seed, file.matrix_seed)
            .unwrap_or(DEFAULT_MATRIX_SEED);

        let merge_interval_ms = pick3(
            cli.merge_interval_ms,
            env.merge_interval_ms,
            file.merge_interval_ms,
        )
        .unwrap_or(DEFAULT_MERGE_INTERVAL_MS);
        if merge_interval_ms == 0 {
            return Err(StartupError::new("merge_interval_ms must be > 0"));
        }

        let max_concurrent_scans = pick3(
            cli.max_concurrent_scans,
            env.max_concurrent_scans,
            file.max_concurrent_scans,
        )
        .unwrap_or(MAX_CONCURRENT_SCANS);
        if max_concurrent_scans == 0 {
            return Err(StartupError::new("max_concurrent_scans must be > 0"));
        }

        let seeds = match pick3(cli.seeds, env.seeds, file.seeds) {
            Some(raw) => parse_u64_triplet(&raw, "seeds")?,
            None => DEFAULT_SEEDS,
        };

        let block_number =
            pick3(cli.block_number, env.block_number, file.block_number).unwrap_or(0);

        let state_root = match pick3(cli.state_root, env.state_root, file.state_root) {
            Some(raw) => parse_fixed_hex::<32>(&raw, "state_root")?,
            None => [0u8; 32],
        };

        let disable_page_pir = pick3(
            cli_disable_page_pir,
            env.disable_page_pir,
            file.disable_page_pir,
        )
        .unwrap_or(false);

        #[cfg(feature = "cuda")]
        if !disable_page_pir {
            let page_size_bytes = morphogen_gpu_dpf::storage::PAGE_SIZE_BYTES;
            if row_size_bytes > page_size_bytes {
                return Err(StartupError::new(format!(
                    "row_size_bytes ({}) must be <= PAGE_SIZE_BYTES ({}) when page PIR is enabled",
                    row_size_bytes, page_size_bytes
                )));
            }
        }

        let page_config = if disable_page_pir {
            None
        } else {
            let domain_bits = pick3(
                cli.page_domain_bits,
                env.page_domain_bits,
                file.page_domain_bits,
            )
            .unwrap_or(DEFAULT_PAGE_DOMAIN_BITS);
            let rows_per_page = pick3(
                cli.page_rows_per_page,
                env.page_rows_per_page,
                file.page_rows_per_page,
            )
            .unwrap_or(DEFAULT_PAGE_ROWS_PER_PAGE);
            if rows_per_page == 0 {
                return Err(StartupError::new("page_rows_per_page must be > 0"));
            }

            let key0_raw = pick3(cli.page_prg_key_0, env.page_prg_key_0, file.page_prg_key_0);
            let key1_raw = pick3(cli.page_prg_key_1, env.page_prg_key_1, file.page_prg_key_1);

            let key0 = if environment == Environment::Prod {
                let raw = key0_raw.ok_or_else(|| {
                    StartupError::new("page_prg_key_0 is required in prod when page PIR is enabled")
                })?;
                parse_fixed_hex::<16>(&raw, "page_prg_key_0")?
            } else {
                match key0_raw {
                    Some(raw) => parse_fixed_hex::<16>(&raw, "page_prg_key_0")?,
                    None => [0u8; 16],
                }
            };
            let key1 = if environment == Environment::Prod {
                let raw = key1_raw.ok_or_else(|| {
                    StartupError::new("page_prg_key_1 is required in prod when page PIR is enabled")
                })?;
                parse_fixed_hex::<16>(&raw, "page_prg_key_1")?
            } else {
                match key1_raw {
                    Some(raw) => parse_fixed_hex::<16>(&raw, "page_prg_key_1")?,
                    None => [0u8; 16],
                }
            };

            if environment == Environment::Prod {
                if key0.iter().all(|b| *b == 0) {
                    return Err(StartupError::new("page_prg_key_0 must be non-zero in prod"));
                }
                if key1.iter().all(|b| *b == 0) {
                    return Err(StartupError::new("page_prg_key_1 must be non-zero in prod"));
                }
            }

            Some(PagePirRuntimeConfig {
                domain_bits,
                rows_per_page,
                prg_keys: [key0, key1],
            })
        };

        #[cfg(feature = "cuda")]
        let gpu_device = pick3(cli.gpu_device, env.gpu_device, file.gpu_device).unwrap_or(0);
        #[cfg(feature = "cuda")]
        let gpu_preload = pick3(cli_gpu_preload, env.gpu_preload, file.gpu_preload).unwrap_or(true);

        Ok(Self {
            bind_addr,
            bind_addr_is_default,
            environment,
            row_size_bytes,
            chunk_size_bytes,
            matrix_size_bytes,
            matrix_file,
            matrix_seed,
            merge_interval: Duration::from_millis(merge_interval_ms),
            max_concurrent_scans,
            seeds,
            block_number,
            state_root,
            page_config,
            #[cfg(feature = "cuda")]
            gpu_device,
            #[cfg(feature = "cuda")]
            gpu_preload,
        })
    }
}

impl CliArgs {
    fn allow_synthetic_matrix_override(&self) -> Option<bool> {
        if self.allow_synthetic_matrix {
            Some(true)
        } else if self.no_allow_synthetic_matrix {
            Some(false)
        } else {
            None
        }
    }

    fn disable_page_pir_override(&self) -> Option<bool> {
        if self.disable_page_pir {
            Some(true)
        } else if self.enable_page_pir {
            Some(false)
        } else {
            None
        }
    }

    #[cfg(feature = "cuda")]
    fn gpu_preload_override(&self) -> Option<bool> {
        if self.gpu_preload {
            Some(true)
        } else if self.no_gpu_preload {
            Some(false)
        } else {
            None
        }
    }

    #[cfg(test)]
    fn default_for_tests() -> Self {
        Self {
            config: None,
            bind_addr: None,
            environment: None,
            row_size_bytes: None,
            chunk_size_bytes: None,
            matrix_size_bytes: None,
            matrix_file: None,
            matrix_seed: None,
            allow_synthetic_matrix: false,
            no_allow_synthetic_matrix: false,
            merge_interval_ms: None,
            max_concurrent_scans: None,
            seeds: None,
            block_number: None,
            state_root: None,
            disable_page_pir: false,
            enable_page_pir: false,
            page_domain_bits: None,
            page_rows_per_page: None,
            page_prg_key_0: None,
            page_prg_key_1: None,
            #[cfg(feature = "cuda")]
            gpu_device: None,
            #[cfg(feature = "cuda")]
            gpu_preload: false,
            #[cfg(feature = "cuda")]
            no_gpu_preload: false,
        }
    }
}

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

fn pick3<T>(cli: Option<T>, env: Option<T>, file: Option<T>) -> Option<T> {
    cli.or(env).or(file)
}

fn parse_environment(raw: &str) -> Result<Environment, StartupError> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "dev" => Ok(Environment::Dev),
        "test" => Ok(Environment::Test),
        "prod" | "production" => Ok(Environment::Prod),
        _ => Err(StartupError::new(format!(
            "invalid environment '{}', expected dev|test|prod",
            raw
        ))),
    }
}

fn parse_u64_triplet(raw: &str, field: &str) -> Result<[u64; 3], StartupError> {
    let parts: Vec<&str> = raw.split(',').map(|p| p.trim()).collect();
    if parts.len() != 3 {
        return Err(StartupError::new(format!(
            "{} must contain exactly 3 comma-separated values",
            field
        )));
    }

    let mut out = [0u64; 3];
    for (i, part) in parts.iter().enumerate() {
        out[i] = parse_u64_value(part, field)?;
    }
    Ok(out)
}

fn parse_u64_value(raw: &str, field: &str) -> Result<u64, StartupError> {
    if let Some(hex) = raw.strip_prefix("0x") {
        u64::from_str_radix(hex, 16)
            .map_err(|e| StartupError::new(format!("invalid {} value '{}': {}", field, raw, e)))
    } else {
        raw.parse::<u64>()
            .map_err(|e| StartupError::new(format!("invalid {} value '{}': {}", field, raw, e)))
    }
}

fn parse_fixed_hex<const N: usize>(raw: &str, field: &str) -> Result<[u8; N], StartupError> {
    let hex = raw.strip_prefix("0x").unwrap_or(raw);
    if hex.len() != N * 2 {
        return Err(StartupError::new(format!(
            "{} must be {} bytes ({} hex chars), got {} hex chars",
            field,
            N,
            N * 2,
            hex.len()
        )));
    }

    if !hex.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(StartupError::new(format!(
            "{} contains non-hex characters",
            field
        )));
    }

    let mut out = [0u8; N];
    hex::decode_to_slice(hex, &mut out)
        .map_err(|e| StartupError::new(format!("failed to decode {} as hex: {}", field, e)))?;
    Ok(out)
}

fn env_var(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

fn parse_env_usize(key: &str) -> Result<Option<usize>, StartupError> {
    let Some(raw) = env_var(key) else {
        return Ok(None);
    };
    raw.parse::<usize>()
        .map(Some)
        .map_err(|e| StartupError::new(format!("invalid {} value '{}': {}", key, raw, e)))
}

fn parse_env_usize_any(keys: &[&str]) -> Result<Option<usize>, StartupError> {
    let Some((legacy_key, preferred_keys)) = keys.split_last() else {
        return Ok(None);
    };

    if preferred_keys.is_empty() {
        return parse_env_usize(legacy_key);
    }

    let mut selected: Option<(&str, usize)> = None;
    for &key in preferred_keys {
        let Some(raw) = env_var(key) else {
            continue;
        };

        let value = raw
            .parse::<usize>()
            .map_err(|e| StartupError::new(format!("invalid {} value '{}': {}", key, raw, e)))?;

        if let Some((selected_key, selected_value)) = selected {
            if value != selected_value {
                return Err(StartupError::new(format!(
                    "conflicting {} ({}) and {} ({}) values; set only one or align them",
                    selected_key, selected_value, key, value
                )));
            }
        } else {
            selected = Some((key, value));
        }
    }

    if let Some((_, value)) = selected {
        return Ok(Some(value));
    }

    parse_env_usize(legacy_key)
}

fn parse_env_u64(key: &str) -> Result<Option<u64>, StartupError> {
    let Some(raw) = env_var(key) else {
        return Ok(None);
    };
    parse_u64_value(&raw, key).map(Some)
}

fn parse_env_bool(key: &str) -> Result<Option<bool>, StartupError> {
    let Some(raw) = env_var(key) else {
        return Ok(None);
    };

    let value = match raw.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => true,
        "0" | "false" | "no" | "off" => false,
        _ => {
            return Err(StartupError::new(format!(
                "invalid {} value '{}': expected true/false",
                key, raw
            )));
        }
    };

    Ok(Some(value))
}

fn validate_admin_mtls_proxy_trust(
    allowed_subjects: &[String],
    trust_proxy_headers: bool,
) -> Result<(), StartupError> {
    if !allowed_subjects.is_empty() && !trust_proxy_headers {
        return Err(StartupError::new(
            "MORPHOGEN_ADMIN_MTLS_ALLOWED_SUBJECTS requires MORPHOGEN_ADMIN_TRUST_PROXY_HEADERS=true",
        ));
    }
    Ok(())
}

fn parse_admin_mtls_subject_header(
    raw: Option<String>,
) -> Result<axum::http::HeaderName, StartupError> {
    let raw = raw.unwrap_or_else(|| DEFAULT_ADMIN_MTLS_SUBJECT_HEADER.to_string());
    axum::http::HeaderName::from_bytes(raw.as_bytes()).map_err(|_| {
        StartupError::new(format!(
            "invalid MORPHOGEN_ADMIN_MTLS_SUBJECT_HEADER '{}'",
            raw
        ))
    })
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
mod runtime_config_tests {
    use super::*;
    use std::fs;

    #[test]
    fn resolve_config_prefers_cli_then_env_then_file() {
        let mut cli = CliArgs::default_for_tests();
        cli.bind_addr = Some("127.0.0.1:4100".to_string());

        let mut env = EnvConfig::default_for_tests();
        env.bind_addr = Some("127.0.0.1:4200".to_string());

        let file = FileConfig {
            bind_addr: Some("127.0.0.1:4300".to_string()),
            ..FileConfig::default()
        };

        cli.environment = Some("dev".to_string());
        cli.allow_synthetic_matrix = true;
        cli.matrix_size_bytes = Some(4096);

        let resolved = RuntimeConfig::resolve(cli, env, Some(file)).expect("config should resolve");
        assert_eq!(resolved.bind_addr, "127.0.0.1:4100".parse().unwrap());
        assert!(!resolved.bind_addr_is_default);
    }

    #[test]
    fn resolve_config_defaults_to_loopback_bind_addr() {
        let mut cli = CliArgs::default_for_tests();
        cli.environment = Some("dev".to_string());
        cli.allow_synthetic_matrix = true;
        cli.matrix_size_bytes = Some(4096);

        let resolved = RuntimeConfig::resolve(cli, EnvConfig::default_for_tests(), None)
            .expect("config should resolve");
        assert_eq!(resolved.bind_addr, "127.0.0.1:3000".parse().unwrap());
        assert!(resolved.bind_addr_is_default);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn resolve_config_rejects_row_size_larger_than_gpu_page_size() {
        use morphogen_gpu_dpf::storage::PAGE_SIZE_BYTES;

        let mut cli = CliArgs::default_for_tests();
        cli.environment = Some("dev".to_string());
        cli.allow_synthetic_matrix = true;
        cli.row_size_bytes = Some(PAGE_SIZE_BYTES + 1);
        cli.chunk_size_bytes = Some((PAGE_SIZE_BYTES + 1) * 2);
        cli.matrix_size_bytes = Some((PAGE_SIZE_BYTES + 1) * 8);

        let err = RuntimeConfig::resolve(cli, EnvConfig::default_for_tests(), None)
            .expect_err("row size larger than PAGE_SIZE_BYTES should be rejected");
        assert!(err.to_string().contains("PAGE_SIZE_BYTES"));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn resolve_config_allows_row_size_equal_to_gpu_page_size() {
        use morphogen_gpu_dpf::storage::PAGE_SIZE_BYTES;

        let mut cli = CliArgs::default_for_tests();
        cli.environment = Some("dev".to_string());
        cli.allow_synthetic_matrix = true;
        cli.row_size_bytes = Some(PAGE_SIZE_BYTES);
        cli.chunk_size_bytes = Some(PAGE_SIZE_BYTES * 2);
        cli.matrix_size_bytes = Some(PAGE_SIZE_BYTES * 8);

        RuntimeConfig::resolve(cli, EnvConfig::default_for_tests(), None).expect(
            "row_size_bytes == PAGE_SIZE_BYTES should be accepted when page PIR is enabled",
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn resolve_config_allows_large_row_size_when_page_pir_disabled() {
        use morphogen_gpu_dpf::storage::PAGE_SIZE_BYTES;

        let mut cli = CliArgs::default_for_tests();
        cli.environment = Some("dev".to_string());
        cli.allow_synthetic_matrix = true;
        cli.disable_page_pir = true;
        cli.row_size_bytes = Some(PAGE_SIZE_BYTES + 1);
        cli.chunk_size_bytes = Some((PAGE_SIZE_BYTES + 1) * 2);
        cli.matrix_size_bytes = Some((PAGE_SIZE_BYTES + 1) * 8);

        let resolved = RuntimeConfig::resolve(cli, EnvConfig::default_for_tests(), None).expect(
            "row size larger than PAGE_SIZE_BYTES should be allowed when page PIR is disabled",
        );
        assert!(
            resolved.page_config.is_none(),
            "page PIR should be disabled in resolved runtime config"
        );
    }

    #[test]
    fn parse_admin_mtls_subject_header_rejects_invalid_name() {
        let err = parse_admin_mtls_subject_header(Some("bad header".to_string()))
            .expect_err("invalid header name should fail");
        assert!(err
            .to_string()
            .contains("MORPHOGEN_ADMIN_MTLS_SUBJECT_HEADER"));
    }

    #[test]
    fn validate_admin_mtls_proxy_trust_rejects_allowlist_without_opt_in() {
        let subjects = vec!["spiffe://morphogenesis/control-plane".to_string()];
        let err = validate_admin_mtls_proxy_trust(&subjects, false)
            .expect_err("mTLS allowlist without trusted-proxy opt-in should fail");
        assert!(err
            .to_string()
            .contains("MORPHOGEN_ADMIN_TRUST_PROXY_HEADERS=true"));
    }

    #[test]
    fn validate_admin_mtls_proxy_trust_accepts_safe_combinations() {
        let subjects = vec!["spiffe://morphogenesis/control-plane".to_string()];
        validate_admin_mtls_proxy_trust(&subjects, true).expect("explicit opt-in should pass");
        validate_admin_mtls_proxy_trust(&Vec::new(), false)
            .expect("empty mTLS allowlist should pass");
    }

    #[test]
    fn resolve_config_requires_matrix_source() {
        let cli = CliArgs::default_for_tests();
        let env = EnvConfig::default_for_tests();
        let err = RuntimeConfig::resolve(cli, env, None)
            .expect_err("should reject missing matrix source");
        assert!(
            err.to_string().contains("matrix_file"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn resolve_config_parses_page_prg_keys() {
        let mut cli = CliArgs::default_for_tests();
        cli.allow_synthetic_matrix = true;
        cli.matrix_size_bytes = Some(4096);
        cli.page_prg_key_0 = Some("00112233445566778899aabbccddeeff".to_string());
        cli.page_prg_key_1 = Some("ffeeddccbbaa99887766554433221100".to_string());

        let resolved = RuntimeConfig::resolve(cli, EnvConfig::default_for_tests(), None)
            .expect("config should resolve");
        let page_cfg = resolved.page_config.expect("page config should be enabled");

        assert_eq!(
            page_cfg.prg_keys[0],
            [
                0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd,
                0xee, 0xff,
            ]
        );
        assert_eq!(
            page_cfg.prg_keys[1],
            [
                0xff, 0xee, 0xdd, 0xcc, 0xbb, 0xaa, 0x99, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22,
                0x11, 0x00,
            ]
        );
    }

    #[test]
    fn resolve_config_rejects_bad_prg_key_len() {
        let mut cli = CliArgs::default_for_tests();
        cli.allow_synthetic_matrix = true;
        cli.matrix_size_bytes = Some(4096);
        cli.page_prg_key_0 = Some("1234".to_string());

        let err = RuntimeConfig::resolve(cli, EnvConfig::default_for_tests(), None)
            .expect_err("config should reject short key");
        assert!(err.to_string().contains("page_prg_key_0"));
    }

    #[test]
    fn resolve_config_prod_requires_explicit_page_prg_keys() {
        let mut cli = CliArgs::default_for_tests();
        cli.environment = Some("prod".to_string());
        cli.allow_synthetic_matrix = true;
        cli.matrix_size_bytes = Some(4096);

        let err = RuntimeConfig::resolve(cli, EnvConfig::default_for_tests(), None)
            .expect_err("prod should require page PRG keys");
        assert!(err.to_string().contains("page_prg_key_0"));
    }

    #[test]
    fn resolve_config_prod_rejects_zero_page_prg_keys() {
        let mut cli = CliArgs::default_for_tests();
        cli.environment = Some("prod".to_string());
        cli.allow_synthetic_matrix = true;
        cli.matrix_size_bytes = Some(4096);
        cli.page_prg_key_0 = Some("00000000000000000000000000000000".to_string());
        cli.page_prg_key_1 = Some("00000000000000000000000000000000".to_string());

        let err = RuntimeConfig::resolve(cli, EnvConfig::default_for_tests(), None)
            .expect_err("prod should reject zero PRG keys");
        assert!(err.to_string().contains("must be non-zero"));
    }

    #[test]
    fn cli_flag_overrides_support_explicit_enable_and_disable() {
        let mut cli = CliArgs::default_for_tests();
        assert_eq!(cli.allow_synthetic_matrix_override(), None);
        assert_eq!(cli.disable_page_pir_override(), None);

        cli.allow_synthetic_matrix = true;
        assert_eq!(cli.allow_synthetic_matrix_override(), Some(true));

        cli = CliArgs::default_for_tests();
        cli.no_allow_synthetic_matrix = true;
        assert_eq!(cli.allow_synthetic_matrix_override(), Some(false));

        cli = CliArgs::default_for_tests();
        cli.disable_page_pir = true;
        assert_eq!(cli.disable_page_pir_override(), Some(true));

        cli = CliArgs::default_for_tests();
        cli.enable_page_pir = true;
        assert_eq!(cli.disable_page_pir_override(), Some(false));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cli_gpu_preload_override_supports_explicit_enable_and_disable() {
        let mut cli = CliArgs::default_for_tests();
        assert_eq!(cli.gpu_preload_override(), None);

        cli.gpu_preload = true;
        assert_eq!(cli.gpu_preload_override(), Some(true));

        cli = CliArgs::default_for_tests();
        cli.no_gpu_preload = true;
        assert_eq!(cli.gpu_preload_override(), Some(false));
    }

    #[test]
    fn cli_rejects_conflicting_allow_synthetic_flags() {
        let result = CliArgs::try_parse_from([
            "server",
            "--allow-synthetic-matrix",
            "--no-allow-synthetic-matrix",
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn cli_rejects_conflicting_page_pir_flags() {
        let result = CliArgs::try_parse_from(["server", "--disable-page-pir", "--enable-page-pir"]);
        assert!(result.is_err());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cli_rejects_conflicting_gpu_preload_flags() {
        let result = CliArgs::try_parse_from(["server", "--gpu-preload", "--no-gpu-preload"]);
        assert!(result.is_err());
    }

    #[test]
    fn load_matrix_from_file_rejects_unaligned_size() {
        let path = unique_temp_path("morphogen_server_unaligned_matrix.bin");
        fs::write(&path, [0u8; 3]).expect("write temp matrix file");

        let err = match load_matrix_from_file(&path, 2, 2) {
            Ok(_) => panic!("file size must be aligned"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("divisible by row_size_bytes"));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn ctrl_c_failure_logging_policy_matches_expected_thresholds() {
        assert!(should_log_ctrl_c_failure(1));
        assert!(should_log_ctrl_c_failure(2));
        assert!(should_log_ctrl_c_failure(3));
        assert!(!should_log_ctrl_c_failure(4));
        assert!(should_log_ctrl_c_failure(10));
        assert!(should_log_ctrl_c_failure(20));
        assert!(!should_log_ctrl_c_failure(21));
    }

    #[test]
    fn ctrl_c_failure_forced_shutdown_policy_matches_expected_thresholds() {
        assert!(!should_force_shutdown_after_ctrl_c_failures(0));
        assert!(!should_force_shutdown_after_ctrl_c_failures(
            CTRL_C_MAX_FAILURES_BEFORE_FORCED_SHUTDOWN - 1
        ));
        assert!(should_force_shutdown_after_ctrl_c_failures(
            CTRL_C_MAX_FAILURES_BEFORE_FORCED_SHUTDOWN
        ));

        #[cfg(unix)]
        assert!(!ctrl_c_failures_force_shutdown());
        #[cfg(not(unix))]
        assert!(ctrl_c_failures_force_shutdown());
    }

    #[test]
    fn ctrl_c_retry_delay_backoff_is_capped() {
        assert_eq!(
            next_ctrl_c_retry_delay_ms(CTRL_C_INITIAL_RETRY_DELAY_MS),
            CTRL_C_INITIAL_RETRY_DELAY_MS * 2
        );
        assert_eq!(
            next_ctrl_c_retry_delay_ms(CTRL_C_MAX_RETRY_DELAY_MS),
            CTRL_C_MAX_RETRY_DELAY_MS
        );
        assert_eq!(
            next_ctrl_c_retry_delay_ms(CTRL_C_MAX_RETRY_DELAY_MS / 2),
            CTRL_C_MAX_RETRY_DELAY_MS
        );
    }

    #[test]
    fn parse_env_usize_any_prefers_prefixed_and_falls_back_to_legacy() {
        let key_a = "MORPHOGEN_SERVER_A_MAX_CONCURRENT_SCANS_TEST";
        let key_b = "MORPHOGEN_SERVER_B_MAX_CONCURRENT_SCANS_TEST";
        let key_legacy = "MORPHOGEN_SERVER_MAX_CONCURRENT_SCANS_TEST";
        std::env::remove_var(key_a);
        std::env::remove_var(key_b);
        std::env::remove_var(key_legacy);

        std::env::set_var(key_legacy, "19");
        let only_legacy =
            parse_env_usize_any(&[key_a, key_b, key_legacy]).expect("legacy parse should succeed");
        assert_eq!(only_legacy, Some(19));

        std::env::set_var(key_b, "23");
        let with_prefixed_b = parse_env_usize_any(&[key_a, key_b, key_legacy])
            .expect("prefixed parse should succeed");
        assert_eq!(with_prefixed_b, Some(23));

        std::env::remove_var(key_b);
        std::env::set_var(key_a, "29");
        let with_prefixed_a = parse_env_usize_any(&[key_a, key_b, key_legacy])
            .expect("prefixed parse should succeed");
        assert_eq!(with_prefixed_a, Some(29));

        std::env::remove_var(key_a);
        std::env::remove_var(key_b);
        std::env::remove_var(key_legacy);
    }

    #[test]
    fn parse_env_usize_any_rejects_conflicting_prefixed_values() {
        let key_a = "MORPHOGEN_SERVER_A_MAX_CONCURRENT_SCANS_CONFLICT";
        let key_b = "MORPHOGEN_SERVER_B_MAX_CONCURRENT_SCANS_CONFLICT";
        let key_legacy = "MORPHOGEN_SERVER_MAX_CONCURRENT_SCANS_CONFLICT";
        std::env::remove_var(key_a);
        std::env::remove_var(key_b);
        std::env::remove_var(key_legacy);

        std::env::set_var(key_a, "21");
        std::env::set_var(key_b, "34");
        let err = parse_env_usize_any(&[key_a, key_b, key_legacy])
            .expect_err("conflicting prefixed values should fail startup");
        let err_text = err.to_string();
        assert!(err_text.contains("conflicting"));
        assert!(err_text.contains(key_a));
        assert!(err_text.contains(key_b));

        std::env::remove_var(key_a);
        std::env::remove_var(key_b);
        std::env::remove_var(key_legacy);
    }

    #[test]
    fn parse_env_usize_any_prefixed_wins_when_legacy_conflicts() {
        let key_a = "MORPHOGEN_SERVER_A_MAX_CONCURRENT_SCANS_LEGACY_CONFLICT";
        let key_b = "MORPHOGEN_SERVER_B_MAX_CONCURRENT_SCANS_LEGACY_CONFLICT";
        let key_legacy = "MORPHOGEN_SERVER_MAX_CONCURRENT_SCANS_LEGACY_CONFLICT";
        std::env::remove_var(key_a);
        std::env::remove_var(key_b);
        std::env::remove_var(key_legacy);

        std::env::set_var(key_b, "31");
        std::env::set_var(key_legacy, "19");
        let parsed = parse_env_usize_any(&[key_a, key_b, key_legacy])
            .expect("prefixed value should take precedence over legacy");
        assert_eq!(parsed, Some(31));

        std::env::remove_var(key_a);
        std::env::remove_var(key_b);
        std::env::remove_var(key_legacy);
    }

    #[test]
    fn parse_env_usize_any_ignores_malformed_legacy_when_prefixed_present() {
        let key_a = "MORPHOGEN_SERVER_A_MAX_CONCURRENT_SCANS_MALFORMED_LEGACY";
        let key_b = "MORPHOGEN_SERVER_B_MAX_CONCURRENT_SCANS_MALFORMED_LEGACY";
        let key_legacy = "MORPHOGEN_SERVER_MAX_CONCURRENT_SCANS_MALFORMED_LEGACY";
        std::env::remove_var(key_a);
        std::env::remove_var(key_b);
        std::env::remove_var(key_legacy);

        std::env::set_var(key_a, "29");
        std::env::set_var(key_legacy, "not-a-number");
        let parsed = parse_env_usize_any(&[key_a, key_b, key_legacy])
            .expect("prefixed value should win when legacy fallback is malformed");
        assert_eq!(parsed, Some(29));

        std::env::remove_var(key_a);
        std::env::remove_var(key_b);
        std::env::remove_var(key_legacy);
    }

    #[test]
    fn parse_env_usize_any_rejects_malformed_preferred_even_with_legacy() {
        let key_a = "MORPHOGEN_SERVER_A_MAX_CONCURRENT_SCANS_MALFORMED_PREFERRED";
        let key_b = "MORPHOGEN_SERVER_B_MAX_CONCURRENT_SCANS_MALFORMED_PREFERRED";
        let key_legacy = "MORPHOGEN_SERVER_MAX_CONCURRENT_SCANS_MALFORMED_PREFERRED";
        std::env::remove_var(key_a);
        std::env::remove_var(key_b);
        std::env::remove_var(key_legacy);

        std::env::set_var(key_a, "not-a-number");
        std::env::set_var(key_legacy, "19");
        let err = parse_env_usize_any(&[key_a, key_b, key_legacy])
            .expect_err("malformed preferred key should fail startup");
        assert!(err.to_string().contains(key_a));

        std::env::remove_var(key_a);
        std::env::remove_var(key_b);
        std::env::remove_var(key_legacy);
    }

    fn expected_ctrl_c_retry_durations(retry_count: u32) -> Vec<Duration> {
        let mut delay_ms = CTRL_C_INITIAL_RETRY_DELAY_MS;
        let mut durations = Vec::with_capacity(retry_count as usize);

        for _ in 0..retry_count {
            durations.push(Duration::from_millis(delay_ms));
            delay_ms = next_ctrl_c_retry_delay_ms(delay_ms);
        }

        durations
    }

    #[tokio::test]
    async fn wait_for_ctrl_c_signal_with_retries_until_success() {
        use std::collections::VecDeque;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Mutex;

        let attempts = Arc::new(AtomicUsize::new(0));
        let sleep_durations = Arc::new(Mutex::new(Vec::new()));
        let outcomes = Arc::new(Mutex::new(VecDeque::from(vec![
            Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "transient ctrl-c stream failure 1",
            )),
            Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "transient ctrl-c stream failure 2",
            )),
            Ok(()),
        ])));

        wait_for_ctrl_c_signal_with(
            {
                let attempts = Arc::clone(&attempts);
                let outcomes = Arc::clone(&outcomes);
                move || {
                    attempts.fetch_add(1, Ordering::Relaxed);
                    let next = outcomes
                        .lock()
                        .expect("lock outcomes")
                        .pop_front()
                        .expect("must provide enough outcomes");
                    async move { next }
                }
            },
            {
                let sleep_durations = Arc::clone(&sleep_durations);
                move |duration| {
                    let sleep_durations = Arc::clone(&sleep_durations);
                    async move {
                        sleep_durations
                            .lock()
                            .expect("lock sleep_durations")
                            .push(duration);
                    }
                }
            },
        )
        .await;

        assert_eq!(attempts.load(Ordering::Relaxed), 3);
        assert_eq!(
            sleep_durations
                .lock()
                .expect("lock sleep_durations")
                .clone(),
            expected_ctrl_c_retry_durations(2)
        );
    }

    #[cfg(not(unix))]
    #[tokio::test]
    async fn wait_for_ctrl_c_signal_with_forces_shutdown_after_persistent_failures() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Mutex;

        let attempts = Arc::new(AtomicUsize::new(0));
        let sleep_durations = Arc::new(Mutex::new(Vec::new()));

        wait_for_ctrl_c_signal_with(
            {
                let attempts = Arc::clone(&attempts);
                move || {
                    attempts.fetch_add(1, Ordering::Relaxed);
                    async {
                        Err(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            "persistent ctrl-c stream failure",
                        ))
                    }
                }
            },
            {
                let sleep_durations = Arc::clone(&sleep_durations);
                move |duration| {
                    let sleep_durations = Arc::clone(&sleep_durations);
                    async move {
                        sleep_durations
                            .lock()
                            .expect("lock sleep_durations")
                            .push(duration);
                    }
                }
            },
        )
        .await;

        assert_eq!(
            attempts.load(Ordering::Relaxed) as u32,
            CTRL_C_MAX_FAILURES_BEFORE_FORCED_SHUTDOWN
        );
        assert_eq!(
            sleep_durations
                .lock()
                .expect("lock sleep_durations")
                .clone(),
            expected_ctrl_c_retry_durations(CTRL_C_MAX_FAILURES_BEFORE_FORCED_SHUTDOWN - 1)
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn wait_for_ctrl_c_signal_with_disables_waiter_after_persistent_failures() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Mutex;

        let attempts = Arc::new(AtomicUsize::new(0));
        let sleep_durations = Arc::new(Mutex::new(Vec::new()));
        let waiter = wait_for_ctrl_c_signal_with(
            {
                let attempts = Arc::clone(&attempts);
                move || {
                    attempts.fetch_add(1, Ordering::Relaxed);
                    async {
                        Err(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            "persistent ctrl-c stream failure",
                        ))
                    }
                }
            },
            {
                let sleep_durations = Arc::clone(&sleep_durations);
                move |duration| {
                    let sleep_durations = Arc::clone(&sleep_durations);
                    async move {
                        sleep_durations
                            .lock()
                            .expect("lock sleep_durations")
                            .push(duration);
                    }
                }
            },
        );

        let timeout = tokio::time::timeout(Duration::from_millis(25), waiter).await;
        assert!(
            timeout.is_err(),
            "ctrl+c waiter should stay pending on unix"
        );
        assert_eq!(
            attempts.load(Ordering::Relaxed) as u32,
            CTRL_C_MAX_FAILURES_BEFORE_FORCED_SHUTDOWN
        );
        assert_eq!(
            sleep_durations
                .lock()
                .expect("lock sleep_durations")
                .clone(),
            expected_ctrl_c_retry_durations(CTRL_C_MAX_FAILURES_BEFORE_FORCED_SHUTDOWN - 1)
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn drive_shutdown_from_futures_sets_channel_on_ctrl_c() {
        let (tx, rx) = watch::channel(false);
        drive_shutdown_from_futures(async {}, std::future::pending::<()>(), tx).await;
        assert!(*rx.borrow());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn drive_shutdown_from_futures_sets_channel_on_sigterm() {
        let (tx, rx) = watch::channel(false);
        drive_shutdown_from_futures(std::future::pending::<()>(), async {}, tx).await;
        assert!(*rx.borrow());
    }

    #[cfg(not(unix))]
    #[tokio::test]
    async fn drive_shutdown_from_futures_sets_channel_on_ctrl_c() {
        let (tx, rx) = watch::channel(false);
        drive_shutdown_from_futures(async {}, tx).await;
        assert!(*rx.borrow());
    }

    fn unique_temp_path(name: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        std::env::temp_dir().join(format!("{}_{}", nanos, name))
    }
}
