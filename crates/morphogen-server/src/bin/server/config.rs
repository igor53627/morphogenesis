//! Server startup config: CLI / env / file layering and resolution.
//!
//! Extracted from `bin/server/main.rs` in TASK-55.3. Holds StartupError,
//! CliArgs, FileConfig, EnvConfig, PagePirRuntimeConfig, RuntimeConfig +
//! their impls, plus validate_server_config / build_page_config. Uses the
//! parse helpers in `super::config_helpers` (already extracted in 55.2).
//!
//! Referenced from main.rs via `mod config; use config::*;`.

use super::config_helpers::*;
use super::*;

#[derive(Debug, Clone)]
pub(super) struct StartupError {
    pub(super) message: String,
}

impl StartupError {
    pub(super) fn new(message: impl Into<String>) -> Self {
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
pub(super) struct CliArgs {
    /// Optional JSON config file path.
    #[arg(long)]
    pub(super) config: Option<PathBuf>,

    /// Bind address, for example 127.0.0.1:3000.
    #[arg(long)]
    pub(super) bind_addr: Option<String>,

    /// Environment profile: dev, test, or prod.
    #[arg(long)]
    pub(super) environment: Option<String>,

    /// Row size in bytes.
    #[arg(long)]
    pub(super) row_size_bytes: Option<usize>,

    /// Chunk size in bytes.
    #[arg(long)]
    pub(super) chunk_size_bytes: Option<usize>,

    /// Synthetic matrix size in bytes.
    #[arg(long)]
    pub(super) matrix_size_bytes: Option<usize>,

    /// Path to a matrix file to load at startup.
    #[arg(long)]
    pub(super) matrix_file: Option<PathBuf>,

    /// Seed used for synthetic matrix generation.
    #[arg(long)]
    pub(super) matrix_seed: Option<u64>,

    /// Allow synthetic matrix generation when no matrix file is provided.
    #[arg(long, action = ArgAction::SetTrue)]
    pub(super) allow_synthetic_matrix: bool,

    /// Forbid synthetic matrix generation even if configured in file/env.
    #[arg(long = "no-allow-synthetic-matrix", action = ArgAction::SetTrue, conflicts_with = "allow_synthetic_matrix")]
    pub(super) no_allow_synthetic_matrix: bool,

    /// Merge worker interval in milliseconds.
    #[arg(long)]
    pub(super) merge_interval_ms: Option<u64>,

    /// Maximum concurrent scan requests.
    #[arg(long)]
    pub(super) max_concurrent_scans: Option<usize>,

    /// Comma-separated epoch seeds (three u64 values).
    #[arg(long)]
    pub(super) seeds: Option<String>,

    /// Initial block number reported by metadata endpoint.
    #[arg(long)]
    pub(super) block_number: Option<u64>,

    /// Initial state root (32-byte hex string).
    #[arg(long)]
    pub(super) state_root: Option<String>,

    /// Disable page PIR metadata and endpoints.
    #[arg(long, action = ArgAction::SetTrue)]
    pub(super) disable_page_pir: bool,

    /// Force-enable page PIR metadata and endpoints.
    #[arg(long, action = ArgAction::SetTrue, conflicts_with = "disable_page_pir")]
    pub(super) enable_page_pir: bool,

    /// Page PIR domain bits.
    #[arg(long)]
    pub(super) page_domain_bits: Option<usize>,

    /// Rows per page in page PIR metadata.
    #[arg(long)]
    pub(super) page_rows_per_page: Option<usize>,

    /// First 16-byte PRG key as hex.
    #[arg(long)]
    pub(super) page_prg_key_0: Option<String>,

    /// Second 16-byte PRG key as hex.
    #[arg(long)]
    pub(super) page_prg_key_1: Option<String>,

    /// CUDA device ordinal.
    #[cfg(feature = "cuda")]
    #[arg(long)]
    pub(super) gpu_device: Option<usize>,

    /// If true, preload GPU matrix from CPU matrix at startup.
    #[cfg(feature = "cuda")]
    #[arg(long, action = ArgAction::SetTrue)]
    pub(super) gpu_preload: bool,

    /// Disable GPU matrix preload even when enabled in file/env.
    #[cfg(feature = "cuda")]
    #[arg(long = "no-gpu-preload", action = ArgAction::SetTrue, conflicts_with = "gpu_preload")]
    pub(super) no_gpu_preload: bool,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub(super) struct FileConfig {
    pub(super) bind_addr: Option<String>,
    pub(super) environment: Option<String>,
    pub(super) row_size_bytes: Option<usize>,
    pub(super) chunk_size_bytes: Option<usize>,
    pub(super) matrix_size_bytes: Option<usize>,
    pub(super) matrix_file: Option<PathBuf>,
    pub(super) matrix_seed: Option<u64>,
    pub(super) allow_synthetic_matrix: Option<bool>,
    pub(super) merge_interval_ms: Option<u64>,
    pub(super) max_concurrent_scans: Option<usize>,
    pub(super) seeds: Option<String>,
    pub(super) block_number: Option<u64>,
    pub(super) state_root: Option<String>,
    pub(super) disable_page_pir: Option<bool>,
    pub(super) page_domain_bits: Option<usize>,
    pub(super) page_rows_per_page: Option<usize>,
    pub(super) page_prg_key_0: Option<String>,
    pub(super) page_prg_key_1: Option<String>,
    #[cfg(feature = "cuda")]
    pub(super) gpu_device: Option<usize>,
    #[cfg(feature = "cuda")]
    pub(super) gpu_preload: Option<bool>,
}

#[derive(Debug, Clone, Default)]
pub(super) struct EnvConfig {
    pub(super) config: Option<PathBuf>,
    pub(super) bind_addr: Option<String>,
    pub(super) environment: Option<String>,
    pub(super) row_size_bytes: Option<usize>,
    pub(super) chunk_size_bytes: Option<usize>,
    pub(super) matrix_size_bytes: Option<usize>,
    pub(super) matrix_file: Option<PathBuf>,
    pub(super) matrix_seed: Option<u64>,
    pub(super) allow_synthetic_matrix: Option<bool>,
    pub(super) merge_interval_ms: Option<u64>,
    pub(super) max_concurrent_scans: Option<usize>,
    pub(super) seeds: Option<String>,
    pub(super) block_number: Option<u64>,
    pub(super) state_root: Option<String>,
    pub(super) disable_page_pir: Option<bool>,
    pub(super) page_domain_bits: Option<usize>,
    pub(super) page_rows_per_page: Option<usize>,
    pub(super) page_prg_key_0: Option<String>,
    pub(super) page_prg_key_1: Option<String>,
    #[cfg(feature = "cuda")]
    pub(super) gpu_device: Option<usize>,
    #[cfg(feature = "cuda")]
    pub(super) gpu_preload: Option<bool>,
}

impl EnvConfig {
    pub(super) fn from_process_env() -> Result<Self, StartupError> {
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
    pub(super) fn default_for_tests() -> Self {
        Self::default()
    }
}

#[derive(Debug, Clone)]
pub(super) struct PagePirRuntimeConfig {
    pub(super) domain_bits: usize,
    pub(super) rows_per_page: usize,
    pub(super) prg_keys: [[u8; 16]; 2],
}

#[derive(Debug, Clone)]
pub(super) struct RuntimeConfig {
    pub(super) bind_addr: SocketAddr,
    pub(super) bind_addr_is_default: bool,
    pub(super) environment: Environment,
    pub(super) row_size_bytes: usize,
    pub(super) chunk_size_bytes: usize,
    pub(super) matrix_size_bytes: Option<usize>,
    pub(super) matrix_file: Option<PathBuf>,
    pub(super) matrix_seed: u64,
    pub(super) merge_interval: Duration,
    pub(super) max_concurrent_scans: usize,
    pub(super) seeds: [u64; 3],
    pub(super) block_number: u64,
    pub(super) state_root: [u8; 32],
    pub(super) page_config: Option<PagePirRuntimeConfig>,
    #[cfg(feature = "cuda")]
    pub(super) gpu_device: usize,
    #[cfg(feature = "cuda")]
    pub(super) gpu_preload: bool,
}

impl RuntimeConfig {
    pub(super) fn resolve(
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
    pub(super) fn allow_synthetic_matrix_override(&self) -> Option<bool> {
        if self.allow_synthetic_matrix {
            Some(true)
        } else if self.no_allow_synthetic_matrix {
            Some(false)
        } else {
            None
        }
    }

    pub(super) fn disable_page_pir_override(&self) -> Option<bool> {
        if self.disable_page_pir {
            Some(true)
        } else if self.enable_page_pir {
            Some(false)
        } else {
            None
        }
    }

    #[cfg(feature = "cuda")]
    pub(super) fn gpu_preload_override(&self) -> Option<bool> {
        if self.gpu_preload {
            Some(true)
        } else if self.no_gpu_preload {
            Some(false)
        } else {
            None
        }
    }

    #[cfg(test)]
    pub(super) fn default_for_tests() -> Self {
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
