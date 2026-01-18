use morphogen_core::ROW_SIZE_BYTES;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Environment {
    Dev,
    Test,
    Prod,
}

#[derive(Clone, Debug)]
pub struct ServerConfig {
    pub environment: Environment,
    pub row_size_bytes: usize,
    pub chunk_size_bytes: usize,
    pub matrix_size_bytes: usize,
    pub bench_fill_seed: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    InvalidRowSize { row_size: usize },
    ChunkNotAlignedToRow { chunk_size: usize, row_size: usize },
    MatrixNotAlignedToRow { matrix_size: usize, row_size: usize },
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::InvalidRowSize { row_size } => {
                write!(f, "row_size_bytes must be > 0, got {}", row_size)
            }
            ConfigError::ChunkNotAlignedToRow {
                chunk_size,
                row_size,
            } => {
                write!(
                    f,
                    "chunk_size_bytes ({}) must be divisible by row_size_bytes ({})",
                    chunk_size, row_size
                )
            }
            ConfigError::MatrixNotAlignedToRow {
                matrix_size,
                row_size,
            } => {
                write!(
                    f,
                    "matrix_size_bytes ({}) must be divisible by row_size_bytes ({})",
                    matrix_size, row_size
                )
            }
        }
    }
}

impl std::error::Error for ConfigError {}

impl ServerConfig {
    pub fn for_env(environment: Environment) -> Self {
        match environment {
            Environment::Dev => Self {
                environment,
                row_size_bytes: ROW_SIZE_BYTES,
                chunk_size_bytes: 16 * 1024 * 1024,
                matrix_size_bytes: 256 * 1024 * 1024,
                bench_fill_seed: None,
            },
            Environment::Test => Self {
                environment,
                row_size_bytes: ROW_SIZE_BYTES,
                chunk_size_bytes: 4 * 1024 * 1024,
                matrix_size_bytes: 32 * 1024 * 1024,
                bench_fill_seed: None,
            },
            Environment::Prod => Self {
                environment,
                row_size_bytes: ROW_SIZE_BYTES,
                chunk_size_bytes: 1024 * 1024 * 1024,
                matrix_size_bytes: 300 * 1024 * 1024 * 1024,
                bench_fill_seed: None,
            },
        }
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.row_size_bytes == 0 {
            return Err(ConfigError::InvalidRowSize {
                row_size: self.row_size_bytes,
            });
        }
        if !self.chunk_size_bytes.is_multiple_of(self.row_size_bytes) {
            return Err(ConfigError::ChunkNotAlignedToRow {
                chunk_size: self.chunk_size_bytes,
                row_size: self.row_size_bytes,
            });
        }
        if !self.matrix_size_bytes.is_multiple_of(self.row_size_bytes) {
            return Err(ConfigError::MatrixNotAlignedToRow {
                matrix_size: self.matrix_size_bytes,
                row_size: self.row_size_bytes,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_rejects_zero_row_size() {
        let config = ServerConfig {
            environment: Environment::Test,
            row_size_bytes: 0,
            chunk_size_bytes: 32,
            matrix_size_bytes: 64,
            bench_fill_seed: None,
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::InvalidRowSize { .. })
        ));
    }

    #[test]
    fn validate_rejects_unaligned_chunk_size() {
        let config = ServerConfig {
            environment: Environment::Test,
            row_size_bytes: 4,
            chunk_size_bytes: 30, // not divisible by 4
            matrix_size_bytes: 64,
            bench_fill_seed: None,
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::ChunkNotAlignedToRow { .. })
        ));
    }

    #[test]
    fn validate_rejects_unaligned_matrix_size() {
        let config = ServerConfig {
            environment: Environment::Test,
            row_size_bytes: 4,
            chunk_size_bytes: 32,
            matrix_size_bytes: 65, // not divisible by 4
            bench_fill_seed: None,
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::MatrixNotAlignedToRow { .. })
        ));
    }

    #[test]
    fn validate_accepts_aligned_config() {
        let config = ServerConfig {
            environment: Environment::Test,
            row_size_bytes: 4,
            chunk_size_bytes: 32,
            matrix_size_bytes: 64,
            bench_fill_seed: None,
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn for_env_configs_are_valid() {
        for env in [Environment::Dev, Environment::Test, Environment::Prod] {
            let config = ServerConfig::for_env(env);
            assert!(
                config.validate().is_ok(),
                "{:?} config should be valid",
                env
            );
        }
    }
}
