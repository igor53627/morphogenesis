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
}
