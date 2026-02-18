use std::sync::Arc;

use morphogen_core::{DeltaBuffer, EpochSnapshot, GlobalState};
use morphogen_dpf::DpfKey;
use morphogen_storage::ChunkedMatrix;

use crate::config::{ConfigError, ServerConfig};
use crate::epoch::{try_build_next_snapshot, MergeError};
#[cfg(feature = "parallel")]
use crate::scan::scan_consistent_parallel;
use crate::scan::{scan_consistent, ScanError};

pub struct MorphogenServer {
    config: ServerConfig,
    state: GlobalState,
}

impl MorphogenServer {
    pub fn new(config: ServerConfig) -> Result<Self, ConfigError> {
        config.validate()?;

        let row_size_bytes = config.row_size_bytes;
        let mut matrix = Arc::new(ChunkedMatrix::new(
            config.matrix_size_bytes,
            config.chunk_size_bytes,
        ));
        if let Some(seed) = config.bench_fill_seed {
            let matrix_mut = Arc::get_mut(&mut matrix)
                .expect("matrix must be uniquely owned during initialization");
            matrix_mut.fill_with_pattern(seed);
        }
        let snapshot = Arc::new(EpochSnapshot {
            epoch_id: 0,
            matrix,
        });

        let pending = Arc::new(DeltaBuffer::new(row_size_bytes));

        Ok(Self {
            config,
            state: GlobalState::new(snapshot, pending),
        })
    }

    pub fn scan<K: DpfKey>(&self, keys: &[K; 3]) -> Result<([Vec<u8>; 3], u64), ScanError> {
        scan_consistent(
            &self.state,
            self.state.load_pending().as_ref(),
            keys,
            self.config.row_size_bytes,
        )
    }

    #[cfg(feature = "parallel")]
    pub fn scan_parallel<K: DpfKey + Sync>(
        &self,
        keys: &[K; 3],
    ) -> Result<([Vec<u8>; 3], u64), ScanError> {
        scan_consistent_parallel(
            &self.state,
            self.state.load_pending().as_ref(),
            keys,
            self.config.row_size_bytes,
        )
    }

    pub fn warmup_matrix(&self) -> u64 {
        let snapshot = self.state.load();
        let matrix = snapshot.matrix.as_ref();
        let mut acc = 0u64;
        for chunk in matrix.chunks() {
            let data = chunk.as_slice();
            let mut i = 0usize;
            while i < data.len() {
                unsafe {
                    acc = acc.wrapping_add(std::ptr::read_volatile(data.as_ptr().add(i)) as u64);
                }
                i = i.saturating_add(4096);
            }
        }
        acc
    }

    pub fn submit_update(
        &self,
        row_idx: usize,
        diff: Vec<u8>,
    ) -> Result<(), morphogen_core::DeltaError> {
        self.state.load_pending().push(row_idx, diff)
    }

    pub fn try_merge_epoch(&mut self) -> Result<u64, MergeError> {
        let current = self.state.load();
        let pending = self.state.load_pending();
        let next_epoch_id = current.epoch_id + 1;
        let next_snapshot =
            try_build_next_snapshot(current.as_ref(), pending.as_ref(), next_epoch_id)?;

        self.state.store(Arc::new(next_snapshot));
        self.state
            .store_pending(Arc::new(DeltaBuffer::new(self.config.row_size_bytes)));
        Ok(next_epoch_id)
    }

    /// Submits a full snapshot from an external source (seed rotation).
    pub fn submit_snapshot(&self, epoch_id: u64, matrix: Arc<ChunkedMatrix>) {
        let snapshot = Arc::new(EpochSnapshot { epoch_id, matrix });
        self.state.store(snapshot);
        // Reset pending buffer for the new epoch
        self.state
            .store_pending(Arc::new(DeltaBuffer::new_with_epoch(
                self.config.row_size_bytes,
                epoch_id,
            )));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> ServerConfig {
        ServerConfig {
            environment: crate::config::Environment::Test,
            matrix_size_bytes: 64,
            chunk_size_bytes: 32,
            row_size_bytes: 4,
            bench_fill_seed: None,
        }
    }

    #[test]
    fn try_merge_epoch_returns_new_epoch_id() {
        let mut server = MorphogenServer::new(test_config()).unwrap();
        server.submit_update(0, vec![1, 2, 3, 4]).unwrap();

        let result = server.try_merge_epoch();
        assert_eq!(result.unwrap(), 1);
    }

    #[test]
    fn try_merge_epoch_errors_on_oob_row() {
        let mut server = MorphogenServer::new(test_config()).unwrap();
        server.submit_update(1000, vec![1, 2, 3, 4]).unwrap();

        let result = server.try_merge_epoch();
        assert!(result.is_err());
    }

    #[test]
    fn new_rejects_unaligned_config() {
        let config = ServerConfig {
            environment: crate::config::Environment::Test,
            matrix_size_bytes: 65, // not divisible by 4
            chunk_size_bytes: 32,
            row_size_bytes: 4,
            bench_fill_seed: None,
        };
        assert!(MorphogenServer::new(config).is_err());
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn scan_parallel_matches_scan() {
        use morphogen_dpf::AesDpfKey;

        let server = MorphogenServer::new(test_config()).unwrap();
        let mut rng = rand::thread_rng();
        let (key0, _) = AesDpfKey::generate_pair(&mut rng, 0);
        let (key1, _) = AesDpfKey::generate_pair(&mut rng, 1);
        let (key2, _) = AesDpfKey::generate_pair(&mut rng, 2);
        let keys = [key0, key1, key2];

        let (scan_results, scan_epoch) = server.scan(&keys).expect("scan should succeed");
        let (parallel_results, parallel_epoch) = server
            .scan_parallel(&keys)
            .expect("parallel scan should succeed");

        assert_eq!(parallel_epoch, scan_epoch);
        assert_eq!(parallel_results, scan_results);
    }
}
