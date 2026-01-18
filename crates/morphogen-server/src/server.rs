use std::sync::Arc;

use morphogen_core::{DeltaBuffer, EpochSnapshot, GlobalState};
use morphogen_dpf::DpfKey;
use morphogen_storage::ChunkedMatrix;

use crate::config::ServerConfig;
use crate::epoch::{try_build_next_snapshot, MergeError};
#[cfg(feature = "parallel")]
use crate::scan::scan_consistent_parallel;
use crate::scan::{scan_consistent, ScanError};

pub struct MorphogenServer {
    config: ServerConfig,
    state: GlobalState,
    pending: Arc<DeltaBuffer>,
}

impl MorphogenServer {
    pub fn new(config: ServerConfig) -> Self {
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

        Self {
            config,
            state: GlobalState::new(snapshot),
            pending: Arc::new(DeltaBuffer::new(row_size_bytes)),
        }
    }

    pub fn scan<K: DpfKey>(&self, keys: &[K; 3]) -> Result<([Vec<u8>; 3], u64), ScanError> {
        scan_consistent(
            &self.state,
            self.pending.as_ref(),
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
            self.pending.as_ref(),
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
        self.pending.push(row_idx, diff)
    }

    pub fn try_merge_epoch(&mut self) -> Result<u64, MergeError> {
        let current = self.state.load();
        let next_epoch_id = current.epoch_id + 1;
        let next_snapshot =
            try_build_next_snapshot(current.as_ref(), self.pending.as_ref(), next_epoch_id)?;

        self.state.store(Arc::new(next_snapshot));
        self.pending = Arc::new(DeltaBuffer::new(self.config.row_size_bytes));
        Ok(next_epoch_id)
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
        let mut server = MorphogenServer::new(test_config());
        server.submit_update(0, vec![1, 2, 3, 4]).unwrap();

        let result = server.try_merge_epoch();
        assert_eq!(result.unwrap(), 1);
    }

    #[test]
    fn try_merge_epoch_errors_on_oob_row() {
        let mut server = MorphogenServer::new(test_config());
        server.submit_update(1000, vec![1, 2, 3, 4]).unwrap();

        let result = server.try_merge_epoch();
        assert!(result.is_err());
    }
}
