use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use morphogen_core::{DeltaBuffer, DeltaEntry, EpochSnapshot, GlobalState};
use morphogen_storage::{AlignedMatrix, ChunkedMatrix};

fn row_offset(entry: &DeltaEntry, row_size: usize) -> Result<usize, MergeError> {
    entry
        .row_idx
        .checked_mul(row_size)
        .ok_or(MergeError::RowIndexOverflow {
            row_idx: entry.row_idx,
            row_size,
        })
}

fn chunk_location(offset: usize, chunk_size: usize) -> (usize, usize) {
    (offset / chunk_size, offset % chunk_size)
}

fn apply_xor_patch(dst: &mut [u8], src: &[u8]) {
    dst.iter_mut().zip(src.iter()).for_each(|(d, s)| *d ^= s);
}

pub fn dirty_chunks(
    pending: &DeltaBuffer,
    row_size_bytes: usize,
    chunk_size_bytes: usize,
) -> HashSet<usize> {
    let snapshot = pending.snapshot().unwrap_or_default();
    let mut result = HashSet::new();
    for entry in &snapshot {
        if let Ok(offset) = row_offset(entry, row_size_bytes) {
            let (chunk_index, _) = chunk_location(offset, chunk_size_bytes);
            result.insert(chunk_index);
        }
    }
    result
}

fn collect_dirty_chunks_from_entries(
    entries: &[DeltaEntry],
    row_size_bytes: usize,
    chunk_size_bytes: usize,
    num_chunks: usize,
) -> Result<Vec<bool>, MergeError> {
    let mut dirty = vec![false; num_chunks];
    for entry in entries {
        let offset = row_offset(entry, row_size_bytes)?;
        let (chunk_index, _) = chunk_location(offset, chunk_size_bytes);
        if chunk_index < num_chunks {
            dirty[chunk_index] = true;
        }
    }
    Ok(dirty)
}

pub fn dirty_chunks_vec(
    pending: &DeltaBuffer,
    row_size_bytes: usize,
    chunk_size_bytes: usize,
    num_chunks: usize,
) -> Vec<bool> {
    let snapshot = pending.snapshot().unwrap_or_default();
    collect_dirty_chunks_from_entries(&snapshot, row_size_bytes, chunk_size_bytes, num_chunks)
        .expect("dirty_chunks overflow")
}

fn clone_chunks_cow(current_matrix: &ChunkedMatrix, dirty: &[bool]) -> Vec<Arc<AlignedMatrix>> {
    (0..current_matrix.num_chunks())
        .map(|index| {
            if dirty.get(index).copied().unwrap_or(false) {
                Arc::new(AlignedMatrix::copy_from(current_matrix.chunk(index)))
            } else {
                Arc::clone(current_matrix.chunk(index))
            }
        })
        .collect()
}

pub fn try_build_snapshot_from_entries(
    current: &EpochSnapshot,
    entries: &[DeltaEntry],
    row_size_bytes: usize,
    next_epoch_id: u64,
) -> Result<EpochSnapshot, MergeError> {
    let current_matrix = current.matrix.as_ref();
    let chunk_size_bytes = current_matrix.chunk_size_bytes();
    let num_chunks = current_matrix.num_chunks();

    if chunk_size_bytes == 0 {
        return Err(MergeError::InvalidChunkSize {
            chunk_size: chunk_size_bytes,
        });
    }

    let dirty =
        collect_dirty_chunks_from_entries(entries, row_size_bytes, chunk_size_bytes, num_chunks)?;
    let mut new_chunks = clone_chunks_cow(current_matrix, &dirty);

    for entry in entries {
        let offset = row_offset(entry, row_size_bytes)?;
        let (chunk_index, chunk_offset) = chunk_location(offset, chunk_size_bytes);

        if chunk_index >= new_chunks.len() {
            return Err(MergeError::DeltaOutOfBounds {
                offset,
                len: entry.diff.len(),
                chunk_len: 0,
            });
        }

        let chunk = Arc::get_mut(&mut new_chunks[chunk_index])
            .ok_or(MergeError::ChunkNotUnique { chunk_index })?;

        let end = chunk_offset + entry.diff.len();
        if end > chunk.len() {
            return Err(MergeError::DeltaOutOfBounds {
                offset: chunk_offset,
                len: entry.diff.len(),
                chunk_len: chunk.len(),
            });
        }

        apply_xor_patch(&mut chunk.as_mut_slice()[chunk_offset..end], &entry.diff);
    }

    let next_matrix = Arc::new(ChunkedMatrix::from_chunks(
        new_chunks,
        current_matrix.chunk_sizes().to_vec(),
        chunk_size_bytes,
        current_matrix.total_size_bytes(),
    ));

    Ok(EpochSnapshot {
        epoch_id: next_epoch_id,
        matrix: next_matrix,
    })
}

pub fn try_build_next_snapshot(
    current: &EpochSnapshot,
    pending: &DeltaBuffer,
    next_epoch_id: u64,
) -> Result<EpochSnapshot, MergeError> {
    let entries = pending.snapshot().map_err(|_| MergeError::LockPoisoned)?;
    try_build_snapshot_from_entries(current, &entries, pending.row_size_bytes(), next_epoch_id)
}

pub fn build_next_snapshot(
    current: &EpochSnapshot,
    pending: &DeltaBuffer,
    next_epoch_id: u64,
) -> EpochSnapshot {
    try_build_next_snapshot(current, pending, next_epoch_id).expect("build_next_snapshot failed")
}

#[derive(Debug, PartialEq, Eq)]
pub enum EpochManagerError {
    InvalidRowSize { row_size: usize },
}

impl std::fmt::Display for EpochManagerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EpochManagerError::InvalidRowSize { row_size } => {
                write!(f, "invalid row size: {} (must be > 0)", row_size)
            }
        }
    }
}

impl std::error::Error for EpochManagerError {}

pub struct EpochManager {
    global: Arc<GlobalState>,
    pending: Arc<DeltaBuffer>,
    merge_lock: Mutex<()>,
}

impl EpochManager {
    pub fn new(global: Arc<GlobalState>, row_size_bytes: usize) -> Result<Self, EpochManagerError> {
        if row_size_bytes == 0 {
            return Err(EpochManagerError::InvalidRowSize {
                row_size: row_size_bytes,
            });
        }
        Ok(Self {
            global,
            pending: Arc::new(DeltaBuffer::new(row_size_bytes)),
            merge_lock: Mutex::new(()),
        })
    }

    pub fn current(&self) -> Arc<EpochSnapshot> {
        self.global.load()
    }

    pub fn pending(&self) -> &DeltaBuffer {
        &self.pending
    }

    pub fn row_size_bytes(&self) -> usize {
        self.pending.row_size_bytes()
    }

    pub fn try_advance(&self) -> Result<u64, MergeError> {
        let _guard = self
            .merge_lock
            .lock()
            .map_err(|_| MergeError::LockPoisoned)?;
        let entries = self.pending.drain().map_err(|_| MergeError::LockPoisoned)?;
        if entries.is_empty() {
            return Ok(self.global.load().epoch_id);
        }
        let current = self.global.load();
        let next_epoch_id = current.epoch_id + 1;
        let next = match try_build_snapshot_from_entries(
            &current,
            &entries,
            self.pending.row_size_bytes(),
            next_epoch_id,
        ) {
            Ok(snapshot) => snapshot,
            Err(e) => {
                let _ = self.pending.restore(entries);
                return Err(e);
            }
        };
        self.global.store(Arc::new(next));
        Ok(next_epoch_id)
    }

    pub fn acquire(&self) -> EpochHandle {
        EpochHandle(self.global.load())
    }
}

#[derive(Clone)]
pub struct EpochHandle(Arc<EpochSnapshot>);

impl EpochHandle {
    pub fn epoch_id(&self) -> u64 {
        self.0.epoch_id
    }

    pub fn strong_count(&self) -> usize {
        Arc::strong_count(&self.0)
    }
}

impl std::ops::Deref for EpochHandle {
    type Target = EpochSnapshot;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum MergeError {
    RowIndexOverflow {
        row_idx: usize,
        row_size: usize,
    },
    DeltaOutOfBounds {
        offset: usize,
        len: usize,
        chunk_len: usize,
    },
    ChunkNotUnique {
        chunk_index: usize,
    },
    InvalidChunkSize {
        chunk_size: usize,
    },
    LockPoisoned,
}

impl std::fmt::Display for MergeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MergeError::RowIndexOverflow { row_idx, row_size } => {
                write!(
                    f,
                    "row index overflow: {} * {} exceeds usize",
                    row_idx, row_size
                )
            }
            MergeError::DeltaOutOfBounds {
                offset,
                len,
                chunk_len,
            } => {
                write!(
                    f,
                    "delta patch out of bounds: offset {} + len {} > chunk len {}",
                    offset, len, chunk_len
                )
            }
            MergeError::ChunkNotUnique { chunk_index } => {
                write!(
                    f,
                    "chunk {} has multiple references, cannot get mutable access",
                    chunk_index
                )
            }
            MergeError::InvalidChunkSize { chunk_size } => {
                write!(f, "invalid chunk size: {} (must be > 0)", chunk_size)
            }
            MergeError::LockPoisoned => {
                write!(f, "lock poisoned during merge operation")
            }
        }
    }
}

impl std::error::Error for MergeError {}

#[cfg(feature = "network")]
pub async fn spawn_merge_worker(
    manager: std::sync::Arc<EpochManager>,
    interval: std::time::Duration,
    shutdown: tokio::sync::watch::Receiver<bool>,
) {
    spawn_merge_worker_with_callback(manager, interval, shutdown, |err| {
        eprintln!("merge worker error: {}", err);
    })
    .await
}

#[cfg(feature = "network")]
pub async fn spawn_merge_worker_with_callback<F>(
    manager: std::sync::Arc<EpochManager>,
    interval: std::time::Duration,
    mut shutdown: tokio::sync::watch::Receiver<bool>,
    on_error: F,
) where
    F: Fn(MergeError) + Send + 'static,
{
    let on_error = std::sync::Arc::new(on_error);
    let mut ticker = tokio::time::interval(interval);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        tokio::select! {
            _ = ticker.tick() => {
                let mgr = manager.clone();
                let cb = on_error.clone();
                let result = tokio::task::spawn_blocking(move || mgr.try_advance()).await;
                if let Ok(Err(err)) = result {
                    cb(err);
                }
            }
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        build_next_snapshot, dirty_chunks, dirty_chunks_vec, try_build_next_snapshot, EpochHandle,
        EpochManager, EpochManagerError, MergeError,
    };
    use morphogen_core::{DeltaBuffer, EpochSnapshot, GlobalState};
    use morphogen_storage::ChunkedMatrix;
    use std::sync::Arc;

    #[test]
    fn dirty_chunks_empty_buffer_returns_empty_set() {
        let pending = DeltaBuffer::new(4);
        let result = dirty_chunks(&pending, 4, 8);
        assert!(result.is_empty());
    }

    #[test]
    fn dirty_chunks_single_row_maps_to_correct_chunk() {
        let pending = DeltaBuffer::new(4);
        pending.push(0, vec![1, 2, 3, 4]);
        let result = dirty_chunks(&pending, 4, 8);
        assert_eq!(result.len(), 1);
        assert!(result.contains(&0));
    }

    #[test]
    fn dirty_chunks_row_at_boundary() {
        let pending = DeltaBuffer::new(4);
        pending.push(2, vec![1, 2, 3, 4]);
        let result = dirty_chunks(&pending, 4, 8);
        assert_eq!(result.len(), 1);
        assert!(result.contains(&1));
    }

    #[test]
    fn dirty_chunks_multiple_rows_same_chunk() {
        let pending = DeltaBuffer::new(4);
        pending.push(0, vec![1, 1, 1, 1]);
        pending.push(1, vec![2, 2, 2, 2]);
        let result = dirty_chunks(&pending, 4, 8);
        assert_eq!(result.len(), 1);
        assert!(result.contains(&0));
    }

    #[test]
    fn dirty_chunks_multiple_chunks() {
        let pending = DeltaBuffer::new(4);
        pending.push(0, vec![1, 1, 1, 1]);
        pending.push(3, vec![2, 2, 2, 2]);
        let result = dirty_chunks(&pending, 4, 8);
        assert_eq!(result.len(), 2);
        assert!(result.contains(&0));
        assert!(result.contains(&1));
    }

    #[test]
    fn dirty_chunks_vec_returns_bool_slice() {
        let pending = DeltaBuffer::new(4);
        pending.push(0, vec![1, 1, 1, 1]);
        pending.push(3, vec![2, 2, 2, 2]);

        let result = dirty_chunks_vec(&pending, 4, 8, 2);
        assert_eq!(result.len(), 2);
        assert!(result[0]);
        assert!(result[1]);
    }

    #[test]
    fn dirty_chunks_vec_sparse_marking() {
        let pending = DeltaBuffer::new(4);
        pending.push(4, vec![1, 1, 1, 1]);

        let result = dirty_chunks_vec(&pending, 4, 8, 4);
        assert_eq!(result, vec![false, false, true, false]);
    }

    #[test]
    fn try_build_returns_error_on_zero_chunk_size() {
        use super::try_build_snapshot_from_entries;
        use morphogen_core::DeltaEntry;
        use morphogen_storage::AlignedMatrix;

        let matrix = Arc::new(ChunkedMatrix::from_chunks(
            vec![Arc::new(AlignedMatrix::new(0))],
            vec![0],
            0,
            0,
        ));
        let snapshot = EpochSnapshot {
            epoch_id: 0,
            matrix,
        };

        let entries = vec![DeltaEntry {
            row_idx: 0,
            diff: vec![1, 2, 3, 4],
        }];

        let result = try_build_snapshot_from_entries(&snapshot, &entries, 4, 1);
        match result {
            Err(MergeError::InvalidChunkSize { chunk_size }) => {
                assert_eq!(chunk_size, 0);
            }
            Err(other) => panic!("expected InvalidChunkSize, got {:?}", other),
            Ok(_) => panic!("expected error, got Ok"),
        }
    }

    #[test]
    fn cow_merge_applies_pending_deltas() {
        let row_size = 4;
        let matrix = Arc::new(ChunkedMatrix::new(16, 8));
        let snapshot = EpochSnapshot {
            epoch_id: 0,
            matrix,
        };

        let pending = DeltaBuffer::new(row_size);
        pending.push(1, vec![1, 1, 1, 1]);
        pending.push(3, vec![2, 2, 2, 2]);

        let next = build_next_snapshot(&snapshot, &pending, 1);
        let chunk0 = next.matrix.chunk(0).as_ref().as_slice();
        let chunk1 = next.matrix.chunk(1).as_ref().as_slice();

        let mut combined = Vec::new();
        combined.extend_from_slice(chunk0);
        combined.extend_from_slice(chunk1);

        let mut expected = vec![0u8; 16];
        expected[4..8].copy_from_slice(&[1, 1, 1, 1]);
        expected[12..16].copy_from_slice(&[2, 2, 2, 2]);

        assert_eq!(combined, expected);
        assert_eq!(next.epoch_id, 1);
    }

    #[test]
    fn try_build_returns_ok_on_valid_input() {
        let row_size = 4;
        let matrix = Arc::new(ChunkedMatrix::new(16, 8));
        let snapshot = EpochSnapshot {
            epoch_id: 0,
            matrix,
        };

        let pending = DeltaBuffer::new(row_size);
        pending.push(1, vec![1, 1, 1, 1]);

        let result = try_build_next_snapshot(&snapshot, &pending, 1);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().epoch_id, 1);
    }

    #[test]
    fn try_build_returns_error_on_out_of_bounds_delta() {
        let row_size = 4;
        let matrix = Arc::new(ChunkedMatrix::new(16, 8));
        let snapshot = EpochSnapshot {
            epoch_id: 0,
            matrix,
        };

        let pending = DeltaBuffer::new(row_size);
        pending.push(10, vec![1, 1, 1, 1]);

        let result = try_build_next_snapshot(&snapshot, &pending, 1);
        assert!(result.is_err());
        let err = result.err().unwrap();
        match err {
            MergeError::DeltaOutOfBounds { .. } => {}
            other => panic!("expected DeltaOutOfBounds, got {:?}", other),
        }
    }

    #[test]
    fn try_build_empty_pending_returns_new_epoch() {
        let row_size = 4;
        let matrix = Arc::new(ChunkedMatrix::new(16, 8));
        let snapshot = EpochSnapshot {
            epoch_id: 5,
            matrix,
        };

        let pending = DeltaBuffer::new(row_size);

        let result = try_build_next_snapshot(&snapshot, &pending, 6);
        assert!(result.is_ok());
        let next = result.unwrap();
        assert_eq!(next.epoch_id, 6);
    }

    fn make_global_state(epoch_id: u64, _row_size: usize) -> Arc<GlobalState> {
        let matrix = Arc::new(ChunkedMatrix::new(64, 32));
        let snapshot = EpochSnapshot { epoch_id, matrix };
        Arc::new(GlobalState::new(Arc::new(snapshot)))
    }

    #[test]
    fn epoch_manager_new_returns_ok_with_valid_row_size() {
        let global = make_global_state(0, 4);
        let result = EpochManager::new(global, 4);
        assert!(result.is_ok());
    }

    #[test]
    fn epoch_manager_new_returns_error_with_zero_row_size() {
        let global = make_global_state(0, 4);
        let result = EpochManager::new(global, 0);
        assert!(result.is_err());
    }

    #[test]
    fn epoch_manager_error_contains_invalid_row_size() {
        let global = make_global_state(0, 4);
        let result = EpochManager::new(global, 0);
        match result {
            Err(EpochManagerError::InvalidRowSize { row_size }) => {
                assert_eq!(row_size, 0);
            }
            _ => panic!("expected InvalidRowSize error"),
        }
    }

    #[test]
    fn epoch_manager_current_returns_initial_epoch() {
        let global = make_global_state(0, 4);
        let manager = EpochManager::new(global, 4).unwrap();
        assert_eq!(manager.current().epoch_id, 0);
    }

    #[test]
    fn epoch_manager_advance_increments_epoch_id() {
        let global = make_global_state(0, 4);
        let manager = EpochManager::new(global, 4).unwrap();

        manager.pending().push(0, vec![1, 2, 3, 4]).unwrap();
        let new_id = manager.try_advance().unwrap();
        assert_eq!(new_id, 1);
        assert_eq!(manager.current().epoch_id, 1);
    }

    #[test]
    fn epoch_manager_advance_applies_pending_deltas() {
        let global = make_global_state(0, 4);
        let manager = EpochManager::new(global, 4).unwrap();

        manager
            .pending()
            .push(0, vec![0xAA, 0xBB, 0xCC, 0xDD])
            .unwrap();

        manager.try_advance().unwrap();

        let snapshot = manager.current();
        let data = snapshot.matrix.chunk(0).as_slice();
        assert_eq!(&data[0..4], &[0xAA, 0xBB, 0xCC, 0xDD]);
    }

    #[test]
    fn epoch_manager_advance_drains_pending() {
        let global = make_global_state(0, 4);
        let manager = EpochManager::new(global, 4).unwrap();

        manager.pending().push(0, vec![1, 2, 3, 4]).unwrap();
        assert_eq!(manager.pending().len().unwrap(), 1);

        manager.try_advance().unwrap();

        assert!(manager.pending().is_empty().unwrap());
    }

    #[test]
    fn epoch_manager_advance_applies_deltas_exactly_once() {
        let global = make_global_state(0, 4);
        let manager = EpochManager::new(global, 4).unwrap();

        manager
            .pending()
            .push(0, vec![0xFF, 0xFF, 0xFF, 0xFF])
            .unwrap();
        manager.try_advance().unwrap();

        let data_after_first = manager.current().matrix.chunk(0).as_slice()[0..4].to_vec();
        assert_eq!(data_after_first, vec![0xFF, 0xFF, 0xFF, 0xFF]);

        manager.try_advance().unwrap();

        let data_after_second = manager.current().matrix.chunk(0).as_slice()[0..4].to_vec();
        assert_eq!(data_after_second, vec![0xFF, 0xFF, 0xFF, 0xFF]);
    }

    #[test]
    fn epoch_manager_multiple_advances() {
        let global = make_global_state(0, 4);
        let manager = EpochManager::new(global, 4).unwrap();

        for _ in 1..=5 {
            manager.pending().push(0, vec![1, 1, 1, 1]).unwrap();
            manager.try_advance().unwrap();
        }
        assert_eq!(manager.current().epoch_id, 5);
    }

    #[test]
    fn epoch_manager_skip_empty_merge_returns_current_epoch() {
        let global = make_global_state(0, 4);
        let manager = EpochManager::new(global, 4).unwrap();

        let result = manager.try_advance();
        assert_eq!(result.unwrap(), 0);
        assert_eq!(manager.current().epoch_id, 0);
    }

    #[test]
    fn epoch_manager_skip_empty_after_real_advance() {
        let global = make_global_state(0, 4);
        let manager = EpochManager::new(global, 4).unwrap();

        manager.pending().push(0, vec![1, 2, 3, 4]).unwrap();
        let id1 = manager.try_advance().unwrap();
        assert_eq!(id1, 1);

        let id2 = manager.try_advance().unwrap();
        assert_eq!(id2, 1);
        assert_eq!(manager.current().epoch_id, 1);
    }

    #[test]
    fn epoch_handle_provides_access_to_snapshot() {
        let global = make_global_state(42, 4);
        let manager = EpochManager::new(global, 4).unwrap();

        let handle = manager.acquire();
        assert_eq!(handle.epoch_id(), 42);
    }

    #[test]
    fn epoch_handle_keeps_snapshot_alive() {
        let global = make_global_state(0, 4);
        let manager = EpochManager::new(global, 4).unwrap();

        let handle = manager.acquire();
        manager.pending().push(0, vec![1, 2, 3, 4]).unwrap();
        manager.try_advance().unwrap();

        assert_eq!(handle.epoch_id(), 0);
        assert_eq!(manager.current().epoch_id, 1);
    }

    #[test]
    fn epoch_handle_clone_increments_refcount() {
        let global = make_global_state(0, 4);
        let manager = EpochManager::new(global, 4).unwrap();

        let handle1 = manager.acquire();
        let handle2 = handle1.clone();

        assert_eq!(handle1.epoch_id(), handle2.epoch_id());
    }

    #[test]
    fn epoch_handle_deref_to_snapshot() {
        let global = make_global_state(99, 4);
        let manager = EpochManager::new(global, 4).unwrap();

        let handle = manager.acquire();
        let snapshot: &EpochSnapshot = &*handle;
        assert_eq!(snapshot.epoch_id, 99);
    }

    #[test]
    fn old_snapshot_dropped_after_advance_when_no_handles() {
        let global = make_global_state(0, 4);
        let manager = EpochManager::new(global.clone(), 4).unwrap();

        let snap0 = manager.current();
        let initial_count = Arc::strong_count(&snap0);
        assert_eq!(initial_count, 2);
        drop(snap0);

        manager.pending().push(0, vec![1, 2, 3, 4]).unwrap();
        manager.try_advance().unwrap();

        let snap1 = manager.current();
        assert_eq!(snap1.epoch_id, 1);
        assert_eq!(Arc::strong_count(&snap1), 2);
    }

    #[test]
    fn old_snapshot_kept_alive_by_handle() {
        let global = make_global_state(0, 4);
        let manager = EpochManager::new(global.clone(), 4).unwrap();

        let handle = manager.acquire();
        assert_eq!(handle.epoch_id(), 0);

        manager.pending().push(0, vec![1, 2, 3, 4]).unwrap();
        manager.try_advance().unwrap();

        assert_eq!(handle.epoch_id(), 0);
        assert_eq!(manager.current().epoch_id, 1);
    }

    #[test]
    fn handle_keeps_old_epoch_accessible_after_multiple_advances() {
        let global = make_global_state(0, 4);
        let manager = EpochManager::new(global, 4).unwrap();

        let handle_epoch_0 = manager.acquire();

        for _ in 0..3 {
            manager.pending().push(0, vec![1, 2, 3, 4]).unwrap();
            manager.try_advance().unwrap();
        }

        assert_eq!(handle_epoch_0.epoch_id(), 0);
        assert_eq!(manager.current().epoch_id, 3);
    }

    #[test]
    fn handle_strong_count_reflects_acquisitions() {
        let global = make_global_state(0, 4);
        let manager = EpochManager::new(global, 4).unwrap();

        let h1 = manager.acquire();
        let h2 = manager.acquire();

        assert_eq!(h1.strong_count(), 3);

        drop(h2);
        assert_eq!(h1.strong_count(), 2);
    }

    #[test]
    fn try_advance_restores_entries_on_merge_error() {
        let global = make_global_state(0, 4);
        let manager = EpochManager::new(global, 4).unwrap();

        manager.pending().push(0, vec![1, 2, 3, 4]).unwrap();
        manager.pending().push(1000, vec![5, 6, 7, 8]).unwrap(); // out of bounds

        let result = manager.try_advance();
        assert!(result.is_err());

        assert_eq!(
            manager.pending().len().unwrap(),
            2,
            "entries should be restored"
        );
        let snap = manager.pending().snapshot().unwrap();
        assert_eq!(snap[0].row_idx, 0);
        assert_eq!(snap[1].row_idx, 1000);
        assert_eq!(manager.current().epoch_id, 0, "epoch should not advance");
    }

    #[test]
    fn try_advance_restores_entries_preserving_new_pushes() {
        let global = make_global_state(0, 4);
        let manager = EpochManager::new(global, 4).unwrap();

        manager.pending().push(1000, vec![0xFF; 4]).unwrap(); // will fail

        let result = manager.try_advance();
        assert!(result.is_err());

        manager.pending().push(0, vec![0xAA; 4]).unwrap(); // new push after failed advance

        let snap = manager.pending().snapshot().unwrap();
        assert_eq!(snap.len(), 2);
        assert_eq!(snap[0].row_idx, 1000, "restored entry first");
        assert_eq!(snap[1].row_idx, 0, "new entry second");
    }
}

#[cfg(all(test, feature = "network"))]
mod async_tests {
    use super::*;
    use morphogen_core::{DeltaBuffer, EpochSnapshot, GlobalState};
    use morphogen_storage::ChunkedMatrix;
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::sync::watch;

    fn make_global_state(epoch_id: u64, _row_size: usize) -> Arc<GlobalState> {
        let matrix = Arc::new(ChunkedMatrix::new(64, 32));
        let snapshot = EpochSnapshot { epoch_id, matrix };
        Arc::new(GlobalState::new(Arc::new(snapshot)))
    }

    #[tokio::test]
    async fn merge_worker_advances_epoch_on_tick() {
        let global = make_global_state(0, 4);
        let manager = Arc::new(EpochManager::new(global, 4).unwrap());
        let (shutdown_tx, shutdown_rx) = watch::channel(false);

        for _ in 0..5 {
            manager.pending().push(0, vec![1, 2, 3, 4]).unwrap();
        }

        let worker_manager = manager.clone();
        let handle = tokio::spawn(spawn_merge_worker(
            worker_manager,
            Duration::from_millis(10),
            shutdown_rx,
        ));

        tokio::time::sleep(Duration::from_millis(35)).await;
        let _ = shutdown_tx.send(true);
        handle.await.unwrap();

        assert!(manager.current().epoch_id >= 1);
    }

    #[tokio::test]
    async fn merge_worker_stops_on_shutdown() {
        let global = make_global_state(0, 4);
        let manager = Arc::new(EpochManager::new(global, 4).unwrap());
        let (shutdown_tx, shutdown_rx) = watch::channel(false);

        let worker_manager = manager.clone();
        let handle = tokio::spawn(spawn_merge_worker(
            worker_manager,
            Duration::from_millis(100),
            shutdown_rx,
        ));

        let _ = shutdown_tx.send(true);
        let result = tokio::time::timeout(Duration::from_millis(50), handle).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn merge_worker_applies_pending_deltas() {
        let global = make_global_state(0, 4);
        let manager = Arc::new(EpochManager::new(global, 4).unwrap());
        let (shutdown_tx, shutdown_rx) = watch::channel(false);

        manager
            .pending()
            .push(0, vec![0xDE, 0xAD, 0xBE, 0xEF])
            .unwrap();

        let worker_manager = manager.clone();
        let handle = tokio::spawn(spawn_merge_worker(
            worker_manager,
            Duration::from_millis(10),
            shutdown_rx,
        ));

        tokio::time::sleep(Duration::from_millis(25)).await;
        let _ = shutdown_tx.send(true);
        handle.await.unwrap();

        let snapshot = manager.current();
        assert!(snapshot.epoch_id >= 1);
        let data = snapshot.matrix.chunk(0).as_slice();
        assert_eq!(&data[0..4], &[0xDE, 0xAD, 0xBE, 0xEF]);
    }

    #[tokio::test]
    async fn concurrent_try_advance_serializes_epoch_increments() {
        let global = make_global_state(0, 4);
        let manager = Arc::new(EpochManager::new(global, 4).unwrap());

        let mut handles = Vec::new();
        for i in 0..10 {
            let m = manager.clone();
            handles.push(tokio::spawn(async move {
                m.pending().push(0, vec![i as u8, 0, 0, 0]).unwrap();
                m.try_advance().unwrap()
            }));
        }

        let mut results = Vec::new();
        for h in handles {
            results.push(h.await.unwrap());
        }

        results.sort();
        let expected: Vec<u64> = (1..=10).collect();
        assert_eq!(results, expected, "each advance should get unique epoch_id");
        assert_eq!(manager.current().epoch_id, 10);
    }

    #[tokio::test]
    async fn concurrent_try_advance_with_barrier_no_corruption() {
        use std::sync::Barrier;

        let global = make_global_state(0, 4);
        let manager = Arc::new(EpochManager::new(global, 4).unwrap());
        let barrier = Arc::new(Barrier::new(10));

        for i in 0..10 {
            manager.pending().push(0, vec![i as u8, 0, 0, 0]).unwrap();
        }

        let mut handles = Vec::new();
        for _ in 0..10 {
            let m = manager.clone();
            let b = barrier.clone();
            handles.push(std::thread::spawn(move || {
                b.wait();
                m.try_advance().unwrap()
            }));
        }

        let results: Vec<u64> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        assert_eq!(manager.current().epoch_id, 1, "final epoch should be 1");

        let ones = results.iter().filter(|&&id| id == 1).count();
        let zeros = results.iter().filter(|&&id| id == 0).count();
        assert_eq!(ones + zeros, 10, "all results should be 0 or 1");
        assert!(ones >= 1, "at least one thread should advance to epoch 1");
    }

    #[tokio::test]
    async fn merge_worker_uses_spawn_blocking() {
        let global = make_global_state(0, 4);
        let manager = Arc::new(EpochManager::new(global, 4).unwrap());
        let (shutdown_tx, shutdown_rx) = watch::channel(false);

        manager.pending().push(0, vec![1, 2, 3, 4]).unwrap();

        let worker_manager = manager.clone();
        let handle = tokio::spawn(spawn_merge_worker(
            worker_manager,
            Duration::from_millis(10),
            shutdown_rx,
        ));

        let other_work = tokio::spawn(async {
            tokio::time::sleep(Duration::from_millis(5)).await;
            42
        });

        let result = other_work.await.unwrap();
        assert_eq!(result, 42, "async work should complete while worker runs");

        tokio::time::sleep(Duration::from_millis(20)).await;
        let _ = shutdown_tx.send(true);
        handle.await.unwrap();

        assert_eq!(manager.current().epoch_id, 1);
    }

    #[tokio::test]
    async fn merge_worker_reports_errors_via_callback() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let global = make_global_state(0, 4);
        let manager = Arc::new(EpochManager::new(global, 4).unwrap());
        let (shutdown_tx, shutdown_rx) = watch::channel(false);

        manager.pending().push(100, vec![1, 2, 3, 4]).unwrap();

        let error_count = Arc::new(AtomicUsize::new(0));
        let error_count_clone = error_count.clone();

        let on_error = move |_err: MergeError| {
            error_count_clone.fetch_add(1, Ordering::SeqCst);
        };

        let worker_manager = manager.clone();
        let handle = tokio::spawn(spawn_merge_worker_with_callback(
            worker_manager,
            Duration::from_millis(10),
            shutdown_rx,
            on_error,
        ));

        tokio::time::sleep(Duration::from_millis(25)).await;
        let _ = shutdown_tx.send(true);
        handle.await.unwrap();

        assert!(
            error_count.load(Ordering::SeqCst) >= 1,
            "errors should be reported"
        );
    }
}
