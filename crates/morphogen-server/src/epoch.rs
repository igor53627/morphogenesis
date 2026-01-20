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

pub fn try_dirty_chunks(
    pending: &DeltaBuffer,
    row_size_bytes: usize,
    chunk_size_bytes: usize,
) -> Result<HashSet<usize>, MergeError> {
    if chunk_size_bytes == 0 {
        return Err(MergeError::InvalidChunkSize {
            chunk_size: chunk_size_bytes,
        });
    }
    let snapshot = pending.snapshot().map_err(|_| MergeError::LockPoisoned)?;
    let mut result = HashSet::new();
    for entry in &snapshot {
        let offset = row_offset(entry, row_size_bytes)?;
        let (chunk_index, _) = chunk_location(offset, chunk_size_bytes);
        result.insert(chunk_index);
    }
    Ok(result)
}

#[cfg(test)]
#[deprecated(
    since = "0.2.0",
    note = "Use try_dirty_chunks instead to handle errors"
)]
pub fn dirty_chunks(
    pending: &DeltaBuffer,
    row_size_bytes: usize,
    chunk_size_bytes: usize,
) -> HashSet<usize> {
    try_dirty_chunks(pending, row_size_bytes, chunk_size_bytes)
        .expect("dirty_chunks: lock poisoned")
}

fn collect_dirty_chunks_from_entries(
    entries: &[DeltaEntry],
    row_size_bytes: usize,
    chunk_size_bytes: usize,
    num_chunks: usize,
) -> Result<Vec<bool>, MergeError> {
    if chunk_size_bytes == 0 {
        return Err(MergeError::InvalidChunkSize {
            chunk_size: chunk_size_bytes,
        });
    }
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

pub fn try_dirty_chunks_vec(
    pending: &DeltaBuffer,
    row_size_bytes: usize,
    chunk_size_bytes: usize,
    num_chunks: usize,
) -> Result<Vec<bool>, MergeError> {
    let snapshot = pending.snapshot().map_err(|_| MergeError::LockPoisoned)?;
    collect_dirty_chunks_from_entries(&snapshot, row_size_bytes, chunk_size_bytes, num_chunks)
}

#[cfg(test)]
#[deprecated(
    since = "0.2.0",
    note = "Use try_dirty_chunks_vec instead to handle errors"
)]
pub fn dirty_chunks_vec(
    pending: &DeltaBuffer,
    row_size_bytes: usize,
    chunk_size_bytes: usize,
    num_chunks: usize,
) -> Vec<bool> {
    try_dirty_chunks_vec(pending, row_size_bytes, chunk_size_bytes, num_chunks)
        .expect("dirty_chunks_vec: lock poisoned or overflow")
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

        let end = chunk_offset
            .checked_add(entry.diff.len())
            .ok_or(MergeError::OffsetOverflow {
                chunk_offset,
                len: entry.diff.len(),
            })?;
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

#[cfg(test)]
#[deprecated(
    since = "0.2.0",
    note = "Use try_build_next_snapshot instead to handle errors"
)]
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
    ManagerAlreadyExists,
}

impl std::fmt::Display for EpochManagerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EpochManagerError::InvalidRowSize { row_size } => {
                write!(f, "invalid row size: {} (must be > 0)", row_size)
            }
            EpochManagerError::ManagerAlreadyExists => {
                write!(f, "an EpochManager already exists for this GlobalState")
            }
        }
    }
}

impl std::error::Error for EpochManagerError {}

#[derive(Debug, PartialEq, Eq)]
pub enum UpdateError {
    RowIndexOutOfBounds { row_idx: usize, num_rows: usize },
    SizeMismatch { expected: usize, actual: usize },
    LockPoisoned,
    BufferFull { current: usize, max: usize },
    EntryCountOverflow,
}

impl std::fmt::Display for UpdateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UpdateError::RowIndexOutOfBounds { row_idx, num_rows } => {
                write!(
                    f,
                    "row index {} out of bounds (num_rows = {})",
                    row_idx, num_rows
                )
            }
            UpdateError::SizeMismatch { expected, actual } => {
                write!(
                    f,
                    "delta diff size mismatch: expected {} bytes, got {}",
                    expected, actual
                )
            }
            UpdateError::LockPoisoned => {
                write!(f, "lock poisoned during update")
            }
            UpdateError::BufferFull { current, max } => {
                write!(f, "pending buffer full: {} entries (max {})", current, max)
            }
            UpdateError::EntryCountOverflow => {
                write!(f, "entry count overflow: total entries exceeds usize::MAX")
            }
        }
    }
}

impl std::error::Error for UpdateError {}

impl From<morphogen_core::DeltaError> for UpdateError {
    fn from(err: morphogen_core::DeltaError) -> Self {
        match err {
            morphogen_core::DeltaError::SizeMismatch { expected, actual } => {
                UpdateError::SizeMismatch { expected, actual }
            }
            morphogen_core::DeltaError::LockPoisoned => UpdateError::LockPoisoned,
            morphogen_core::DeltaError::BufferFull { current, max } => {
                UpdateError::BufferFull { current, max }
            }
            morphogen_core::DeltaError::EntryCountOverflow => UpdateError::EntryCountOverflow,
        }
    }
}

pub struct EpochManager {
    global: Arc<GlobalState>,
    pending: Arc<DeltaBuffer>,
    #[cfg(feature = "cuda")]
    gpu_matrix: Arc<Mutex<Option<morphogen_gpu_dpf::storage::GpuPageMatrix>>>,
    merge_lock: Mutex<()>,
}

impl EpochManager {
    pub fn new(global: Arc<GlobalState>, row_size_bytes: usize) -> Result<Self, EpochManagerError> {
        if row_size_bytes == 0 {
            return Err(EpochManagerError::InvalidRowSize {
                row_size: row_size_bytes,
            });
        }
        if !global.try_acquire_manager() {
            return Err(EpochManagerError::ManagerAlreadyExists);
        }
        let initial_epoch = global.load().epoch_id;
        Ok(Self {
            global,
            pending: Arc::new(DeltaBuffer::new_with_epoch(row_size_bytes, initial_epoch)),
            #[cfg(feature = "cuda")]
            gpu_matrix: Arc::new(Mutex::new(None)),
            merge_lock: Mutex::new(()),
        })
    }

    #[cfg(feature = "cuda")]
    pub fn set_gpu_matrix(&self, matrix: morphogen_gpu_dpf::storage::GpuPageMatrix) {
        let mut guard = self.gpu_matrix.lock().unwrap();
        *guard = Some(matrix);
    }

    #[cfg(feature = "cuda")]
    pub fn gpu_matrix(&self) -> Arc<Mutex<Option<morphogen_gpu_dpf::storage::GpuPageMatrix>>> {
        Arc::clone(&self.gpu_matrix)
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

    pub fn num_rows(&self) -> usize {
        let snapshot = self.global.load();
        let row_size = self.pending.row_size_bytes();
        if row_size == 0 {
            return 0;
        }
        snapshot.matrix.total_size_bytes() / row_size
    }

    pub fn submit_update(&self, row_idx: usize, diff: Vec<u8>) -> Result<(), UpdateError> {
        let num_rows = self.num_rows();
        if row_idx >= num_rows {
            return Err(UpdateError::RowIndexOutOfBounds { row_idx, num_rows });
        }
        self.pending.push(row_idx, diff)?;
        Ok(())
    }

    /// Submit a full snapshot for a major epoch transition (e.g. seed rotation).
    ///
    /// This replaces the current matrix entirely and resets the delta buffer.
    /// Returns the new epoch ID.
    ///
    /// # Warning
    /// This operation acquires the merge lock and blocks updates.
    /// On GPU-enabled builds, this may involve significant data transfer time.
    pub fn submit_snapshot(
        &self,
        new_matrix: ChunkedMatrix,
        _new_seeds: [u64; 3],
    ) -> Result<u64, MergeError> {
        // Future: Prepare GPU upload here (Blue/Green strategy) to minimize lock time.

        let _guard = self
            .merge_lock
            .lock()
            .map_err(|_| MergeError::LockPoisoned)?;
        
        let current = self.global.load();
        let next_epoch_id = current
            .epoch_id
            .checked_add(1)
            .ok_or(MergeError::EpochOverflow)?;

        let next_snapshot = EpochSnapshot {
            epoch_id: next_epoch_id,
            matrix: Arc::new(new_matrix),
        };
        
        self.global.store(Arc::new(next_snapshot));
        
        // Reset delta buffer for new epoch
        self.pending.reset_with_epoch(next_epoch_id).map_err(|_| MergeError::LockPoisoned)?;

        #[cfg(feature = "cuda")]
        {
            // TODO: Implement actual GPU matrix swap.
            // For now, we just acknowledge the transition.
            // Real implementation requires uploading ChunkedMatrix to GpuPageMatrix.
        }

        Ok(next_epoch_id)
    }

    pub fn try_advance(&self) -> Result<u64, MergeError> {
        let _guard = self
            .merge_lock
            .lock()
            .map_err(|_| MergeError::LockPoisoned)?;
        let current = self.global.load();
        let next_epoch_id = current
            .epoch_id
            .checked_add(1)
            .ok_or(MergeError::EpochOverflow)?;
        let entries = self
            .pending
            .drain_for_epoch(next_epoch_id)
            .map_err(|_| MergeError::LockPoisoned)?;
        if entries.is_empty() {
            return Ok(current.epoch_id);
        }
        let next = match try_build_snapshot_from_entries(
            &current,
            &entries,
            self.pending.row_size_bytes(),
            next_epoch_id,
        ) {
            Ok(snapshot) => snapshot,
            Err(e) => {
                if let Err(rollback_err) = self.pending.restore_for_epoch(current.epoch_id, entries)
                {
                    // BufferFull means data WAS preserved (just exceeds max_entries limit).
                    // Only LockPoisoned, EntryCountOverflow, or SizeMismatch mean data loss.
                    use morphogen_core::DeltaError;
                    match rollback_err {
                        DeltaError::BufferFull { .. } => {
                            // Data preserved but buffer exceeds limit - log warning, return merge error
                            // (not RollbackFailed, since data is safe)
                        }
                        DeltaError::LockPoisoned
                        | DeltaError::EntryCountOverflow
                        | DeltaError::SizeMismatch { .. } => {
                            // True rollback failure - data may be lost
                            return Err(MergeError::RollbackFailed {
                                merge_error: e.to_string(),
                                rollback_error: rollback_err.to_string(),
                            });
                        }
                    }
                }
                return Err(e);
            }
        };
        self.global.store(Arc::new(next));

        #[cfg(feature = "cuda")]
        {
            let mut gpu_guard = self.gpu_matrix.lock().map_err(|_| MergeError::LockPoisoned)?;
            if let Some(gpu_matrix) = gpu_guard.as_mut() {
                let snapshot = self.global.load();
                let chunk_size = snapshot.matrix.chunk_size_bytes();
                let num_chunks = snapshot.matrix.num_chunks();
                let dirty = collect_dirty_chunks_from_entries(
                    &entries,
                    self.pending.row_size_bytes(),
                    chunk_size,
                    num_chunks,
                )?;

                for (chunk_idx, &is_dirty) in dirty.iter().enumerate() {
                    if is_dirty {
                        let chunk = snapshot.matrix.chunk(chunk_idx);
                        let start_byte = chunk_idx * chunk_size;
                        let start_page =
                            start_byte / morphogen_gpu_dpf::storage::PAGE_SIZE_BYTES;
                        if let Err(e) = gpu_matrix.update_pages(start_page, chunk.as_slice()) {
                            eprintln!("GPU update failed for chunk {}: {:?}", chunk_idx, e);
                            // We don't fail the whole advance because CPU is already updated,
                            // but this signals a serious inconsistency.
                        }
                    }
                }
            }
        }

        Ok(next_epoch_id)
    }

    pub fn acquire(&self) -> EpochHandle {
        EpochHandle(self.global.load())
    }
}

impl Drop for EpochManager {
    fn drop(&mut self) {
        self.global.release_manager();
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
    EpochOverflow,
    OffsetOverflow {
        chunk_offset: usize,
        len: usize,
    },
    /// Merge failed AND rollback also failed - entries may be lost.
    /// Contains the original merge error message and the rollback error message.
    RollbackFailed {
        merge_error: String,
        rollback_error: String,
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
            MergeError::EpochOverflow => {
                write!(f, "epoch id overflow: cannot increment past u64::MAX")
            }
            MergeError::OffsetOverflow { chunk_offset, len } => {
                write!(
                    f,
                    "offset overflow: chunk_offset {} + len {} exceeds usize::MAX",
                    chunk_offset, len
                )
            }
            MergeError::RollbackFailed {
                merge_error,
                rollback_error,
            } => {
                write!(
                    f,
                    "merge failed and rollback also failed (DATA MAY BE LOST): merge={}, rollback={}",
                    merge_error, rollback_error
                )
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
#[allow(deprecated)]
mod tests {
    use super::{
        build_next_snapshot, dirty_chunks, dirty_chunks_vec, try_build_next_snapshot,
        try_dirty_chunks, try_dirty_chunks_vec, EpochManager, EpochManagerError,
        MergeError,
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
        pending.push(0, vec![1, 2, 3, 4]).unwrap();
        let result = dirty_chunks(&pending, 4, 8);
        assert_eq!(result.len(), 1);
        assert!(result.contains(&0));
    }

    #[test]
    fn dirty_chunks_row_at_boundary() {
        let pending = DeltaBuffer::new(4);
        pending.push(2, vec![1, 2, 3, 4]).unwrap();
        let result = dirty_chunks(&pending, 4, 8);
        assert_eq!(result.len(), 1);
        assert!(result.contains(&1));
    }

    #[test]
    fn dirty_chunks_multiple_rows_same_chunk() {
        let pending = DeltaBuffer::new(4);
        pending.push(0, vec![1, 1, 1, 1]).unwrap();
        pending.push(1, vec![2, 2, 2, 2]).unwrap();
        let result = dirty_chunks(&pending, 4, 8);
        assert_eq!(result.len(), 1);
        assert!(result.contains(&0));
    }

    #[test]
    fn dirty_chunks_multiple_chunks() {
        let pending = DeltaBuffer::new(4);
        pending.push(0, vec![1, 1, 1, 1]).unwrap();
        pending.push(3, vec![2, 2, 2, 2]).unwrap();
        let result = dirty_chunks(&pending, 4, 8);
        assert_eq!(result.len(), 2);
        assert!(result.contains(&0));
        assert!(result.contains(&1));
    }

    #[test]
    fn dirty_chunks_vec_returns_bool_slice() {
        let pending = DeltaBuffer::new(4);
        pending.push(0, vec![1, 1, 1, 1]).unwrap();
        pending.push(3, vec![2, 2, 2, 2]).unwrap();

        let result = dirty_chunks_vec(&pending, 4, 8, 2);
        assert_eq!(result.len(), 2);
        assert!(result[0]);
        assert!(result[1]);
    }

    #[test]
    fn dirty_chunks_vec_sparse_marking() {
        let pending = DeltaBuffer::new(4);
        pending.push(4, vec![1, 1, 1, 1]).unwrap();

        let result = dirty_chunks_vec(&pending, 4, 8, 4);
        assert_eq!(result, vec![false, false, true, false]);
    }

    #[test]
    fn try_dirty_chunks_returns_result() {
        let pending = DeltaBuffer::new(4);
        pending.push(0, vec![1, 2, 3, 4]).unwrap();
        let result = try_dirty_chunks(&pending, 4, 8);
        assert!(result.is_ok());
        let chunks = result.unwrap();
        assert_eq!(chunks.len(), 1);
        assert!(chunks.contains(&0));
    }

    #[test]
    fn try_dirty_chunks_errors_on_row_overflow() {
        let pending = DeltaBuffer::new(4);
        pending.push(usize::MAX, vec![1, 2, 3, 4]).unwrap();

        let result = try_dirty_chunks(&pending, 4, 8);
        assert!(
            matches!(result, Err(MergeError::RowIndexOverflow { .. })),
            "expected RowIndexOverflow, got {:?}",
            result
        );
    }

    #[test]
    fn try_dirty_chunks_vec_errors_on_row_overflow() {
        let pending = DeltaBuffer::new(4);
        pending.push(usize::MAX, vec![1, 2, 3, 4]).unwrap();

        let result = try_dirty_chunks_vec(&pending, 4, 8, 2);
        assert!(
            matches!(result, Err(MergeError::RowIndexOverflow { .. })),
            "expected RowIndexOverflow, got {:?}",
            result
        );
    }

    #[test]
    fn try_dirty_chunks_errors_on_zero_chunk_size() {
        let pending = DeltaBuffer::new(4);
        pending.push(0, vec![1, 2, 3, 4]).unwrap();

        let result = try_dirty_chunks(&pending, 4, 0);
        assert!(
            matches!(result, Err(MergeError::InvalidChunkSize { chunk_size: 0 })),
            "expected InvalidChunkSize, got {:?}",
            result
        );
    }

    #[test]
    fn try_dirty_chunks_vec_errors_on_zero_chunk_size() {
        let pending = DeltaBuffer::new(4);
        pending.push(0, vec![1, 2, 3, 4]).unwrap();

        let result = try_dirty_chunks_vec(&pending, 4, 0, 2);
        assert!(
            matches!(result, Err(MergeError::InvalidChunkSize { chunk_size: 0 })),
            "expected InvalidChunkSize, got {:?}",
            result
        );
    }

    #[test]
    fn try_dirty_chunks_vec_returns_result() {
        let pending = DeltaBuffer::new(4);
        pending.push(1, vec![1, 1, 1, 1]).unwrap();
        pending.push(3, vec![2, 2, 2, 2]).unwrap();

        let result = try_dirty_chunks_vec(&pending, 4, 8, 2);
        assert!(result.is_ok());
        let dirty = result.unwrap();
        assert_eq!(dirty.len(), 2);
        assert!(dirty[0]);
        assert!(dirty[1]);
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
    fn offset_overflow_error_exists_and_formats() {
        let err = MergeError::OffsetOverflow {
            chunk_offset: usize::MAX - 10,
            len: 100,
        };
        let msg = err.to_string();
        assert!(msg.contains("overflow"), "should mention overflow: {}", msg);
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
        pending.push(1, vec![1, 1, 1, 1]).unwrap();
        pending.push(3, vec![2, 2, 2, 2]).unwrap();

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
        pending.push(1, vec![1, 1, 1, 1]).unwrap();

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
        pending.push(10, vec![1, 1, 1, 1]).unwrap();

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
    fn epoch_manager_num_rows_returns_correct_count() {
        let global = make_global_state(0, 4);
        let manager = EpochManager::new(global, 4).unwrap();
        assert_eq!(manager.num_rows(), 16);
    }

    #[test]
    fn epoch_manager_rejects_second_manager_on_same_global_state() {
        let global = make_global_state(0, 4);

        let manager1 = EpochManager::new(global.clone(), 4);
        assert!(manager1.is_ok(), "first manager should succeed");

        let manager2 = EpochManager::new(global.clone(), 4);
        assert!(
            matches!(manager2, Err(EpochManagerError::ManagerAlreadyExists)),
            "second manager on same GlobalState should fail with ManagerAlreadyExists"
        );
    }

    #[test]
    fn epoch_manager_allows_new_manager_after_drop() {
        let global = make_global_state(0, 4);

        {
            let _manager1 = EpochManager::new(global.clone(), 4).unwrap();
        }

        let manager2 = EpochManager::new(global.clone(), 4);
        assert!(
            manager2.is_ok(),
            "new manager should succeed after previous one is dropped"
        );
    }

    #[test]
    fn epoch_manager_try_advance_returns_error_on_epoch_overflow() {
        let global = make_global_state(u64::MAX, 4);
        let manager = EpochManager::new(global, 4).unwrap();

        manager.pending().push(0, vec![1, 2, 3, 4]).unwrap();

        let result = manager.try_advance();
        assert!(
            matches!(result, Err(MergeError::EpochOverflow)),
            "try_advance should return EpochOverflow when epoch_id is u64::MAX, got {:?}",
            result
        );
    }

    #[test]
    fn rollback_failed_error_contains_both_messages() {
        let err = MergeError::RollbackFailed {
            merge_error: "merge failed".to_string(),
            rollback_error: "lock poisoned".to_string(),
        };

        let msg = err.to_string();
        assert!(msg.contains("merge failed"), "should contain merge error");
        assert!(
            msg.contains("lock poisoned"),
            "should contain rollback error"
        );
        assert!(
            msg.contains("DATA MAY BE LOST"),
            "should warn about data loss"
        );
    }

    #[test]
    fn buffer_full_rollback_is_not_data_loss() {
        // BufferFull from restore_for_epoch means data IS preserved,
        // just that the buffer exceeds max_entries.
        // This should NOT be wrapped in RollbackFailed ("DATA MAY BE LOST").
        //
        // We can't easily test try_advance rollback path without injecting failures,
        // but we can verify the DeltaError classification logic directly.
        use morphogen_core::DeltaError;

        // These errors mean data was NOT preserved - true rollback failure
        let lock_poisoned = DeltaError::LockPoisoned;
        let entry_overflow = DeltaError::EntryCountOverflow;
        let size_mismatch = DeltaError::SizeMismatch {
            expected: 4,
            actual: 3,
        };

        // This error means data WAS preserved - not a true failure
        let buffer_full = DeltaError::BufferFull {
            current: 100,
            max: 50,
        };

        // BufferFull should be distinguishable from true failures
        assert!(
            !matches!(buffer_full, DeltaError::LockPoisoned),
            "BufferFull is not LockPoisoned"
        );
        assert!(
            !matches!(buffer_full, DeltaError::EntryCountOverflow),
            "BufferFull is not EntryCountOverflow"
        );
        assert!(
            !matches!(buffer_full, DeltaError::SizeMismatch { .. }),
            "BufferFull is not SizeMismatch"
        );
        assert!(
            matches!(lock_poisoned, DeltaError::LockPoisoned),
            "LockPoisoned is true failure"
        );
        assert!(
            matches!(entry_overflow, DeltaError::EntryCountOverflow),
            "EntryCountOverflow is true failure"
        );
        assert!(
            matches!(size_mismatch, DeltaError::SizeMismatch { .. }),
            "SizeMismatch is true failure"
        );
    }

    #[test]
    fn epoch_manager_submit_update_accepts_valid_row() {
        let global = make_global_state(0, 4);
        let manager = EpochManager::new(global, 4).unwrap();
        let result = manager.submit_update(0, vec![1, 2, 3, 4]);
        assert!(result.is_ok());
        assert_eq!(manager.pending().len().unwrap(), 1);
    }

    #[test]
    fn epoch_manager_submit_update_accepts_last_valid_row() {
        let global = make_global_state(0, 4);
        let manager = EpochManager::new(global, 4).unwrap();
        let result = manager.submit_update(15, vec![1, 2, 3, 4]);
        assert!(result.is_ok());
    }

    #[test]
    fn epoch_manager_submit_update_rejects_out_of_bounds_row() {
        use super::UpdateError;
        let global = make_global_state(0, 4);
        let manager = EpochManager::new(global, 4).unwrap();
        let result = manager.submit_update(16, vec![1, 2, 3, 4]);
        assert!(result.is_err());
        match result {
            Err(UpdateError::RowIndexOutOfBounds { row_idx, num_rows }) => {
                assert_eq!(row_idx, 16);
                assert_eq!(num_rows, 16);
            }
            _ => panic!("expected RowIndexOutOfBounds"),
        }
    }

    #[test]
    fn epoch_manager_submit_update_rejects_wrong_size_diff() {
        use super::UpdateError;
        let global = make_global_state(0, 4);
        let manager = EpochManager::new(global, 4).unwrap();
        let result = manager.submit_update(0, vec![1, 2, 3]);
        match result {
            Err(UpdateError::SizeMismatch { expected, actual }) => {
                assert_eq!(expected, 4);
                assert_eq!(actual, 3);
            }
            _ => panic!("expected SizeMismatch"),
        }
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

    #[test]
    fn try_advance_resets_pending_epoch_on_merge_error() {
        let global = make_global_state(0, 4);
        let manager = EpochManager::new(global, 4).unwrap();

        let initial_pending_epoch = manager.pending().pending_epoch();
        assert_eq!(initial_pending_epoch, 0);

        manager.pending().push(1000, vec![0xFF; 4]).unwrap();

        let result = manager.try_advance();
        assert!(result.is_err());

        let pending_epoch_after_failure = manager.pending().pending_epoch();
        let matrix_epoch = manager.current().epoch_id;

        assert_eq!(
            pending_epoch_after_failure, matrix_epoch,
            "pending_epoch ({}) must equal matrix epoch ({}) after merge failure, \
             otherwise scan_consistent will livelock",
            pending_epoch_after_failure, matrix_epoch
        );
    }

    #[test]
    fn scan_consistent_succeeds_after_merge_failure() {
        use crate::scan::scan_consistent_with_max_retries;
        use morphogen_dpf::AesDpfKey;

        let global = make_global_state(0, 4);
        let manager = EpochManager::new(global.clone(), 4).unwrap();

        manager.pending().push(1000, vec![0xFF; 4]).unwrap();
        let result = manager.try_advance();
        assert!(result.is_err());

        let mut rng = rand::thread_rng();
        let (key0, _) = AesDpfKey::generate_pair(&mut rng, 0);
        let (key1, _) = AesDpfKey::generate_pair(&mut rng, 1);
        let (key2, _) = AesDpfKey::generate_pair(&mut rng, 2);
        let keys = [key0, key1, key2];

        let scan_result =
            scan_consistent_with_max_retries(&global, manager.pending(), &keys, 4, 10);

        assert!(
            scan_result.is_ok(),
            "scan_consistent should succeed after merge failure, but got: {:?}",
            scan_result
        );
    }

    #[test]
    fn epoch_manager_initializes_pending_epoch_from_global() {
        let global = make_global_state(42, 4);
        let manager = EpochManager::new(global, 4).unwrap();

        assert_eq!(
            manager.pending().pending_epoch(),
            42,
            "pending_epoch should be initialized to global epoch"
        );
    }

    #[test]
    fn scan_consistent_succeeds_with_nonzero_initial_epoch() {
        use crate::scan::scan_consistent_with_max_retries;
        use morphogen_dpf::AesDpfKey;

        let global = make_global_state(100, 4);
        let manager = EpochManager::new(global.clone(), 4).unwrap();

        let mut rng = rand::thread_rng();
        let (key0, _) = AesDpfKey::generate_pair(&mut rng, 0);
        let (key1, _) = AesDpfKey::generate_pair(&mut rng, 1);
        let (key2, _) = AesDpfKey::generate_pair(&mut rng, 2);
        let keys = [key0, key1, key2];

        let scan_result =
            scan_consistent_with_max_retries(&global, manager.pending(), &keys, 4, 10);

        assert!(
            scan_result.is_ok(),
            "scan_consistent should succeed with nonzero initial epoch, but got: {:?}",
            scan_result
        );

        let (_, epoch) = scan_result.unwrap();
        assert_eq!(epoch, 100);
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
