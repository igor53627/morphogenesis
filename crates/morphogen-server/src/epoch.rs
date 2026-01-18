use std::collections::HashSet;
use std::sync::Arc;

use morphogen_core::{DeltaBuffer, EpochSnapshot};
use morphogen_storage::{AlignedMatrix, ChunkedMatrix};

pub fn build_next_snapshot(
    current: &EpochSnapshot,
    pending: &DeltaBuffer,
    next_epoch_id: u64,
) -> EpochSnapshot {
    let row_size_bytes = pending.row_size_bytes();
    let current_matrix = current.matrix.as_ref();
    let chunk_size_bytes = current_matrix.chunk_size_bytes();

    let mut dirty_chunks = HashSet::new();
    for entry in pending.snapshot() {
        let offset = entry.row_idx * row_size_bytes;
        dirty_chunks.insert(offset / chunk_size_bytes.max(1));
    }

    let mut new_chunks = Vec::with_capacity(current_matrix.num_chunks());
    for index in 0..current_matrix.num_chunks() {
        if dirty_chunks.contains(&index) {
            let cloned = AlignedMatrix::copy_from(current_matrix.chunk(index));
            new_chunks.push(Arc::new(cloned));
        } else {
            new_chunks.push(Arc::clone(current_matrix.chunk(index)));
        }
    }

    let chunk_sizes = current_matrix.chunk_sizes().to_vec();
    let total_size_bytes = current_matrix.total_size_bytes();

    for entry in pending.snapshot() {
        let offset = entry.row_idx * row_size_bytes;
        let chunk_index = offset / chunk_size_bytes.max(1);
        let chunk_offset = offset % chunk_size_bytes.max(1);

        let chunk = Arc::get_mut(&mut new_chunks[chunk_index]).expect("dirty chunk must be unique");

        let end = chunk_offset + entry.diff.len();
        assert!(end <= chunk.len(), "delta patch out of bounds");
        chunk.as_mut_slice()[chunk_offset..end]
            .iter_mut()
            .zip(entry.diff.iter())
            .for_each(|(dst, src)| {
                *dst ^= src;
            });
    }

    let next_matrix = Arc::new(ChunkedMatrix::from_chunks(
        new_chunks,
        chunk_sizes,
        chunk_size_bytes,
        total_size_bytes,
    ));

    EpochSnapshot {
        epoch_id: next_epoch_id,
        matrix: next_matrix,
        delta: Arc::new(DeltaBuffer::new(row_size_bytes)),
    }
}

#[cfg(test)]
mod tests {
    use super::build_next_snapshot;
    use morphogen_core::{DeltaBuffer, EpochSnapshot};
    use morphogen_storage::ChunkedMatrix;
    use std::sync::Arc;

    #[test]
    fn cow_merge_applies_pending_deltas() {
        let row_size = 4;
        let matrix = Arc::new(ChunkedMatrix::new(16, 8));
        let delta = Arc::new(DeltaBuffer::new(row_size));
        let snapshot = EpochSnapshot {
            epoch_id: 0,
            matrix,
            delta,
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
        assert_eq!(next.delta.len(), 0);
    }
}
