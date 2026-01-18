use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeltaEntry {
    pub row_idx: usize,
    pub diff: Vec<u8>,
}

#[derive(Debug)]
pub enum DeltaError {
    SizeMismatch { expected: usize, actual: usize },
    LockPoisoned,
    BufferFull { current: usize, max: usize },
    EntryCountOverflow,
}

impl std::fmt::Display for DeltaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeltaError::SizeMismatch { expected, actual } => {
                write!(
                    f,
                    "delta diff size mismatch: expected {} bytes, got {}",
                    expected, actual
                )
            }
            DeltaError::LockPoisoned => {
                write!(f, "delta buffer lock poisoned")
            }
            DeltaError::BufferFull { current, max } => {
                write!(f, "delta buffer full: {} entries (max {})", current, max)
            }
            DeltaError::EntryCountOverflow => {
                write!(f, "entry count overflow: total entries exceeds usize::MAX")
            }
        }
    }
}

impl std::error::Error for DeltaError {}

#[deprecated(since = "0.2.0", note = "Use DeltaError instead")]
pub type DeltaPushError = DeltaError;

/// Thread-safe buffer for accumulating delta entries before epoch merge.
///
/// # Concurrency Invariant
///
/// The `pending_epoch` field must only be modified while holding the `entries` write lock.
/// This ensures that `snapshot_with_epoch()` returns a consistent (epoch, entries) pair:
/// readers holding the read lock will see both the entries AND the epoch that corresponds
/// to those entries, preventing torn reads where we see an old epoch with empty entries
/// (or vice versa).
pub struct DeltaBuffer {
    row_size_bytes: usize,
    max_entries: Option<usize>,
    entries: RwLock<Vec<DeltaEntry>>,
    /// The epoch that pending entries will be merged into.
    ///
    /// # Invariant
    ///
    /// This field must only be modified while holding the `entries` write lock.
    /// This is critical for `snapshot_with_epoch()` correctness - it reads this field
    /// while holding the read lock to ensure atomicity with the entries snapshot.
    pending_epoch: AtomicU64,
}

impl DeltaBuffer {
    pub fn new(row_size_bytes: usize) -> Self {
        Self {
            row_size_bytes,
            max_entries: None,
            entries: RwLock::new(Vec::new()),
            pending_epoch: AtomicU64::new(0),
        }
    }

    pub fn with_max_entries(row_size_bytes: usize, max_entries: usize) -> Self {
        Self {
            row_size_bytes,
            max_entries: Some(max_entries),
            entries: RwLock::new(Vec::new()),
            pending_epoch: AtomicU64::new(0),
        }
    }

    pub fn new_with_epoch(row_size_bytes: usize, initial_epoch: u64) -> Self {
        Self {
            row_size_bytes,
            max_entries: None,
            entries: RwLock::new(Vec::new()),
            pending_epoch: AtomicU64::new(initial_epoch),
        }
    }

    pub fn with_max_entries_and_epoch(
        row_size_bytes: usize,
        max_entries: usize,
        initial_epoch: u64,
    ) -> Self {
        Self {
            row_size_bytes,
            max_entries: Some(max_entries),
            entries: RwLock::new(Vec::new()),
            pending_epoch: AtomicU64::new(initial_epoch),
        }
    }

    pub fn max_entries(&self) -> Option<usize> {
        self.max_entries
    }

    pub fn row_size_bytes(&self) -> usize {
        self.row_size_bytes
    }

    fn validate_entries(&self, entries: &[DeltaEntry]) -> Result<(), DeltaError> {
        for entry in entries {
            if entry.diff.len() != self.row_size_bytes {
                return Err(DeltaError::SizeMismatch {
                    expected: self.row_size_bytes,
                    actual: entry.diff.len(),
                });
            }
        }
        Ok(())
    }

    pub fn push(&self, row_idx: usize, diff: Vec<u8>) -> Result<(), DeltaError> {
        if diff.len() != self.row_size_bytes {
            return Err(DeltaError::SizeMismatch {
                expected: self.row_size_bytes,
                actual: diff.len(),
            });
        }
        let mut guard = self.entries.write().map_err(|_| DeltaError::LockPoisoned)?;
        if let Some(max) = self.max_entries {
            if guard.len() >= max {
                return Err(DeltaError::BufferFull {
                    current: guard.len(),
                    max,
                });
            }
        }
        guard.push(DeltaEntry { row_idx, diff });
        Ok(())
    }

    pub fn snapshot(&self) -> Result<Vec<DeltaEntry>, DeltaError> {
        let guard = self.entries.read().map_err(|_| DeltaError::LockPoisoned)?;
        Ok(guard.clone())
    }

    /// Returns the current pending epoch and a snapshot of all entries atomically.
    ///
    /// # Atomicity
    ///
    /// Reads `pending_epoch` while holding the `entries` read lock. This ensures
    /// we never see a "torn read" where the epoch is updated but entries are
    /// already drained (or vice versa). The invariant that `pending_epoch` is only
    /// modified while holding the write lock guarantees consistency.
    pub fn snapshot_with_epoch(&self) -> Result<(u64, Vec<DeltaEntry>), DeltaError> {
        let guard = self.entries.read().map_err(|_| DeltaError::LockPoisoned)?;
        // Read epoch while holding lock - see struct doc for invariant
        let epoch = self.pending_epoch.load(Ordering::Acquire);
        Ok((epoch, guard.clone()))
    }

    pub fn pending_epoch(&self) -> u64 {
        self.pending_epoch.load(Ordering::Acquire)
    }

    pub fn len(&self) -> Result<usize, DeltaError> {
        let guard = self.entries.read().map_err(|_| DeltaError::LockPoisoned)?;
        Ok(guard.len())
    }

    pub fn is_empty(&self) -> Result<bool, DeltaError> {
        Ok(self.len()? == 0)
    }

    pub fn drain(&self) -> Result<Vec<DeltaEntry>, DeltaError> {
        let mut guard = self.entries.write().map_err(|_| DeltaError::LockPoisoned)?;
        Ok(std::mem::take(&mut *guard))
    }

    /// Atomically drains all entries and updates the pending epoch.
    ///
    /// # Invariant
    ///
    /// Updates `pending_epoch` while holding the `entries` write lock, ensuring
    /// `snapshot_with_epoch()` always sees a consistent (epoch, entries) pair.
    pub fn drain_for_epoch(&self, new_epoch: u64) -> Result<Vec<DeltaEntry>, DeltaError> {
        let mut guard = self.entries.write().map_err(|_| DeltaError::LockPoisoned)?;
        // SAFETY: pending_epoch modification while holding write lock - see struct doc
        self.pending_epoch.store(new_epoch, Ordering::Release);
        Ok(std::mem::take(&mut *guard))
    }

    pub fn restore(&self, entries: Vec<DeltaEntry>) -> Result<(), DeltaError> {
        if entries.is_empty() {
            return Ok(());
        }
        self.validate_entries(&entries)?;
        let mut guard = self.entries.write().map_err(|_| DeltaError::LockPoisoned)?;
        let existing = std::mem::take(&mut *guard);
        *guard = entries;
        guard.extend(existing);
        Ok(())
    }

    /// Atomically restores entries (prepending them) and resets pending epoch.
    ///
    /// Used to roll back after a failed epoch merge. The epoch is always updated,
    /// and all entries are always restored, even if this would exceed `max_entries`.
    /// If the limit is exceeded, returns `BufferFull` error (but data is preserved).
    ///
    /// Returns `SizeMismatch` error if any entry has wrong diff size. In this case,
    /// neither the epoch nor the buffer is modified.
    ///
    /// # Invariant
    ///
    /// Updates `pending_epoch` while holding the `entries` write lock, ensuring
    /// `snapshot_with_epoch()` always sees a consistent (epoch, entries) pair.
    pub fn restore_for_epoch(
        &self,
        epoch: u64,
        entries: Vec<DeltaEntry>,
    ) -> Result<(), DeltaError> {
        // Validate before acquiring lock - reject malformed entries without side effects
        self.validate_entries(&entries)?;
        let mut guard = self.entries.write().map_err(|_| DeltaError::LockPoisoned)?;

        // Check for overflow BEFORE modifying any state
        let total = entries
            .len()
            .checked_add(guard.len())
            .ok_or(DeltaError::EntryCountOverflow)?;

        // SAFETY: pending_epoch modification while holding write lock - see struct doc
        self.pending_epoch.store(epoch, Ordering::Release);

        // Merge entries (even if entries is empty, we still checked overflow above)
        let existing = std::mem::take(&mut *guard);
        if entries.is_empty() {
            *guard = existing;
        } else {
            *guard = entries;
            guard.extend(existing);
        }

        // Check max_entries after merge - report but preserve data
        if let Some(max) = self.max_entries {
            if total > max {
                return Err(DeltaError::BufferFull {
                    current: total,
                    max,
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_with_epoch_returns_current_epoch_and_entries() {
        let buf = DeltaBuffer::new(4);
        buf.push(0, vec![1, 2, 3, 4]).unwrap();
        buf.push(1, vec![5, 6, 7, 8]).unwrap();

        let (epoch, entries) = buf.snapshot_with_epoch().unwrap();
        assert_eq!(epoch, 0, "initial epoch should be 0");
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn drain_for_epoch_advances_pending_epoch() {
        let buf = DeltaBuffer::new(4);
        buf.push(0, vec![1, 2, 3, 4]).unwrap();

        let entries = buf.drain_for_epoch(1).unwrap();
        assert_eq!(entries.len(), 1);

        let (epoch, _) = buf.snapshot_with_epoch().unwrap();
        assert_eq!(epoch, 1, "epoch should advance after drain_for_epoch");
    }

    #[test]
    fn pending_epoch_reflects_last_drain() {
        let buf = DeltaBuffer::new(4);
        assert_eq!(buf.pending_epoch(), 0);

        buf.push(0, vec![1, 2, 3, 4]).unwrap();
        let _ = buf.drain_for_epoch(5).unwrap();
        assert_eq!(buf.pending_epoch(), 5);

        buf.push(1, vec![5, 6, 7, 8]).unwrap();
        let _ = buf.drain_for_epoch(10).unwrap();
        assert_eq!(buf.pending_epoch(), 10);
    }

    #[test]
    fn drain_returns_all_entries_and_clears_buffer() {
        let buf = DeltaBuffer::new(4);
        buf.push(0, vec![1, 2, 3, 4]).unwrap();
        buf.push(1, vec![5, 6, 7, 8]).unwrap();

        let drained = buf.drain().unwrap();
        assert_eq!(drained.len(), 2);
        assert_eq!(drained[0].row_idx, 0);
        assert_eq!(drained[1].row_idx, 1);
        assert!(buf.is_empty().unwrap());
    }

    #[test]
    fn drain_on_empty_buffer_returns_empty_vec() {
        let buf = DeltaBuffer::new(4);
        let drained = buf.drain().unwrap();
        assert!(drained.is_empty());
        assert!(buf.is_empty().unwrap());
    }

    #[test]
    fn drain_allows_new_pushes_after() {
        let buf = DeltaBuffer::new(4);
        buf.push(0, vec![1, 2, 3, 4]).unwrap();
        buf.drain().unwrap();
        buf.push(5, vec![9, 9, 9, 9]).unwrap();
        assert_eq!(buf.len().unwrap(), 1);
        assert_eq!(buf.snapshot().unwrap()[0].row_idx, 5);
    }

    #[test]
    fn restore_prepends_entries_before_existing() {
        let buf = DeltaBuffer::new(4);
        buf.push(10, vec![0xAA; 4]).unwrap();

        let to_restore = vec![
            DeltaEntry {
                row_idx: 0,
                diff: vec![1, 2, 3, 4],
            },
            DeltaEntry {
                row_idx: 1,
                diff: vec![5, 6, 7, 8],
            },
        ];
        buf.restore(to_restore).unwrap();

        let snap = buf.snapshot().unwrap();
        assert_eq!(snap.len(), 3);
        assert_eq!(snap[0].row_idx, 0, "restored entries should be first");
        assert_eq!(snap[1].row_idx, 1);
        assert_eq!(snap[2].row_idx, 10, "existing entry should be last");
    }

    #[test]
    fn restore_empty_vec_is_noop() {
        let buf = DeltaBuffer::new(4);
        buf.push(5, vec![1, 2, 3, 4]).unwrap();
        buf.restore(vec![]).unwrap();
        assert_eq!(buf.len().unwrap(), 1);
    }

    #[test]
    fn push_returns_ok_on_correct_size() {
        let buf = DeltaBuffer::new(4);
        let result = buf.push(0, vec![1, 2, 3, 4]);
        assert!(result.is_ok());
        assert_eq!(buf.len().unwrap(), 1);
    }

    #[test]
    fn push_returns_error_on_wrong_size() {
        let buf = DeltaBuffer::new(4);
        let result = buf.push(0, vec![1, 2, 3]); // 3 bytes, expected 4
        assert!(result.is_err());
        assert!(
            buf.is_empty().unwrap(),
            "buffer should remain unchanged on error"
        );
    }

    #[test]
    fn push_error_contains_expected_and_actual_sizes() {
        let buf = DeltaBuffer::new(4);
        let result = buf.push(0, vec![1, 2]);
        match result {
            Err(DeltaError::SizeMismatch { expected, actual }) => {
                assert_eq!(expected, 4);
                assert_eq!(actual, 2);
            }
            _ => panic!("expected SizeMismatch error"),
        }
    }

    #[test]
    fn snapshot_returns_result() {
        let buf = DeltaBuffer::new(4);
        buf.push(0, vec![1, 2, 3, 4]).unwrap();
        let result = buf.snapshot();
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn len_returns_result() {
        let buf = DeltaBuffer::new(4);
        buf.push(0, vec![1, 2, 3, 4]).unwrap();
        let result = buf.len();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1);
    }

    #[test]
    fn drain_returns_result() {
        let buf = DeltaBuffer::new(4);
        buf.push(0, vec![1, 2, 3, 4]).unwrap();
        let result = buf.drain();
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn restore_returns_result() {
        let buf = DeltaBuffer::new(4);
        let entries = vec![DeltaEntry {
            row_idx: 0,
            diff: vec![1, 2, 3, 4],
        }];
        let result = buf.restore(entries);
        assert!(result.is_ok());
    }

    #[test]
    fn with_max_entries_enforces_limit() {
        let buf = DeltaBuffer::with_max_entries(4, 2);
        assert_eq!(buf.max_entries(), Some(2));

        buf.push(0, vec![1, 2, 3, 4]).unwrap();
        buf.push(1, vec![5, 6, 7, 8]).unwrap();

        let result = buf.push(2, vec![9, 10, 11, 12]);
        assert!(matches!(
            result,
            Err(DeltaError::BufferFull { current: 2, max: 2 })
        ));
    }

    #[test]
    fn with_max_entries_allows_push_after_drain() {
        let buf = DeltaBuffer::with_max_entries(4, 1);
        buf.push(0, vec![1, 2, 3, 4]).unwrap();

        assert!(buf.push(1, vec![5, 6, 7, 8]).is_err());

        buf.drain().unwrap();
        assert!(buf.push(2, vec![9, 10, 11, 12]).is_ok());
    }

    #[test]
    fn restore_for_epoch_warns_on_overflow_but_restores_data() {
        let buf = DeltaBuffer::with_max_entries(4, 2);

        buf.push(0, vec![1, 2, 3, 4]).unwrap();
        buf.push(1, vec![5, 6, 7, 8]).unwrap();

        let drained = buf.drain_for_epoch(1).unwrap();
        assert_eq!(drained.len(), 2);

        buf.push(2, vec![9, 10, 11, 12]).unwrap();
        buf.push(3, vec![13, 14, 15, 16]).unwrap();

        let result = buf.restore_for_epoch(0, drained);

        assert!(
            result.is_err(),
            "restore_for_epoch should return error when exceeding max_entries"
        );
        match result {
            Err(DeltaError::BufferFull { current, max }) => {
                assert_eq!(max, 2);
                assert_eq!(current, 4);
            }
            _ => panic!("expected BufferFull error"),
        }

        assert_eq!(
            buf.pending_epoch(),
            0,
            "pending_epoch must be reset even on overflow"
        );
        assert_eq!(
            buf.len().unwrap(),
            4,
            "all entries must be preserved (restored + existing)"
        );

        let snap = buf.snapshot().unwrap();
        assert_eq!(snap[0].row_idx, 0, "restored entries come first");
        assert_eq!(snap[1].row_idx, 1);
        assert_eq!(snap[2].row_idx, 2, "existing entries come after");
        assert_eq!(snap[3].row_idx, 3);
    }

    #[test]
    fn restore_for_epoch_succeeds_when_within_limit() {
        let buf = DeltaBuffer::with_max_entries(4, 4);

        buf.push(0, vec![1, 2, 3, 4]).unwrap();
        let drained = buf.drain_for_epoch(1).unwrap();

        buf.push(1, vec![5, 6, 7, 8]).unwrap();

        let result = buf.restore_for_epoch(0, drained);
        assert!(result.is_ok());
        assert_eq!(buf.len().unwrap(), 2);
    }

    #[test]
    fn unbounded_buffer_has_no_limit() {
        let buf = DeltaBuffer::new(4);
        assert_eq!(buf.max_entries(), None);

        for i in 0..1000 {
            buf.push(i, vec![1, 2, 3, 4]).unwrap();
        }
        assert_eq!(buf.len().unwrap(), 1000);
    }

    #[test]
    fn with_max_entries_and_epoch_sets_both() {
        let buf = DeltaBuffer::with_max_entries_and_epoch(4, 100, 42);

        assert_eq!(buf.row_size_bytes(), 4);
        assert_eq!(buf.max_entries(), Some(100));
        assert_eq!(buf.pending_epoch(), 42);
        assert!(buf.is_empty().unwrap());
    }

    #[test]
    fn with_max_entries_and_epoch_enforces_limit() {
        let buf = DeltaBuffer::with_max_entries_and_epoch(4, 2, 10);

        buf.push(0, vec![1, 2, 3, 4]).unwrap();
        buf.push(1, vec![5, 6, 7, 8]).unwrap();

        let result = buf.push(2, vec![9, 10, 11, 12]);
        assert!(matches!(
            result,
            Err(DeltaError::BufferFull { current: 2, max: 2 })
        ));
    }

    #[test]
    fn restore_rejects_wrong_size_entry() {
        let buf = DeltaBuffer::new(4);

        let bad_entries = vec![DeltaEntry {
            row_idx: 0,
            diff: vec![1, 2, 3], // 3 bytes, expected 4
        }];

        let result = buf.restore(bad_entries);
        assert!(matches!(
            result,
            Err(DeltaError::SizeMismatch {
                expected: 4,
                actual: 3
            })
        ));
        assert!(
            buf.is_empty().unwrap(),
            "buffer should remain empty on error"
        );
    }

    #[test]
    fn restore_for_epoch_rejects_wrong_size_entry() {
        let buf = DeltaBuffer::new(4);

        let bad_entries = vec![DeltaEntry {
            row_idx: 0,
            diff: vec![1, 2, 3, 4, 5], // 5 bytes, expected 4
        }];

        let result = buf.restore_for_epoch(10, bad_entries);
        assert!(matches!(
            result,
            Err(DeltaError::SizeMismatch {
                expected: 4,
                actual: 5
            })
        ));
        assert!(
            buf.is_empty().unwrap(),
            "buffer should remain empty on error"
        );
        assert_eq!(buf.pending_epoch(), 0, "epoch should not change on error");
    }

    #[test]
    fn restore_rejects_mixed_valid_and_invalid_entries() {
        let buf = DeltaBuffer::new(4);

        let mixed_entries = vec![
            DeltaEntry {
                row_idx: 0,
                diff: vec![1, 2, 3, 4], // valid
            },
            DeltaEntry {
                row_idx: 1,
                diff: vec![5, 6], // invalid - 2 bytes
            },
        ];

        let result = buf.restore(mixed_entries);
        assert!(matches!(
            result,
            Err(DeltaError::SizeMismatch {
                expected: 4,
                actual: 2
            })
        ));
        assert!(
            buf.is_empty().unwrap(),
            "buffer should remain empty on error"
        );
    }

    #[test]
    fn restore_for_epoch_empty_entries_ok_when_within_limit() {
        let buf = DeltaBuffer::with_max_entries(4, 2);

        // Add one entry (below max)
        buf.push(0, vec![1, 2, 3, 4]).unwrap();

        // Call restore_for_epoch with EMPTY entries
        // Buffer has 1 entry, max is 2 - no overflow
        let result = buf.restore_for_epoch(5, vec![]);

        assert!(result.is_ok());
        assert_eq!(buf.len().unwrap(), 1);
        assert_eq!(buf.pending_epoch(), 5);
    }

    #[test]
    fn restore_for_epoch_empty_entries_checks_existing_overflow() {
        let buf = DeltaBuffer::with_max_entries(4, 2);

        // Fill buffer beyond max using restore_for_epoch (which allows overflow)
        buf.push(0, vec![1, 2, 3, 4]).unwrap();
        buf.push(1, vec![5, 6, 7, 8]).unwrap();
        let drained = buf.drain_for_epoch(1).unwrap();

        buf.push(2, vec![9, 10, 11, 12]).unwrap();
        buf.push(3, vec![13, 14, 15, 16]).unwrap();

        // This will exceed max (4 entries, max 2) but still restore
        let _ = buf.restore_for_epoch(0, drained);
        assert_eq!(buf.len().unwrap(), 4); // 4 entries, max is 2

        // Now call restore_for_epoch with EMPTY entries
        // Existing buffer already exceeds max - should report overflow
        let result = buf.restore_for_epoch(5, vec![]);

        assert!(
            matches!(result, Err(DeltaError::BufferFull { current: 4, max: 2 })),
            "should report BufferFull even with empty entries when existing exceeds max"
        );
        // Epoch should still be updated
        assert_eq!(buf.pending_epoch(), 5);
    }

    #[test]
    fn entry_count_overflow_error_exists_and_formats() {
        let err = DeltaError::EntryCountOverflow;
        let msg = err.to_string();
        assert!(
            msg.contains("overflow"),
            "error message should mention overflow: {}",
            msg
        );
    }
}

#[cfg(test)]
mod concurrency_tests {
    use super::*;
    use std::sync::{Arc, Barrier};
    use std::thread;

    #[test]
    fn snapshot_with_epoch_is_atomic_with_drain() {
        let buf = Arc::new(DeltaBuffer::new(4));
        let iterations = 1000;
        let torn_reads = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        for _ in 0..iterations {
            buf.push(0, vec![0xAA; 4]).unwrap();

            let buf_clone = buf.clone();
            let torn_clone = torn_reads.clone();
            let barrier = Arc::new(Barrier::new(2));
            let barrier_clone = barrier.clone();

            let drainer = thread::spawn(move || {
                barrier_clone.wait();
                let _ = buf_clone.drain_for_epoch(buf_clone.pending_epoch() + 1);
            });

            barrier.wait();
            let (epoch, entries) = buf.snapshot_with_epoch().unwrap();

            drainer.join().unwrap();

            let current_epoch = buf.pending_epoch();
            if epoch < current_epoch && entries.is_empty() {
                torn_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }

        let torn = torn_reads.load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(
            torn, 0,
            "snapshot_with_epoch had {} torn reads out of {} iterations - \
             saw old epoch with empty entries (race condition!)",
            torn, iterations
        );
    }

    #[test]
    fn snapshot_with_epoch_never_returns_old_epoch_with_empty_entries_after_drain() {
        let buf = Arc::new(DeltaBuffer::new(4));

        let step1 = Arc::new(Barrier::new(2));
        let step2 = Arc::new(Barrier::new(2));

        buf.push(0, vec![0xFF; 4]).unwrap();
        let initial_epoch = buf.pending_epoch();

        let buf_clone = buf.clone();
        let step1_clone = step1.clone();
        let step2_clone = step2.clone();

        let drainer = thread::spawn(move || {
            step1_clone.wait();
            let _ = buf_clone.drain_for_epoch(initial_epoch + 1);
            step2_clone.wait();
        });

        step1.wait();
        step2.wait();

        let (epoch, entries) = buf.snapshot_with_epoch().unwrap();

        drainer.join().unwrap();

        if entries.is_empty() {
            assert_eq!(
                epoch,
                initial_epoch + 1,
                "if entries are empty after drain, epoch must be the new epoch, not old"
            );
        }
    }

    #[test]
    fn snapshot_with_epoch_lock_ordering_prevents_torn_read() {
        use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

        let old_epoch = Arc::new(AtomicU64::new(0));
        let entries_empty = Arc::new(AtomicBool::new(false));
        let new_epoch_set = Arc::new(AtomicBool::new(false));

        let buf = Arc::new(DeltaBuffer::new(4));
        buf.push(0, vec![0xAA; 4]).unwrap();

        let buf_clone = buf.clone();
        let old_epoch_clone = old_epoch.clone();
        let entries_empty_clone = entries_empty.clone();
        let new_epoch_clone = new_epoch_set.clone();

        let reader = thread::spawn(move || {
            while !new_epoch_clone.load(Ordering::Acquire) {
                let (epoch, entries) = buf_clone.snapshot_with_epoch().unwrap();

                if entries.is_empty() && epoch == old_epoch_clone.load(Ordering::Acquire) {
                    entries_empty_clone.store(true, Ordering::Release);
                    return true;
                }
            }
            false
        });

        old_epoch.store(buf.pending_epoch(), Ordering::Release);

        for _ in 0..100 {
            std::thread::yield_now();
        }

        let _ = buf.drain_for_epoch(buf.pending_epoch() + 1);
        new_epoch_set.store(true, Ordering::Release);

        let saw_torn = reader.join().unwrap();
        assert!(
            !saw_torn,
            "Reader saw torn read: old epoch with empty entries"
        );
    }
}
