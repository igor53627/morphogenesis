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
        }
    }
}

impl std::error::Error for DeltaError {}

#[deprecated(since = "0.2.0", note = "Use DeltaError instead")]
pub type DeltaPushError = DeltaError;

pub struct DeltaBuffer {
    row_size_bytes: usize,
    entries: RwLock<Vec<DeltaEntry>>,
}

impl DeltaBuffer {
    pub fn new(row_size_bytes: usize) -> Self {
        Self {
            row_size_bytes,
            entries: RwLock::new(Vec::new()),
        }
    }

    pub fn row_size_bytes(&self) -> usize {
        self.row_size_bytes
    }

    pub fn push(&self, row_idx: usize, diff: Vec<u8>) -> Result<(), DeltaError> {
        if diff.len() != self.row_size_bytes {
            return Err(DeltaError::SizeMismatch {
                expected: self.row_size_bytes,
                actual: diff.len(),
            });
        }
        let mut guard = self.entries.write().map_err(|_| DeltaError::LockPoisoned)?;
        guard.push(DeltaEntry { row_idx, diff });
        Ok(())
    }

    pub fn snapshot(&self) -> Result<Vec<DeltaEntry>, DeltaError> {
        let guard = self.entries.read().map_err(|_| DeltaError::LockPoisoned)?;
        Ok(guard.clone())
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

    pub fn restore(&self, entries: Vec<DeltaEntry>) -> Result<(), DeltaError> {
        if entries.is_empty() {
            return Ok(());
        }
        let mut guard = self.entries.write().map_err(|_| DeltaError::LockPoisoned)?;
        let existing = std::mem::take(&mut *guard);
        *guard = entries;
        guard.extend(existing);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
