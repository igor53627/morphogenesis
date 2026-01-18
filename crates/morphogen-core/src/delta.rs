use std::sync::RwLock;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeltaEntry {
    pub row_idx: usize,
    pub diff: Vec<u8>,
}

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

    pub fn push(&self, row_idx: usize, diff: Vec<u8>) {
        assert_eq!(
            diff.len(),
            self.row_size_bytes,
            "delta diff must match row size"
        );
        let mut guard = self.entries.write().expect("delta write lock poisoned");
        guard.push(DeltaEntry { row_idx, diff });
    }

    pub fn snapshot(&self) -> Vec<DeltaEntry> {
        let guard = self.entries.read().expect("delta read lock poisoned");
        guard.clone()
    }

    pub fn len(&self) -> usize {
        let guard = self.entries.read().expect("delta read lock poisoned");
        guard.len()
    }
}
