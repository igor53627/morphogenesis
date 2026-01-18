use std::sync::Arc;

use arc_swap::ArcSwap;

use morphogen_storage::ChunkedMatrix;

use crate::DeltaBuffer;

pub struct EpochSnapshot {
    pub epoch_id: u64,
    pub matrix: Arc<ChunkedMatrix>,
    pub delta: Arc<DeltaBuffer>,
}

pub struct GlobalState {
    current_snapshot: ArcSwap<EpochSnapshot>,
}

impl GlobalState {
    pub fn new(initial: Arc<EpochSnapshot>) -> Self {
        Self {
            current_snapshot: ArcSwap::from(initial),
        }
    }

    pub fn load(&self) -> Arc<EpochSnapshot> {
        self.current_snapshot.load_full()
    }

    pub fn store(&self, new_snapshot: Arc<EpochSnapshot>) {
        self.current_snapshot.store(new_snapshot);
    }
}
