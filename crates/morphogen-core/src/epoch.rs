use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use arc_swap::ArcSwap;

use morphogen_storage::ChunkedMatrix;

pub struct EpochSnapshot {
    pub epoch_id: u64,
    pub matrix: Arc<ChunkedMatrix>,
}

pub struct GlobalState {
    current_snapshot: ArcSwap<EpochSnapshot>,
    has_manager: AtomicBool,
}

impl GlobalState {
    pub fn new(initial: Arc<EpochSnapshot>) -> Self {
        Self {
            current_snapshot: ArcSwap::from(initial),
            has_manager: AtomicBool::new(false),
        }
    }

    pub fn load(&self) -> Arc<EpochSnapshot> {
        self.current_snapshot.load_full()
    }

    pub fn store(&self, new_snapshot: Arc<EpochSnapshot>) {
        self.current_snapshot.store(new_snapshot);
    }

    pub fn try_acquire_manager(&self) -> bool {
        self.has_manager
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    pub fn release_manager(&self) {
        self.has_manager.store(false, Ordering::Release);
    }
}
