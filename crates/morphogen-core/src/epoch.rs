use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use arc_swap::ArcSwap;

use crate::DeltaBuffer;
use morphogen_storage::ChunkedMatrix;

pub struct EpochSnapshot {
    pub epoch_id: u64,
    pub matrix: Arc<ChunkedMatrix>,
}

pub struct GlobalState {
    current_snapshot: ArcSwap<EpochSnapshot>,
    pending: ArcSwap<DeltaBuffer>,
    has_manager: AtomicBool,
}

impl GlobalState {
    pub fn new(initial: Arc<EpochSnapshot>, pending: Arc<DeltaBuffer>) -> Self {
        Self {
            current_snapshot: ArcSwap::from(initial),
            pending: ArcSwap::from(pending),
            has_manager: AtomicBool::new(false),
        }
    }

    pub fn load(&self) -> Arc<EpochSnapshot> {
        self.current_snapshot.load_full()
    }

    pub fn store(&self, new_snapshot: Arc<EpochSnapshot>) {
        self.current_snapshot.store(new_snapshot);
    }

    pub fn load_pending(&self) -> Arc<DeltaBuffer> {
        self.pending.load_full()
    }

    pub fn store_pending(&self, new_pending: Arc<DeltaBuffer>) {
        self.pending.store(new_pending);
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
