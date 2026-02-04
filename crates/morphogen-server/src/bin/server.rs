//! Production server binary.

use morphogen_core::{DeltaBuffer, EpochSnapshot, GlobalState};
use morphogen_server::{
    epoch::EpochManager,
    network::{
        create_router_with_concurrency, telemetry, AppState, EpochMetadata, PagePirConfig,
        MAX_CONCURRENT_SCANS,
    },
    Environment, ServerConfig,
};
use morphogen_storage::ChunkedMatrix;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::watch;

/// Type alias for stub GPU scanner (non-CUDA builds).
type StubGpuScanner = Option<Arc<()>>;

/// Type alias for stub GPU matrix (non-CUDA builds).
type StubGpuMatrix = Option<Arc<std::sync::Mutex<Option<()>>>>;

#[tokio::main]
async fn main() {
    // 1. Initialize Telemetry
    telemetry::init_tracing();
    let metrics_handle = telemetry::init_metrics();

    // 2. Configuration (Hardcoded for now, or env vars)
    let config = ServerConfig {
        environment: Environment::Prod,
        matrix_size_bytes: 108 * 1024 * 1024 * 1024, // 108GB
        chunk_size_bytes: 1024 * 1024 * 1024,        // 1GB
        row_size_bytes: 4096,                        // Page size
        bench_fill_seed: Some(42),
    };

    // 3. Initialize State
    let matrix = Arc::new(ChunkedMatrix::new(
        config.matrix_size_bytes,
        config.chunk_size_bytes,
    ));
    // Fill if needed...

    let snapshot = EpochSnapshot {
        epoch_id: 0,
        matrix,
    };
    let pending = Arc::new(DeltaBuffer::new_with_epoch(config.row_size_bytes, 0));
    let global = Arc::new(GlobalState::new(Arc::new(snapshot), pending.clone()));

    let (_epoch_tx, epoch_rx) = watch::channel(EpochMetadata {
        epoch_id: 0,
        num_rows: (config.matrix_size_bytes / config.row_size_bytes),
        seeds: [0x1234, 0x5678, 0x9ABC],
        block_number: 0,
        state_root: [0u8; 32],
    });

    // 4. GPU Setup
    #[cfg(feature = "cuda")]
    let (gpu_scanner, gpu_matrix) = {
        use morphogen_gpu_dpf::kernel::GpuScanner;
        use morphogen_gpu_dpf::storage::GpuPageMatrix;
        use std::sync::Mutex;

        tracing::info!("Initializing GPU...");
        let scanner = Arc::new(GpuScanner::new(0).expect("Failed to init GPU scanner"));
        // Allocate empty GPU matrix (lazy load or sync from CPU)
        // For production, we'd load data. Here we alloc empty for demo.
        let num_pages = config.matrix_size_bytes / 4096;
        let matrix = GpuPageMatrix::alloc_empty(scanner.device.clone(), num_pages)
            .expect("VRAM alloc failed");
        (Some(scanner), Some(Arc::new(Mutex::new(Some(matrix)))))
    };

    #[cfg(not(feature = "cuda"))]
    let (_gpu_scanner, _gpu_matrix): (StubGpuScanner, StubGpuMatrix) = (None, None);

    let state = Arc::new(AppState {
        global: global.clone(),
        row_size_bytes: config.row_size_bytes,
        num_rows: config.matrix_size_bytes / config.row_size_bytes,
        seeds: [0x1234, 0x5678, 0x9ABC],
        block_number: 0,
        state_root: [0u8; 32],
        epoch_rx,
        page_config: Some(PagePirConfig {
            rows_per_page: 1, // Since row_size = page_size
            domain_bits: 25,
            prg_keys: [[0u8; 16], [0u8; 16]],
        }),
        #[cfg(feature = "cuda")]
        gpu_scanner,
        #[cfg(feature = "cuda")]
        gpu_matrix,
    });

    // 5. Epoch Manager (Background worker)
    let _epoch_manager = Arc::new(EpochManager::new(global, config.row_size_bytes).unwrap());
    // Start merge worker... (omitted for brevity)

    // 6. Start Server
    let app = create_router_with_concurrency(state, MAX_CONCURRENT_SCANS, Some(metrics_handle));

    let addr = SocketAddr::from(([0, 0, 0, 0], 3000));
    tracing::info!("Listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
