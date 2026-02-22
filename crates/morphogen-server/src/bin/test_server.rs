//! Test server for local development and E2E testing.
//! Runs in CPU mode with a small synthetic dataset.

use clap::Parser;
use morphogen_core::{DeltaBuffer, EpochSnapshot, GlobalState};
use morphogen_server::epoch::EpochManager;
use morphogen_server::network::{create_router, AppState, EpochMetadata};
use morphogen_storage::ChunkedMatrix;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::watch;

#[derive(Parser)]
struct Args {
    /// Port to listen on
    #[arg(short, long, default_value_t = 3000)]
    port: u16,

    /// Domain size in bits (e.g. 10 = 1024 pages)
    #[arg(short, long, default_value_t = 10)]
    domain: usize,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let num_pages = 1usize << args.domain;
    let row_size_bytes = 4096; // Standard page size
    let total_size = num_pages * row_size_bytes;

    tracing::info!(
        "Initializing Test Server with {} pages ({:.2} MB)",
        num_pages,
        total_size as f64 / 1e6
    );

    // 1. Initialize State with synthetic data
    let mut matrix = ChunkedMatrix::new(total_size, 1024 * 1024 * 64); // 64MB chunks

    // Fill Page 42 with some recognizable data
    // Balance = 100 ETH (100 * 10^18)
    // 100 ETH = 0x56bc75e2d63100000
    let mut balance = [0u8; 16];
    balance.copy_from_slice(&0x56bc75e2d63100000u128.to_be_bytes());

    let mut row_data = vec![0u8; row_size_bytes];
    row_data[0..16].copy_from_slice(&balance);
    row_data[16..24].copy_from_slice(&123u64.to_be_bytes()); // Nonce = 123
    row_data[24..28].copy_from_slice(&1u32.to_be_bytes()); // CodeID = 1

    matrix.write_row(42, row_size_bytes, &row_data);

    let snapshot = EpochSnapshot {
        epoch_id: 1,
        matrix: Arc::new(matrix),
    };

    let pending = Arc::new(DeltaBuffer::new_with_epoch(row_size_bytes, 1));
    let global = Arc::new(GlobalState::new(Arc::new(snapshot), pending.clone()));

    let seeds = [0x1234, 0x5678, 0x9ABC];
    let (epoch_tx, epoch_rx) = watch::channel(EpochMetadata {
        epoch_id: 1,
        num_rows: num_pages,
        seeds,
        block_number: 1000,
        state_root: [0u8; 32],
    });
    let epoch_manager =
        Arc::new(EpochManager::new(global.clone(), row_size_bytes).expect("epoch manager"));

    let state = Arc::new(AppState {
        global,
        epoch_manager,
        epoch_tx,
        snapshot_rotation_lock: Arc::new(tokio::sync::Mutex::new(())),
        admin_snapshot_token: None,
        admin_snapshot_allow_local_paths: false,
        admin_snapshot_allowed_hosts: vec![],
        admin_snapshot_max_bytes: 1_073_741_824,
        row_size_bytes,
        num_rows: num_pages,
        seeds,
        block_number: 1000,
        state_root: [0u8; 32],
        epoch_rx,
        page_config: None, // No page PIR for this test server
        #[cfg(feature = "cuda")]
        gpu_scanner: None,
        #[cfg(feature = "cuda")]
        gpu_matrix: None,
    });

    // 2. Start Server
    let app = create_router(state);

    let addr = SocketAddr::from(([127, 0, 0, 1], args.port));
    tracing::info!("Test Server listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
