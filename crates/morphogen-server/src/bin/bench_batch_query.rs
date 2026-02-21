#[cfg(not(feature = "network"))]
fn main() {
    eprintln!("bench_batch_query requires --features network");
    std::process::exit(1);
}

#[cfg(feature = "network")]
use std::env;
#[cfg(feature = "network")]
use std::sync::Arc;
#[cfg(feature = "network")]
use std::time::Instant;

#[cfg(feature = "network")]
use axum::extract::State;
#[cfg(feature = "network")]
use axum::Json;
#[cfg(feature = "network")]
use morphogen_core::{DeltaBuffer, EpochSnapshot, GlobalState, ROW_SIZE_BYTES};
#[cfg(feature = "network")]
use morphogen_dpf::AesDpfKey;
#[cfg(feature = "network")]
use morphogen_server::network::api::{
    batch_query_handler, AppState, BatchQueryRequest, EpochMetadata, QueryRequest, MAX_BATCH_SIZE,
};
#[cfg(feature = "network")]
use morphogen_storage::ChunkedMatrix;
#[cfg(feature = "network")]
use rand::{rngs::StdRng, SeedableRng};
#[cfg(feature = "network")]
use tokio::sync::watch;

#[cfg(feature = "network")]
const DEFAULT_ROWS: usize = 262_144;
#[cfg(feature = "network")]
const DEFAULT_ITERATIONS: usize = 10;
#[cfg(feature = "network")]
const DEFAULT_WARMUP_ITERATIONS: usize = 2;
#[cfg(feature = "network")]
const DEFAULT_CHUNK_SIZE_BYTES: usize = 4 * 1024 * 1024;
#[cfg(feature = "network")]
const DEFAULT_BATCH_SIZES: &str = "1,2,4,8,16,32";
#[cfg(feature = "network")]
const DEFAULT_SEED: u64 = 0xBAD5_EED;

#[cfg(feature = "network")]
#[derive(Debug)]
struct BenchConfig {
    rows: usize,
    iterations: usize,
    warmup_iterations: usize,
    chunk_size_bytes: usize,
    batch_sizes: Vec<usize>,
    matrix_seed: u64,
    with_delta: bool,
}

#[cfg(feature = "network")]
#[tokio::main(flavor = "current_thread")]
async fn main() {
    let cfg = parse_config(env::args().collect());
    let row_size_bytes = ROW_SIZE_BYTES;
    let matrix_size_bytes = cfg
        .rows
        .checked_mul(row_size_bytes)
        .expect("rows * row_size_bytes overflow");

    println!(
        "batch_query_bench rows={} row_size={} iterations={} warmup={} chunk_size={} with_delta={} matrix_seed={}",
        cfg.rows,
        row_size_bytes,
        cfg.iterations,
        cfg.warmup_iterations,
        cfg.chunk_size_bytes,
        cfg.with_delta,
        cfg.matrix_seed
    );
    println!(
        "q,total_ms,ms_per_batch,ms_per_query,p95_batch_ms,queries_per_sec,effective_gbps,checksum"
    );

    let state = build_state(&cfg, matrix_size_bytes, row_size_bytes);
    let mut rng = StdRng::seed_from_u64(cfg.matrix_seed ^ 0xA11C_E0DE);

    for &batch_size in &cfg.batch_sizes {
        let template = build_query_template(cfg.rows, batch_size, &mut rng);

        for _ in 0..cfg.warmup_iterations {
            let request = build_batch_request(&template);
            let _ = batch_query_handler(State(state.clone()), Json(request))
                .await
                .expect("warmup batch query failed");
        }

        let bench_start = Instant::now();
        let mut iter_ms = Vec::with_capacity(cfg.iterations);
        let mut checksum = 0u64;

        for _ in 0..cfg.iterations {
            let request = build_batch_request(&template);
            let iter_start = Instant::now();
            let response = batch_query_handler(State(state.clone()), Json(request))
                .await
                .expect("batch query failed")
                .0;
            let elapsed = iter_start.elapsed();
            iter_ms.push(elapsed.as_secs_f64() * 1000.0);
            checksum = checksum.wrapping_add(checksum_response(&response));
        }

        let total = bench_start.elapsed();
        let total_ms = total.as_secs_f64() * 1000.0;
        let total_queries = (batch_size * cfg.iterations) as f64;
        let ms_per_batch = total_ms / cfg.iterations as f64;
        let ms_per_query = total_ms / total_queries;
        let queries_per_sec = total_queries / total.as_secs_f64().max(1e-9);
        let total_scan_bytes = (matrix_size_bytes * batch_size * cfg.iterations) as f64;
        let effective_gbps =
            total_scan_bytes / (1024.0 * 1024.0 * 1024.0) / total.as_secs_f64().max(1e-9);
        let p95_batch_ms = percentile_ms(&iter_ms, 0.95);

        println!(
            "{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{}",
            batch_size,
            total_ms,
            ms_per_batch,
            ms_per_query,
            p95_batch_ms,
            queries_per_sec,
            effective_gbps,
            checksum
        );
    }
}

#[cfg(feature = "network")]
fn parse_config(args: Vec<String>) -> BenchConfig {
    let rows = parse_arg(&args, "--rows").unwrap_or(DEFAULT_ROWS);
    assert!(rows > 0, "--rows must be > 0");

    let iterations = parse_arg(&args, "--iterations").unwrap_or(DEFAULT_ITERATIONS);
    assert!(iterations > 0, "--iterations must be > 0");

    let warmup_iterations =
        parse_arg(&args, "--warmup-iterations").unwrap_or(DEFAULT_WARMUP_ITERATIONS);
    let chunk_size_bytes = parse_arg(&args, "--chunk-size").unwrap_or(DEFAULT_CHUNK_SIZE_BYTES);
    assert!(
        chunk_size_bytes > 0 && chunk_size_bytes.is_multiple_of(ROW_SIZE_BYTES),
        "--chunk-size must be > 0 and divisible by row_size ({ROW_SIZE_BYTES})"
    );

    let batch_sizes_raw =
        parse_arg_string(&args, "--batch-sizes").unwrap_or_else(|| DEFAULT_BATCH_SIZES.to_string());
    let batch_sizes = parse_batch_sizes(&batch_sizes_raw).expect("invalid --batch-sizes");

    let matrix_seed = parse_arg(&args, "--matrix-seed").unwrap_or(DEFAULT_SEED);
    let with_delta = args.contains(&"--with-delta".to_string());

    BenchConfig {
        rows,
        iterations,
        warmup_iterations,
        chunk_size_bytes,
        batch_sizes,
        matrix_seed,
        with_delta,
    }
}

#[cfg(feature = "network")]
fn parse_batch_sizes(value: &str) -> Result<Vec<usize>, String> {
    let mut out = Vec::new();
    for token in value.split(',') {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }

        let size = token
            .parse::<usize>()
            .map_err(|_| format!("invalid batch size: {token}"))?;
        if size == 0 {
            return Err("batch sizes must be >= 1".to_string());
        }
        if size > MAX_BATCH_SIZE {
            return Err(format!(
                "batch size {size} exceeds MAX_BATCH_SIZE ({MAX_BATCH_SIZE})"
            ));
        }
        out.push(size);
    }

    if out.is_empty() {
        return Err("at least one batch size is required".to_string());
    }

    Ok(out)
}

#[cfg(feature = "network")]
fn parse_arg<T: std::str::FromStr>(args: &[String], name: &str) -> Option<T> {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|idx| args.get(idx + 1))
        .and_then(|value| value.parse::<T>().ok())
}

#[cfg(feature = "network")]
fn parse_arg_string(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|idx| args.get(idx + 1))
        .cloned()
}

#[cfg(feature = "network")]
fn build_state(
    cfg: &BenchConfig,
    matrix_size_bytes: usize,
    row_size_bytes: usize,
) -> Arc<AppState> {
    let mut matrix = Arc::new(ChunkedMatrix::new(matrix_size_bytes, cfg.chunk_size_bytes));
    Arc::get_mut(&mut matrix)
        .expect("matrix is uniquely owned at initialization")
        .fill_with_pattern(cfg.matrix_seed);

    let snapshot = Arc::new(EpochSnapshot {
        epoch_id: 1,
        matrix,
    });
    let pending = Arc::new(DeltaBuffer::new_with_epoch(row_size_bytes, 1));
    if cfg.with_delta {
        pending
            .push(cfg.rows / 2, vec![0xA5; row_size_bytes])
            .expect("failed to insert synthetic pending delta");
    }
    let global = Arc::new(GlobalState::new(snapshot, pending));

    let metadata = EpochMetadata {
        epoch_id: 1,
        num_rows: cfg.rows,
        seeds: [0x1234, 0x5678, 0x9ABC],
        block_number: 1,
        state_root: [0xAB; 32],
    };
    let (_tx, rx) = watch::channel(metadata);

    Arc::new(AppState {
        global,
        row_size_bytes,
        num_rows: cfg.rows,
        seeds: [0x1234, 0x5678, 0x9ABC],
        block_number: 1,
        state_root: [0xAB; 32],
        epoch_rx: rx,
        page_config: None,
        #[cfg(feature = "cuda")]
        gpu_scanner: None,
        #[cfg(feature = "cuda")]
        gpu_matrix: None,
    })
}

#[cfg(feature = "network")]
fn build_query_template(rows: usize, batch_size: usize, rng: &mut StdRng) -> Vec<[Vec<u8>; 3]> {
    let offset1 = rows / 3;
    let offset2 = (2 * rows) / 3;
    let rows_u64 = rows as u64;

    (0..batch_size)
        .map(|i| {
            let base = ((i as u64).wrapping_mul(1_146_959_810_393_466_559) % rows_u64) as usize;
            let k0 = AesDpfKey::new_single(rng, base);
            let k1 = AesDpfKey::new_single(rng, (base + offset1) % rows);
            let k2 = AesDpfKey::new_single(rng, (base + offset2) % rows);
            [
                k0.to_bytes().to_vec(),
                k1.to_bytes().to_vec(),
                k2.to_bytes().to_vec(),
            ]
        })
        .collect()
}

#[cfg(feature = "network")]
fn build_batch_request(template: &[[Vec<u8>; 3]]) -> BatchQueryRequest {
    let queries = template
        .iter()
        .map(|keys| QueryRequest {
            keys: keys.to_vec(),
        })
        .collect();
    BatchQueryRequest { queries }
}

#[cfg(feature = "network")]
fn checksum_response(response: &morphogen_server::network::api::BatchQueryResponse) -> u64 {
    response
        .results
        .iter()
        .flat_map(|result| result.payloads.iter())
        .flat_map(|payload| payload.iter())
        .fold(0u64, |acc, b| acc.wrapping_add(*b as u64))
}

#[cfg(feature = "network")]
fn percentile_ms(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let idx = ((sorted.len() as f64 * p).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[idx]
}

#[cfg(all(test, feature = "network"))]
mod tests {
    use super::*;

    #[test]
    fn parse_batch_sizes_accepts_csv() {
        let parsed = parse_batch_sizes("1,2,4,8").unwrap();
        assert_eq!(parsed, vec![1, 2, 4, 8]);
    }

    #[test]
    fn parse_batch_sizes_rejects_zero() {
        let err = parse_batch_sizes("1,0,2").unwrap_err();
        assert!(err.contains(">= 1"));
    }

    #[test]
    fn parse_batch_sizes_rejects_above_limit() {
        let err = parse_batch_sizes("1,64").unwrap_err();
        assert!(err.contains("MAX_BATCH_SIZE"));
    }
}
