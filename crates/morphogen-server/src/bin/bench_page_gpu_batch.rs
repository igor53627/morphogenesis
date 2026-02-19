#[cfg(not(feature = "network"))]
fn main() {
    eprintln!("bench_page_gpu_batch requires --features network");
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
use axum::http::StatusCode;
#[cfg(feature = "network")]
use axum::Json;
#[cfg(feature = "network")]
use morphogen_core::{DeltaBuffer, EpochSnapshot, GlobalState};
#[cfg(feature = "network")]
use morphogen_gpu_dpf::dpf::{generate_chacha_dpf_keys, ChaChaParams};
#[cfg(feature = "network")]
use morphogen_gpu_dpf::storage::PAGE_SIZE_BYTES;
#[cfg(feature = "network")]
use morphogen_server::network::api::{
    page_query_gpu_batch_handler, page_query_gpu_handler, AppState, BatchGpuPageQueryRequest,
    EpochMetadata, GpuPageQueryRequest, MAX_BATCH_SIZE,
};
#[cfg(feature = "network")]
use morphogen_storage::ChunkedMatrix;
#[cfg(all(feature = "network", feature = "cuda"))]
use std::sync::Mutex;
#[cfg(feature = "network")]
use tokio::sync::watch;

#[cfg(feature = "network")]
const DEFAULT_NUM_PAGES: usize = 256;
#[cfg(feature = "network")]
const DEFAULT_DOMAIN_BITS: usize = 8;
#[cfg(feature = "network")]
const DEFAULT_ITERATIONS: usize = 20;
#[cfg(feature = "network")]
const DEFAULT_WARMUP: usize = 3;
#[cfg(feature = "network")]
const DEFAULT_BATCH_SIZES: &str = "1,2,4,8,16,32";
#[cfg(feature = "network")]
const DEFAULT_GPU_STREAMS: usize = 1;
#[cfg(feature = "network")]
const DEFAULT_ROW_SIZE_BYTES: usize = 256;
#[cfg(feature = "network")]
const DEFAULT_CHUNK_SIZE_BYTES: usize = 1024 * 1024;

#[cfg(feature = "network")]
#[derive(Debug)]
struct BenchConfig {
    num_pages: usize,
    domain_bits: usize,
    iterations: usize,
    warmup_iterations: usize,
    batch_sizes: Vec<usize>,
    gpu_streams: usize,
}

#[cfg(feature = "network")]
#[tokio::main(flavor = "current_thread")]
async fn main() {
    let cfg = parse_config(env::args().collect());
    env::set_var("MORPHOGEN_GPU_STREAMS", cfg.gpu_streams.to_string());
    let state = build_state(cfg.num_pages);
    let params = ChaChaParams::new(cfg.domain_bits).expect("invalid domain bits");
    let mode = bench_mode(state.as_ref());

    println!(
        "gpu_page_batch_bench mode={} pages={} domain_bits={} iterations={} warmup={} gpu_streams={}",
        mode,
        cfg.num_pages,
        cfg.domain_bits,
        cfg.iterations,
        cfg.warmup_iterations,
        cfg.gpu_streams
    );
    println!("gpu_page_batch_bench_backend={mode}");
    println!(
        "q,single_ms_per_batch,batch_ms_per_batch,batch_vs_single_speedup,single_qps,batch_qps,checksum_single,checksum_batch"
    );

    for &q in &cfg.batch_sizes {
        let key_sets = build_key_sets(&params, cfg.num_pages, q);

        for _ in 0..cfg.warmup_iterations {
            let _ = run_single_loop_once(state.clone(), &key_sets).await;
            let _ = run_batch_once(state.clone(), &key_sets).await;
        }

        let single_start = Instant::now();
        let mut single_checksum = 0u64;
        for _ in 0..cfg.iterations {
            single_checksum = single_checksum.wrapping_add(
                run_single_loop_once(state.clone(), &key_sets)
                    .await
                    .expect("single loop failed"),
            );
        }
        let single_elapsed = single_start.elapsed();

        let batch_start = Instant::now();
        let mut batch_checksum = 0u64;
        for _ in 0..cfg.iterations {
            batch_checksum = batch_checksum.wrapping_add(
                run_batch_once(state.clone(), &key_sets)
                    .await
                    .expect("batch endpoint failed"),
            );
        }
        let batch_elapsed = batch_start.elapsed();

        let single_ms_per_batch = single_elapsed.as_secs_f64() * 1000.0 / cfg.iterations as f64;
        let batch_ms_per_batch = batch_elapsed.as_secs_f64() * 1000.0 / cfg.iterations as f64;
        let speedup = single_ms_per_batch / batch_ms_per_batch.max(1e-9);
        let single_qps = (q * cfg.iterations) as f64 / single_elapsed.as_secs_f64().max(1e-9);
        let batch_qps = (q * cfg.iterations) as f64 / batch_elapsed.as_secs_f64().max(1e-9);

        println!(
            "{},{:.2},{:.2},{:.2},{:.2},{:.2},{},{}",
            q,
            single_ms_per_batch,
            batch_ms_per_batch,
            speedup,
            single_qps,
            batch_qps,
            single_checksum,
            batch_checksum
        );
    }
}

#[cfg(feature = "network")]
fn parse_config(args: Vec<String>) -> BenchConfig {
    let num_pages = parse_arg(&args, "--num-pages").unwrap_or(DEFAULT_NUM_PAGES);
    assert!(num_pages > 0, "--num-pages must be > 0");

    let domain_bits = parse_arg(&args, "--domain-bits").unwrap_or(DEFAULT_DOMAIN_BITS);
    let iterations = parse_arg(&args, "--iterations").unwrap_or(DEFAULT_ITERATIONS);
    assert!(iterations > 0, "--iterations must be > 0");

    let warmup_iterations = parse_arg(&args, "--warmup-iterations").unwrap_or(DEFAULT_WARMUP);
    let gpu_streams = parse_arg(&args, "--gpu-streams").unwrap_or(DEFAULT_GPU_STREAMS);
    assert!(gpu_streams > 0, "--gpu-streams must be > 0");
    let batch_sizes_raw =
        parse_arg_string(&args, "--batch-sizes").unwrap_or_else(|| DEFAULT_BATCH_SIZES.to_string());
    let batch_sizes = parse_batch_sizes(&batch_sizes_raw).expect("invalid --batch-sizes");

    BenchConfig {
        num_pages,
        domain_bits,
        iterations,
        warmup_iterations,
        batch_sizes,
        gpu_streams,
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
fn build_state(num_pages: usize) -> Arc<AppState> {
    let total_size = num_pages * PAGE_SIZE_BYTES;
    let mut matrix = ChunkedMatrix::new(total_size, DEFAULT_CHUNK_SIZE_BYTES);

    // Fill pages with deterministic bytes so checksum is stable.
    let mut page = vec![0u8; PAGE_SIZE_BYTES];
    for page_idx in 0..num_pages {
        page.fill((page_idx & 0xFF) as u8);
        matrix.write_row(page_idx, PAGE_SIZE_BYTES, &page);
    }
    let matrix = Arc::new(matrix);

    let snapshot = EpochSnapshot {
        epoch_id: 1,
        matrix: matrix.clone(),
    };
    let pending = Arc::new(DeltaBuffer::new_with_epoch(DEFAULT_ROW_SIZE_BYTES, 1));
    let global = Arc::new(GlobalState::new(Arc::new(snapshot), pending));

    let metadata = EpochMetadata {
        epoch_id: 1,
        num_rows: num_pages * 16,
        seeds: [0x1234, 0x5678, 0x9ABC],
        block_number: 1,
        state_root: [0xAB; 32],
    };
    let (_tx, rx) = watch::channel(metadata);

    #[cfg(feature = "cuda")]
    let (gpu_scanner, gpu_matrix) = {
        use morphogen_gpu_dpf::kernel::GpuScanner;
        use morphogen_gpu_dpf::storage::GpuPageMatrix;

        match GpuScanner::new(0) {
            Ok(scanner) => {
                let scanner = Arc::new(scanner);
                match GpuPageMatrix::from_chunked_matrix(scanner.device.clone(), matrix.as_ref()) {
                    Ok(matrix) => (Some(scanner), Some(Arc::new(Mutex::new(Some(matrix))))),
                    Err(e) => {
                        eprintln!(
                            "warning: failed to initialize GPU matrix (falling back to CPU path): {e}"
                        );
                        (None, None)
                    }
                }
            }
            Err(e) => {
                eprintln!(
                    "warning: failed to initialize GPU scanner (falling back to CPU path): {e}"
                );
                (None, None)
            }
        }
    };

    Arc::new(AppState {
        global,
        row_size_bytes: DEFAULT_ROW_SIZE_BYTES,
        num_rows: num_pages * 16,
        seeds: [0x1234, 0x5678, 0x9ABC],
        block_number: 1,
        state_root: [0xAB; 32],
        epoch_rx: rx,
        page_config: None,
        #[cfg(feature = "cuda")]
        gpu_scanner,
        #[cfg(feature = "cuda")]
        gpu_matrix,
    })
}

#[cfg(all(feature = "network", feature = "cuda"))]
fn bench_mode(state: &AppState) -> &'static str {
    if state.gpu_scanner.is_some() && state.gpu_matrix.is_some() {
        "cuda"
    } else {
        "cpu_fallback"
    }
}

#[cfg(all(feature = "network", not(feature = "cuda")))]
fn bench_mode(_state: &AppState) -> &'static str {
    "cpu_fallback"
}

#[cfg(feature = "network")]
fn build_key_sets(params: &ChaChaParams, num_pages: usize, q: usize) -> Vec<Vec<Vec<u8>>> {
    (0..q)
        .map(|i| {
            let target = (i * 97) % num_pages;
            let (k0, _) = generate_chacha_dpf_keys(params, target).expect("key generation failed");
            let key = k0.to_bytes().to_vec();
            vec![key.clone(), key.clone(), key]
        })
        .collect()
}

#[cfg(feature = "network")]
async fn run_single_loop_once(
    state: Arc<AppState>,
    key_sets: &[Vec<Vec<u8>>],
) -> Result<u64, StatusCode> {
    let mut checksum = 0u64;
    for keys in key_sets {
        let response = page_query_gpu_handler(
            State(state.clone()),
            Json(GpuPageQueryRequest { keys: keys.clone() }),
        )
        .await?
        .0;
        checksum = checksum.wrapping_add(checksum_pages(&response.pages));
    }
    Ok(checksum)
}

#[cfg(feature = "network")]
async fn run_batch_once(
    state: Arc<AppState>,
    key_sets: &[Vec<Vec<u8>>],
) -> Result<u64, StatusCode> {
    let queries = key_sets
        .iter()
        .cloned()
        .map(|keys| GpuPageQueryRequest { keys })
        .collect();
    let response =
        page_query_gpu_batch_handler(State(state), Json(BatchGpuPageQueryRequest { queries }))
            .await?
            .0;
    let checksum = response
        .results
        .iter()
        .map(|result| checksum_pages(&result.pages))
        .fold(0u64, |acc, v| acc.wrapping_add(v));
    Ok(checksum)
}

#[cfg(feature = "network")]
fn checksum_pages(pages: &[Vec<u8>]) -> u64 {
    pages
        .iter()
        .flat_map(|p| p.iter())
        .fold(0u64, |acc, b| acc.wrapping_add(*b as u64))
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
        let err = parse_batch_sizes("1,0").unwrap_err();
        assert!(err.contains(">= 1"));
    }

    #[test]
    fn parse_batch_sizes_rejects_above_limit() {
        let err = parse_batch_sizes("33").unwrap_err();
        assert!(err.contains("MAX_BATCH_SIZE"));
    }

    #[test]
    fn parse_config_reads_gpu_streams() {
        let args = vec![
            "bench_page_gpu_batch".to_string(),
            "--gpu-streams".to_string(),
            "4".to_string(),
        ];
        let cfg = parse_config(args);
        assert_eq!(cfg.gpu_streams, 4);
    }
}
