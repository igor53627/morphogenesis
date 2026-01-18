use std::env;
use std::time::Instant;

use morphogen_dpf::AesDpfKey;
use morphogen_server::{Environment, MorphogenServer, ServerConfig};

#[cfg(feature = "parallel")]
fn do_scan(server: &MorphogenServer, keys: &[AesDpfKey; 3], use_parallel: bool) -> [Vec<u8>; 3] {
    let (results, _epoch_id) = if use_parallel {
        server.scan_parallel(keys).expect("scan failed")
    } else {
        server.scan(keys).expect("scan failed")
    };
    results
}

#[cfg(not(feature = "parallel"))]
fn do_scan(server: &MorphogenServer, keys: &[AesDpfKey; 3], _use_parallel: bool) -> [Vec<u8>; 3] {
    let (results, _epoch_id) = server.scan(keys).expect("scan failed");
    results
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let rows = parse_arg(&args, "--rows").unwrap_or(1024);
    let iterations = parse_arg(&args, "--iterations").unwrap_or(5);
    let warmup_iterations = parse_arg(&args, "--warmup-iterations").unwrap_or(0);
    let row_size_override = parse_arg(&args, "--row-size");
    let no_fill = args.contains(&"--no-fill".to_string());
    let scan_only = args.contains(&"--scan-only".to_string());
    let use_parallel = args.contains(&"--parallel".to_string());

    let mut config = ServerConfig::for_env(Environment::Test);
    if let Some(rs) = row_size_override {
        config.row_size_bytes = rs;
    }
    config.matrix_size_bytes = rows * config.row_size_bytes;
    if no_fill {
        config.bench_fill_seed = None;
    } else {
        config.bench_fill_seed = Some(1);
    }
    let row_size_bytes = config.row_size_bytes;

    let setup_start = Instant::now();
    let server = MorphogenServer::new(config).expect("invalid config");
    let setup_elapsed = setup_start.elapsed();
    let mut rng = rand::thread_rng();
    let keys = [
        AesDpfKey::new_single(&mut rng, 0),
        AesDpfKey::new_single(&mut rng, rows.saturating_sub(1)),
        AesDpfKey::new_single(&mut rng, rows / 2),
    ];

    if warmup_iterations > 0 {
        for _ in 0..warmup_iterations {
            let _ = server.warmup_matrix();
        }
    }

    let snapshot = do_scan(&server, &keys, use_parallel);
    let checksum: u64 = snapshot
        .iter()
        .flat_map(|buf| buf.iter())
        .fold(0u64, |acc, v| acc.wrapping_add(*v as u64));

    #[cfg(feature = "profiling")]
    {
        let use_perf = args.contains(&"--perf".to_string());
        let use_flamegraph = args.contains(&"--flamegraph".to_string());
        if use_perf {
            eprintln!("Note: Run with 'perf record' for detailed CPU profiling");
            eprintln!(
                "Example: perf record -g -- ./target/release/bench_scan --rows {} --iterations {}",
                rows, iterations
            );
        }
        if use_flamegraph {
            eprintln!("Note: Install cargo-flamegraph for flamegraph generation");
            eprintln!(
                "Example: cargo flamegraph --bin bench_scan -- --rows {} --iterations {}",
                rows, iterations
            );
        }
    }

    let start = Instant::now();
    let mut iteration_ms = Vec::with_capacity(iterations);
    for iter in 0..iterations {
        let iter_start = Instant::now();
        let _ = do_scan(&server, &keys, use_parallel);
        let iter_elapsed = iter_start.elapsed();
        let iter_ms = iter_elapsed.as_secs_f64() * 1000.0;
        iteration_ms.push(iter_ms);
        let iter_bytes = (rows * row_size_bytes) as f64;
        let iter_gb_per_sec =
            iter_bytes / (1024.0 * 1024.0 * 1024.0) / iter_elapsed.as_secs_f64().max(1e-9);
        println!(
            "scan_iter {} ms={:.2} gb_per_sec={:.2}",
            iter + 1,
            iter_ms,
            iter_gb_per_sec
        );
    }
    let elapsed = start.elapsed();
    let elapsed_secs = elapsed.as_secs_f64().max(1e-9);
    let total_bytes = (rows * row_size_bytes) as f64 * iterations as f64;
    let gb_per_sec = total_bytes / (1024.0 * 1024.0 * 1024.0) / elapsed_secs;

    if scan_only {
        println!(
            "scan_bench rows={} iterations={} scan_ms={} gb_per_sec={:.2} checksum={}",
            rows,
            iterations,
            elapsed.as_millis(),
            gb_per_sec,
            checksum
        );
    } else {
        println!(
            "scan_bench rows={} iterations={} setup_ms={} scan_ms={} gb_per_sec={:.2} checksum={}",
            rows,
            iterations,
            setup_elapsed.as_millis(),
            elapsed.as_millis(),
            gb_per_sec,
            checksum
        );
    }
}

fn parse_arg(args: &[String], name: &str) -> Option<usize> {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|idx| args.get(idx + 1))
        .and_then(|value| value.parse::<usize>().ok())
}
