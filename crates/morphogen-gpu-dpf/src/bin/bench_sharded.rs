//! Sharded GPU PIR Benchmark (2x GPUs).
//!
//! Simulates splitting the database across 2 GPUs to measure latency reduction.

use morphogen_gpu_dpf::dpf::{generate_chacha_dpf_keys, ChaChaKey, ChaChaParams};
use morphogen_gpu_dpf::kernel::PAGE_SIZE_BYTES;
use std::env;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

#[cfg(feature = "cuda")]
use morphogen_gpu_dpf::kernel::GpuScanner;
#[cfg(feature = "cuda")]
use morphogen_gpu_dpf::storage::GpuPageMatrix;

fn main() {
    let args: Vec<String> = env::args().collect();
    if !args.iter().any(|arg| arg == "--gpu") {
        panic!("This benchmark requires --gpu");
    }

    let domain_bits = 25; // Mainnet scale (137 GB total)
    let num_shards = 2;

    println!("=== Morphogenesis Sharded Benchmark (2x GPUs) ===");
    println!("Total Domain: {} bits (137 GB)", domain_bits);
    println!("Shards: {}", num_shards);

    let params = ChaChaParams::new(domain_bits).unwrap();
    let total_pages = params.max_pages();
    let pages_per_shard = total_pages / num_shards;
    let shard_size_bytes = pages_per_shard * PAGE_SIZE_BYTES;
    let shard_size_gb = shard_size_bytes as f64 / 1e9;

    println!("Pages per shard: {}", pages_per_shard);
    println!("Data per shard: {:.2} GB", shard_size_gb);

    #[cfg(feature = "cuda")]
    {
        // 1. Setup Keys (Single query for test)
        let target = 12345; // Target in first shard
        let (k0, _) = generate_chacha_dpf_keys(&params, target).unwrap();
        // Use same key triple for simplicity
        let query = [k0.clone(), k0.clone(), k0.clone()];

        // 2. Spawn threads for each GPU
        let barrier = Arc::new(Barrier::new(num_shards + 1)); // +1 for main thread
        let mut handles = vec![];

        for gpu_id in 0..num_shards {
            let barrier = barrier.clone();
            let query = query.clone();

            handles.push(thread::spawn(move || {
                println!("[GPU {}] Initializing...", gpu_id);
                let scanner = GpuScanner::new(gpu_id).expect("Failed to create GpuScanner");

                println!("[GPU {}] Allocating {:.2} GB...", gpu_id, shard_size_gb);
                // Allocate empty zeros to save host RAM/Time (we only care about GPU VRAM/BW)
                let gpu_db = GpuPageMatrix::alloc_empty(scanner.device.clone(), pages_per_shard)
                    .expect("Failed to allocate VRAM");

                println!("[GPU {}] Ready.", gpu_id);

                // Wait for all GPUs to be ready
                barrier.wait();

                // Wait for start signal
                barrier.wait();

                let start = Instant::now();
                let _result = unsafe { scanner.scan(&gpu_db, [&query[0], &query[1], &query[2]]) }
                    .expect("Scan failed");
                let elapsed = start.elapsed();

                println!(
                    "[GPU {}] Scan finished in {:.2} ms",
                    gpu_id,
                    elapsed.as_secs_f64() * 1000.0
                );
                elapsed
            }));
        }

        // Wait for initialization
        barrier.wait();
        println!("\nAll GPUs initialized and data loaded. Starting synchronized scan...");

        // Start race
        let start = Instant::now();
        barrier.wait(); // Release threads

        // Collect results
        let mut max_latency = Duration::new(0, 0);
        for h in handles {
            let latency = h.join().unwrap();
            if latency > max_latency {
                max_latency = latency;
            }
        }
        let total_elapsed = start.elapsed(); // Includes thread overhead

        // Note: total_elapsed includes the overhead of join(), but since scan is sync,
        // the max_latency from threads is the real "parallel" time.
        // Actually, wall clock is determined by the slowest thread.

        let ms = max_latency.as_secs_f64() * 1000.0;
        let total_bw = (137.44 / ms) * 1000.0;

        println!("\n=== Results ===");
        println!("Sharded Latency (Max): {:.2} ms", ms);
        println!("Effective Throughput:  {:.2} GB/s", total_bw);
        println!("(Single GPU baseline was ~1600 GB/s)");

        // Single GPU estimated latency for 137GB: ~87ms (from previous H200 bench)
        // If we get ~43ms here, linear scaling is proven.
    }

    #[cfg(not(feature = "cuda"))]
    panic!("Requires cuda feature");
}
