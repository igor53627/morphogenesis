//! Batching benchmark for GPU PIR.
//!
//! Measures QPS and Latency for different batch sizes.

use morphogen_gpu_dpf::dpf::{generate_chacha_dpf_keys, ChaChaParams};
use morphogen_gpu_dpf::kernel::PAGE_SIZE_BYTES;
use std::env;
use std::time::Instant;

#[cfg(feature = "cuda")]
use morphogen_gpu_dpf::kernel::GpuScanner;
#[cfg(feature = "cuda")]
use morphogen_gpu_dpf::storage::GpuPageMatrix;

fn main() {
    let args: Vec<String> = env::args().collect();
    let use_gpu = args.iter().any(|arg| arg == "--gpu");

    let mut domain_bits = 24; // Default to 24 (fit in H100)
    if let Some(idx) = args.iter().position(|arg| arg == "--domain") {
        if let Some(val) = args.get(idx + 1) {
            domain_bits = val.parse().expect("Invalid domain bits");
        }
    }

    let params = ChaChaParams::new(domain_bits).unwrap();
    let num_pages = params.max_pages();
    let total_size = num_pages * PAGE_SIZE_BYTES;
    let size_gb = total_size as f64 / 1_000_000_000.0;

    println!("=== Morphogenesis PIR Batching Benchmark ===");
    println!("Domain: {} bits", domain_bits);
    println!("Data Size: {:.2} GB", size_gb);

    if !use_gpu {
        panic!("Batching benchmark requires --gpu");
    }

    #[cfg(feature = "cuda")]
    {
        // 1. Database Setup
        println!("\nAllocating {:.2}GB synthetic database...", size_gb);
        let db_data = vec![0u8; total_size];

        println!("Initializing GPU...");
        let scanner = GpuScanner::new(0).expect("Failed to create GpuScanner");
        let device = scanner.device.clone();

        println!("Uploading to GPU...");
        let gpu_db = GpuPageMatrix::new(device, &db_data).expect("Failed to upload to GPU");

        // Test batch sizes
        // Limited to 4 because:
        // 1. Default dynamic shared memory per block is 48KB. Batch 8 requires 96KB.
        //    We cannot easily increase this limit without unsafe access to raw CUDA handles.
        // 2. Benchmarks show no throughput gain from batching (compute bound).
        let batch_sizes = [1, 2, 4];

        println!(
            "\n{:<10} {:<15} {:<15} {:<15}",
            "Batch", "Avg Latency", "Throughput", "Mem BW"
        );
        println!("{:-<60}", "");

        for &batch_size in &batch_sizes {
            // Generate unique queries for the batch
            let mut queries = Vec::with_capacity(batch_size);
            for i in 0..batch_size {
                let target = (i * 12345) % num_pages;
                let (k0, _) = generate_chacha_dpf_keys(&params, target).unwrap();
                // Use same key for all 3 slots for simplicity in bench
                queries.push([k0.clone(), k0.clone(), k0.clone()]);
            }

            // Warmup
            unsafe { scanner.scan_batch(&gpu_db, &queries) }.expect("Warmup failed");

            let iterations = 5;
            let start = Instant::now();
            for _ in 0..iterations {
                unsafe { scanner.scan_batch(&gpu_db, &queries) }.expect("Scan failed");
            }
            let elapsed = start.elapsed();

            let avg_latency_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
            // QPS = (batch_size * iterations) / total_seconds
            let qps = (batch_size * iterations) as f64 / elapsed.as_secs_f64();
            // Effective Bandwidth = (Data Size * iterations) / total_seconds
            // Wait, batched scan reads DB once per batch.
            // So we read (Data Size) per batch.
            // BW = Data Size / (latency per batch / 1000)
            let mem_bw_gbs = size_gb / (avg_latency_ms / 1000.0);

            println!(
                "{:<10} {:<15.2} {:<15.2} {:<15.2}",
                batch_size, avg_latency_ms, qps, mem_bw_gbs
            );
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        panic!("Compile with --features cuda");
    }
}
