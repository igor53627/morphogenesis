//! Benchmark comparing original vs optimized CUDA kernels
//!
//! Recommendation: Use Optimized v1 - it provides 3-8% speedup without
//! the serialization penalty of v2/v3.

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

    let mut domain_bits = 24;
    if let Some(idx) = args.iter().position(|arg| arg == "--domain") {
        if let Some(val) = args.get(idx + 1) {
            domain_bits = val.parse().expect("Invalid domain bits");
        }
    }

    let params = ChaChaParams::new(domain_bits).unwrap();
    let num_pages = params.max_pages();
    let total_size = num_pages * PAGE_SIZE_BYTES;
    let size_gb = total_size as f64 / 1_000_000_000.0;

    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║    Morphogenesis: Original vs Optimized v1                         ║");
    println!("╚════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Configuration:");
    println!(
        "  Domain:        {} bits ({} pages)",
        domain_bits, num_pages
    );
    println!("  Data Size:     {:.2} GB", size_gb);
    println!();

    if !use_gpu {
        panic!("This benchmark requires --gpu");
    }

    #[cfg(feature = "cuda")]
    {
        println!("Allocating {:.2}GB synthetic database...", size_gb);
        let db_data = vec![0u8; total_size];

        println!("Initializing GPU...");
        let scanner = GpuScanner::new(0).expect("Failed to create GpuScanner");
        let device = scanner.device.clone();

        if let Ok(name) = device.name() {
            println!("  Device: {}", name);
        }
        println!();

        println!("Uploading to GPU...");
        let gpu_db = GpuPageMatrix::new(device, &db_data).expect("Failed to upload to GPU");
        println!("✓ Upload complete");
        println!();

        let batch_sizes = [1, 2, 4];
        let iterations = 10;

        println!("Running benchmarks ({} iterations each)...", iterations);
        println!();

        println!("┌─────────┬────────────────────┬────────────────────┬────────────────┐");
        println!("│         │ Original           │ Optimized v1       │ Speedup        │");
        println!(
            "│ {:>7} │ {:>8} {:>8} │ {:>8} {:>8} │ {:>14} │",
            "Batch", "Lat(ms)", "GB/s", "Lat(ms)", "GB/s", ""
        );
        println!("├─────────┼────────────────────┼────────────────────┼────────────────┤");

        for &batch_size in &batch_sizes {
            let mut queries = Vec::with_capacity(batch_size);
            for i in 0..batch_size {
                let target = (i * 12345) % num_pages;
                let (k0, _) = generate_chacha_dpf_keys(&params, target).unwrap();
                queries.push([k0.clone(), k0.clone(), k0.clone()]);
            }

            // Benchmark Original
            unsafe { scanner.scan_batch(&gpu_db, &queries) }.expect("Warmup failed");
            let start = Instant::now();
            for _ in 0..iterations {
                unsafe { scanner.scan_batch(&gpu_db, &queries) }.expect("Scan failed");
            }
            let orig_lat = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
            let orig_bw = size_gb * batch_size as f64 / (orig_lat / 1000.0);

            // Benchmark v1
            let (v1_lat, v1_bw, speedup) =
                if unsafe { scanner.scan_batch_optimized(&gpu_db, &queries) }.is_ok() {
                    let start = Instant::now();
                    for _ in 0..iterations {
                        unsafe { scanner.scan_batch_optimized(&gpu_db, &queries) }.unwrap();
                    }
                    let lat = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
                    let bw = size_gb * batch_size as f64 / (lat / 1000.0);
                    let sp = orig_lat / lat;
                    (lat, bw, format!("{:.2}×", sp))
                } else {
                    (0.0, 0.0, "N/A".to_string())
                };

            println!(
                "│ {:>7} │ {:>8.2} {:>8.0} │ {:>8.2} {:>8.0} │ {:>14} │",
                batch_size, orig_lat, orig_bw, v1_lat, v1_bw, speedup
            );
        }

        println!("└─────────┴────────────────────┴────────────────────┴────────────────┘");
        println!();
        println!("Note: Batch sizes > 2 may show slowdown due to shared memory constraints.");
        println!("For production use, recommend batch=1 or batch=2 with Optimized v1.");
        println!();
    }
    #[cfg(not(feature = "cuda"))]
    {
        panic!("Compile with --features cuda");
    }
}
