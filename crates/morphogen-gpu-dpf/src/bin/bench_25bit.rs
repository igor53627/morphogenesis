//! High-scale benchmark for GPU PIR.
//!
//! Measures End-to-End latency at Mainnet scale (25-bit domain, 108GB).
//! Supports both CPU and GPU paths.

use morphogen_gpu_dpf::dpf::{generate_chacha_dpf_keys, ChaChaParams};
use morphogen_gpu_dpf::kernel::{eval_fused_3dpf_cpu, PAGE_SIZE_BYTES};
use std::env;
use std::time::Instant;

#[cfg(feature = "cuda")]
use morphogen_gpu_dpf::kernel::GpuScanner;
#[cfg(feature = "cuda")]
use morphogen_gpu_dpf::storage::GpuPageMatrix;

fn main() {
    let args: Vec<String> = env::args().collect();
    let use_gpu = args.iter().any(|arg| arg == "--gpu");
    let domain_bits: usize = 25; // Mainnet scale

    println!("=== Morphogenesis PIR Mainnet Benchmark (25-bit) ===");
    println!("Pages: {} (2^25)", 1usize << domain_bits);
    println!("Data Size: 108.00 GB");
    println!("Mode: {}", if use_gpu { "GPU (CUDA)" } else { "CPU (Rayon)" });

    let params = ChaChaParams::new(domain_bits).unwrap();
    let num_pages = params.max_pages();
    let total_size = num_pages * PAGE_SIZE_BYTES;

    // 1. Database Setup
    println!("\nAllocating 108GB synthetic database...");
    let start = Instant::now();
    // Use a simpler pattern to avoid slow random/mapped initialization
    let db_data = vec![0u8; total_size]; 
    println!("Allocation took {:.2}s", start.elapsed().as_secs_f64());

    // 2. Query Setup
    let targets = [1337, num_pages / 2, num_pages - 1];
    let (k0_0, k0_1) = generate_chacha_dpf_keys(&params, targets[0]).unwrap();
    let (k1_0, k1_1) = generate_chacha_dpf_keys(&params, targets[1]).unwrap();
    let (k2_0, k2_1) = generate_chacha_dpf_keys(&params, targets[2]).unwrap();

    // 3. Execution
    if use_gpu {
        #[cfg(feature = "cuda")]
        {
            println!("\nInitializing GPU...");
            let scanner = GpuScanner::new(0).expect("Failed to create GpuScanner");
            let device = scanner.device.clone();
            
            println!("Uploading 108GB to GPU (this will take a while over PCIe)...");
            let upload_start = Instant::now();
            let gpu_db = GpuPageMatrix::new(device, &db_data).expect("Failed to upload to GPU");
            println!("Upload took {:.2}s ({:.2} GB/s)", 
                upload_start.elapsed().as_secs_f64(),
                108.0 / upload_start.elapsed().as_secs_f64()
            );

            println!("\n=== Starting GPU Timed Runs (5 iterations) ===");
            for i in 0..5 {
                let start = Instant::now();
                let _result = unsafe { scanner.scan(&gpu_db, [&k0_0, &k1_0, &k2_0]) }.expect("GPU scan failed");
                let elapsed = start.elapsed();
                println!("Iter {}: {:.2} ms ({:.2} GB/s)", 
                    i, 
                    elapsed.as_secs_f64() * 1000.0,
                    108.0 / elapsed.as_secs_f64()
                );
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            panic!("Binary must be compiled with --features cuda to use --gpu");
        }
    } else {
        println!("\nPreparing page references...");
        let pages: Vec<&[u8]> = db_data.chunks_exact(PAGE_SIZE_BYTES).collect();

        println!("\n=== Starting CPU Timed Runs (3 iterations) ===");
        for i in 0..3 {
            let start = Instant::now();
            let _result = eval_fused_3dpf_cpu([&k0_0, &k1_0, &k2_0], &pages).expect("CPU scan failed");
            let elapsed = start.elapsed();
            println!("Iter {}: {:.2} ms ({:.2} GB/s)", 
                i, 
                elapsed.as_secs_f64() * 1000.0,
                108.0 / elapsed.as_secs_f64()
            );
        }
    }
}
