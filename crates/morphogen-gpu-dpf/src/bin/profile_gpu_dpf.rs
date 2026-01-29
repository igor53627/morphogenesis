//! Profiling binary for ChaCha8-based GPU DPF.
//!
//! Usage: cargo run --release --package morphogen-gpu-dpf --bin profile_gpu_dpf -- [domain_bits]

use morphogen_gpu_dpf::dpf::{generate_chacha_dpf_keys, ChaChaParams};
use morphogen_gpu_dpf::kernel::{eval_fused_3dpf_cpu, PAGE_SIZE_BYTES};
use std::env;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    let domain_bits: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(12);

    println!("=== ChaCha8 GPU-DPF CPU Baseline ===");
    println!("Domain bits: {}", domain_bits);

    let params = ChaChaParams::new(domain_bits).unwrap();
    let num_pages = params.max_pages();
    let data_size_mb = (num_pages * PAGE_SIZE_BYTES) / (1024 * 1024);

    println!("Pages: {}", num_pages);
    println!("Data size: {} MB", data_size_mb);

    // Generate 3 key pairs for Cuckoo hashing
    let targets = [num_pages / 4, num_pages / 2, 3 * num_pages / 4];
    println!("Targets: {:?}", targets);

    let (k0_0, k0_1) = generate_chacha_dpf_keys(&params, targets[0]).unwrap();
    let (k1_0, k1_1) = generate_chacha_dpf_keys(&params, targets[1]).unwrap();
    let (k2_0, k2_1) = generate_chacha_dpf_keys(&params, targets[2]).unwrap();

    // Allocate pages
    println!("\nAllocating {} pages...", num_pages);
    let alloc_start = Instant::now();
    let pages_data: Vec<Vec<u8>> = (0..num_pages)
        .map(|i| {
            let mut page = vec![0u8; PAGE_SIZE_BYTES];
            page[0] = (i & 0xFF) as u8;
            page[1] = ((i >> 8) & 0xFF) as u8;
            for j in 2..PAGE_SIZE_BYTES {
                page[j] = ((i + j) & 0xFF) as u8;
            }
            page
        })
        .collect();
    println!("Allocation took {:?}", alloc_start.elapsed());

    let pages: Vec<&[u8]> = pages_data.iter().map(|p| p.as_slice()).collect();

    // Warmup
    println!("\nWarmup...");
    let _ = eval_fused_3dpf_cpu([&k0_0, &k1_0, &k2_0], &pages);

    // Timed runs
    println!("\n=== Timed Runs (5 iterations) ===");
    println!(
        "{:>6} {:>10} {:>10} {:>10} {:>8}",
        "Iter", "DPF(ms)", "XOR(ms)", "Total(ms)", "DPF%"
    );
    println!("{:-<50}", "");

    for iter in 0..5 {
        let result = eval_fused_3dpf_cpu([&k0_0, &k1_0, &k2_0], &pages).unwrap();
        let timing = &result.timing;
        let dpf_ms = timing.dpf_eval_ns as f64 / 1_000_000.0;
        let xor_ms = timing.xor_accumulate_ns as f64 / 1_000_000.0;
        let total_ms = timing.total_ns as f64 / 1_000_000.0;
        let dpf_pct = (dpf_ms / total_ms) * 100.0;

        println!(
            "{:>6} {:>10.2} {:>10.2} {:>10.2} {:>7.1}%",
            iter, dpf_ms, xor_ms, total_ms, dpf_pct
        );
    }

    // Verify correctness
    println!("\n=== Correctness Verification ===");
    let result0 = eval_fused_3dpf_cpu([&k0_0, &k1_0, &k2_0], &pages).unwrap();
    let result1 = eval_fused_3dpf_cpu([&k0_1, &k1_1, &k2_1], &pages).unwrap();

    for (i, target) in targets.iter().enumerate() {
        let page0 = match i {
            0 => &result0.page0,
            1 => &result0.page1,
            _ => &result0.page2,
        };
        let page1 = match i {
            0 => &result1.page0,
            1 => &result1.page1,
            _ => &result1.page2,
        };

        let mut recovered = vec![0u8; PAGE_SIZE_BYTES];
        for j in 0..PAGE_SIZE_BYTES {
            recovered[j] = page0[j] ^ page1[j];
        }

        let expected = &pages_data[*target];
        if recovered == *expected {
            println!("Target {}: [PASS]", target);
        } else {
            println!("Target {}: [FAIL]", target);
        }
    }

    // Throughput calculation
    println!("\n=== Throughput ===");
    let iterations = 20;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = eval_fused_3dpf_cpu([&k0_0, &k1_0, &k2_0], &pages);
    }
    let elapsed = start.elapsed();
    let ms_per_iter = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    let throughput_mbs = (data_size_mb as f64) / (ms_per_iter / 1000.0);
    let throughput_gbs = throughput_mbs / 1024.0;

    println!("Average latency: {:.2} ms", ms_per_iter);
    println!(
        "Throughput: {:.1} MB/s ({:.2} GB/s)",
        throughput_mbs, throughput_gbs
    );

    // Extrapolate to 25-bit (mainnet)
    if domain_bits < 25 {
        let scale = (1usize << 25) as f64 / num_pages as f64;
        let projected_ms = ms_per_iter * scale;
        println!("\nExtrapolated 25-bit (27M pages, 108GB):");
        println!(
            "  Projected latency: {:.0} ms ({:.1} s)",
            projected_ms,
            projected_ms / 1000.0
        );
    }
}
