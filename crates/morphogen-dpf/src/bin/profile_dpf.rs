//! Profiling binary for DPF evaluation
//!
//! Run with: cargo build --release --package morphogen-dpf --features fss --bin profile_dpf
//! Profile: samply record ./target/release/profile_dpf 16
//! Or: cargo flamegraph --bin profile_dpf --features fss -- 16

use morphogen_dpf::page::{generate_page_dpf_keys, PageDpfParams, PAGE_SIZE_BYTES};
use std::env;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    let domain_bits: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(16);

    println!("=== DPF Profile: {}-bit domain ===", domain_bits);

    let num_pages: usize = 1 << domain_bits;
    let data_size_mb = (num_pages * PAGE_SIZE_BYTES) / (1024 * 1024);
    println!("Pages: {}, Data: {} MB", num_pages, data_size_mb);

    // Setup
    println!("Allocating {} MB page data...", data_size_mb);
    let alloc_start = Instant::now();
    let page_data: Vec<Vec<u8>> = (0..num_pages)
        .map(|i| {
            let mut page = vec![0u8; PAGE_SIZE_BYTES];
            page[0] = (i & 0xFF) as u8;
            page[1] = ((i >> 8) & 0xFF) as u8;
            page
        })
        .collect();
    let pages: Vec<&[u8]> = page_data.iter().map(|p| p.as_slice()).collect();
    println!("Allocation took {:?}", alloc_start.elapsed());

    // Generate keys
    println!("Generating DPF keys...");
    let params = PageDpfParams::new(domain_bits).unwrap();
    let target_page = num_pages / 2;
    let (k0, _k1) = generate_page_dpf_keys(&params, target_page).unwrap();

    // Warmup
    println!("Warmup...");
    let _ = k0.eval_and_accumulate_chunked(&pages, 4096);

    // Timed run with breakdown
    println!("\n=== Timed Run (5 iterations) ===");
    for iter in 0..5 {
        let timing = k0.eval_and_accumulate_chunked_timed(&pages, 4096);
        let total = timing.dpf_eval_ns + timing.xor_accumulate_ns;
        let dpf_pct = (timing.dpf_eval_ns as f64 / total as f64) * 100.0;
        println!(
            "Iter {}: DPF={:.1}% ({:.2}ms), XOR={:.1}% ({:.2}ms), Total={:.2}ms",
            iter,
            dpf_pct,
            timing.dpf_eval_ns as f64 / 1_000_000.0,
            100.0 - dpf_pct,
            timing.xor_accumulate_ns as f64 / 1_000_000.0,
            total as f64 / 1_000_000.0
        );
    }

    // Profile loop - run many iterations for flamegraph
    println!("\n=== Profile Loop (100 iterations) ===");
    let profile_start = Instant::now();
    for _ in 0..100 {
        let result = k0.eval_and_accumulate_chunked(&pages, 4096);
        std::hint::black_box(result);
    }
    let elapsed = profile_start.elapsed();
    println!(
        "100 iterations: {:?} ({:.2}ms/iter)",
        elapsed,
        elapsed.as_millis() as f64 / 100.0
    );

    // Extrapolate to 25-bit
    if domain_bits < 25 {
        let scale = (1usize << 25) as f64 / num_pages as f64;
        let projected_ms = (elapsed.as_millis() as f64 / 100.0) * scale;
        println!(
            "\nExtrapolated 25-bit latency: {:.0}ms ({:.1}s)",
            projected_ms,
            projected_ms / 1000.0
        );
    }
}
