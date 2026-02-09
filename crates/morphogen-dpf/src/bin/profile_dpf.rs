//! Profiling binary for DPF evaluation
//!
//! Run with: cargo build --release --package morphogen-dpf --features fss --bin profile_dpf
//! Profile: samply record ./target/release/profile_dpf 16
//! Or: cargo flamegraph --bin profile_dpf --features fss -- 16

use morphogen_dpf::page::{generate_page_dpf_keys, PageDpfParams, PAGE_SIZE_BYTES};
use std::env;
use std::time::Instant;

/// Contiguous page storage - avoids pointer chasing and enables prefetching.
/// Pages are stored in a single allocation indexed as base + page_idx * PAGE_SIZE_BYTES.
struct ContiguousPages {
    data: Vec<u8>,
    num_pages: usize,
}

impl ContiguousPages {
    fn new(num_pages: usize) -> Self {
        let total_bytes = num_pages * PAGE_SIZE_BYTES;
        let mut data = vec![0u8; total_bytes];

        for i in 0..num_pages {
            let offset = i * PAGE_SIZE_BYTES;
            data[offset] = (i & 0xFF) as u8;
            data[offset + 1] = ((i >> 8) & 0xFF) as u8;
        }

        Self { data, num_pages }
    }

    fn page(&self, idx: usize) -> &[u8] {
        let offset = idx * PAGE_SIZE_BYTES;
        &self.data[offset..offset + PAGE_SIZE_BYTES]
    }

    fn as_page_refs(&self) -> Vec<&[u8]> {
        (0..self.num_pages).map(|i| self.page(i)).collect()
    }
}

/// Fragmented page storage - simulates Vec<Vec<u8>> pattern (for comparison).
struct FragmentedPages {
    pages: Vec<Vec<u8>>,
}

impl FragmentedPages {
    fn new(num_pages: usize) -> Self {
        let pages: Vec<Vec<u8>> = (0..num_pages)
            .map(|i| {
                let mut page = vec![0u8; PAGE_SIZE_BYTES];
                page[0] = (i & 0xFF) as u8;
                page[1] = ((i >> 8) & 0xFF) as u8;
                page
            })
            .collect();
        Self { pages }
    }

    fn as_page_refs(&self) -> Vec<&[u8]> {
        self.pages.iter().map(|p| p.as_slice()).collect()
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let domain_bits: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(16);
    let mode = args.get(2).map(|s| s.as_str()).unwrap_or("both");
    let chunk_size: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(4096);

    println!("=== DPF Profile: {}-bit domain ===", domain_bits);

    let num_pages: usize = 1 << domain_bits;
    let data_size_mb = (num_pages * PAGE_SIZE_BYTES) / (1024 * 1024);
    println!("Pages: {}, Data: {} MB", num_pages, data_size_mb);

    let params = PageDpfParams::new(domain_bits).unwrap();
    let target_page = num_pages / 2;
    let (k0, _k1) = generate_page_dpf_keys(&params, target_page).unwrap();

    match mode {
        "contiguous" => {
            run_contiguous_benchmark(&k0, num_pages, domain_bits, chunk_size);
        }
        "fragmented" => {
            run_fragmented_benchmark(&k0, num_pages, domain_bits, chunk_size);
        }
        "sweep" => {
            run_chunk_size_sweep(&k0, num_pages);
        }
        "optimal" => {
            println!("--- OPTIMAL CHUNK SIZE (131072) ---");
            run_contiguous_benchmark(&k0, num_pages, domain_bits, 131072);
        }
        "masks" => {
            analyze_mask_distribution(&k0, num_pages);
        }
        _ => {
            println!("\n--- CONTIGUOUS ALLOCATION ---");
            run_contiguous_benchmark(&k0, num_pages, domain_bits, chunk_size);

            println!("\n--- FRAGMENTED ALLOCATION (Vec<Vec<u8>>) ---");
            run_fragmented_benchmark(&k0, num_pages, domain_bits, chunk_size);

            println!("\n--- CHUNK SIZE SWEEP ---");
            run_chunk_size_sweep(&k0, num_pages);
        }
    }
}

fn analyze_mask_distribution(k0: &morphogen_dpf::page::PageDpfKey, num_pages: usize) {
    use fss_rs::group::byte::ByteGroup;
    use fss_rs::group::Group;

    println!("Analyzing mask distribution...");

    let mut dpf_output = vec![ByteGroup::zero(); num_pages];

    k0.full_eval(&mut dpf_output).unwrap();

    let mut zero_masks = 0usize;
    let mut nonzero_masks = 0usize;

    for output in &dpf_output {
        if output.0[0] == 0 {
            zero_masks += 1;
        } else {
            nonzero_masks += 1;
        }
    }

    println!("Total pages: {}", num_pages);
    println!("Zero masks (first byte = 0): {}", zero_masks);
    println!(
        "Non-zero masks: {} ({:.1}%)",
        nonzero_masks,
        nonzero_masks as f64 / num_pages as f64 * 100.0
    );
    println!(
        "\nThis means the XOR loop processes {} pages (should ideally be 1)",
        nonzero_masks
    );
}

fn run_contiguous_benchmark(
    k0: &morphogen_dpf::page::PageDpfKey,
    num_pages: usize,
    domain_bits: usize,
    chunk_size: usize,
) {
    println!("Allocating contiguous page data...");
    let alloc_start = Instant::now();
    let storage = ContiguousPages::new(num_pages);
    println!("Allocation took {:?}", alloc_start.elapsed());
    println!("Chunk size: {}", chunk_size);

    let pages = storage.as_page_refs();

    println!("Warmup...");
    let _ = k0.eval_and_accumulate_chunked(&pages, chunk_size);

    println!("\n=== Timed Run (5 iterations) ===");
    for iter in 0..5 {
        let timing = k0.eval_and_accumulate_chunked_timed(&pages, chunk_size);
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

    println!("\n=== Profile Loop (100 iterations) ===");
    let profile_start = Instant::now();
    for _ in 0..100 {
        let result = k0.eval_and_accumulate_chunked(&pages, chunk_size);
        std::hint::black_box(result);
    }
    let elapsed = profile_start.elapsed();
    let ms_per_iter = elapsed.as_secs_f64() * 1000.0 / 100.0;
    println!("100 iterations: {:?} ({:.2}ms/iter)", elapsed, ms_per_iter);

    if domain_bits < 25 {
        let scale = (1usize << 25) as f64 / num_pages as f64;
        let projected_ms = ms_per_iter * scale;
        println!(
            "\nExtrapolated 25-bit latency: {:.0}ms ({:.1}s)",
            projected_ms,
            projected_ms / 1000.0
        );
    }
}

fn run_fragmented_benchmark(
    k0: &morphogen_dpf::page::PageDpfKey,
    num_pages: usize,
    domain_bits: usize,
    chunk_size: usize,
) {
    println!("Allocating fragmented page data...");
    let alloc_start = Instant::now();
    let storage = FragmentedPages::new(num_pages);
    println!("Allocation took {:?}", alloc_start.elapsed());
    println!("Chunk size: {}", chunk_size);

    let pages = storage.as_page_refs();

    println!("Warmup...");
    let _ = k0.eval_and_accumulate_chunked(&pages, chunk_size);

    println!("\n=== Timed Run (5 iterations) ===");
    for iter in 0..5 {
        let timing = k0.eval_and_accumulate_chunked_timed(&pages, chunk_size);
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

    println!("\n=== Profile Loop (100 iterations) ===");
    let profile_start = Instant::now();
    for _ in 0..100 {
        let result = k0.eval_and_accumulate_chunked(&pages, chunk_size);
        std::hint::black_box(result);
    }
    let elapsed = profile_start.elapsed();
    let ms_per_iter = elapsed.as_secs_f64() * 1000.0 / 100.0;
    println!("100 iterations: {:?} ({:.2}ms/iter)", elapsed, ms_per_iter);

    if domain_bits < 25 {
        let scale = (1usize << 25) as f64 / num_pages as f64;
        let projected_ms = ms_per_iter * scale;
        println!(
            "\nExtrapolated 25-bit latency: {:.0}ms ({:.1}s)",
            projected_ms,
            projected_ms / 1000.0
        );
    }
}

fn run_chunk_size_sweep(k0: &morphogen_dpf::page::PageDpfKey, num_pages: usize) {
    println!("Using contiguous allocation for chunk sweep...");
    let storage = ContiguousPages::new(num_pages);
    let pages = storage.as_page_refs();

    let chunk_sizes = [
        256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072,
    ];

    println!(
        "\n{:>8} {:>10} {:>10} {:>10} {:>10}",
        "Chunk", "DPF(ms)", "XOR(ms)", "Total(ms)", "DPF%"
    );
    println!("{:-<52}", "");

    for &chunk_size in &chunk_sizes {
        if chunk_size > num_pages {
            continue;
        }

        let _ = k0.eval_and_accumulate_chunked(&pages, chunk_size);

        let mut total_dpf_ns: u64 = 0;
        let mut total_xor_ns: u64 = 0;
        let iterations = 20;

        for _ in 0..iterations {
            let timing = k0.eval_and_accumulate_chunked_timed(&pages, chunk_size);
            total_dpf_ns += timing.dpf_eval_ns;
            total_xor_ns += timing.xor_accumulate_ns;
        }

        let avg_dpf_ms = (total_dpf_ns as f64) / (iterations as f64 * 1_000_000.0);
        let avg_xor_ms = (total_xor_ns as f64) / (iterations as f64 * 1_000_000.0);
        let avg_total = avg_dpf_ms + avg_xor_ms;
        let dpf_pct = (avg_dpf_ms / avg_total) * 100.0;

        println!(
            "{:>8} {:>10.2} {:>10.2} {:>10.2} {:>9.1}%",
            chunk_size, avg_dpf_ms, avg_xor_ms, avg_total, dpf_pct
        );
    }
}
