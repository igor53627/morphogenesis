//! Benchmark comparing current AesDpfKey vs fss-rs DPF
//!
//! Run with: cargo bench --package morphogen-dpf --features fss

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn bench_current_dpf(c: &mut Criterion) {
    use morphogen_dpf::{AesDpfKey, DpfKey};
    use rand::SeedableRng;

    let mut group = c.benchmark_group("current_dpf_eval_range");

    for domain_bits in [8u32, 10, 12, 14, 16] {
        let domain_size: usize = 1 << domain_bits;
        let mut rng = rand::rngs::StdRng::seed_from_u64(12345);
        let target = domain_size / 2;
        let key = AesDpfKey::new_single(&mut rng, target);

        group.throughput(Throughput::Elements(domain_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(domain_size),
            &domain_size,
            |b, &size| {
                let mut out = vec![0u8; size];
                b.iter(|| {
                    key.eval_range_masks(0, black_box(&mut out));
                });
            },
        );
    }

    group.finish();
}

#[cfg(feature = "fss")]
fn bench_fss_dpf(c: &mut Criterion) {
    use fss_rs::dpf::{Dpf, DpfImpl, PointFn};
    use fss_rs::group::byte::ByteGroup;
    use fss_rs::group::Group;
    use fss_rs::prg::Aes128MatyasMeyerOseasPrg;
    use fss_rs::Share;

    let mut group = c.benchmark_group("fss_dpf_full_eval");

    // Test up to 1M for page-level PIR analysis
    // Page PIR: 250M rows / 16 rows per page = 15.6M pages
    // We test: 64K, 256K, 1M to extrapolate to 16M
    for (domain_bits, label) in [(16u32, "64K"), (18, "256K"), (20, "1M")] {
        let domain_size: usize = 1 << domain_bits;

        group.throughput(Throughput::Elements(domain_size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(label), &domain_bits, |b, _| {
            // Setup: create DPF with 16-byte output
            let prg_keys: [[u8; 16]; 2] = rand::random();
            let prg =
                Aes128MatyasMeyerOseasPrg::<16, 1, 2>::new(&std::array::from_fn(|i| &prg_keys[i]));

            // Use 3-byte input domain (up to 16M) with filter
            let dpf = DpfImpl::<3, 16, _>::new_with_filter(prg, domain_bits as usize);

            let target: u32 = 42;
            let alpha = [(target >> 16) as u8, (target >> 8) as u8, target as u8];
            let beta = ByteGroup([0xFF; 16]);
            let point_fn = PointFn { alpha, beta };

            let s0s: [[u8; 16]; 2] = rand::random();
            let share = dpf.gen(&point_fn, [&s0s[0], &s0s[1]]);

            let k0 = Share {
                s0s: vec![share.s0s[0]],
                cws: share.cws.clone(),
                cw_np1: share.cw_np1.clone(),
            };

            let mut ys: Vec<ByteGroup<16>> = vec![ByteGroup::zero(); domain_size];

            b.iter(|| {
                // Reset output
                for y in ys.iter_mut() {
                    *y = ByteGroup::zero();
                }
                let mut ys_refs: Vec<&mut ByteGroup<16>> = ys.iter_mut().collect();
                dpf.full_eval(false, &k0, black_box(&mut ys_refs));
            });
        });
    }

    group.finish();
}

#[cfg(feature = "fss")]
fn bench_page_pir_comparison(c: &mut Criterion) {
    use fss_rs::dpf::{Dpf, DpfImpl, PointFn};
    use fss_rs::group::byte::ByteGroup;
    use fss_rs::group::Group;
    use fss_rs::prg::Aes128MatyasMeyerOseasPrg;
    use fss_rs::Share;
    use morphogen_dpf::{AesDpfKey, DpfKey};
    use rand::SeedableRng;

    let mut group = c.benchmark_group("page_pir_comparison");

    // Compare row-level vs page-level at equivalent data sizes
    // 1M rows at 256B = 256MB data
    // 1M pages at 4KB = 4GB data (but same DPF domain cost)

    let domain_size: usize = 1 << 20; // 1M

    // Benchmark 1: Current AesDpfKey at 1M rows
    group.bench_function("aes_dpf_1M_rows", |b| {
        let mut rng = rand::rngs::StdRng::seed_from_u64(12345);
        let key = AesDpfKey::new_single(&mut rng, domain_size / 2);
        let mut out = vec![0u8; domain_size];
        b.iter(|| {
            key.eval_range_masks(0, black_box(&mut out));
        });
    });

    // Benchmark 2: fss-rs proper DPF at 1M pages
    group.bench_function("fss_dpf_1M_pages", |b| {
        let prg_keys: [[u8; 16]; 2] = rand::random();
        let prg =
            Aes128MatyasMeyerOseasPrg::<16, 1, 2>::new(&std::array::from_fn(|i| &prg_keys[i]));
        let dpf = DpfImpl::<3, 16, _>::new_with_filter(prg, 20);

        let target: u32 = 42;
        let alpha = [(target >> 16) as u8, (target >> 8) as u8, target as u8];
        let beta = ByteGroup([0xFF; 16]);
        let point_fn = PointFn { alpha, beta };

        let s0s: [[u8; 16]; 2] = rand::random();
        let share = dpf.gen(&point_fn, [&s0s[0], &s0s[1]]);
        let k0 = Share {
            s0s: vec![share.s0s[0]],
            cws: share.cws.clone(),
            cw_np1: share.cw_np1.clone(),
        };

        let mut ys: Vec<ByteGroup<16>> = vec![ByteGroup::zero(); domain_size];

        b.iter(|| {
            for y in ys.iter_mut() {
                *y = ByteGroup::zero();
            }
            let mut ys_refs: Vec<&mut ByteGroup<16>> = ys.iter_mut().collect();
            dpf.full_eval(false, &k0, black_box(&mut ys_refs));
        });
    });

    group.finish();
}

#[cfg(feature = "fss")]
fn bench_chunked_vs_full_memory(c: &mut Criterion) {
    use morphogen_dpf::page::{generate_page_dpf_keys, PageDpfParams, PAGE_SIZE_BYTES};

    let mut group = c.benchmark_group("chunked_vs_full_memory");

    // Test at 16-bit domain (64K pages) to keep benchmark fast
    // Memory analysis:
    // - Full eval: 64K * 16 bytes = 1MB DPF output buffer
    // - Chunked (4096): 4096 * 16 bytes = 64KB DPF output buffer
    let domain_bits = 16;
    let num_pages: usize = 1 << domain_bits;

    let params = PageDpfParams::new(domain_bits).unwrap();
    let target_page = num_pages / 2;
    let (k0, _k1) = generate_page_dpf_keys(&params, target_page).unwrap();

    // Create dummy page data
    let page_data: Vec<[u8; PAGE_SIZE_BYTES]> = (0..num_pages)
        .map(|i| {
            let mut page = [0u8; PAGE_SIZE_BYTES];
            page[0] = i as u8;
            page
        })
        .collect();
    let pages: Vec<&[u8]> = page_data.iter().map(|p| p.as_slice()).collect();

    group.throughput(Throughput::Elements(num_pages as u64));

    // Full eval: allocates O(N) = 1MB for DPF output
    #[allow(deprecated)]
    group.bench_function("full_eval_1MB_alloc", |b| {
        b.iter(|| {
            let result = k0.eval_and_accumulate(black_box(pages.iter().copied()));
            black_box(result)
        });
    });

    // Chunked eval with various chunk sizes
    for chunk_size in [64, 256, 1024, 4096] {
        let alloc_kb = chunk_size * 16 / 1024;
        group.bench_function(format!("chunked_{}KB_alloc", alloc_kb), |b| {
            b.iter(|| {
                let result = k0.eval_and_accumulate_chunked(black_box(&pages), chunk_size);
                black_box(result)
            });
        });
    }

    group.finish();

    // Print memory summary
    println!("\n=== Memory Usage Summary (16-bit domain = 64K pages) ===");
    println!("Full eval DPF buffer:     {} KB", num_pages * 16 / 1024);
    println!("Chunked (64) DPF buffer:  {} KB", 64 * 16 / 1024);
    println!("Chunked (256) DPF buffer: {} KB", 256 * 16 / 1024);
    println!("Chunked (1024) DPF buffer:{} KB", 1024 * 16 / 1024);
    println!("Chunked (4096) DPF buffer:{} KB", 4096 * 16 / 1024);
    println!("\n=== Projected Memory at 25-bit domain (27M pages) ===");
    let pages_27m: usize = 1 << 25;
    println!("Full eval DPF buffer:     {} MB", pages_27m * 16 / 1024 / 1024);
    println!("Chunked (4096) DPF buffer:{} KB", 4096 * 16 / 1024);
    println!("Memory reduction:         {}x", pages_27m / 4096);
}

#[cfg(feature = "fss")]
criterion_group!(
    benches,
    bench_current_dpf,
    bench_fss_dpf,
    bench_page_pir_comparison,
    bench_chunked_vs_full_memory
);

#[cfg(not(feature = "fss"))]
criterion_group!(benches, bench_current_dpf);

criterion_main!(benches);
