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
    for (domain_bits, label) in [
        (16u32, "64K"),
        (18, "256K"),
        (20, "1M"),
    ] {
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
            let alpha = [
                (target >> 16) as u8,
                (target >> 8) as u8,
                target as u8,
            ];
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
criterion_group!(benches, bench_current_dpf, bench_fss_dpf, bench_page_pir_comparison);

#[cfg(not(feature = "fss"))]
criterion_group!(benches, bench_current_dpf);

criterion_main!(benches);
