use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn bench_current_dpf(c: &mut Criterion) {
    use morphogen_dpf::{AesDpfKey, DpfKey};
    use rand::SeedableRng;

    let mut group = c.benchmark_group("current_dpf");

    for domain_bits in [8, 12, 16] {
        let domain_size = 1 << domain_bits;
        let mut rng = rand::rngs::StdRng::seed_from_u64(12345);
        let target = domain_size / 2;
        let key = AesDpfKey::new_single(&mut rng, target);

        group.bench_with_input(
            BenchmarkId::new("eval_range", domain_size),
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

    let mut group = c.benchmark_group("fss_dpf");

    for domain_bits in [8u8, 12, 16] {
        let domain_size: usize = 1 << domain_bits;

        group.bench_with_input(
            BenchmarkId::new("full_eval", domain_size),
            &domain_bits,
            |b, &bits| {
                let prg_keys: [[u8; 16]; 2] = rand::random();

                b.iter_with_setup(
                    || match bits {
                        8 => {
                            let prg = Aes128MatyasMeyerOseasPrg::<16, 1, 2>::new(
                                std::array::from_fn(|i| &prg_keys[i]),
                            );
                            let dpf = DpfImpl::<8, 1, _>::new(prg);
                            let target: u8 = 42;
                            let alpha = [target];
                            let beta = ByteGroup([0xFF]);
                            let point_fn = PointFn { alpha, beta };
                            let s0s: [[u8; 16]; 2] = rand::random();
                            let share = dpf.gen(&point_fn, [&s0s[0], &s0s[1]]);
                            let (k0, _k1) = share.split();
                            (dpf, k0, 256usize)
                        }
                        _ => panic!("unsupported bits for this simple bench"),
                    },
                    |(dpf, k0, size)| {
                        let mut ys: Vec<ByteGroup<1>> = vec![ByteGroup::zero(); size];
                        dpf.full_eval(false, &k0, black_box(&mut ys));
                    },
                );
            },
        );
    }

    group.finish();
}

#[cfg(feature = "fss")]
criterion_group!(benches, bench_current_dpf, bench_fss_dpf);

#[cfg(not(feature = "fss"))]
criterion_group!(benches, bench_current_dpf);

criterion_main!(benches);
