//! Benchmark for ChaCha8 PRG performance.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use morphogen_gpu_dpf::chacha_prg::{ChaCha8Prg, Seed128};
use morphogen_gpu_dpf::dpf::{generate_chacha_dpf_keys, ChaChaParams};
use morphogen_gpu_dpf::kernel::{eval_fused_3dpf_cpu, PAGE_SIZE_BYTES};

fn bench_prg_expand(c: &mut Criterion) {
    let seed = Seed128::random();

    c.bench_function("chacha8_prg_expand", |b| {
        b.iter(|| ChaCha8Prg::expand(black_box(&seed)))
    });
}

fn bench_dpf_eval_point(c: &mut Criterion) {
    let params = ChaChaParams::new(16).unwrap();
    let (k0, _) = generate_chacha_dpf_keys(&params, 1000).unwrap();

    c.bench_function("chacha_dpf_eval_point_16bit", |b| {
        b.iter(|| k0.eval(black_box(1000)))
    });
}

fn bench_dpf_full_eval(c: &mut Criterion) {
    let mut group = c.benchmark_group("chacha_dpf_full_eval");

    for bits in [8, 10, 12] {
        let params = ChaChaParams::new(bits).unwrap();
        let num_pages = params.max_pages();
        let (k0, _) = generate_chacha_dpf_keys(&params, num_pages / 2).unwrap();

        group.throughput(Throughput::Elements(num_pages as u64));
        group.bench_function(format!("{}bit", bits), |b| {
            let mut output = vec![Seed128::ZERO; num_pages];
            b.iter(|| {
                k0.full_eval(black_box(&mut output)).unwrap();
            })
        });
    }

    group.finish();
}

fn bench_fused_3dpf(c: &mut Criterion) {
    let mut group = c.benchmark_group("fused_3dpf_cpu");

    for bits in [8, 10, 12] {
        let params = ChaChaParams::new(bits).unwrap();
        let num_pages = params.max_pages();

        let (k0_0, _) = generate_chacha_dpf_keys(&params, 0).unwrap();
        let (k1_0, _) = generate_chacha_dpf_keys(&params, num_pages / 3).unwrap();
        let (k2_0, _) = generate_chacha_dpf_keys(&params, 2 * num_pages / 3).unwrap();

        let pages_data: Vec<Vec<u8>> = (0..num_pages)
            .map(|i| {
                let mut page = vec![0u8; PAGE_SIZE_BYTES];
                page[0] = (i & 0xFF) as u8;
                page
            })
            .collect();
        let pages: Vec<&[u8]> = pages_data.iter().map(|p| p.as_slice()).collect();

        let data_size = num_pages * PAGE_SIZE_BYTES;
        group.throughput(Throughput::Bytes(data_size as u64));

        group.bench_function(
            format!("{}bit_{}MB", bits, data_size / (1024 * 1024)),
            |b| b.iter(|| eval_fused_3dpf_cpu([&k0_0, &k1_0, &k2_0], black_box(&pages))),
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_prg_expand,
    bench_dpf_eval_point,
    bench_dpf_full_eval,
    bench_fused_3dpf,
);
criterion_main!(benches);
