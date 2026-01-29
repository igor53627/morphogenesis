use binius_field::{BinaryField128b, Field};
use std::time::Instant;

fn main() {
    // 1. Setup
    // We want to benchmark Dot Product of size N = 2^20 (approx 1M) for quick test.
    // Real scale is 2^28.
    let log_n = 20;
    let n = 1 << log_n;

    println!("=== Binius Dot Product Benchmark ===");
    println!("Vector Size: 2^{} = {} elements", log_n, n);
    println!("Field: BinaryField128b");

    // Allocate vectors
    // specific type: PackedBinaryField1x128b is a wrapper around u128 that implements PackedField
    // actually, let's use the standard packed types if possible or just Vec<BinaryField128b>

    // Using simple Vec for baseline
    let mut vec_a = Vec::with_capacity(n);
    let mut vec_b = Vec::with_capacity(n);

    // Fill with random data (dummy)
    let val = BinaryField128b::new(12345);
    for _ in 0..n {
        vec_a.push(val);
        vec_b.push(val);
    }

    println!("Allocated vectors. Running benchmark...");

    // 2. Benchmark Dot Product (Naive)
    let start = Instant::now();
    let mut sum = BinaryField128b::ZERO;

    for i in 0..n {
        sum += vec_a[i] * vec_b[i];
    }

    let elapsed = start.elapsed();
    let seconds = elapsed.as_secs_f64();
    let throughput = n as f64 / seconds;

    println!("Result: {:?}", sum);
    println!("Time: {:.4} seconds", seconds);
    println!("Throughput: {:.2} M ops/sec", throughput / 1_000_000.0);

    // 3. Estimate for 2^28
    let est_time = (1 << 28) as f64 / throughput;
    println!("Estimated time for 2^28 items: {:.2} seconds", est_time);
}
