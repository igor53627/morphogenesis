#![cfg(feature = "verifiable-pir")]

use binius_field::BinaryField128b;
use morphogen_core::sumcheck::SumCheckProver;
use std::time::Instant;

fn main() {
    let log_n = 20; // 1M elements
    let n = 1 << log_n;

    println!("=== Sum-Check Prover Benchmark ===");
    println!("Vector Size: 2^{} = {} elements", log_n, n);

    let db = vec![BinaryField128b::new(12345); n];
    let query = vec![BinaryField128b::new(67890); n];
    let challenges: Vec<_> = (0..log_n)
        .map(|i| BinaryField128b::new(i as u128 + 1))
        .collect();

    let prover = SumCheckProver::new(db, query);

    println!("Starting prover...");
    let start = Instant::now();
    let proof = prover.prove(&challenges);
    let elapsed = start.elapsed();

    println!("Proving time: {:.4} seconds", elapsed.as_secs_f64());
    println!(
        "Proof size: {} round polynomials",
        proof.round_polynomials.len()
    );

    let throughput = n as f64 / elapsed.as_secs_f64();
    println!("Throughput: {:.2} M elements/sec", throughput / 1_000_000.0);

    // Estimate for 2^28
    let est_time = (1 << 28) as f64 / throughput;
    println!("Estimated time for 2^28 items: {:.2} seconds", est_time);
}
