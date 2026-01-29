use binius_field::{BinaryField128b, Field};
use morphogen_core::sumcheck::SumCheckProver;
use rayon::prelude::*;
use std::time::Instant;

const TRACE_WIDTH: usize = 16; // Simulating 16 columns for PRG state

fn main() {
    let log_n = 20; // Start with 2^20 (1M rows)
    let n = 1 << log_n;

    println!("=== Verifiable DPF Benchmark (Binius/SumCheck) ===");
    println!("Parameters:");
    println!("  Rows (N):       2^{} = {}", log_n, n);
    println!("  Trace Width:    {} columns (128-bit fields)", TRACE_WIDTH);
    println!("  Parallelism:    Rayon (Auto)");

    // 1. Trace Generation
    // Simulate generating the full DPF expansion trace.
    // In reality, this involves ChaCha8/AES PRG calls.
    // Here we just fill memory to simulate the bandwidth/allocation cost.
    println!("\n[1] Trace Generation (Simulation)...");
    let start_gen = Instant::now();

    // We use a flat vector to be cache-friendly, logically divided into chunks of TRACE_WIDTH
    let mut trace = vec![BinaryField128b::ZERO; n * TRACE_WIDTH];

    trace
        .par_chunks_mut(TRACE_WIDTH)
        .enumerate()
        .for_each(|(i, row)| {
            // Simulate some PRG work: fill with pseudo-random data based on index
            let seed = i as u128;
            for j in 0..TRACE_WIDTH {
                row[j] = BinaryField128b::new(
                    seed.wrapping_add(j as u128)
                        .wrapping_mul(6364136223846793005),
                );
            }
        });

    let time_gen = start_gen.elapsed();
    println!("    Time: {:.4} s", time_gen.as_secs_f64());
    println!(
        "    Throughput: {:.2} M rows/sec",
        (n as f64 / time_gen.as_secs_f64()) / 1e6
    );

    // 2. RLC Compression (Batching)
    // Reduce the Wide Trace (16 cols) to a Single Column using random weights.
    // T' = \sum alpha^j * T_j
    println!("\n[2] RLC Compression (Batching)...");
    let alpha = BinaryField128b::new(123456789); // Random challenge
    let mut alphas = [BinaryField128b::ONE; TRACE_WIDTH];
    let mut curr = BinaryField128b::ONE;
    for i in 1..TRACE_WIDTH {
        curr *= alpha;
        alphas[i] = curr;
    }

    let start_rlc = Instant::now();

    // We produce a single column "compressed_trace"
    let compressed_trace: Vec<BinaryField128b> = trace
        .par_chunks(TRACE_WIDTH)
        .map(|row| {
            let mut acc = BinaryField128b::ZERO;
            for j in 0..TRACE_WIDTH {
                acc += row[j] * alphas[j];
            }
            acc
        })
        .collect();

    let time_rlc = start_rlc.elapsed();
    println!("    Time: {:.4} s", time_rlc.as_secs_f64());
    println!(
        "    Throughput: {:.2} M rows/sec",
        (n as f64 / time_rlc.as_secs_f64()) / 1e6
    );

    // 3. SumCheck Proving
    // Run the standard SumCheck on the compressed trace against a dummy selector/query vector.
    println!("\n[3] SumCheck Proving...");
    let query_vector = vec![BinaryField128b::new(0xCAFEBABE); n];
    let challenges: Vec<_> = (0..log_n)
        .map(|i| BinaryField128b::new(i as u128 + 1))
        .collect();

    let prover = SumCheckProver::new(compressed_trace, query_vector);

    let start_prove = Instant::now();
    let proof = prover.prove(&challenges);
    let time_prove = start_prove.elapsed();

    println!("    Time: {:.4} s", time_prove.as_secs_f64());
    println!("    Proof Size: {} rounds", proof.round_polynomials.len());
    println!(
        "    Throughput: {:.2} M rows/sec",
        (n as f64 / time_prove.as_secs_f64()) / 1e6
    );

    // --- Totals & Projections ---
    let total_time = time_gen + time_rlc + time_prove;
    println!("\n=== Results ===");
    println!(
        "Total Time (2^{}): {:.4} s",
        log_n,
        total_time.as_secs_f64()
    );

    let scale_factor = (1 << 28) as f64 / n as f64; // Scale to 2^28 (268M)
    let projected_time = total_time.as_secs_f64() * scale_factor;

    println!("\n>>> Projection for Mainnet (2^28 rows) <<<");
    println!(
        "Trace Gen:   {:.2} s",
        time_gen.as_secs_f64() * scale_factor
    );
    println!(
        "RLC Compress: {:.2} s",
        time_rlc.as_secs_f64() * scale_factor
    );
    println!(
        "SumCheck:    {:.2} s",
        time_prove.as_secs_f64() * scale_factor
    );
    println!("--------------------------");
    println!("TOTAL:       {:.2} s", projected_time);

    // Binius optimizations typically fuse steps.
    // This is a naive "allocate-then-process" benchmark.
    // A fused implementation would be significantly faster (memory bandwidth).
    println!("\nNote: This is a memory-bound baseline (separate passes).");
    println!("      A fused GPU kernel would likely be 3-5x faster.");
}
