use clap::Parser;
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, DriverError, LaunchAsync, LaunchConfig};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::Ptx;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Log2 of the vector size (e.g., 20 for 1M rows)
    #[arg(short, long, default_value_t = 24)]
    log_n: u32,

    /// Number of iterations for benchmarking
    #[arg(short, long, default_value_t = 10)]
    iterations: u32,
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("Error: This benchmark requires the 'cuda' feature to be enabled.");
    println!("Please run with: cargo run --release --features cuda --bin bench_binius_gpu");
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let n = 1 << args.log_n;
    let size_bytes = n * 16; // 128-bit elements

    println!("=== Binius GPU Dot Product Benchmark ===");
    println!("N = 2^{} = {} elements", args.log_n, n);
    println!("Data Size: {:.2} GB", size_bytes as f64 / 1e9);

    let dev = CudaDevice::new(0)?;
    println!("GPU: {}", dev.name()?);

    // Load the PTX (compiled at runtime or embedded)
    // For this bench, we assume the user has compiled `binius_test.cu` to PTX or uses nvrtc.
    // Simpler: use nvrtc to compile the source code string directly.

    let ptx = Ptx::from_src(include_str!("../../cuda/binius_test.cu"));

    // Note: binius_test.cu has a main() and extern "C" run_bench.
    // We can't easily call the C++ run_bench from Rust via cudarc without binding.
    // Instead, we should extract the kernels into a separate .cu file or string.

    // BETTER APPROACH for user:
    // Just tell them to compile the .cu file with nvcc!
    // running `nvcc -O3 -arch=sm_90 binius_test.cu -o bench_binius_cu` is simpler than rust shim.

    println!("\n[NOTE] Ideally, run the standalone CUDA binary for raw kernel performance:");
    println!(
        "       nvcc -O3 -arch=native cuda/binius_test.cu -o bench_binius_cu && ./bench_binius_cu"
    );

    Ok(())
}
