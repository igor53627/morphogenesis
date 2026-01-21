use clap::Parser;
use morphogen_gpu_dpf::dpf::ChaChaParams;
use morphogen_gpu_dpf::kernel::GpuScanner;
use morphogen_gpu_dpf::storage::GpuPageMatrix;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
struct Args {
    /// Path to matrix file
    #[arg(short, long)]
    file: PathBuf,

    /// Number of iterations
    #[arg(short, long, default_value_t = 10)]
    iterations: usize,
    
    /// GPU Device ID
    #[arg(long, default_value_t = 0)]
    gpu: usize,
}

fn main() {
    let args = Args::parse();
    println!("=== Morphogenesis Real Data Benchmark ===");
    println!("File: {:?}", args.file);

    // 1. Load File
    let start_load = Instant::now();
    let mut data = std::fs::read(&args.file).expect("Failed to read file");
    let load_time = start_load.elapsed();
    let total_bytes = data.len();
    println!("Loaded {:.2} GB in {:.2}s ({:.2} GB/s)", 
        total_bytes as f64 / 1e9, 
        load_time.as_secs_f64(),
        (total_bytes as f64 / 1e9) / load_time.as_secs_f64()
    );

    // Pad to 4096 bytes (GpuPageMatrix requirement)
    let remainder = data.len() % 4096;
    if remainder != 0 {
        let padding = 4096 - remainder;
        println!("Padding data with {} bytes to align to 4KB page...", padding);
        data.extend(std::iter::repeat(0).take(padding));
    }

    // 2. Init GPU
    println!("Initializing GPU {}...", args.gpu);
    let scanner = GpuScanner::new(args.gpu).expect("Failed to init scanner");
    
    // Auto-detect parameters
    let row_size = 32; // We know it's compact
    let num_rows = total_bytes / row_size;
    let domain_bits = (num_rows as f64).log2().ceil() as usize;
    
    println!("Detected Matrix: {} rows, {} bits domain, {} bytes/row", 
        num_rows, domain_bits, row_size);

    // 3. Upload
    println!("Uploading to VRAM...");
    let start_upload = Instant::now();
    let gpu_db = GpuPageMatrix::new(scanner.device.clone(), &data).expect("Failed to upload");
    let upload_time = start_upload.elapsed();
    println!("Uploaded in {:.2}s ({:.2} GB/s)",
        upload_time.as_secs_f64(),
        (total_bytes as f64 / 1e9) / upload_time.as_secs_f64()
    );

    // 4. Benchmark
    println!("Running {} iterations...", args.iterations);
    let params = ChaChaParams::new(domain_bits).unwrap();
    let (k0, _k1) = morphogen_gpu_dpf::dpf::generate_chacha_dpf_keys(&params, 12345).unwrap();
    let keys = [&k0, &k0, &k0]; // Batch of 3 queries

    // Warmup
    unsafe { scanner.scan(&gpu_db, keys).unwrap(); }

    let mut latencies = Vec::new();
    for i in 0..args.iterations {
        let start = Instant::now();
        unsafe { scanner.scan(&gpu_db, keys).unwrap(); }
        latencies.push(start.elapsed());
        print!(".");
        use std::io::Write;
        std::io::stdout().flush().unwrap();
    }
    println!();

    let avg_latency = latencies.iter().sum::<std::time::Duration>() / args.iterations as u32;
    let avg_ms = avg_latency.as_secs_f64() * 1000.0;
    let throughput = (total_bytes as f64 / 1e9) / avg_latency.as_secs_f64();

    println!("Results:");
    println!("  Avg Latency: {:.2} ms", avg_ms);
    println!("  Throughput:  {:.2} GB/s", throughput);
}
