use clap::Parser;
use ubt_exex::{AccountSource, build_matrix};
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    /// Path to Reth DB
    #[arg(short, long)]
    db: PathBuf,

    /// Output path for matrix.bin
    #[arg(short, long, default_value = "matrix.bin")]
    output: PathBuf,

    /// Number of rows in Cuckoo table
    #[arg(short, long, default_value_t = 27000000)]
    rows: usize,

    /// Row size in bytes (256 or 2048)
    #[arg(short, long, default_value_t = 256)]
    size: usize,

    /// Build with UBT proofs (Trustless mode)
    #[arg(short, long)]
    trustless: bool,

    /// Estimate optimal table size instead of extracting
    #[arg(short, long)]
    estimate: bool,
}

fn main() {
    let args = Args::parse();

    println!("=== Morphogenesis Reth Extraction Tool ===");
    println!("DB Path: {:?}", args.db);

    if args.estimate {
        println!("Mode: ESTIMATE");
        #[cfg(feature = "reth")]
        {
            // let mut source = ubt_exex::RethSource::new(args.db.to_str().unwrap());
            // let count = source.count_all();
            // let recommended = (count as f64 / 0.85).ceil() as usize;
            // println!("Found {} accounts.", count);
            // println!("Recommended --rows: {}", recommended);
            println!("Reth estimation would run here.");
        }
        #[cfg(not(feature = "reth"))]
        {
            println!("Error: Compile with --features reth for real estimation.");
            let count = (args.rows as f64 * 0.8) as usize; // Demo count
            let recommended = (count as f64 / 0.85).ceil() as usize;
            println!("Found {} accounts (synthetic).", count);
            println!("Recommended --rows: {}", recommended);
        }
        return;
    }

    println!("Mode: EXTRACT");
    println!("Output: {:?}", args.output);
    println!("Target Rows: {}", args.rows);
    println!("Row Size: {}", args.size);

    #[cfg(feature = "reth")]
    {
        // let mut source = ubt_exex::RethSource::new(args.db.to_str().unwrap());
        // let (matrix, seeds) = build_matrix(&mut source, args.rows, args.size, args.trustless);
        // matrix.write_to_file(args.output).unwrap();
        println!("Reth extraction logic would run here.");
    }

    #[cfg(not(feature = "reth"))]
    {
        println!("Error: This binary must be compiled with --features reth to use RethSource.");
        println!("Running with SyntheticSource for demo...");
        let mut source = ubt_exex::SyntheticSource::new((args.rows as f64 * 0.8) as usize);
        let (matrix, seeds) = build_matrix(&mut source, args.rows, args.size, args.trustless);
        // matrix.write_to_file(args.output).unwrap();
        println!("Synthetic matrix generated. Seeds: {:x?}", seeds);
    }
}
