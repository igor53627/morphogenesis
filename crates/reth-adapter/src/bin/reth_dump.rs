use clap::Parser;
use reth_adapter::{build_matrix, RowScheme};
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    /// Path to Reth DB
    #[arg(short, long)]
    db: PathBuf,

    /// Output path for matrix.bin
    #[arg(short, long, default_value = "matrix.bin")]
    output: PathBuf,

    /// Row scheme (compact=32B, full=64B, optimized48=48B)
    #[arg(short, long, value_enum, default_value_t = RowScheme::Optimized48)]
    scheme: RowScheme,

    /// Number of rows in Cuckoo table
    #[arg(short, long, default_value_t = 27000000)]
    rows: usize,

    /// Build with UBT proofs (Trustless mode)
    #[arg(short, long)]
    trustless: bool,

    /// Estimate optimal table size instead of extracting
    #[arg(short, long)]
    estimate: bool,

    /// Sample data to analyze compression potential
    #[arg(long)]
    sample: bool,

    /// Verify 16-byte balance compression safety
    #[arg(long)]
    verify: bool,

    /// Extract bytecode using dictionary (pass path to .dict file)
    #[arg(long)]
    extract_code: Option<PathBuf>,

    /// Estimate Merkle Proof sizes (MPT) by sampling
    #[arg(long)]
    estimate_proofs: bool,
}

fn main() {
    let args = Args::parse();

    println!("=== Morphogenesis Reth Extraction Tool ===");
    println!("DB Path: {:?}", args.db);

    if args.estimate_proofs {
        println!("Mode: ESTIMATE PROOFS");
        #[cfg(feature = "reth")]
        {
            let source = reth_adapter::RethSource::new(args.db.to_str().unwrap());
            source.estimate_proof_sizes(100); // Sample 100 accounts
        }
        #[cfg(not(feature = "reth"))]
        {
            println!("Error: Compile with --features reth for proof estimation.");
        }
        return;
    }

    if let Some(_dict_path) = args.extract_code {
        println!("Mode: EXTRACT CODE");
        let _out_dir = args.output; // Reuse output as directory
        #[cfg(feature = "reth")]
        {
            reth_adapter::extract_code_from_dict(args.db.to_str().unwrap(), &_dict_path, &_out_dir);
        }
        #[cfg(not(feature = "reth"))]
        {
            println!("Error: Compile with --features reth for code extraction.");
        }
        return;
    }

    if args.verify {
        println!("Mode: VERIFY COMPRESSION");
        #[cfg(feature = "reth")]
        {
            let source = reth_adapter::RethSource::new(args.db.to_str().unwrap());
            source.verify_compression();
        }
        return;
    }

    if args.sample {
        println!("Mode: SAMPLE");
        #[cfg(feature = "reth")]
        {
            let source = reth_adapter::RethSource::new(args.db.to_str().unwrap());
            source.sample_data(10); // Sample 10 items
        }
        #[cfg(not(feature = "reth"))]
        {
            println!("Error: Compile with --features reth for sampling.");
        }
        return;
    }

    if args.estimate {
        println!("Mode: ESTIMATE");
        #[cfg(feature = "reth")]
        {
            let source = reth_adapter::RethSource::new(args.db.to_str().unwrap());
            let count = source.count_items();
            let recommended = (count as f64 / 0.85).ceil() as usize;
            println!("Found {} items (Accounts + Storage).", count);
            println!("Recommended --rows: {}", recommended);
        }
        #[cfg(not(feature = "reth"))]
        {
            println!("Error: Compile with --features reth for real estimation.");
            let count = (args.rows as f64 * 0.8) as usize; // Demo count
            let recommended = (count as f64 / 0.85).ceil() as usize;
            println!("Found {} items (synthetic).", count);
            println!("Recommended --rows: {}", recommended);
        }
        return;
    }

    println!("Mode: EXTRACT");
    println!("Output: {:?}", args.output);
    println!("Scheme: {:?}", args.scheme);
    println!("Target Rows: {}", args.rows);

    #[cfg(feature = "reth")]
    {
        let (matrix, manifest, indexer) = reth_adapter::dump_reth_to_matrix(
            args.db.to_str().unwrap(),
            args.rows,
            args.scheme,
            args.trustless,
        );
        println!("Saving matrix to {:?}...", args.output);
        matrix.write_to_file(&args.output).unwrap();

        let manifest_path = args.output.with_extension("json");
        let file = std::fs::File::create(manifest_path).unwrap();
        serde_json::to_writer_pretty(file, &manifest).unwrap();

        if let RowScheme::Compact = args.scheme {
            let dict_path = args.output.with_extension("dict");
            println!(
                "Saving code dictionary to {:?} ({} entries)...",
                dict_path,
                indexer.list.len()
            );
            // Serialize as flat list of 32B hashes
            let mut dict_file = std::fs::File::create(dict_path).unwrap();
            use std::io::Write;
            for hash in indexer.list {
                dict_file.write_all(&hash).unwrap();
            }
        }

        println!("Extraction complete.");
    }

    #[cfg(not(feature = "reth"))]
    {
        println!("Error: This binary must be compiled with --features reth to use RethSource.");
        println!("Running with SyntheticSource for demo...");
        let mut source = reth_adapter::SyntheticSource::new((args.rows as f64 * 0.8) as usize);
        let (_matrix, manifest, _indexer) =
            build_matrix(&mut source, args.rows, args.scheme, args.trustless);

        let manifest_path = args.output.with_extension("json");
        println!("Writing manifest to {:?}", manifest_path);
        println!(
            "Manifest: {}",
            serde_json::to_string_pretty(&manifest).unwrap()
        );
    }
}
