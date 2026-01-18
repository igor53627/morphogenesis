use morphogen_core::CuckooTable;
use std::time::Instant;

fn main() {
    let sizes = [
        (1_000_000, "1M"),
        (10_000_000, "10M"),
        (78_000_000, "78M (Ethereum scale)"),
    ];

    println!("Cuckoo Hash Table Load Factor Benchmark");
    println!("========================================\n");

    for (num_entries, label) in sizes {
        println!("Target entries: {} ({})", num_entries, label);

        for target_load in [0.85, 0.90, 0.91] {
            let table_size = (num_entries as f64 / target_load) as usize;
            let mut table: CuckooTable<u64> = CuckooTable::new(table_size);

            let start = Instant::now();
            let mut inserted = 0u64;
            let mut failed = false;

            for i in 0..num_entries {
                let key = format!("0x{:040x}", i).into_bytes();
                match table.insert(key, i as u64) {
                    Ok(_) => inserted += 1,
                    Err(_) => {
                        failed = true;
                        break;
                    }
                }
            }

            let elapsed = start.elapsed();
            let actual_load = table.load_factor();
            let stash_used = table.stash_len();

            println!(
                "  Target {:.0}%: table_size={}, inserted={}, actual_load={:.2}%, stash={}, time={:.2}s {}",
                target_load * 100.0,
                table_size,
                inserted,
                actual_load * 100.0,
                stash_used,
                elapsed.as_secs_f64(),
                if failed { "[FAILED]" } else { "[OK]" }
            );
        }
        println!();
    }

    println!("Memory implications for 78M accounts:");
    println!("--------------------------------------");
    for target_load in [0.85, 0.90] {
        let accounts = 78_000_000u64;
        let table_size = (accounts as f64 / target_load) as usize;
        let row_size_kb = 1;
        let matrix_gb = (table_size * row_size_kb) as f64 / (1024.0 * 1024.0);
        let scan_time_ms = matrix_gb / 393.0 * 1000.0;

        println!(
            "  {:.0}% load: {} rows, {:.1} GB matrix, ~{:.0}ms scan",
            target_load * 100.0,
            table_size,
            matrix_gb,
            scan_time_ms
        );
    }
}
