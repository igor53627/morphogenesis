use ubt_exex::{build_matrix, SyntheticSource};

fn main() {
    // Small scale demo
    let num_rows = 100_000; 
    let row_size = 256;
    let load_factor = 0.85;
    let num_accounts = (num_rows as f64 * load_factor) as usize;
    
    println!("=== Morphogenesis UBT/Matrix Generator ===");
    println!("Target Rows: {}", num_rows);
    println!("Input Accounts: {}", num_accounts);
    println!("Row Size: {} bytes", row_size);

    let mut source = SyntheticSource::new(num_accounts);

    let (matrix, seeds) = build_matrix(&mut source, num_rows, row_size, false);
    
    println!("Generation Complete.");
    println!("Matrix Size: {} bytes", matrix.total_size_bytes());
    println!("Seeds: {:x?}", seeds);
    
    // In a real execution, we would serialize `matrix` to a file here.
    // e.g. matrix.write_to_file("snapshot_epoch_0.bin");
}
