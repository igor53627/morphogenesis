use morphogen_core::cuckoo::CuckooAddresser;

fn main() {
    let num_rows = 1024;
    let seeds = [0x1234, 0x5678, 0x9ABC];
    let target = 42;
    let addresser = CuckooAddresser::with_seeds(num_rows, seeds);

    println!("Searching for address that hashes to {}...", target);

    for i in 0..1000000 {
        let addr = format!("{:040x}", i);
        let addr_bytes = hex::decode(&addr).unwrap();
        let indices = addresser.hash_indices(&addr_bytes);
        if indices.contains(&target) {
            println!("Found! Address: 0x{}", addr);
            println!("Indices: {:?}", indices);
            return;
        }
    }
}
