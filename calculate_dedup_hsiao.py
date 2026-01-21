import modal
import os
import struct
import numpy as np

app = modal.App("morphogen-dedup-stats")
volume = modal.Volume.from_name("morphogenesis-data")

image = modal.Image.debian_slim().pip_install("numpy", "tqdm")

@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=3600,
    memory=32768, # 32GB for reference counts
)
def calculate_dedup_stats():
    matrix_path = "/root/morphogenesis/mainnet_compact.bin"
    dict_path = "/root/morphogenesis/mainnet_compact.dict"
    
    if not os.path.exists(matrix_path):
        return "Error: matrix.bin not found"
    
    # 1. Load Dictionary to get Unique Count
    print("Reading code dictionary...")
    with open(dict_path, "rb") as f:
        dict_data = f.read()
    
    unique_contracts = len(dict_data) // 32
    print(f"Unique Contracts in dictionary: {unique_contracts:,}")
    
    # 2. Scan Matrix for Reference Counts
    print("Scanning 68.8GB matrix for code references...")
    # CodeID is at offset 24 in every 32-byte row.
    # It's a 4-byte big-endian integer.
    
    # We use numpy for fast scanning
    file_size = os.path.getsize(matrix_path)
    num_rows = file_size // 32
    print(f"Total Rows: {num_rows:,}")
    
    # We'll use a 1D array to store counts for each CodeID
    # Most CodeIDs are < unique_contracts
    counts = np.zeros(unique_contracts + 1000, dtype=np.uint64)
    
    chunk_size_rows = 10_000_000 # ~320MB chunks
    
    with open(matrix_path, "rb") as f:
        for i in range(0, num_rows, chunk_size_rows):
            batch_size = min(chunk_size_rows, num_rows - i)
            data = f.read(batch_size * 32)
            if not data:
                break
            
            # View as uint32 array with stride
            # We want bytes 24, 25, 26, 27 of every 32-byte block
            # In numpy: reshape to (batch, 32), take col [24:28]
            arr = np.frombuffer(data, dtype=np.uint8).reshape(-1, 32)
            # CodeID is big-endian uint32 at [24:28]
            code_ids_bytes = arr[:, 24:28]
            # Convert to uint32 (vectorized)
            code_ids = (code_ids_bytes[:, 0].astype(np.uint32) << 24) |
                       (code_ids_bytes[:, 1].astype(np.uint32) << 16) |
                       (code_ids_bytes[:, 2].astype(np.uint32) << 8)  |
                       (code_ids_bytes[:, 3].astype(np.uint32))
            
            # Count occurrences of each ID
            # Ignore ID 0 (EOA)
            mask = code_ids > 0
            active_ids = code_ids[mask]
            
            # Use bincount to update counts
            if active_ids.size > 0:
                batch_counts = np.bincount(active_ids, minlength=len(counts))
                counts[:len(batch_counts)] += batch_counts
            
            if (i // chunk_size_rows) % 10 == 0:
                print(f"  Processed {i:,} rows...")

    # 3. Analyze results
    total_references = np.sum(counts[1:])
    entries_with_code = total_references
    unique_referenced = np.count_nonzero(counts[1:])
    
    # We need sizes to calculate space. 
    # If we don't have code_store, we can't get exact sizes.
    # But we can estimate based on average contract size (approx 10KB)
    # OR we can check if code_store exists.
    
    code_store_path = "/root/morphogenesis/code_store"
    total_unique_size = 0
    total_redundant_size = 0
    
    if os.path.exists(code_store_path):
        print("Calculating exact sizes from code_store...")
        # Dictionary index matches CodeID
        for i in range(1, unique_contracts):
            if counts[i] == 0: continue
            
            hash_bytes = dict_data[i*32 : (i+1)*32]
            hex_hash = hash_bytes.hex()
            shard1 = hex_hash[0:2]
            shard2 = hex_hash[2:4]
            path = os.path.join(code_store_path, shard1, shard2, f"{hex_hash}.bin")
            
            if os.path.exists(path):
                size = os.path.getsize(path)
                total_unique_size += size
                total_redundant_size += size * counts[i]
    else:
        print("Warning: code_store not found. Using estimates.")
        # Average contract size ~8KB
        avg_size = 8000 
        total_unique_size = unique_referenced * avg_size
        total_redundant_size = total_references * avg_size

    # Report
    print("\n" + "="*40)
    print("=== DEDUPLICATION REPORT ===")
    print("="*40)
    print(f"Total Account Entries Scanned: {num_rows:,}")
    print(f"Accounts with Code (Contracts): {entries_with_code:,}")
    print(f"Unique Bytecodes Found:        {unique_referenced:,}")
    print(f"Deduplicated Entries:          {entries_with_code - unique_referenced:,}")
    
    if total_unique_size > 0:
        print(f"\nDeduplicated (CAS) Size:       {total_unique_size / 1e9:.2f} GB")
        print(f"Non-Deduplicated Size (est):   {total_redundant_size / 1e9:.2f} GB")
        print(f"Space Saved:                   {(total_redundant_size - total_unique_size) / 1e9:.2f} GB")
        print(f"Deduplication Factor:          {total_redundant_size / total_unique_size:.1f}x")
    
    return "Report generated successfully."

@app.local_entrypoint()
def main():
    calculate_dedup_stats.remote()
