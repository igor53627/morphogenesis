"""
Generate a 108GB dummy database directly on a Modal Volume.

This avoids uploading massive data from the client.
Target: 27M pages * 4KB = 108 GB.

Run with: modal run modal_create_volume.py
"""
import modal
import os

app = modal.App("morphogen-data-gen")
volume = modal.Volume.from_name("morphogen-data", create_if_missing=True)

# 27 million pages * 4096 bytes = ~108 GB
NUM_PAGES = 27_000_000
PAGE_SIZE = 4096
TOTAL_SIZE = NUM_PAGES * PAGE_SIZE
CHUNK_SIZE = 1024 * 1024 * 1024  # 1GB chunks

@app.function(
    image=modal.Image.debian_slim(),
    volumes={"/data": volume},
    timeout=3600  # Allow 1 hour for generation
)
def generate_data():
    print(f"Generating {TOTAL_SIZE / 1e9:.2f} GB of data in /data/db.bin...")
    
    file_path = "/data/db.bin"
    
    # Check if file exists and has correct size
    if os.path.exists(file_path):
        current_size = os.path.getsize(file_path)
        if current_size == TOTAL_SIZE:
            print("Database already exists and has correct size. Skipping.")
            return
        else:
            print(f"Existing file size {current_size} mismatch. Regenerating...")
    
    # Generate data in chunks
    with open(file_path, "wb") as f:
        bytes_written = 0
        while bytes_written < TOTAL_SIZE:
            remaining = TOTAL_SIZE - bytes_written
            to_write = min(remaining, CHUNK_SIZE)
            
            # Write zeros or pattern (faster than random)
            # Just using zeros for performance benchmark is sufficient for memory BW
            # But let's add a repeating pattern to verify correctness if needed
            chunk = b"\xAA" * to_write 
            f.write(chunk)
            
            bytes_written += to_write
            print(f"Progress: {bytes_written / 1e9:.2f} / {TOTAL_SIZE / 1e9:.2f} GB")
            
    print("Generation complete.")
    volume.commit()  # Persist changes

@app.local_entrypoint()
def main():
    generate_data.remote()
