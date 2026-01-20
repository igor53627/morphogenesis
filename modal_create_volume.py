import modal
import requests
import os

app = modal.App("morphogen-setup")
volume = modal.Volume.from_name("morphogenesis-data", create_if_missing=True)

@app.function(volumes={"/data": volume}, timeout=7200) # 2 hours timeout
def download_matrix():
    url = "http://104.204.142.13:12345/mainnet_compact.bin"
    output_path = "/data/mainnet_compact.bin"
    
    if os.path.exists(output_path):
        print(f"File {output_path} already exists. Checking size...")
        if os.path.getsize(output_path) > 50 * 1024 * 1024 * 1024:
            print("File seems complete (size > 50GB). Skipping.")
            return

    print(f"Downloading {url} to {output_path}...")
    
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        downloaded = 0
        
        with open(output_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024*1024*16): # 16MB chunks
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded % (1024*1024*1024) == 0: # Log every 1GB
                        print(f"Downloaded {downloaded / (1024*1024*1024):.2f} GB / {total_size / (1024*1024*1024):.2f} GB")
                
    print("Download complete!")
    volume.commit()

@app.local_entrypoint()
def main():
    download_matrix.remote()