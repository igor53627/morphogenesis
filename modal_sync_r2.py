import modal
import os

app = modal.App("morphogen-sync")
volume = modal.Volume.from_name("morphogenesis-data", create_if_missing=True)

# Image with rclone installed
image = modal.Image.debian_slim().apt_install("rclone", "curl", "unzip")

@app.function(
    image=image,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_dict({
        "RCLONE_CONFIG_R2_TYPE": "s3",
        "RCLONE_CONFIG_R2_PROVIDER": "Cloudflare",
        "RCLONE_CONFIG_R2_ACCESS_KEY_ID": "5fbfe317536b096f78d27a2b90f8da84",
        "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY": "716c9584a2f9b9fb9ceab32f8df2d1d3c872634d3a3b20028cf62394a59be38b",
        "RCLONE_CONFIG_R2_ENDPOINT": "https://c6ffe0823b48a4b7689d9b9e9045e465.r2.cloudflarestorage.com",
        "RCLONE_CONFIG_R2_ACL": "private",
    })],
    timeout=7200
)
def sync_from_r2():
    print("Starting sync from Cloudflare R2...")
    
    # Check if file exists and size matches
    # R2 path: pir/matrix/mainnet_compact.bin
    
    # We use --size-only to avoid checksumming 60GB if size matches
    exit_code = os.system("rclone copy r2:pir/matrix/mainnet_compact.bin /data/ --progress --transfers 16")
    
    if exit_code != 0:
        raise Exception("Rclone failed")
        
    print("Sync complete!")
    
    # Also sync metadata
    os.system("rclone copy r2:pir/matrix/mainnet_compact.json /data/")
    os.system("rclone copy r2:pir/matrix/mainnet_compact.dict /data/")
    
    volume.commit()

@app.local_entrypoint()
def main():
    sync_from_r2.remote()
