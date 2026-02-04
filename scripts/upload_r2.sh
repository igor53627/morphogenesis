#!/bin/bash
set -e

# R2 Credentials (from memory)
export RCLONE_CONFIG_R2_TYPE=s3
export RCLONE_CONFIG_R2_PROVIDER=Cloudflare
export RCLONE_CONFIG_R2_ACCESS_KEY_ID=5fbfe317536b096f78d27a2b90f8da84
export RCLONE_CONFIG_R2_SECRET_ACCESS_KEY=716c9584a2f9b9fb9ceab32f8df2d1d3c872634d3a3b20028cf62394a59be38b
export RCLONE_CONFIG_R2_ENDPOINT=https://c6ffe0823b48a4b7689d9b9e9045e465.r2.cloudflarestorage.com
export RCLONE_CONFIG_R2_ACL=private
export RCLONE_CONFIG_R2_REGION=auto

BUCKET="pir/matrix"

echo "=== Uploading to Cloudflare R2 ($BUCKET) ==="

if [ -f "mainnet_compact.bin" ]; then
    echo "Uploading mainnet_compact.bin..."
    rclone copy mainnet_compact.bin r2:$BUCKET --progress --transfers 16
else
    echo "mainnet_compact.bin not found!"
fi

if [ -f "mainnet_compact.ubt" ]; then
    echo "Uploading mainnet_compact.ubt..."
    rclone copy mainnet_compact.ubt r2:$BUCKET --progress --transfers 16
else
    echo "mainnet_compact.ubt not found (process still running?)"
fi

echo "Uploads complete."
