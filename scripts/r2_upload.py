import boto3
import os
import sys
from botocore.config import Config

# R2 Credentials
ACCESS_KEY = "5fbfe317536b096f78d27a2b90f8da84"
SECRET_KEY = "716c9584a2f9b9fb9ceab32f8df2d1d3c872634d3a3b20028cf62394a59be38b"
ENDPOINT = "https://c6ffe0823b48a4b7689d9b9e9045e465.r2.cloudflarestorage.com"
BUCKET = "pir"

def upload_file(file_name, object_name=None):
    if object_name is None:
        object_name = f"matrix/{file_name}"

    print(f"Uploading {file_name} to {BUCKET}/{object_name}...")
    
    s3 = boto3.client(
        's3',
        endpoint_url=ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        region_name="auto",
        config=Config(signature_version='s3v4')
    )

    try:
        # Multi-part upload for large files
        config = boto3.s3.transfer.TransferConfig(
            multipart_threshold=1024 * 25, # 25MB
            max_concurrency=10,
            use_threads=True
        )
        
        s3.upload_file(
            file_name, BUCKET, object_name,
            Config=config,
            Callback=ProgressPercentage(file_name)
        )
        print(f"\nSuccessfully uploaded {file_name}")
    except Exception as e:
        print(f"\nError uploading {file_name}: {e}")

class ProgressPercentage(object):
    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0

    def __call__(self, bytes_amount):
        self._seen_so_far += bytes_amount
        percentage = (self._seen_so_far / self._size) * 100
        sys.stdout.write(
            f"\r{self._filename}  {self._seen_so_far} / {self._size}  ({percentage:.2f}%)"
        )
        sys.stdout.flush()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 r2_upload.py <file_name>")
        sys.exit(1)
    
    file_to_upload = sys.argv[1]
    if os.path.exists(file_to_upload):
        upload_file(file_to_upload)
    else:
        print(f"File {file_to_upload} not found.")
