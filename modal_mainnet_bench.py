import modal
import os

app = modal.App("morphogen-mainnet-bench")
volume = modal.Volume.from_name("morphogenesis-data", create_if_missing=True)

# Image with Rust, CUDA, and dependencies
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("curl", "build-essential", "pkg-config", "libssl-dev", "git")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        "echo 'source $HOME/.cargo/env' >> $HOME/.bashrc",
    )
    .env({"PATH": "/root/.cargo/bin:$PATH"})
    .add_local_dir(
        ".",
        remote_path="/root/morphogen",
        ignore=["target", ".git", ".jj", "venv", "__pycache__", ".DS_Store", "crates/morphogen-gpu-dpf/target"]
    )
)

@app.function(
    image=image,
    gpu="H100", # Start with H100
    timeout=3600,
    memory=131072, # 128GB RAM needed to load 60GB file
    volumes={"/data": volume}
)
def bench_full_mainnet():
    import subprocess
    import os
    
    os.chdir("/root/morphogen")
    
    # Check if file exists
    if not os.path.exists("/data/mainnet_compact.bin"):
        print("Error: /data/mainnet_compact.bin not found! Run modal_sync_r2.py first.")
        return

    print("--- Running Full Mainnet Benchmark on H100 ---")
    
    env = os.environ.copy()
    env["CUDA_ARCH"] = "sm_90"
    
    subprocess.run(
        [
            "cargo", "run", "--release", 
            "--package", "morphogen-gpu-dpf", 
            "--bin", "bench_real_data", 
            "--features", "cuda", 
            "--", 
            "--file", "/data/mainnet_compact.bin",
            "--iterations", "20"
        ],
        check=True,
        env=env
    )

@app.local_entrypoint()
def main():
    bench_full_mainnet.remote()