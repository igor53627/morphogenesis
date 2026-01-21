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

def run_benchmark(gpu_name, arch, batch_size=1):
    import subprocess
    import os
    
    os.chdir("/root/morphogen")
    
    if not os.path.exists("/data/mainnet_compact.bin"):
        print("Error: /data/mainnet_compact.bin not found! Run modal_sync_r2.py first.")
        return

    print(f"--- Running Full Mainnet Benchmark on {gpu_name} (Batch Size: {batch_size}) ---")
    
    env = os.environ.copy()
    env["CUDA_ARCH"] = arch
    
    # We use iterations=50 for stable measurement
    subprocess.run(
        [
            "cargo", "run", "--release", 
            "--package", "morphogen-gpu-dpf", 
            "--bin", "bench_real_data", 
            "--features", "cuda", 
            "--", 
            "--file", "/data/mainnet_compact.bin",
            "--iterations", "50",
            "--batch-size", str(batch_size)
        ],
        check=True,
        env=env
    )

@app.function(image=image, gpu="H100", timeout=3600, memory=131072, volumes={"/data": volume})
def bench_h100(batch_size: int = 1):
    run_benchmark("H100", "sm_90", batch_size)

@app.function(image=image, gpu="A100-80GB", timeout=3600, memory=131072, volumes={"/data": volume})
def bench_a100(batch_size: int = 1):
    run_benchmark("A100-80GB", "sm_80", batch_size)

# Explicitly request H200
@app.function(image=image, gpu="H200", timeout=3600, memory=131072, volumes={"/data": volume})
def bench_h200(batch_size: int = 1):
    run_benchmark("H200", "sm_90", batch_size)

# Explicitly request B200 (Note: Might fail if not in catalog)
@app.function(image=image, gpu="B200", timeout=3600, memory=131072, volumes={"/data": volume})
def bench_b200(batch_size: int = 1):
    run_benchmark("B200", "sm_90", batch_size)

@app.local_entrypoint()
def main(gpu: str = "H100", batch: int = 4):
    """
    Run benchmark on specified GPU.
    Usage: modal run modal_mainnet_bench.py --gpu H200 --batch 4
    Options: H100, A100, H200, B200
    """
    if gpu.upper() == "H100":
        bench_h100.remote(batch)
    elif gpu.upper() == "A100":
        bench_a100.remote(batch)
    elif gpu.upper() == "H200":
        bench_h200.remote(batch) 
    elif gpu.upper() == "B200":
        bench_b200.remote(batch)
    else:
        print(f"Unknown GPU: {gpu}. Defaulting to H100.")
        bench_h100.remote(batch)
