import modal

app = modal.App("morphogen-mainnet-bench")

# Image with Rust and CUDA
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

def run_bench(gpu_type: str, arch: str, domain_bits: int):
    import subprocess
    import os
    
    os.chdir("/root/morphogen")
    os.environ["CUDA_ARCH"] = arch
    
    print(f"--- Benchmarking on {gpu_type} ({arch}) - Domain: {domain_bits} bits ---")
    
    # Run the benchmark binary with --gpu flag
    # Note: We use --release for peak performance
    subprocess.run(
        ["cargo", "run", "--release", "--package", "morphogen-gpu-dpf", "--bin", "bench_25bit", "--features", "cuda", "--", "--gpu", "--domain", str(domain_bits)],
        check=True
    )

@app.function(image=image, gpu="H100", timeout=1800, memory=131072) # Need 128GB host RAM
def bench_h100():
    run_bench("H100", "sm_90", 24)

@app.function(image=image, gpu="H200", timeout=1800, memory=131072)
def bench_h200():
    run_bench("H200", "sm_90", 25)

@app.local_entrypoint()
def main():
    print("Starting Mainnet-Scale GPU Benchmarks (108GB)...")
    
    try:
        bench_h100.remote()
    except Exception as e:
        print(f"H100 Error: {e}")
        
    print("\n" + "="*50 + "\n")
    
    try:
        bench_h200.remote()
    except Exception as e:
        print(f"H200 Error: {e}")
