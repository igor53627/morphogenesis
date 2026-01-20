import modal

app = modal.App("morphogen-sharded-bench")

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

@app.function(image=image, gpu="H100:2", timeout=1200)
def bench_sharded_h100():
    import subprocess
    import os
    
    os.chdir("/root/morphogen")
    os.environ["CUDA_ARCH"] = "sm_90"
    
    print("--- Running 2x H100 Sharded Benchmark ---")
    subprocess.run(
        ["cargo", "run", "--release", "--package", "morphogen-gpu-dpf", "--bin", "bench_sharded", "--features", "cuda", "--", "--gpu"],
        check=True
    )

@app.local_entrypoint()
def main():
    bench_sharded_h100.remote()
