import modal

app = modal.App("morphogen-rust-test")

# Image with Rust and CUDA
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04")
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

@app.function(image=image, gpu="T4", timeout=600)
def test_gpu_integration():
    import subprocess
    import os
    
    os.chdir("/root/morphogen")
    
    print("--- Checking nvcc ---")
    subprocess.run(["nvcc", "--version"], check=True)
    
    print("\n--- Running morphogen-gpu-dpf tests (CUDA) ---")
    # This compiles the kernel and runs tests that use GpuScanner
    # Note: We need to ensure build.rs runs successfully with nvcc
    subprocess.run(
        ["cargo", "test", "-p", "morphogen-gpu-dpf", "--features", "cuda", "--", "--nocapture"],
        check=True
    )

    print("\n--- Running morphogen-server tests (CUDA feature enabled) ---")
    # This validates that the server compiles with cuda feature and fallback logic works
    subprocess.run(
        ["cargo", "test", "-p", "morphogen-server", "--features", "cuda", "--", "--nocapture"],
        check=True
    )

@app.local_entrypoint()
def main():
    test_gpu_integration.remote()
