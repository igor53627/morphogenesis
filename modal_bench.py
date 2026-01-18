import modal

app = modal.App("morphogenesis-bench")

# Copy source files directly into the image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "build-essential", "procps")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
    )
    .env({"PATH": "/root/.cargo/bin:$PATH"})
    .add_local_dir("/Users/user/pse/morphogenesis/crates", "/root/morphogenesis/crates")
    .add_local_file("/Users/user/pse/morphogenesis/Cargo.toml", "/root/morphogenesis/Cargo.toml")
    .add_local_file("/Users/user/pse/morphogenesis/Cargo.lock", "/root/morphogenesis/Cargo.lock")
)


@app.function(
    image=image,
    cpu=16.0,
    memory=131072,  # 128GB
    timeout=900,
)
def run_cpu_benchmark():
    import subprocess
    import os
    
    os.environ["PATH"] = "/root/.cargo/bin:" + os.environ.get("PATH", "")
    os.chdir("/root/morphogenesis")
    
    # Get CPU info
    cpu_info = subprocess.run(["cat", "/proc/cpuinfo"], capture_output=True, text=True)
    cpu_models = [l for l in cpu_info.stdout.split('\n') if 'model name' in l]
    print(f"CPU: {cpu_models[0] if cpu_models else 'unknown'}")
    print(f"CPU cores: {len(cpu_models)}")
    
    # Get memory info
    mem_info = subprocess.run(["free", "-h"], capture_output=True, text=True)
    print("Memory:\n", mem_info.stdout)
    
    # Check for AVX-512
    avx_check = subprocess.run(["sh", "-c", "grep -o 'avx512[a-z]*' /proc/cpuinfo | sort -u"], capture_output=True, text=True)
    avx_features = avx_check.stdout.strip().split('\n') if avx_check.stdout.strip() else []
    print(f"AVX-512 features: {avx_features}")
    
    has_avx512 = len(avx_features) > 0 and avx_features[0] != ''
    
    # Build
    features = "parallel"
    if has_avx512:
        features += ",avx512"
    
    print(f"\nBuilding with features: {features}")
    result = subprocess.run(
        ["cargo", "build", "-p", "morphogen-server", "--bin", "bench_scan", "--release", "--features", features],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print("Build stdout:", result.stdout)
        print("Build stderr:", result.stderr)
        return f"Build failed"
    print("Build complete")
    
    # Run benchmarks at different sizes
    results = []
    for rows in [1024000, 10240000, 39321600]:  # ~1GB, ~10GB, ~37.5GB
        size_gb = rows * 1024 / (1024**3)
        print(f"\n--- Testing {rows:,} rows ({size_gb:.2f} GB) ---")
        
        result = subprocess.run(
            ["./target/release/bench_scan", 
             "--rows", str(rows), 
             "--iterations", "3", 
             "--warmup-iterations", "1",
             "--scan-only",
             "--parallel"],
            capture_output=True,
            text=True,
            timeout=300
        )
        print(result.stdout)
        results.append(result.stdout)
        if result.stderr:
            print("stderr:", result.stderr)
    
    return "\n".join(results)


@app.function(
    image=image,
    gpu="H200",
    memory=65536,
    timeout=900,
)
def run_h200_host_benchmark():
    import subprocess
    import os
    
    os.environ["PATH"] = "/root/.cargo/bin:" + os.environ.get("PATH", "")
    os.chdir("/root/morphogenesis")
    
    # GPU info
    gpu_info = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv"], capture_output=True, text=True)
    print("GPU:\n", gpu_info.stdout)
    
    # CPU info on GPU host
    cpu_info = subprocess.run(["cat", "/proc/cpuinfo"], capture_output=True, text=True)
    cpu_models = [l for l in cpu_info.stdout.split('\n') if 'model name' in l]
    print(f"CPU: {cpu_models[0] if cpu_models else 'unknown'}")
    print(f"CPU cores: {len(cpu_models)}")
    
    mem_info = subprocess.run(["free", "-h"], capture_output=True, text=True)
    print("Memory:\n", mem_info.stdout)
    
    # Check AVX-512 on GPU host
    avx_check = subprocess.run(["sh", "-c", "grep -o 'avx512[a-z]*' /proc/cpuinfo | sort -u"], capture_output=True, text=True)
    avx_features = avx_check.stdout.strip().split('\n') if avx_check.stdout.strip() else []
    print(f"AVX-512 features: {avx_features}")
    
    has_avx512 = len(avx_features) > 0 and avx_features[0] != ''
    features = "parallel"
    if has_avx512:
        features += ",avx512"
    
    print(f"\nBuilding with features: {features}")
    result = subprocess.run(
        ["cargo", "build", "-p", "morphogen-server", "--bin", "bench_scan", "--release", "--features", features],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print("Build stderr:", result.stderr)
        return f"Build failed"
    print("Build complete")
    
    # Run on H200 host CPU (not GPU) - just to see what CPU/memory the H200 hosts have
    for rows in [1024000, 10240000]:
        size_gb = rows * 1024 / (1024**3)
        print(f"\n--- Testing {rows:,} rows ({size_gb:.2f} GB) on H200 host CPU ---")
        
        result = subprocess.run(
            ["./target/release/bench_scan", 
             "--rows", str(rows), 
             "--iterations", "3", 
             "--warmup-iterations", "1",
             "--scan-only",
             "--parallel"],
            capture_output=True,
            text=True,
            timeout=300
        )
        print(result.stdout)
    
    return "H200 host benchmark complete"


@app.local_entrypoint()
def main():
    print("=" * 60)
    print("=== Modal CPU Instance Benchmark ===")
    print("=" * 60)
    result = run_cpu_benchmark.remote()
    print(result)
    
    print("\n" + "=" * 60)
    print("=== Modal H200 Host Benchmark (CPU, not GPU) ===")
    print("=" * 60)
    result = run_h200_host_benchmark.remote()
    print(result)
