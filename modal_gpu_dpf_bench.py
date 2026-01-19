"""
Benchmark Facebook GPU-DPF on Modal GPU instances.

Tests proper 2-server DPF (BGI 2014) on various GPU types to measure
performance for page-level PIR.

Run with: modal run modal_gpu_dpf_bench.py
"""
import modal

app = modal.App("gpu-dpf-bench")

gpu_dpf_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential", "cmake", "libssl-dev", "g++", "ninja-build")
    .pip_install("torch==2.2.0", "numpy<2", "pybind11", "ninja")  # Use older torch/numpy for CUDA 12.4 compat
    .run_commands(
        # Clone Facebook GPU-DPF
        "git clone https://github.com/facebookresearch/GPU-DPF.git /root/GPU-DPF",
    )
    .run_commands(
        # Build GPU-DPF with g++
        "cd /root/GPU-DPF && CXX=g++ python setup.py build_ext --inplace",
        gpu="T4",  # Need GPU to compile CUDA
    )
)


@app.function(
    image=gpu_dpf_image,
    gpu="T4",
    timeout=600,
)
def bench_gpu_dpf_t4():
    return run_gpu_dpf_benchmark("T4")


@app.function(
    image=gpu_dpf_image,
    gpu="A10G",
    timeout=600,
)
def bench_gpu_dpf_a10g():
    return run_gpu_dpf_benchmark("A10G")


@app.function(
    image=gpu_dpf_image,
    gpu="A100",
    timeout=600,
)
def bench_gpu_dpf_a100():
    return run_gpu_dpf_benchmark("A100")


@app.function(
    image=gpu_dpf_image,
    gpu="H100",
    timeout=600,
)
def bench_gpu_dpf_h100():
    return run_gpu_dpf_benchmark("H100")


def run_gpu_dpf_benchmark(gpu_name: str) -> dict:
    import subprocess
    import time
    import sys
    import os
    
    # GPU info
    gpu_info = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv"],
        capture_output=True, text=True
    )
    print(f"GPU Info:\n{gpu_info.stdout}")
    
    # Add GPU-DPF to path
    sys.path.insert(0, "/root/GPU-DPF")
    os.chdir("/root/GPU-DPF")
    
    try:
        import torch
        # Import the DPF class from dpf.py
        from dpf import DPF
        
        print(f"\n=== GPU-DPF Benchmark on {gpu_name} ===\n")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        
        results = {}
        
        # Test various domain sizes relevant to page-level PIR
        # Mainnet: 368M accounts → 433M Cuckoo slots → 27M pages
        test_sizes = [
            (14, "16K"),      # 2^14 = 16,384 (minimum for GPU-DPF)
            (16, "64K"),      # 2^16 = 65,536
            (18, "256K"),     # 2^18 = 262,144  
            (20, "1M"),       # 2^20 = 1,048,576
        ]
        
        for log_n, label in test_sizes:
            n = 1 << log_n
            print(f"\n--- Domain size: {label} ({n:,} entries) ---")
            
            try:
                dpf_instance = DPF(prf=DPF.PRF_AES128)
                
                # Create a dummy table for evaluation
                table = torch.randint(0, 256, (n, 16), dtype=torch.int32)
                dpf_instance.eval_init(table)
                
                # Generate keys
                target = n // 2
                k1, k2 = dpf_instance.gen(target, n)
                
                # Warmup
                for _ in range(3):
                    dpf_instance.eval_gpu([k1])
                torch.cuda.synchronize()
                
                # Benchmark with CUDA events for accurate GPU timing
                num_iters = 20
                
                # Method 1: Wall clock (includes transfer overhead)
                start = time.perf_counter()
                for _ in range(num_iters):
                    dpf_instance.eval_gpu([k1])
                torch.cuda.synchronize()
                elapsed_wall = time.perf_counter() - start
                
                # Method 2: CUDA events (GPU kernel time only)
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                gpu_times = []
                for _ in range(num_iters):
                    start_event.record()
                    dpf_instance.eval_gpu([k1])
                    end_event.record()
                    torch.cuda.synchronize()
                    gpu_times.append(start_event.elapsed_time(end_event))
                
                avg_wall_ms = (elapsed_wall / num_iters) * 1000
                avg_gpu_ms = sum(gpu_times) / len(gpu_times)
                transfer_overhead_ms = avg_wall_ms - avg_gpu_ms
                
                throughput_wall = n / (elapsed_wall / num_iters) / 1e6
                throughput_gpu = n / (avg_gpu_ms / 1000) / 1e6
                
                print(f"  Wall clock:     {avg_wall_ms:.2f} ms ({throughput_wall:.1f} M/s)")
                print(f"  GPU kernel:     {avg_gpu_ms:.2f} ms ({throughput_gpu:.1f} M/s)")
                print(f"  Transfer overhead: {transfer_overhead_ms:.2f} ms ({transfer_overhead_ms/avg_wall_ms*100:.1f}%)")
                
                results[label] = {
                    "domain_size": n,
                    "wall_ms": avg_wall_ms,
                    "gpu_ms": avg_gpu_ms,
                    "transfer_ms": transfer_overhead_ms,
                    "throughput_wall_meps": throughput_wall,
                    "throughput_gpu_meps": throughput_gpu,
                }
                
            except Exception as e:
                import traceback
                print(f"  Error: {e}")
                print(traceback.format_exc())
                results[label] = {"error": str(e)}
        
        # Extrapolate to mainnet (27M pages)
        if "1M" in results and "gpu_ms" in results["1M"]:
            mainnet_pages = 27_000_000
            ratio = mainnet_pages / (1 << 20)
            est_wall = results["1M"]["wall_ms"] * ratio
            est_gpu = results["1M"]["gpu_ms"] * ratio
            print(f"\n=== Mainnet Extrapolation (27M pages) ===")
            print(f"Wall clock estimated: {est_wall:.0f} ms")
            print(f"GPU kernel estimated: {est_gpu:.0f} ms")
            results["mainnet_wall_ms"] = est_wall
            results["mainnet_gpu_ms"] = est_gpu
        
        return {"gpu": gpu_name, "results": results}
        
    except Exception as e:
        import traceback
        return {"gpu": gpu_name, "error": str(e), "traceback": traceback.format_exc()}


@app.local_entrypoint()
def main():
    print("=" * 60)
    print("=== Facebook GPU-DPF Benchmark on Modal ===")
    print("=" * 60)
    print("\nTesting proper 2-server DPF (BGI 2014) for page-level PIR")
    print("Target: 27M pages for Ethereum mainnet (368M accounts)\n")
    
    # Run on available GPUs (start with cheaper ones)
    print("\n--- T4 GPU ---")
    try:
        result = bench_gpu_dpf_t4.remote()
        print_results(result)
    except Exception as e:
        print(f"T4 failed: {e}")
    
    print("\n--- A10G GPU ---")
    try:
        result = bench_gpu_dpf_a10g.remote()
        print_results(result)
    except Exception as e:
        print(f"A10G failed: {e}")
    
    print("\n--- A100 GPU ---")
    try:
        result = bench_gpu_dpf_a100.remote()
        print_results(result)
    except Exception as e:
        print(f"A100 failed: {e}")
        
    print("\n--- H100 GPU ---")
    try:
        result = bench_gpu_dpf_h100.remote()
        print_results(result)
    except Exception as e:
        print(f"H100 failed: {e}")


def print_results(result: dict):
    if "error" in result:
        print(f"Error: {result['error']}")
        if "traceback" in result:
            print(result["traceback"])
        return
    
    print(f"\nGPU: {result['gpu']}")
    print("-" * 50)
    print(f"{'Size':<8} {'Wall(ms)':<10} {'GPU(ms)':<10} {'Transfer%':<10}")
    print("-" * 50)
    
    for label, data in result.get("results", {}).items():
        if label.startswith("mainnet"):
            continue
        if "error" in data:
            print(f"{label}: Error - {data['error']}")
        else:
            xfer_pct = data['transfer_ms'] / data['wall_ms'] * 100 if data['wall_ms'] > 0 else 0
            print(f"{label:<8} {data['wall_ms']:<10.2f} {data['gpu_ms']:<10.2f} {xfer_pct:<10.1f}%")
    
    results = result.get("results", {})
    if "mainnet_wall_ms" in results and "mainnet_gpu_ms" in results:
        print(f"\n>>> Mainnet (27M): Wall={results['mainnet_wall_ms']:.0f}ms, GPU={results['mainnet_gpu_ms']:.0f}ms")
