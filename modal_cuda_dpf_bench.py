"""
Compile and benchmark ChaCha8 CUDA kernel on Modal GPU instances.

Run with: modal run modal_cuda_dpf_bench.py
"""
import modal

app = modal.App("cuda-dpf-bench")

# Embed CUDA source directly
CUDA_SOURCE = r'''
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

__device__ __forceinline__ uint32_t rotl32(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

__device__ __forceinline__ void chacha_qr(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
    a += b; d ^= a; d = rotl32(d, 16);
    c += d; b ^= c; b = rotl32(b, 12);
    a += b; d ^= a; d = rotl32(d, 8);
    c += d; b ^= c; b = rotl32(b, 7);
}

__device__ void chacha8_block(uint32_t out[16], const uint32_t key[8], uint32_t counter) {
    uint32_t s[16] = {
        0x61707865, 0x3320646e, 0x79622d32, 0x6b206574,
        key[0], key[1], key[2], key[3], key[4], key[5], key[6], key[7],
        counter, 0, 0, 0
    };
    uint32_t init[16];
    for (int i = 0; i < 16; i++) init[i] = s[i];
    
    for (int r = 0; r < 4; r++) {
        chacha_qr(s[0], s[4], s[8], s[12]);
        chacha_qr(s[1], s[5], s[9], s[13]);
        chacha_qr(s[2], s[6], s[10], s[14]);
        chacha_qr(s[3], s[7], s[11], s[15]);
        chacha_qr(s[0], s[5], s[10], s[15]);
        chacha_qr(s[1], s[6], s[11], s[12]);
        chacha_qr(s[2], s[7], s[8], s[13]);
        chacha_qr(s[3], s[4], s[9], s[14]);
    }
    for (int i = 0; i < 16; i++) out[i] = s[i] + init[i];
}

__global__ void bench_chacha8(const uint32_t* seeds, uint32_t* outputs, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    uint32_t key[8];
    for (int i = 0; i < 4; i++) {
        key[i] = seeds[idx * 4 + i];
        key[4 + i] = seeds[idx * 4 + i];
    }
    
    uint32_t block[16];
    chacha8_block(block, key, 0);
    
    for (int i = 0; i < 4; i++) outputs[idx * 4 + i] = block[i];
}

// Fused kernel: DPF mask + XOR with DB page + accumulate
// Each thread handles one 16-byte chunk of a page
// Block handles one page, accumulates into shared memory
__global__ void bench_fused_dpf_xor(
    const uint32_t* __restrict__ db,     // [num_pages][256][4] = [num_pages][4KB]
    const uint32_t* __restrict__ seeds,  // [num_pages][4]
    uint32_t* __restrict__ acc,          // [256][4] = 4KB accumulator
    int num_pages
) {
    __shared__ uint32_t s_acc[256][4];  // 4KB shared accumulator
    
    int chunk = threadIdx.x;  // 0-255
    int page = blockIdx.x;
    
    // Init shared accumulator
    for (int i = 0; i < 4; i++) s_acc[chunk][i] = 0;
    __syncthreads();
    
    // Process this page
    if (page < num_pages) {
        // Generate DPF mask from seed
        uint32_t key[8];
        for (int i = 0; i < 4; i++) {
            key[i] = seeds[page * 4 + i];
            key[4 + i] = seeds[page * 4 + i];
        }
        uint32_t block[16];
        chacha8_block(block, key, chunk);  // Different mask per chunk
        
        // Load DB chunk (16 bytes = 4 uint32)
        int db_offset = page * 256 * 4 + chunk * 4;
        uint32_t d0 = db[db_offset + 0];
        uint32_t d1 = db[db_offset + 1];
        uint32_t d2 = db[db_offset + 2];
        uint32_t d3 = db[db_offset + 3];
        
        // XOR with mask and accumulate
        s_acc[chunk][0] ^= d0 & block[0];
        s_acc[chunk][1] ^= d1 & block[1];
        s_acc[chunk][2] ^= d2 & block[2];
        s_acc[chunk][3] ^= d3 & block[3];
    }
    __syncthreads();
    
    // Write to global accumulator (atomic XOR)
    if (page == 0) {
        atomicXor(&acc[chunk * 4 + 0], s_acc[chunk][0]);
        atomicXor(&acc[chunk * 4 + 1], s_acc[chunk][1]);
        atomicXor(&acc[chunk * 4 + 2], s_acc[chunk][2]);
        atomicXor(&acc[chunk * 4 + 3], s_acc[chunk][3]);
    }
}

void benchmark(int log_n, int iters) {
    int n = 1 << log_n;
    printf("ChaCha8 %dK: ", n/1024);
    fflush(stdout);

    uint32_t *d_seeds, *d_outputs;
    cudaMalloc(&d_seeds, n * 4 * sizeof(uint32_t));
    cudaMalloc(&d_outputs, n * 4 * sizeof(uint32_t));

    uint32_t* h = new uint32_t[n * 4];
    for (int i = 0; i < n * 4; i++) h[i] = i;
    cudaMemcpy(d_seeds, h, n * 4 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    delete[] h;

    int threads = 256, blocks = (n + 255) / 256;
    bench_chacha8<<<blocks, threads>>>(d_seeds, d_outputs, n);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) bench_chacha8<<<blocks, threads>>>(d_seeds, d_outputs, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float avg = ms / iters;
    float tp = n / (avg / 1000.0f) / 1e6f;
    printf("%.3f ms (%.1f M/s) -> 27M: %.0fms\n", avg, tp, avg * (27e6f / n));

    cudaFree(d_seeds);
    cudaFree(d_outputs);
}

void benchmark_fused(int log_n, int iters) {
    int n = 1 << log_n;
    size_t db_size = (size_t)n * 4096;  // 4KB per page
    printf("Fused %dK pages (%.0f MB): ", n/1024, db_size/1e6);
    fflush(stdout);

    uint32_t *d_db, *d_seeds, *d_acc;
    cudaMalloc(&d_db, db_size);
    cudaMalloc(&d_seeds, n * 4 * sizeof(uint32_t));
    cudaMalloc(&d_acc, 256 * 4 * sizeof(uint32_t));
    cudaMemset(d_db, 0x42, db_size);
    cudaMemset(d_acc, 0, 256 * 4 * sizeof(uint32_t));

    uint32_t* h = new uint32_t[n * 4];
    for (int i = 0; i < n * 4; i++) h[i] = i;
    cudaMemcpy(d_seeds, h, n * 4 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    delete[] h;

    // Warmup
    bench_fused_dpf_xor<<<n, 256>>>(d_db, d_seeds, d_acc, n);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        cudaMemset(d_acc, 0, 256 * 4 * sizeof(uint32_t));
        bench_fused_dpf_xor<<<n, 256>>>(d_db, d_seeds, d_acc, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float avg = ms / iters;
    float bw = db_size / (avg / 1000.0f) / 1e9f;
    printf("%.2f ms (%.1f GB/s)", avg, bw);
    if (log_n < 25) {
        float est = avg * (27e6f / n);
        printf(" -> 27M: %.0fms", est);
    }
    printf("\n");

    cudaFree(d_db);
    cudaFree(d_seeds);
    cudaFree(d_acc);
}

int main() {
    printf("=== ChaCha8 GPU Benchmark ===\n");
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("GPU: %s (sm_%d%d)\n", p.name, p.major, p.minor);
    printf("Memory BW: %.0f GB/s\n\n", 2.0 * p.memoryClockRate * (p.memoryBusWidth / 8) / 1e6);
    
    printf("--- PRG Only ---\n");
    benchmark(14, 10);
    benchmark(16, 10);
    benchmark(18, 5);
    
    printf("\n--- Fused DPF+XOR ---\n");
    benchmark_fused(14, 10);  // 64MB
    benchmark_fused(16, 5);   // 256MB
    benchmark_fused(18, 3);   // 1GB
    
    return 0;
}
'''

cuda_image = modal.Image.from_registry(
    "nvidia/cuda:12.4.0-devel-ubuntu22.04", 
    add_python="3.11"
)


@app.function(image=cuda_image, gpu="T4", timeout=120)
def bench_t4():
    return run_bench("T4", "75")


@app.function(image=cuda_image, gpu="A100", timeout=120)
def bench_a100():
    return run_bench("A100", "80")


@app.function(image=cuda_image, gpu="H100", timeout=120)
def bench_h100():
    return run_bench("H100", "90")


def run_bench(gpu_name: str, sm: str) -> dict:
    import subprocess
    
    with open("/tmp/chacha.cu", "w") as f:
        f.write(CUDA_SOURCE)
    
    result = subprocess.run(
        ["nvcc", "-O3", f"-arch=sm_{sm}", "/tmp/chacha.cu", "-o", "/tmp/bench"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        return {"gpu": gpu_name, "error": result.stderr}
    
    result = subprocess.run(["/tmp/bench"], capture_output=True, text=True)
    return {"gpu": gpu_name, "output": result.stdout}


@app.local_entrypoint()
def main():
    print("=== CUDA ChaCha8 Benchmark ===\n")
    
    print("--- T4 ---")
    r = bench_t4.remote()
    print(r.get("output") or r.get("error"))
    
    print("--- A100 ---")
    r = bench_a100.remote()
    print(r.get("output") or r.get("error"))
    
    print("--- H100 ---")
    r = bench_h100.remote()
    print(r.get("output") or r.get("error"))
