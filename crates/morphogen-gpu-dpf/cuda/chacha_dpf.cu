/**
 * ChaCha8-based DPF kernel for GPU PIR.
 *
 * This implements a fused DPF+DB kernel that:
 * 1. Evaluates 3 DPF keys (Cuckoo hashing) in parallel
 * 2. XORs with database pages in a single pass
 * 3. Accumulates results per-tile
 *
 * Compile: nvcc -O3 -arch=sm_80 chacha_dpf.cu -o chacha_dpf_bench
 */

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

// Constants
#define PAGE_SIZE 4096
#define SUBTREE_BITS 10
#define SUBTREE_SIZE (1 << SUBTREE_BITS)  // 1024 pages per tile
#define MAX_DOMAIN_BITS 25
#define THREADS_PER_BLOCK 256
#define CHUNKS_PER_PAGE (PAGE_SIZE / 16)  // 256 chunks of 16 bytes

// ChaCha constants
__constant__ uint32_t CHACHA_CONSTANTS[4] = {
    0x61707865, 0x3320646e, 0x79622d32, 0x6b206574
};

// DPF key structure (stored in constant memory for single query)
struct DpfKeyGpu {
    uint32_t root_seed[4];
    uint8_t root_t;
    uint8_t domain_bits;
    uint32_t cw_seed[MAX_DOMAIN_BITS][4];  // Correction word seeds
    uint8_t cw_t_left[MAX_DOMAIN_BITS];
    uint8_t cw_t_right[MAX_DOMAIN_BITS];
    uint32_t final_cw[4];
};

// Store 3 DPF keys in constant memory
__constant__ DpfKeyGpu c_keys[3];

// Device functions
__device__ __forceinline__ uint32_t rotl32(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

__device__ __forceinline__ void chacha_quarter_round(
    uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d
) {
    a += b; d ^= a; d = rotl32(d, 16);
    c += d; b ^= c; b = rotl32(b, 12);
    a += b; d ^= a; d = rotl32(d, 8);
    c += d; b ^= c; b = rotl32(b, 7);
}

// ChaCha8 block function
__device__ void chacha8_block(
    uint32_t out[16],
    const uint32_t key[8],
    uint32_t counter
) {
    // Initialize state
    uint32_t s[16];
    s[0] = CHACHA_CONSTANTS[0];
    s[1] = CHACHA_CONSTANTS[1];
    s[2] = CHACHA_CONSTANTS[2];
    s[3] = CHACHA_CONSTANTS[3];
    #pragma unroll
    for (int i = 0; i < 8; i++) s[4 + i] = key[i];
    s[12] = counter;
    s[13] = 0;
    s[14] = 0;
    s[15] = 0;

    // Save initial state
    uint32_t init[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) init[i] = s[i];

    // 8 rounds (4 double-rounds)
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        // Column rounds
        chacha_quarter_round(s[0], s[4], s[8],  s[12]);
        chacha_quarter_round(s[1], s[5], s[9],  s[13]);
        chacha_quarter_round(s[2], s[6], s[10], s[14]);
        chacha_quarter_round(s[3], s[7], s[11], s[15]);
        // Diagonal rounds
        chacha_quarter_round(s[0], s[5], s[10], s[15]);
        chacha_quarter_round(s[1], s[6], s[11], s[12]);
        chacha_quarter_round(s[2], s[7], s[8],  s[13]);
        chacha_quarter_round(s[3], s[4], s[9],  s[14]);
    }

    // Add initial state
    #pragma unroll
    for (int i = 0; i < 16; i++) out[i] = s[i] + init[i];
}

// PRG expansion: seed -> (left_seed, right_seed, left_t, right_t)
__device__ void prg_expand(
    const uint32_t seed[4],
    uint32_t left_seed[4],
    uint32_t right_seed[4],
    uint8_t& left_t,
    uint8_t& right_t
) {
    // Double seed for 256-bit key
    uint32_t key[8];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        key[i] = seed[i];
        key[4 + i] = seed[i];
    }

    uint32_t block0[16], block1[16];
    chacha8_block(block0, key, 0);
    chacha8_block(block1, key, 1);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        left_seed[i] = block0[i];
        right_seed[i] = block0[4 + i];
    }
    left_t = block1[0] & 1;
    right_t = block1[1] & 1;
}

// Evaluate DPF at single point (for testing)
__device__ void dpf_eval_point(
    const DpfKeyGpu& key,
    uint32_t index,
    uint32_t out_mask[4]
) {
    uint32_t seed[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) seed[i] = key.root_seed[i];
    uint8_t t = key.root_t;

    for (int level = 0; level < key.domain_bits; level++) {
        uint32_t left_seed[4], right_seed[4];
        uint8_t left_t, right_t;
        prg_expand(seed, left_seed, right_seed, left_t, right_t);

        // Get direction bit (MSB first)
        int bit = (index >> (key.domain_bits - 1 - level)) & 1;

        // Select child
        uint32_t* child_seed = bit ? right_seed : left_seed;
        uint8_t child_t = bit ? right_t : left_t;

        // Apply correction word if t == 1
        if (t == 1) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                child_seed[i] ^= key.cw_seed[level][i];
            }
            child_t ^= bit ? key.cw_t_right[level] : key.cw_t_left[level];
        }

        #pragma unroll
        for (int i = 0; i < 4; i++) seed[i] = child_seed[i];
        t = child_t;
    }

    // Apply final correction word
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        out_mask[i] = seed[i];
        if (t == 1) out_mask[i] ^= key.final_cw[i];
    }
}

// Benchmark kernel: evaluate DPF at all points
__global__ void bench_dpf_eval_kernel(
    uint32_t* __restrict__ outputs,  // [num_pages][4]
    int num_pages,
    int key_idx
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pages) return;

    uint32_t mask[4];
    dpf_eval_point(c_keys[key_idx], idx, mask);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        outputs[idx * 4 + i] = mask[i];
    }
}

// Simple throughput benchmark
__global__ void bench_chacha8_throughput(
    const uint32_t* __restrict__ seeds,  // [n][4]
    uint32_t* __restrict__ outputs,       // [n][4]
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Load seed and double for key
    uint32_t key[8];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        key[i] = seeds[idx * 4 + i];
        key[4 + i] = seeds[idx * 4 + i];
    }

    uint32_t block[16];
    chacha8_block(block, key, 0);

    // Store first 4 words
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        outputs[idx * 4 + i] = block[i];
    }
}

// Host function to benchmark ChaCha8 throughput
void benchmark_chacha8(int log_n, int iterations) {
    int n = 1 << log_n;
    printf("ChaCha8 %dK: ", n/1024);
    fflush(stdout);

    uint32_t *d_seeds, *d_outputs;
    cudaMalloc(&d_seeds, n * 4 * sizeof(uint32_t));
    cudaMalloc(&d_outputs, n * 4 * sizeof(uint32_t));

    uint32_t* h_seeds = new uint32_t[n * 4];
    for (int i = 0; i < n * 4; i++) h_seeds[i] = i;
    cudaMemcpy(d_seeds, h_seeds, n * 4 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    delete[] h_seeds;

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // Warmup
    bench_chacha8_throughput<<<blocks, threads>>>(d_seeds, d_outputs, n);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        bench_chacha8_throughput<<<blocks, threads>>>(d_seeds, d_outputs, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    float avg_ms = elapsed_ms / iterations;
    float throughput = n / (avg_ms / 1000.0f) / 1e6f;

    printf("%.3f ms (%.1f M/s)", avg_ms, throughput);
    if (log_n < 25) {
        float est_ms = avg_ms * (27e6f / n);
        printf(" -> 27M: %.0fms", est_ms);
    }
    printf("\n");

    cudaFree(d_seeds);
    cudaFree(d_outputs);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char** argv) {
    printf("=== ChaCha8 GPU-DPF Benchmark ===\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (sm_%d%d)\n\n", prop.name, prop.major, prop.minor);

    // Quick test with small sizes
    benchmark_chacha8(10, 10);  // 1K - instant
    benchmark_chacha8(14, 10);  // 16K
    benchmark_chacha8(16, 10);  // 64K
    benchmark_chacha8(18, 5);   // 256K
    benchmark_chacha8(20, 3);   // 1M

    return 0;
}
