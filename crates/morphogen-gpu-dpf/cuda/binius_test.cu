#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <chrono>

#define POLY_LOW 0x0000000000000087ULL

// Define constants
#define BLOCK_SIZE 256
#define ITEMS_PER_THREAD 16
#define CHUNK_SIZE (BLOCK_SIZE * ITEMS_PER_THREAD)

struct uint128 {
    uint64_t lo;
    uint64_t hi;
};

void cuda_check(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", file, line, cudaGetErrorString(code));
        exit(1);
    }
}
#define CHECK(x) cuda_check(x, __FILE__, __LINE__)

// --- Baseline XOR Kernel ---
__global__ void kernel_xor_baseline(const uint128* __restrict__ data, const uint128* __restrict__ query, uint128* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    uint128 d = data[idx];
    uint128 q = query[idx];
    uint128 res = { d.lo ^ q.lo, d.hi ^ q.hi };
    
    if (idx == 0) out[0] = res;
    if (res.lo == 0xDEADBEEF) out[0].hi = res.lo;
}

// --- Strategy 1: Shared Memory Lookup Table (Tower Field) ---

__device__ uint8_t mul_gf2_8_slow(uint8_t a, uint8_t b) {
    uint8_t p = 0;
    for(int i=0; i<8; i++) {
        if((b >> i) & 1) p ^= a;
        bool cy = (a >> 7) & 1;
        a = (a << 1);
        if(cy) a ^= 0x1B;
    }
    return p;
}

// Dynamic shared memory
extern __shared__ uint8_t smem_raw[];

__device__ __forceinline__ uint8_t mul_gf2_8_fast(uint8_t (*smem)[256], uint8_t a, uint8_t b) {
    return smem[a][b];
}

__device__ __forceinline__ uint32_t mul_gf2_16(uint8_t (*smem)[256], uint32_t a, uint32_t b) {
    uint8_t al = a & 0xFF; uint8_t ah = a >> 8;
    uint8_t bl = b & 0xFF; uint8_t bh = b >> 8;
    uint8_t ll = mul_gf2_8_fast(smem, al, bl);
    uint8_t hh = mul_gf2_8_fast(smem, ah, bh);
    uint8_t sum = mul_gf2_8_fast(smem, al ^ ah, bl ^ bh);
    return ((uint32_t)(sum ^ ll ^ hh) << 8) | (ll ^ hh);
}

__device__ __forceinline__ uint32_t mul_gf2_32(uint8_t (*smem)[256], uint32_t a, uint32_t b) {
    uint32_t al = a & 0xFFFF; uint32_t ah = a >> 16;
    uint32_t bl = b & 0xFFFF; uint32_t bh = b >> 16;
    uint32_t ll = mul_gf2_16(smem, al, bl);
    uint32_t hh = mul_gf2_16(smem, ah, bh);
    uint32_t sum = mul_gf2_16(smem, al ^ ah, bl ^ bh);
    return ((sum ^ ll ^ hh) << 16) | (ll ^ hh);
}

__device__ __forceinline__ uint64_t mul_gf2_64(uint8_t (*smem)[256], uint64_t a, uint64_t b) {
    uint32_t al = a & 0xFFFFFFFF; uint32_t ah = a >> 32;
    uint32_t bl = b & 0xFFFFFFFF; uint32_t bh = b >> 32;
    uint64_t ll = mul_gf2_32(smem, al, bl);
    uint64_t hh = mul_gf2_32(smem, ah, bh);
    uint64_t sum = mul_gf2_32(smem, al ^ ah, bl ^ bh);
    return ((uint64_t)(sum ^ ll ^ hh) << 32) | (ll ^ hh);
}

__device__ __forceinline__ uint128 gf128_mul_tower(uint8_t (*smem)[256], uint128 a, uint128 b) {
    uint64_t al = a.lo; uint64_t ah = a.hi;
    uint64_t bl = b.lo; uint64_t bh = b.hi;
    uint64_t ll = mul_gf2_64(smem, al, bl);
    uint64_t hh = mul_gf2_64(smem, ah, bh);
    uint64_t sum = mul_gf2_64(smem, al ^ ah, bl ^ bh);
    return { ll ^ hh, (sum ^ ll ^ hh) };
}

// --- Strategy 2: Radix-16 Branchless (GCM Polynomial Basis) ---
__device__ __forceinline__ uint128 shift_left_4_mod(uint128 v) {
    uint32_t top = (uint32_t)(v.hi >> 60);
    uint128 res;
    res.hi = (v.hi << 4) | (v.lo >> 60);
    res.lo = (v.lo << 4);
    uint64_t poly = POLY_LOW;
    uint64_t mask = 0;
    if (top & 8) mask ^= (poly << 3);
    if (top & 4) mask ^= (poly << 2);
    if (top & 2) mask ^= (poly << 1);
    if (top & 1) mask ^= poly;
    res.lo ^= mask;
    return res;
}

__device__ __forceinline__ uint128 gf128_mul_radix16(uint128 a, uint128 b) {
    uint128 T[16];
    T[0] = {0, 0};
    T[1] = a;
    #pragma unroll
    for(int i=2; i<16; i++) {
        if (i%2 == 0) {
            uint128 prev = T[i/2];
            uint128 next = {prev.lo << 1, (prev.hi << 1) | (prev.lo >> 63)};
            if (prev.hi >> 63) next.lo ^= POLY_LOW;
            T[i] = next;
        } else {
            T[i].lo = T[i-1].lo ^ T[1].lo;
            T[i].hi = T[i-1].hi ^ T[1].hi;
        }
    }
    uint128 res = {0, 0};
    #pragma unroll 32
    for (int i = 31; i >= 0; i--) {
        res = shift_left_4_mod(res);
        int shift = i * 4;
        uint32_t nibble = (shift >= 64) ? (b.hi >> (shift - 64)) : (b.lo >> shift);
        nibble &= 0xF;
        res.lo ^= T[nibble].lo;
        res.hi ^= T[nibble].hi;
    }
    return res;
}

// --- Kernels ---
__global__ void kernel_gf128_tower(const uint128* __restrict__ data, const uint128* __restrict__ query, uint128* __restrict__ out, int n) {
    // Initialize SMEM
    uint8_t (*smem)[256] = (uint8_t (*)[256])smem_raw;
    int tid = threadIdx.x;
    for (int i = tid; i < 65536; i += blockDim.x) {
        uint8_t a = i >> 8;
        uint8_t b = i & 0xFF;
        smem[a][b] = mul_gf2_8_slow(a, b);
    }
    __syncthreads();

    int bid = blockIdx.x;
    int base_idx = bid * CHUNK_SIZE;
    uint128 acc = {0, 0};
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int idx = base_idx + tid + i * BLOCK_SIZE;
        if (idx < n) {
            uint128 d = data[idx];
            uint128 q = query[idx];
            uint128 p = gf128_mul_tower(smem, d, q);
            acc.lo ^= p.lo;
            acc.hi ^= p.hi;
        }
    }
    if (tid == 0 && bid == 0) out[0] = acc;
}

__global__ void kernel_gf128_radix16(const uint128* __restrict__ data, const uint128* __restrict__ query, uint128* __restrict__ out, int n) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int base_idx = bid * CHUNK_SIZE;
    uint128 acc = {0, 0};
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int idx = base_idx + tid + i * BLOCK_SIZE;
        if (idx < n) {
            uint128 p = gf128_mul_radix16(data[idx], query[idx]);
            acc.lo ^= p.lo;
            acc.hi ^= p.hi;
        }
    }
    if (tid == 0 && bid == 0) out[0] = acc;
}

extern "C" void run_bench(int log_n) {
    int n = 1 << log_n;
    size_t size = n * sizeof(uint128);
    printf("N = 2^%d (%.2f GB)\n", log_n, (float)size / 1e9);

    uint128 *d_data, *d_query, *d_out;
    CHECK(cudaMalloc(&d_data, size));
    CHECK(cudaMalloc(&d_query, size));
    CHECK(cudaMalloc(&d_out, sizeof(uint128)));
    CHECK(cudaMemset(d_data, 0xAB, size));
    CHECK(cudaMemset(d_query, 0xCD, size));

    int threads = BLOCK_SIZE;
    int blocks = (n + CHUNK_SIZE - 1) / CHUNK_SIZE;
    int iterations = 10;
    
    // 1. XOR
    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<iterations; i++) {
        kernel_xor_baseline<<<n/threads, threads>>>(d_data, d_query, d_out, n);
    }
    CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    double ms_xor = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
    printf("  XOR:      %8.3f ms (%7.2f GB/s)\n", ms_xor, (size * 2.0 / 1e9) / (ms_xor / 1000.0));

    // 2. GF128 Tower (SMEM)
    size_t smem_size = 65536; 
    CHECK(cudaFuncSetAttribute(kernel_gf128_tower, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<iterations; i++) {
        kernel_gf128_tower<<<blocks, threads, smem_size>>>(d_data, d_query, d_out, n);
    }
    CHECK(cudaDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    double ms_tower = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
    printf("  GF-Tower: %8.3f ms (%7.2f GB/s) [%.1fx slow]\n", ms_tower, (size * 2.0 / 1e9) / (ms_tower / 1000.0), ms_tower / ms_xor);

    // 3. GF128 Radix-16
    start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<iterations; i++) {
        kernel_gf128_radix16<<<blocks, threads>>>(d_data, d_query, d_out, n);
    }
    CHECK(cudaDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    double ms_radix = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
    printf("  GF-Radix: %8.3f ms (%7.2f GB/s) [%.1fx slow]\n", ms_radix, (size * 2.0 / 1e9) / (ms_radix / 1000.0), ms_radix / ms_xor);

    CHECK(cudaFree(d_data)); CHECK(cudaFree(d_query)); CHECK(cudaFree(d_out));
}

int main() {
    run_bench(24);
    run_bench(28); // Enable for full memory test
    return 0;
}