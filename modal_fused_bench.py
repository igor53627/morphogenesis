"""
Benchmark Fused DPF+XOR Kernel on H200/B200 using a 108GB dataset from Modal Volume.
"""
import modal

app = modal.App("fused-dpf-full-bench")
volume = modal.Volume.from_name("morphogen-data")

# Embed updated CUDA source with file loading logic
CUDA_SOURCE = r'''
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

#define PAGE_SIZE_BYTES 4096
#define SUBTREE_BITS 10
#define SUBTREE_SIZE (1 << SUBTREE_BITS)
#define MAX_DOMAIN_BITS 25
#define THREADS_PER_BLOCK 256
#define VECS_PER_PAGE (PAGE_SIZE_BYTES / 16)

__constant__ uint32_t CHACHA_CONSTANTS[4] = {0x61707865, 0x3320646e, 0x79622d32, 0x6b206574};

struct DpfKeyGpu {
    uint32_t root_seed[4];
    uint8_t root_t;
    uint8_t domain_bits;
    uint32_t cw_seed[MAX_DOMAIN_BITS][4];
    uint8_t cw_t_left[MAX_DOMAIN_BITS];
    uint8_t cw_t_right[MAX_DOMAIN_BITS];
    uint32_t final_cw[4];
};

__constant__ DpfKeyGpu c_keys[3];

__device__ __forceinline__ uint32_t rotl32(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }

__device__ __forceinline__ void chacha_qr(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
    a += b; d ^= a; d = rotl32(d, 16);
    c += d; b ^= c; b = rotl32(b, 12);
    a += b; d ^= a; d = rotl32(d, 8);
    c += d; b ^= c; b = rotl32(b, 7);
}

__device__ void chacha8_block(uint32_t out[16], const uint32_t key[8], uint32_t counter) {
    uint32_t s[16] = {CHACHA_CONSTANTS[0], CHACHA_CONSTANTS[1], CHACHA_CONSTANTS[2], CHACHA_CONSTANTS[3],
                      key[0], key[1], key[2], key[3], key[4], key[5], key[6], key[7],
                      counter, 0, 0, 0};
    uint32_t init[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) init[i] = s[i];
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        chacha_qr(s[0], s[4], s[8],  s[12]); chacha_qr(s[1], s[5], s[9],  s[13]);
        chacha_qr(s[2], s[6], s[10], s[14]); chacha_qr(s[3], s[7], s[11], s[15]);
        chacha_qr(s[0], s[5], s[10], s[15]); chacha_qr(s[1], s[6], s[11], s[12]);
        chacha_qr(s[2], s[7], s[8],  s[13]); chacha_qr(s[3], s[4], s[9],  s[14]);
    }
    #pragma unroll
    for (int i = 0; i < 16; i++) out[i] = s[i] + init[i];
}

__device__ void prg_expand(const uint32_t seed[4], uint32_t left_seed[4], uint32_t right_seed[4], uint8_t& left_t, uint8_t& right_t) {
    uint32_t key[8] = {seed[0], seed[1], seed[2], seed[3], seed[0], seed[1], seed[2], seed[3]};
    uint32_t b0[16], b1[16];
    chacha8_block(b0, key, 0); chacha8_block(b1, key, 1);
    #pragma unroll
    for (int i = 0; i < 4; i++) { left_seed[i] = b0[i]; right_seed[i] = b0[4 + i]; }
    left_t = b1[0] & 1; right_t = b1[1] & 1;
}

__device__ void dpf_eval_point_inline(const DpfKeyGpu& key, uint32_t global_idx, uint32_t out_mask[4]) {
    uint32_t seed[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) seed[i] = key.root_seed[i];
    uint8_t t = key.root_t;
    for (int level = 0; level < key.domain_bits; level++) {
        uint32_t left_seed[4], right_seed[4];
        uint8_t left_t, right_t;
        prg_expand(seed, left_seed, right_seed, left_t, right_t);
        int bit = (global_idx >> (key.domain_bits - 1 - level)) & 1;
        uint32_t* child_seed = bit ? right_seed : left_seed;
        uint8_t child_t = bit ? right_t : left_t;
        if (t == 1) {
            #pragma unroll
            for (int i = 0; i < 4; i++) child_seed[i] ^= key.cw_seed[level][i];
            child_t ^= bit ? key.cw_t_right[level] : key.cw_t_left[level];
        }
        #pragma unroll
        for (int i = 0; i < 4; i++) seed[i] = child_seed[i];
        t = child_t;
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        out_mask[i] = seed[i];
        if (t == 1) out_mask[i] ^= key.final_cw[i];
    }
}

extern "C" __global__ void fused_pir_kernel(const uint4* __restrict__ db_pages, uint4* __restrict__ out_acc0, uint4* __restrict__ out_acc1, uint4* __restrict__ out_acc2, int num_pages) {
    int tile_start_page = blockIdx.x * SUBTREE_SIZE;
    int tid = threadIdx.x;
    
    // Debug print for first thread
    if (blockIdx.x == 0 && tid == 0) {
        printf("Kernel running: num_pages=%d, grid=%d\n", num_pages, gridDim.x);
    }

    __shared__ uint4 s_acc0[VECS_PER_PAGE]; __shared__ uint4 s_acc1[VECS_PER_PAGE]; __shared__ uint4 s_acc2[VECS_PER_PAGE];
    for (int i = tid; i < VECS_PER_PAGE; i += blockDim.x) { s_acc0[i] = s_acc1[i] = s_acc2[i] = make_uint4(0,0,0,0); }
    __syncthreads();
    for (int i = tid; i < SUBTREE_SIZE; i += blockDim.x) {
        int global_page_idx = tile_start_page + i;
        if (global_page_idx >= num_pages) break;
        uint32_t mask0[4], mask1[4], mask2[4];
        dpf_eval_point_inline(c_keys[0], global_page_idx, mask0);
        dpf_eval_point_inline(c_keys[1], global_page_idx, mask1);
        dpf_eval_point_inline(c_keys[2], global_page_idx, mask2);
        uint4 m0 = make_uint4(mask0[0], mask0[1], mask0[2], mask0[3]);
        uint4 m1 = make_uint4(mask1[0], mask1[1], mask1[2], mask1[3]);
        uint4 m2 = make_uint4(mask2[0], mask2[1], mask2[2], mask2[3]);
        // Cast to size_t to prevent 32-bit overflow for >16GB databases
        const uint4* page_ptr = &db_pages[(size_t)global_page_idx * VECS_PER_PAGE];
        for (int v = 0; v < VECS_PER_PAGE; v++) {
            uint4 data = page_ptr[v];
            atomicXor(&s_acc0[v].x, data.x & m0.x); atomicXor(&s_acc0[v].y, data.y & m0.y); atomicXor(&s_acc0[v].z, data.z & m0.z); atomicXor(&s_acc0[v].w, data.w & m0.w);
            atomicXor(&s_acc1[v].x, data.x & m1.x); atomicXor(&s_acc1[v].y, data.y & m1.y); atomicXor(&s_acc1[v].z, data.z & m1.z); atomicXor(&s_acc1[v].w, data.w & m1.w);
            atomicXor(&s_acc2[v].x, data.x & m2.x); atomicXor(&s_acc2[v].y, data.y & m2.y); atomicXor(&s_acc2[v].z, data.z & m2.z); atomicXor(&s_acc2[v].w, data.w & m2.w);
        }
    }
    __syncthreads();
    for (int i = tid; i < VECS_PER_PAGE; i += blockDim.x) {
        if (s_acc0[i].x|s_acc0[i].y|s_acc0[i].z|s_acc0[i].w) { atomicXor(&out_acc0[i].x, s_acc0[i].x); atomicXor(&out_acc0[i].y, s_acc0[i].y); atomicXor(&out_acc0[i].z, s_acc0[i].z); atomicXor(&out_acc0[i].w, s_acc0[i].w); }
        if (s_acc1[i].x|s_acc1[i].y|s_acc1[i].z|s_acc1[i].w) { atomicXor(&out_acc1[i].x, s_acc1[i].x); atomicXor(&out_acc1[i].y, s_acc1[i].y); atomicXor(&out_acc1[i].z, s_acc1[i].z); atomicXor(&out_acc1[i].w, s_acc1[i].w); }
        if (s_acc2[i].x|s_acc2[i].y|s_acc2[i].z|s_acc2[i].w) { atomicXor(&out_acc2[i].x, s_acc2[i].x); atomicXor(&out_acc2[i].y, s_acc2[i].y); atomicXor(&out_acc2[i].z, s_acc2[i].z); atomicXor(&out_acc2[i].w, s_acc2[i].w); }
    }
}

int main(int argc, char** argv) {
    const char* db_path = "/data/db.bin";
    if (argc > 1) db_path = argv[1];

    struct stat st;
    if (stat(db_path, &st) != 0) { printf("Error: Could not stat %s\n", db_path); return 1; }
    size_t db_size = st.st_size;
    int num_pages = db_size / PAGE_SIZE_BYTES;
    printf("Loading %d pages (%.2f GB) from %s...\n", num_pages, db_size / 1e9, db_path);

    uint4 *d_db, *d_acc0, *d_acc1, *d_acc2;
    cudaError_t malloc_err = cudaMalloc(&d_db, db_size);
    if (malloc_err != cudaSuccess) {
        printf("cudaMalloc failed for DB (%zu bytes): %s\n", db_size, cudaGetErrorString(malloc_err));
        return 1;
    }
    printf("Allocated DB at %p\n", d_db);
    cudaMalloc(&d_acc0, 4096); cudaMalloc(&d_acc1, 4096); cudaMalloc(&d_acc2, 4096);

    int fd = open(db_path, O_RDONLY);
    size_t chunk_size = 1024 * 1024 * 512; // 512MB chunks for loading
    uint8_t* h_buf = (uint8_t*)malloc(chunk_size);
    for (size_t off = 0; off < db_size; off += chunk_size) {
        size_t to_read = (db_size - off < chunk_size) ? db_size - off : chunk_size;
        if (read(fd, h_buf, to_read) != (ssize_t)to_read) {
            printf("\nError: Failed to read from %s at offset %zu\n", db_path, off);
            return 1;
        }
        cudaMemcpy((uint8_t*)d_db + off, h_buf, to_read, cudaMemcpyHostToDevice);
        printf("Loaded %.2f / %.2f GB\r", (off + to_read) / 1e9, db_size / 1e9); fflush(stdout);
    }
    printf("\nLoad complete.\n");
    free(h_buf); close(fd);

    DpfKeyGpu h_keys[3]; memset(h_keys, 0, sizeof(h_keys));
    for(int i=0; i<3; i++) h_keys[i].domain_bits = 25;
    cudaMemcpyToSymbol(c_keys, h_keys, sizeof(h_keys));

    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
    printf("Benchmarking full mainnet scan (%d pages)...\n", num_pages);
    
    cudaEventRecord(start);
    fused_pir_kernel<<<(num_pages + SUBTREE_SIZE - 1) / SUBTREE_SIZE, THREADS_PER_BLOCK>>>(d_db, d_acc0, d_acc1, d_acc2, num_pages);
    
    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    cudaEventRecord(stop); 
    err = cudaEventSynchronize(stop);
    if (err != cudaSuccess) {
        printf("Event sync failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    float ms; cudaEventElapsedTime(&ms, start, stop);
    printf(">>> Mainnet Scan Time: %.2f ms (%.2f GB/s)\n", ms, db_size / (ms / 1000.0f) / 1e9f);

    cudaFree(d_db); cudaFree(d_acc0); cudaFree(d_acc1); cudaFree(d_acc2);
    return 0;
}
'''

cuda_image = modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")

@app.function(image=cuda_image, gpu="H200", timeout=1200, volumes={"/data": volume})
def bench_h200():
    return run_bench("H200", "90")

@app.function(image=cuda_image, gpu="B200", timeout=1200, volumes={"/data": volume})
def bench_b200():
    # Fallback to sm_90 (Hopper) for B200 as CUDA 12.4 doesn't support sm_100 yet
    return run_bench("B200", "90")

def run_bench(gpu_name: str, sm: str) -> dict:
    import subprocess
    with open("/tmp/fused.cu", "w") as f: f.write(CUDA_SOURCE)
    subprocess.run(["nvcc", "-O3", f"-arch=sm_{sm}", "/tmp/fused.cu", "-o", "/tmp/bench"], check=True)
    result = subprocess.run(["/tmp/bench"], capture_output=True, text=True)
    return {"gpu": gpu_name, "output": result.stdout}

@app.local_entrypoint()
def main():
    print("---" + " H200 ---")
    try:
        r = bench_h200.remote()
        print(r.get("output"))
    except Exception as e:
        print(f"H200 Error: {e}")
        
    print("\n---" + " B200 ---")
    try:
        r = bench_b200.remote()
        print(r.get("output"))
    except Exception as e:
        print(f"B200 Error: {e}")