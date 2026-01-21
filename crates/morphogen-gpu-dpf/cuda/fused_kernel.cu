/**
 * Fused DPF+XOR Kernel for GPU PIR.
 *
 * Plan K: Subtree Optimization + Parallel DPF + Warp-per-Page
 * 
 * Optimization:
 * 1. Subtree Caching: Evaluate the top of the DPF tree (Prefix) ONCE per tile.
 *    - Amortizes ~60% of DPF compute.
 *    - Example: For 25-bit domain and 10-bit tiles, we skip 15 levels per page.
 * 2. Parallel Suffix: All threads evaluate the bottom 10 levels in parallel.
 * 3. Coalesced Memory: Warp-per-Page access.
 */

#include <cstdint>
#include <cuda_runtime.h>

// Constants
#define PAGE_SIZE_BYTES 4096
#define MAX_DOMAIN_BITS 25
#define SUBTREE_BITS 11
#define SUBTREE_SIZE (1 << SUBTREE_BITS) // 2048 pages

// Use uint4 (16 bytes) for vectorized memory access
#define VEC_SIZE 16
#define VECS_PER_PAGE (PAGE_SIZE_BYTES / VEC_SIZE) // 256

// Warp Constants
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8 // 256 threads / 32
#define VECS_PER_THREAD 8 // 256 / 32

// ChaCha8 constants
__constant__ uint32_t CHACHA_CONSTANTS[4] = {
    0x61707865, 0x3320646e, 0x79622d32, 0x6b206574
};

struct DpfKeyGpu {
    uint32_t root_seed[4];
    uint8_t root_t;
    uint8_t domain_bits;
    uint8_t _pad[2];
    uint32_t cw_seed[MAX_DOMAIN_BITS][4];
    uint8_t cw_t_left[MAX_DOMAIN_BITS];
    uint8_t cw_t_right[MAX_DOMAIN_BITS];
    uint8_t _pad2[2];
    uint32_t final_cw[4];
};

struct DpfSeed {
    uint32_t seed[4];
    uint8_t t;
};

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

__device__ void chacha8_block(
    uint32_t out[16],
    const uint32_t key[8],
    uint32_t counter
) {
    uint32_t s[16];
    s[0] = CHACHA_CONSTANTS[0];
    s[1] = CHACHA_CONSTANTS[1];
    s[2] = CHACHA_CONSTANTS[2];
    s[3] = CHACHA_CONSTANTS[3];
    #pragma unroll
    for (int i = 0; i < 8; i++) s[4 + i] = key[i];
    s[12] = counter; s[13] = 0; s[14] = 0; s[15] = 0;

    uint32_t init[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) init[i] = s[i];

    #pragma unroll
    for (int r = 0; r < 4; r++) {
        chacha_quarter_round(s[0], s[4], s[8],  s[12]);
        chacha_quarter_round(s[1], s[5], s[9],  s[13]);
        chacha_quarter_round(s[2], s[6], s[10], s[14]);
        chacha_quarter_round(s[3], s[7], s[11], s[15]);
        chacha_quarter_round(s[0], s[5], s[10], s[15]);
        chacha_quarter_round(s[1], s[6], s[11], s[12]);
        chacha_quarter_round(s[2], s[7], s[8],  s[13]);
        chacha_quarter_round(s[3], s[4], s[9],  s[14]);
    }

    #pragma unroll
    for (int i = 0; i < 16; i++) out[i] = s[i] + init[i];
}

__device__ void prg_expand(
    const uint32_t seed[4],
    uint32_t left_seed[4],
    uint32_t right_seed[4],
    uint8_t& left_t,
    uint8_t& right_t
) {
    uint32_t key[8];
    #pragma unroll
    for (int i = 0; i < 4; i++) { key[i] = seed[i]; key[4 + i] = seed[i]; }

    uint32_t b0[16], b1[16];
    chacha8_block(b0, key, 0);
    chacha8_block(b1, key, 1);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        left_seed[i] = b0[i];
        right_seed[i] = b0[4 + i];
    }
    left_t = b1[0] & 1;
    right_t = b1[1] & 1;
}

// Evaluate DPF from Root down to `target_level`
__device__ void dpf_eval_prefix(
    const DpfKeyGpu& key,
    uint32_t global_idx,
    int target_level,
    DpfSeed& out_seed
) {
    uint32_t seed[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) seed[i] = key.root_seed[i];
    uint8_t t = key.root_t;

    for (int level = 0; level < target_level; level++) {
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
    for (int i = 0; i < 4; i++) out_seed.seed[i] = seed[i];
    out_seed.t = t;
}

// Evaluate DPF from `start_level` to Leaf
__device__ void dpf_eval_suffix(
    const DpfKeyGpu& key,
    uint32_t global_idx,
    int start_level,
    const DpfSeed& in_seed,
    uint32_t out_mask[4]
) {
    uint32_t seed[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) seed[i] = in_seed.seed[i];
    uint8_t t = in_seed.t;

    for (int level = start_level; level < (int)key.domain_bits; level++) {
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

template<int BATCH_SIZE>
__device__ void fused_batch_pir_kernel_tmpl(
    const uint4* __restrict__ db_pages,
    const DpfKeyGpu* __restrict__ keys,
    uint4* __restrict__ out_accumulators,
    int num_pages,
    int batch_size_arg
) {
    (void)batch_size_arg;
    
    extern __shared__ uint4 s_mem[];
    // Shared Memory Layout:
    // 1. Accumulators: BATCH * 3 * 256 uint4s
    // 2. Tile Seeds (Prefix): BATCH * 3 * sizeof(DpfSeed) [Aligned to uint4]
    // 3. Mask Buffer: 256 * BATCH * 3 uint4s
    
    int accum_stride = 3 * VECS_PER_PAGE;
    int total_vecs = BATCH_SIZE * accum_stride;
    
    // Pointers
    uint4* s_acc = s_mem;
    int accum_size_vecs = BATCH_SIZE * 3 * VECS_PER_PAGE;
    
    // Use raw pointer for seeds to handle DpfSeed struct alignment
    DpfSeed* s_tile_seeds = (DpfSeed*)&s_mem[accum_size_vecs];
    int seed_slots = BATCH_SIZE * 3;
    
    // Align mask buffer start to 16 bytes (sizeof(uint4))
    size_t seeds_end = (size_t)(s_tile_seeds + seed_slots);
    size_t masks_start = (seeds_end + 15) & ~15;
    uint4* s_masks = (uint4*)masks_start;

    // 1. Initialize shared accumulators
    for (int i = threadIdx.x; i < total_vecs; i += blockDim.x) {
        s_acc[i] = make_uint4(0, 0, 0, 0);
    }
    __syncthreads();

    // 2. Compute Prefix (Tile Seeds) - One per Key per Batch
    int tile_start_page = blockIdx.x * SUBTREE_SIZE;
    
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int q = 0; q < BATCH_SIZE * 3; q++) {
            int split_level = keys[q].domain_bits - SUBTREE_BITS;
            if (split_level < 0) split_level = 0;
            dpf_eval_prefix(keys[q], tile_start_page, split_level, s_tile_seeds[q]);
        }
    }
    __syncthreads();

    // 3. Warp-per-Page Logic
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    uint4 local_acc[BATCH_SIZE * 3][VECS_PER_THREAD];
    #pragma unroll
    for (int q = 0; q < BATCH_SIZE * 3; q++) {
        #pragma unroll
        for (int v = 0; v < VECS_PER_THREAD; v++) {
            local_acc[q][v] = make_uint4(0, 0, 0, 0);
        }
    }

    const int PAGES_PER_WARP = SUBTREE_SIZE / WARPS_PER_BLOCK; // 128
    const int WARP_BATCH_SIZE = WARP_SIZE; // 32

    for (int batch_start = 0; batch_start < PAGES_PER_WARP; batch_start += WARP_BATCH_SIZE) {
        // Phase 1: Parallel Suffix DPF Compute
        int page_offset = batch_start + lane_id;
        int global_page_idx = tile_start_page + warp_id * PAGES_PER_WARP + page_offset;
        bool page_valid = (global_page_idx < num_pages);

        int mask_base_idx = threadIdx.x * BATCH_SIZE * 3;

        #pragma unroll
        for (int q = 0; q < BATCH_SIZE * 3; q++) {
            uint32_t raw_mask[4] = {0, 0, 0, 0};
            if (page_valid) {
                int split_level = keys[q].domain_bits - SUBTREE_BITS;
                if (split_level < 0) split_level = 0;
                dpf_eval_suffix(keys[q], global_page_idx, split_level, s_tile_seeds[q], raw_mask);
            }
            s_masks[mask_base_idx + q] = make_uint4(raw_mask[0], raw_mask[1], raw_mask[2], raw_mask[3]);
        }
        
        __syncwarp(); 

        // Phase 2: Memory Access
        for (int p = 0; p < WARP_BATCH_SIZE; p++) {
            int current_page_offset = batch_start + p;
            int current_global_page = tile_start_page + warp_id * PAGES_PER_WARP + current_page_offset;
            
            if (current_global_page >= num_pages) break;

            const uint4* page_ptr = db_pages + (unsigned long long)current_global_page * VECS_PER_PAGE;

            int target_thread_idx = warp_id * WARP_SIZE + p;
            int target_mask_base = target_thread_idx * BATCH_SIZE * 3;

            uint4 local_masks[BATCH_SIZE * 3];
            #pragma unroll
            for (int q = 0; q < BATCH_SIZE * 3; q++) {
                local_masks[q] = s_masks[target_mask_base + q];
            }

            #pragma unroll
            for (int v = 0; v < VECS_PER_THREAD; v++) {
                int vec_idx = v * WARP_SIZE + lane_id;
                uint4 data = page_ptr[vec_idx];

                #pragma unroll
                for (int q = 0; q < BATCH_SIZE * 3; q++) {
                    uint4 m = local_masks[q];
                    local_acc[q][v].x ^= (data.x & m.x);
                    local_acc[q][v].y ^= (data.y & m.y);
                    local_acc[q][v].z ^= (data.z & m.z);
                    local_acc[q][v].w ^= (data.w & m.w);
                }
            }
        }
    }

    // 2c. Flush
    #pragma unroll
    for (int q = 0; q < BATCH_SIZE * 3; q++) {
        #pragma unroll
        for (int v = 0; v < VECS_PER_THREAD; v++) {
            int vec_idx = v * WARP_SIZE + lane_id;
            int acc_idx = q * VECS_PER_PAGE + vec_idx;
            uint4 val = local_acc[q][v];
            
            if (val.x | val.y | val.z | val.w) {
                atomicXor(&s_acc[acc_idx].x, val.x);
                atomicXor(&s_acc[acc_idx].y, val.y);
                atomicXor(&s_acc[acc_idx].z, val.z);
                atomicXor(&s_acc[acc_idx].w, val.w);
            }
        }
    }
    __syncthreads();

    // 3. Write Back
    for (int i = threadIdx.x; i < total_vecs; i += blockDim.x) {
        uint4 val = s_acc[i];
        if (val.x | val.y | val.z | val.w) {
            atomicXor(&out_accumulators[i].x, val.x);
            atomicXor(&out_accumulators[i].y, val.y);
            atomicXor(&out_accumulators[i].z, val.z);
            atomicXor(&out_accumulators[i].w, val.w);
        }
    }
}

extern "C" __global__ void fused_batch_pir_kernel_1(const uint4* d, const DpfKeyGpu* k, uint4* o, int n, int b) { fused_batch_pir_kernel_tmpl<1>(d, k, o, n, b); }
extern "C" __global__ void fused_batch_pir_kernel_2(const uint4* d, const DpfKeyGpu* k, uint4* o, int n, int b) { fused_batch_pir_kernel_tmpl<2>(d, k, o, n, b); }
extern "C" __global__ void fused_batch_pir_kernel_4(const uint4* d, const DpfKeyGpu* k, uint4* o, int n, int b) { fused_batch_pir_kernel_tmpl<4>(d, k, o, n, b); }
extern "C" __global__ void fused_batch_pir_kernel_8(const uint4* d, const DpfKeyGpu* k, uint4* o, int n, int b) { fused_batch_pir_kernel_tmpl<8>(d, k, o, n, b); }
extern "C" __global__ void fused_batch_pir_kernel_16(const uint4* d, const DpfKeyGpu* k, uint4* o, int n, int b) { fused_batch_pir_kernel_tmpl<16>(d, k, o, n, b); }

extern "C" __global__ void fused_batch_pir_kernel_transposed_1(const uint4* d, const DpfKeyGpu* k, uint4* o, int n, int b) { fused_batch_pir_kernel_tmpl<1>(d, k, o, n, b); }
extern "C" __global__ void fused_batch_pir_kernel_transposed_2(const uint4* d, const DpfKeyGpu* k, uint4* o, int n, int b) { fused_batch_pir_kernel_tmpl<2>(d, k, o, n, b); }
extern "C" __global__ void fused_batch_pir_kernel_transposed_4(const uint4* d, const DpfKeyGpu* k, uint4* o, int n, int b) { fused_batch_pir_kernel_tmpl<4>(d, k, o, n, b); }
extern "C" __global__ void fused_batch_pir_kernel_transposed_8(const uint4* d, const DpfKeyGpu* k, uint4* o, int n, int b) { fused_batch_pir_kernel_tmpl<8>(d, k, o, n, b); }
extern "C" __global__ void fused_batch_pir_kernel_transposed_16(const uint4* d, const DpfKeyGpu* k, uint4* o, int n, int b) { fused_batch_pir_kernel_tmpl<16>(d, k, o, n, b); }
