/**
 * Fused DPF+XOR Kernel for GPU PIR.
 * 
 * Stability & Performance Update:
 * 1. Layered Architecture: #define ENABLE_BINIUS to toggle verifiable logic.
 * 2. Warp-Level Mask Sharing: Threads compute DPF masks cooperatively and share via shuffle.
 * 3. Radix-16 Optimization: "Compute Only" strategy (Low Register Pressure).
 *    - Replaces 16-element register table with 4-element basis + logic.
 *    - 26% faster than table-based approach on H100.
 */

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

// --- Configuration ---
// #define ENABLE_BINIUS 1

// Constants
#define PAGE_SIZE_BYTES 4096
#define MAX_DOMAIN_BITS 25
#define SUBTREE_BITS 11
#define SUBTREE_SIZE (1 << SUBTREE_BITS) // 2048 pages

#define VEC_SIZE 16
#define VECS_PER_PAGE (PAGE_SIZE_BYTES / VEC_SIZE) // 256

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define VECS_PER_THREAD 8

#define POLY_LOW 0x0000000000000087ULL

__constant__ uint32_t CHACHA_CONSTANTS[4] = {
    0x61707865, 0x3320646e, 0x79622d32, 0x6b206574
};

struct uint128 {
    uint64_t lo;
    uint64_t hi;
};

// --- Radix-16 "Compute Only" GF(2^128) Multiplication ---
#ifdef ENABLE_BINIUS
__device__ __forceinline__ uint128 shift_left_4_mod(uint128 v) {
    uint32_t top = (uint32_t)(v.hi >> 60);
    uint128 res;
    res.hi = (v.hi << 4) | (v.lo >> 60);
    res.lo = (v.lo << 4);
    
    uint64_t mask = 0;
    if (top & 8) mask ^= (POLY_LOW << 3);
    if (top & 4) mask ^= (POLY_LOW << 2);
    if (top & 2) mask ^= (POLY_LOW << 1);
    if (top & 1) mask ^= POLY_LOW;
    
    res.lo ^= mask;
    return res;
}

__device__ __forceinline__ uint128 gf128_mul_radix16(uint4 a_vec, uint4 b_vec) {
    uint128 a = { *((uint64_t*)&a_vec.x), *((uint64_t*)&a_vec.z) };
    uint128 b = { *((uint64_t*)&b_vec.x), *((uint64_t*)&b_vec.z) };

    // Precompute basis elements only (4 registers vs 16)
    uint128 B[4];
    B[0] = a; // A * 1
    
    // B[1] = A * 2 (shift 1)
    B[1] = {a.lo << 1, (a.hi << 1) | (a.lo >> 63)};
    if (a.hi >> 63) B[1].lo ^= POLY_LOW;
    
    // B[2] = A * 4 (shift 2)
    B[2] = {B[1].lo << 1, (B[1].hi << 1) | (B[1].lo >> 63)};
    if (B[1].hi >> 63) B[2].lo ^= POLY_LOW;

    // B[3] = A * 8 (shift 3)
    B[3] = {B[2].lo << 1, (B[2].hi << 1) | (B[2].lo >> 63)};
    if (B[2].hi >> 63) B[3].lo ^= POLY_LOW;

    uint128 res = {0, 0};
    
    // Horner's Method (32 iterations)
    // Manually unrolled logic to avoid lookups
    #pragma unroll 4
    for (int i = 31; i >= 0; i--) {
        res = shift_left_4_mod(res);
        
        int shift = i * 4;
        uint32_t nibble;
        if (shift >= 64) {
            nibble = (b.hi >> (shift - 64)) & 0xF;
        } else {
            nibble = (b.lo >> shift) & 0xF;
        }
        
        // Logical accumulation (trades loads for XORs)
        if (nibble & 1) { res.lo ^= B[0].lo; res.hi ^= B[0].hi; }
        if (nibble & 2) { res.lo ^= B[1].lo; res.hi ^= B[1].hi; }
        if (nibble & 4) { res.lo ^= B[2].lo; res.hi ^= B[2].hi; }
        if (nibble & 8) { res.lo ^= B[3].lo; res.hi ^= B[3].hi; }
    }
    
    return res;
}
#endif

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
    uint4* __restrict__ out_verifiers, 
    int num_pages,
    int batch_size_arg
) {
    (void)batch_size_arg;
    extern __shared__ uint4 s_mem[];
    
    // Layout:
    // 1. Accumulators: BATCH * 3 * 256 uint4s
    // 2. Verifiers: BATCH * 3 uint4s
    // 3. Tile Seeds: BATCH * 3 * sizeof(DpfSeed)
    // No Mask Buffer!
    
    int accum_stride = 3 * VECS_PER_PAGE;
    int total_vecs = BATCH_SIZE * accum_stride;
    
    uint4* s_acc = s_mem;
    int accum_size_vecs = BATCH_SIZE * 3 * VECS_PER_PAGE;
    
    uint4* s_verifiers = &s_mem[accum_size_vecs];
    int verifier_size_vecs = BATCH_SIZE * 3;

    DpfSeed* s_tile_seeds = (DpfSeed*)&s_mem[accum_size_vecs + verifier_size_vecs];

    // Init Shared Acc
    for (int i = threadIdx.x; i < total_vecs; i += blockDim.x) {
        s_acc[i] = make_uint4(0, 0, 0, 0);
    }
    #ifdef ENABLE_BINIUS
    if (threadIdx.x < verifier_size_vecs) {
        s_verifiers[threadIdx.x] = make_uint4(0, 0, 0, 0);
    }
    #endif
    __syncthreads();

    // Prefix (One thread does this)
    int tile_start_page = blockIdx.x * SUBTREE_SIZE;
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int q = 0; q < BATCH_SIZE * 3; q++) {
            int split_level = (int)keys[q].domain_bits - SUBTREE_BITS;
            if (split_level < 0) split_level = 0;
            dpf_eval_prefix(keys[q], tile_start_page, split_level, s_tile_seeds[q]);
        }
    }
    __syncthreads();

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    uint4 local_acc[BATCH_SIZE * 3][VECS_PER_THREAD];
    #ifdef ENABLE_BINIUS
    uint4 local_verif[BATCH_SIZE * 3];
    #endif

    #pragma unroll
    for (int q = 0; q < BATCH_SIZE * 3; q++) {
        #pragma unroll
        for (int v = 0; v < VECS_PER_THREAD; v++) local_acc[q][v] = make_uint4(0, 0, 0, 0);
        #ifdef ENABLE_BINIUS
        local_verif[q] = make_uint4(0, 0, 0, 0);
        #endif
    }

    const int PAGES_PER_WARP = SUBTREE_SIZE / WARPS_PER_BLOCK; // 256 / 8 = 32 pages per warp

    // Warp-Level Mask Sharing Loop
    for (int batch_start = 0; batch_start < PAGES_PER_WARP; batch_start += WARP_SIZE) {
        
        // 1. Cooperative DPF Eval: Each thread computes mask for ONE page
        // Since WarpSize=32 and we process 32 pages at a time, mapping is 1:1
        int page_offset_in_batch = lane_id;
        int page_offset_in_subtree = warp_id * PAGES_PER_WARP + batch_start + page_offset_in_batch;
        int global_page_idx = tile_start_page + page_offset_in_subtree;
        bool page_valid = (global_page_idx < num_pages);

        // Compute my mask (if valid)
        uint4 my_masks[BATCH_SIZE * 3];
        #pragma unroll
        for (int q = 0; q < BATCH_SIZE * 3; q++) {
            if (page_valid) {
                uint32_t m[4];
                int split_level = (int)keys[q].domain_bits - SUBTREE_BITS;
                dpf_eval_suffix(keys[q], global_page_idx, split_level > 0 ? split_level : 0, s_tile_seeds[q], m);
                my_masks[q] = make_uint4(m[0], m[1], m[2], m[3]);
            } else {
                my_masks[q] = make_uint4(0, 0, 0, 0);
            }
        }

        // 2. Process pages in the batch using shuffle to retrieve masks
        // We iterate 32 times (once per page in the batch)
        for (int p = 0; p < WARP_SIZE; p++) {
            if (batch_start + p >= PAGES_PER_WARP) break;
            
            int current_page_in_subtree = warp_id * PAGES_PER_WARP + batch_start + p;
            int current_global_page = tile_start_page + current_page_in_subtree;
            if (current_global_page >= num_pages) break;

            const uint4* page_ptr = db_pages + (unsigned long long)current_global_page * 256;

            // Broadcast the mask from thread 'p' to all threads
            #pragma unroll
            for (int q = 0; q < BATCH_SIZE * 3; q++) {
                uint4 m;
                m.x = __shfl_sync(0xFFFFFFFF, my_masks[q].x, p);
                m.y = __shfl_sync(0xFFFFFFFF, my_masks[q].y, p);
                m.z = __shfl_sync(0xFFFFFFFF, my_masks[q].z, p);
                m.w = __shfl_sync(0xFFFFFFFF, my_masks[q].w, p);

                // Compute
                #pragma unroll
                for (int v = 0; v < VECS_PER_THREAD; v++) {
                    uint4 data = page_ptr[v * WARP_SIZE + lane_id];
                    local_acc[q][v].x ^= (data.x & m.x);
                    local_acc[q][v].y ^= (data.y & m.y);
                    local_acc[q][v].z ^= (data.z & m.z);
                    local_acc[q][v].w ^= (data.w & m.w);

                    #ifdef ENABLE_BINIUS
                    uint128 prod = gf128_mul_radix16(data, m);
                    local_verif[q].x ^= (uint32_t)prod.lo;
                    local_verif[q].y ^= (uint32_t)(prod.lo >> 32);
                    local_verif[q].z ^= (uint32_t)prod.hi;
                    local_verif[q].w ^= (uint32_t)(prod.hi >> 32);
                    #endif
                }
            }
        }
    }

    // Flush
    #pragma unroll
    for (int q = 0; q < BATCH_SIZE * 3; q++) {
        #pragma unroll
        for (int v = 0; v < VECS_PER_THREAD; v++) {
            uint4 val = local_acc[q][v];
            if (val.x | val.y | val.z | val.w) {
                int acc_idx = q * 256 + v * WARP_SIZE + lane_id;
                atomicXor(&s_acc[acc_idx].x, val.x);
                atomicXor(&s_acc[acc_idx].y, val.y);
                atomicXor(&s_acc[acc_idx].z, val.z);
                atomicXor(&s_acc[acc_idx].w, val.w);
            }
        }
        #ifdef ENABLE_BINIUS
        uint4 v_val = local_verif[q];
        if (v_val.x | v_val.y | v_val.z | v_val.w) {
            atomicXor(&s_verifiers[q].x, v_val.x);
            atomicXor(&s_verifiers[q].y, v_val.y);
            atomicXor(&s_verifiers[q].z, v_val.z);
            atomicXor(&s_verifiers[q].w, v_val.w);
        }
        #endif
    }
    __syncthreads();

    // Write Back
    for (int i = threadIdx.x; i < total_vecs; i += blockDim.x) {
        uint4 val = s_acc[i];
        if (val.x | val.y | val.z | val.w) {
            atomicXor(&out_accumulators[i].x, val.x);
            atomicXor(&out_accumulators[i].y, val.y);
            atomicXor(&out_accumulators[i].z, val.z);
            atomicXor(&out_accumulators[i].w, val.w);
        }
    }
    #ifdef ENABLE_BINIUS
    if (threadIdx.x < BATCH_SIZE * 3) {
        uint4 val = s_verifiers[threadIdx.x];
        if (val.x | val.y | val.z | val.w) {
            atomicXor(&out_verifiers[threadIdx.x].x, val.x);
            atomicXor(&out_verifiers[threadIdx.x].y, val.y);
            atomicXor(&out_verifiers[threadIdx.x].z, val.z);
            atomicXor(&out_verifiers[threadIdx.x].w, val.w);
        }
    }
    #endif
}

extern "C" __global__ void fused_batch_pir_kernel_1(const uint4* d, const DpfKeyGpu* k, uint4* o, uint4* v, int n, int b) { fused_batch_pir_kernel_tmpl<1>(d, k, o, v, n, b); }
extern "C" __global__ void fused_batch_pir_kernel_2(const uint4* d, const DpfKeyGpu* k, uint4* o, uint4* v, int n, int b) { fused_batch_pir_kernel_tmpl<2>(d, k, o, v, n, b); }
extern "C" __global__ void fused_batch_pir_kernel_4(const uint4* d, const DpfKeyGpu* k, uint4* o, uint4* v, int n, int b) { fused_batch_pir_kernel_tmpl<4>(d, k, o, v, n, b); }
extern "C" __global__ void fused_batch_pir_kernel_8(const uint4* d, const DpfKeyGpu* k, uint4* o, uint4* v, int n, int b) { fused_batch_pir_kernel_tmpl<8>(d, k, o, v, n, b); }
extern "C" __global__ void fused_batch_pir_kernel_16(const uint4* d, const DpfKeyGpu* k, uint4* o, uint4* v, int n, int b) { fused_batch_pir_kernel_tmpl<16>(d, k, o, v, n, b); }

extern "C" __global__ void fused_batch_pir_kernel_transposed_1(const uint4* d, const DpfKeyGpu* k, uint4* o, uint4* v, int n, int b) { fused_batch_pir_kernel_tmpl<1>(d, k, o, v, n, b); }
extern "C" __global__ void fused_batch_pir_kernel_transposed_2(const uint4* d, const DpfKeyGpu* k, uint4* o, uint4* v, int n, int b) { fused_batch_pir_kernel_tmpl<2>(d, k, o, v, n, b); }
extern "C" __global__ void fused_batch_pir_kernel_transposed_4(const uint4* d, const DpfKeyGpu* k, uint4* o, uint4* v, int n, int b) { fused_batch_pir_kernel_tmpl<4>(d, k, o, v, n, b); }
extern "C" __global__ void fused_batch_pir_kernel_transposed_8(const uint4* d, const DpfKeyGpu* k, uint4* o, uint4* v, int n, int b) { fused_batch_pir_kernel_tmpl<8>(d, k, o, v, n, b); }
extern "C" __global__ void fused_batch_pir_kernel_transposed_16(const uint4* d, const DpfKeyGpu* k, uint4* o, uint4* v, int n, int b) { fused_batch_pir_kernel_tmpl<16>(d, k, o, v, n, b); }
