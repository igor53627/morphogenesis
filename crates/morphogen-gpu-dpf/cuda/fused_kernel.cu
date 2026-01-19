/**
 * Fused DPF+XOR Kernel for GPU PIR.
 *
 * Implements the full PIR scan logic:
 * 1. Expand 3 DPF keys for a subtree (tile) of pages.
 * 2. Load database pages from global memory.
 * 3. Apply masks (AND) and accumulate (XOR) into output buffers.
 *
 * Each thread block processes one subtree (e.g., 1024 pages).
 */

#include <cstdint>
#include <cuda_runtime.h>

// Constants
#define PAGE_SIZE_BYTES 4096
#define SUBTREE_BITS 10
#define SUBTREE_SIZE (1 << SUBTREE_BITS) // 1024 pages per tile
#define MAX_DOMAIN_BITS 25
#define THREADS_PER_BLOCK 256

// Use uint4 (16 bytes) for vectorized memory access
#define VEC_SIZE 16
#define VECS_PER_PAGE (PAGE_SIZE_BYTES / VEC_SIZE)

// ChaCha8 constants
__constant__ uint32_t CHACHA_CONSTANTS[4] = {
    0x61707865, 0x3320646e, 0x79622d32, 0x6b206574
};

struct DpfKeyGpu {
    uint32_t root_seed[4];
    uint8_t root_t;
    uint8_t domain_bits;
    uint32_t cw_seed[MAX_DOMAIN_BITS][4];
    uint8_t cw_t_left[MAX_DOMAIN_BITS];
    uint8_t cw_t_right[MAX_DOMAIN_BITS];
    uint32_t final_cw[4];
};

// Removed __constant__ c_keys[3] to simplify host-side invocation

// Helper: Rotate Left
__device__ __forceinline__ uint32_t rotl32(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

// Helper: ChaCha Quarter Round
__device__ __forceinline__ void chacha_quarter_round(
    uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d
) {
    a += b; d ^= a; d = rotl32(d, 16);
    c += d; b ^= c; b = rotl32(b, 12);
    a += b; d ^= a; d = rotl32(d, 8);
    c += d; b ^= c; b = rotl32(b, 7);
}

// Helper: Full ChaCha8 Block
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

// PRG Expansion
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

// Single Point Eval (used inside fused kernel)
__device__ void dpf_eval_point_inline(
    const DpfKeyGpu& key,
    uint32_t global_idx,
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

// ----------------------------------------------------------------------------
// Fused Kernel
// ----------------------------------------------------------------------------
// Each block processes 'SUBTREE_SIZE' (1024) pages.
// Thread layout: 256 threads per block.
// Each thread processes 4 pages (1024 / 256 = 4) to cover the subtree.
//
// Arguments:
// - db_pages: Pointer to contiguous database pages (uint4*)
// - output_accumulators: Pointer to output buffers (3 x PAGE_SIZE)
// - num_pages: Total number of pages
// ----------------------------------------------------------------------------
extern "C" __global__ void fused_pir_kernel(
    const uint4* __restrict__ db_pages,
    const DpfKeyGpu* __restrict__ keys,
    uint4* __restrict__ out_acc0,
    uint4* __restrict__ out_acc1,
    uint4* __restrict__ out_acc2,
    int num_pages
) {
    // 1. Identify which page this thread is processing
    // We stride by blockDim.x to cover the tile.
    int tile_start_page = blockIdx.x * SUBTREE_SIZE;
    int tid = threadIdx.x;

    // Shared memory accumulators for this block?
    // No, PAGE_SIZE (4KB) * 3 is too big for shared memory if we do it per thread.
    // Instead, we can't reduce across threads easily for different pages.
    // BUT: The goal is 3 output pages TOTAL (accumulated across all DB pages).
    // So we need a reduction.
    
    // Strategy:
    // 1. Each block accumulates its tile's contribution into a per-block buffer? Too much memory.
    // 2. Or: Atomic XOR into global memory? No native atomic XOR for 128-bit.
    // 3. Optimized: Each block computes partial result in registers/shared, then writes to temporary global buffer.
    //    Then a second kernel reduces.
    
    // Simplified for Prototype:
    // Atomic XOR on 32-bit words in global memory is slow but correct.
    // Better: Each grid stride accumulates to a private buffer, then merge.
    
    // For now, let's implement the "Block Reduce" strategy:
    // Each thread loads a page, masks it, and adds to a SHARED memory accumulator for the BLOCK.
    // Wait, 3 output pages = 12KB. Shared memory is ~48KB or 96KB on H100. It FITS!
    
    __shared__ uint4 s_acc0[VECS_PER_PAGE]; // 4KB
    __shared__ uint4 s_acc1[VECS_PER_PAGE]; // 4KB
    __shared__ uint4 s_acc2[VECS_PER_PAGE]; // 4KB

    // Initialize shared accumulators
    for (int i = tid; i < VECS_PER_PAGE; i += blockDim.x) {
        s_acc0[i] = make_uint4(0, 0, 0, 0);
        s_acc1[i] = make_uint4(0, 0, 0, 0);
        s_acc2[i] = make_uint4(0, 0, 0, 0);
    }
    __syncthreads();

    // Loop over pages in this tile assigned to this thread
    for (int i = tid; i < SUBTREE_SIZE; i += blockDim.x) {
        int global_page_idx = tile_start_page + i;
        if (global_page_idx >= num_pages) break;

        // Generate Masks for this page (3 keys)
        uint32_t mask0[4], mask1[4], mask2[4];
        dpf_eval_point_inline(keys[0], global_page_idx, mask0);
        dpf_eval_point_inline(keys[1], global_page_idx, mask1);
        dpf_eval_point_inline(keys[2], global_page_idx, mask2);
        
        uint4 m0 = make_uint4(mask0[0], mask0[1], mask0[2], mask0[3]);
        uint4 m1 = make_uint4(mask1[0], mask1[1], mask1[2], mask1[3]);
        uint4 m2 = make_uint4(mask2[0], mask2[1], mask2[2], mask2[3]);

        // Load page data (vectorized loop)
        // Cast to size_t to prevent 32-bit overflow for large databases (>16GB)
        const uint4* page_ptr = &db_pages[(size_t)global_page_idx * VECS_PER_PAGE];
        
        for (int v = 0; v < VECS_PER_PAGE; v++) {
            uint4 data = page_ptr[v];
            
            // Mask AND (simulated with bitwise ops)
            // Note: uint4 doesn't support & operator directly in CUDA C++, need helper
            uint4 masked0, masked1, masked2;
            
            masked0.x = data.x & m0.x; masked0.y = data.y & m0.y; masked0.z = data.z & m0.z; masked0.w = data.w & m0.w;
            masked1.x = data.x & m1.x; masked1.y = data.y & m1.y; masked1.z = data.z & m1.z; masked1.w = data.w & m1.w;
            masked2.x = data.x & m2.x; masked2.y = data.y & m2.y; masked2.z = data.z & m2.z; masked2.w = data.w & m2.w;

            // Atomic XOR into shared memory
            // CUDA supports atomicXor for 32-bit.
            atomicXor(&s_acc0[v].x, masked0.x); atomicXor(&s_acc0[v].y, masked0.y); atomicXor(&s_acc0[v].z, masked0.z); atomicXor(&s_acc0[v].w, masked0.w);
            atomicXor(&s_acc1[v].x, masked1.x); atomicXor(&s_acc1[v].y, masked1.y); atomicXor(&s_acc1[v].z, masked1.z); atomicXor(&s_acc1[v].w, masked1.w);
            atomicXor(&s_acc2[v].x, masked2.x); atomicXor(&s_acc2[v].y, masked2.y); atomicXor(&s_acc2[v].z, masked2.z); atomicXor(&s_acc2[v].w, masked2.w);
        }
    }
    __syncthreads();

    // Write final shared accumulator to global output (using Atomic XOR)
    // NOTE: For multi-block reduction, global atomics are needed.
    for (int i = tid; i < VECS_PER_PAGE; i += blockDim.x) {
        uint4 val0 = s_acc0[i];
        uint4 val1 = s_acc1[i];
        uint4 val2 = s_acc2[i];

        if (val0.x | val0.y | val0.z | val0.w) {
            atomicXor(&out_acc0[i].x, val0.x); atomicXor(&out_acc0[i].y, val0.y); atomicXor(&out_acc0[i].z, val0.z); atomicXor(&out_acc0[i].w, val0.w);
        }
        if (val1.x | val1.y | val1.z | val1.w) {
            atomicXor(&out_acc1[i].x, val1.x); atomicXor(&out_acc1[i].y, val1.y); atomicXor(&out_acc1[i].z, val1.z); atomicXor(&out_acc1[i].w, val1.w);
        }
        if (val2.x | val2.y | val2.z | val2.w) {
            atomicXor(&out_acc2[i].x, val2.x); atomicXor(&out_acc2[i].y, val2.y); atomicXor(&out_acc2[i].z, val2.z); atomicXor(&out_acc2[i].w, val2.w);
        }
    }
}
