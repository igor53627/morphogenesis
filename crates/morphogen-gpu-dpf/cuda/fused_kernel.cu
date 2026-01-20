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
// Fused Batch Kernel
// ----------------------------------------------------------------------------
extern "C" __global__ void fused_batch_pir_kernel(
    const uint4* __restrict__ db_pages,
    const DpfKeyGpu* __restrict__ keys, // [batch_size * 3]
    uint4* __restrict__ out_accumulators, // [batch_size * 3 * VECS_PER_PAGE]
    int num_pages,
    int batch_size
) {
    extern __shared__ uint4 s_mem[];
    // Layout: s_mem[query_idx * 3 + key_idx][vec_idx]
    // Stride: 3 * VECS_PER_PAGE
    // Total size: batch_size * 3 * VECS_PER_PAGE * 16 bytes
    
    int accum_stride = 3 * VECS_PER_PAGE;
    int total_vecs = batch_size * accum_stride;

    // Initialize shared accumulators
    for (int i = threadIdx.x; i < total_vecs; i += blockDim.x) {
        s_mem[i] = make_uint4(0, 0, 0, 0);
    }
    __syncthreads();

    // Tile processing
    int tile_start_page = blockIdx.x * SUBTREE_SIZE;
    
    for (int i = threadIdx.x; i < SUBTREE_SIZE; i += blockDim.x) {
        int global_page_idx = tile_start_page + i;
        if (global_page_idx >= num_pages) break;

        // Compute masks for all queries in batch for this page
        // Limit batch size to avoid register spill. 
        // 16 queries * 3 keys = 48 masks. 48*4*4 = 768 bytes.
        // This is high register pressure. Compiler might spill to local mem (L1).
        // This is fine as L1 is fast.
        
        // We process the batch in a loop if it's large? 
        // No, we rely on the loop below.
        
        // Optimization: Compute masks on demand or precompute?
        // If we precompute, we store in registers.
        // If we don't, we re-eval DPF for every vector v (BAD).
        // So we MUST precompute.
        
        // Dynamic allocation of masks array in registers is not possible in C.
        // We assume MAX_BATCH <= 16 for unrolling? 
        // Or we just loop and let compiler handle local mem spilling.
        
        const uint4* page_ptr = &db_pages[(size_t)global_page_idx * VECS_PER_PAGE];

        // To reduce register pressure, we can invert loops:
        // Loop v (vectors)
        //   Load data
        //   Loop q (queries)
        //     Compute mask? No, mask is constant for all v.
        
        // Hybrid:
        // Loop q (queries)
        //   Compute mask
        //   Loop v (vectors)
        //     atomicXor(s_acc, data & mask)
        // This re-loads `data` 3*Batch times! Bad for memory bandwidth.
        
        // We MUST load `data` once.
        // We MUST compute `mask` once.
        
        // Let's rely on local memory caching.
        // Max 16 queries.
        
        uint4 local_masks[16 * 3]; 
        int effective_batch = (batch_size > 16) ? 16 : batch_size; // Clamp for safety
        
        for (int q = 0; q < effective_batch; q++) {
            for (int k = 0; k < 3; k++) {
                uint32_t raw_mask[4];
                dpf_eval_point_inline(keys[q * 3 + k], global_page_idx, raw_mask);
                local_masks[q * 3 + k] = make_uint4(raw_mask[0], raw_mask[1], raw_mask[2], raw_mask[3]);
            }
        }

        // Apply masks
        for (int v = 0; v < VECS_PER_PAGE; v++) {
            uint4 data = page_ptr[v];
            
            for (int q = 0; q < effective_batch; q++) {
                for (int k = 0; k < 3; k++) {
                    uint4 m = local_masks[q * 3 + k];
                    uint4 masked;
                    masked.x = data.x & m.x; masked.y = data.y & m.y; 
                    masked.z = data.z & m.z; masked.w = data.w & m.w;
                    
                    // Atomic add to shared memory
                    int acc_idx = q * accum_stride + k * VECS_PER_PAGE + v;
                    atomicXor(&s_mem[acc_idx].x, masked.x);
                    atomicXor(&s_mem[acc_idx].y, masked.y);
                    atomicXor(&s_mem[acc_idx].z, masked.z);
                    atomicXor(&s_mem[acc_idx].w, masked.w);
                }
            }
        }
    }
    __syncthreads();

    // Write back to global
    for (int i = threadIdx.x; i < total_vecs; i += blockDim.x) {
        uint4 val = s_mem[i];
        if (val.x | val.y | val.z | val.w) {
            atomicXor(&out_accumulators[i].x, val.x);
            atomicXor(&out_accumulators[i].y, val.y);
            atomicXor(&out_accumulators[i].z, val.z);
            atomicXor(&out_accumulators[i].w, val.w);
        }
    }
}
