/**
 * Optimized v3 - Hybrid Shared/Global Memory with Adaptive Sizing
 * 
 * Strategy:
 * 1. Use SMALL fixed shared memory buffer (never overflows)
 * 2. Stream pages through shared memory, flush to global
 * 3. Process queries in parallel (not sequentially)
 */

#include <cstdint>
#include <cuda_runtime.h>

#define PAGE_SIZE_BYTES 4096
#define MAX_DOMAIN_BITS 25
#define SUBTREE_BITS 11
#define SUBTREE_SIZE 2048
#define VECS_PER_PAGE 256
#define WARP_SIZE 32
#define THREADS_PER_BLOCK 256

// Fixed small shared memory - never overflows regardless of batch size
// Only used for: seeds + small scratch buffer
#define MAX_SHARED_QUERIES 4  // Process at most 4 queries at a time in shared memory
#define SHARED_ACCUM_VECTORS 64  // 64 uint4s = 1KB per query

__constant__ uint32_t CHACHA_CONSTANTS[4] = {0x61707865, 0x3320646e, 0x79622d32, 0x6b206574};

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

struct DpfSeed { uint32_t seed[4]; uint8_t t; };

__device__ __forceinline__ uint32_t rotl32(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }

__device__ __forceinline__ void chacha_qr(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
    a += b; d ^= a; d = rotl32(d, 16);
    c += d; b ^= c; b = rotl32(b, 12);
    a += b; d ^= a; d = rotl32(d, 8);
    c += d; b ^= c; b = rotl32(b, 7);
}

__device__ void prg_expand_fast(const uint32_t seed[4], uint32_t ls[4], uint32_t rs[4], uint8_t& lt, uint8_t& rt) {
    uint32_t key[8];
    for (int i = 0; i < 4; i++) { key[i] = seed[i]; key[4+i] = seed[i]; }
    uint32_t s[16] = {0x61707865, 0x3320646e, 0x79622d32, 0x6b206574, key[0], key[1], key[2], key[3], 
                      key[4], key[5], key[6], key[7], 0, 0, 0, 0};
    uint32_t i0=s[0], i1=s[1], i2=s[2], i3=s[3], i4=s[4], i5=s[5], i6=s[6], i7=s[7];
    for (int r = 0; r < 4; r++) {
        chacha_qr(s[0], s[4], s[8], s[12]); chacha_qr(s[1], s[5], s[9], s[13]);
        chacha_qr(s[2], s[6], s[10], s[14]); chacha_qr(s[3], s[7], s[11], s[15]);
        chacha_qr(s[0], s[5], s[10], s[15]); chacha_qr(s[1], s[6], s[11], s[12]);
        chacha_qr(s[2], s[7], s[8], s[13]); chacha_qr(s[3], s[4], s[9], s[14]);
    }
    ls[0]=s[0]+i0; ls[1]=s[1]+i1; ls[2]=s[2]+i2; ls[3]=s[3]+i3;
    rs[0]=s[4]+i4; rs[1]=s[5]+i5; rs[2]=s[6]+i6; rs[3]=s[7]+i7;
    uint32_t s2[16] = {0x61707865, 0x3320646e, 0x79622d32, 0x6b206574, key[0], key[1], key[2], key[3],
                       key[4], key[5], key[6], key[7], 1, 0, 0, 0};
    uint32_t j0=s2[0], j1=s2[1];
    for (int r = 0; r < 4; r++) {
        chacha_qr(s2[0], s2[4], s2[8], s2[12]); chacha_qr(s2[1], s2[5], s2[9], s2[13]);
        chacha_qr(s2[2], s2[6], s2[10], s2[14]); chacha_qr(s2[3], s2[7], s2[11], s2[15]);
        chacha_qr(s2[0], s2[5], s2[10], s2[15]); chacha_qr(s2[1], s2[6], s2[11], s2[12]);
        chacha_qr(s2[2], s2[7], s2[8], s2[13]); chacha_qr(s2[3], s2[4], s2[9], s2[14]);
    }
    lt = (s2[0]+j0) & 1; rt = (s2[1]+j1) & 1;
}

__device__ void dpf_eval_prefix(const DpfKeyGpu& key, uint32_t gidx, int tl, DpfSeed& out) {
    uint32_t sd[4]; for (int i = 0; i < 4; i++) sd[i] = key.root_seed[i];
    uint8_t t = key.root_t;
    for (int lv = 0; lv < tl; lv++) {
        uint32_t ls[4], rs[4]; uint8_t lt, rt;
        prg_expand_fast(sd, ls, rs, lt, rt);
        int b = (gidx >> (key.domain_bits - 1 - lv)) & 1;
        uint32_t* cs = b ? rs : ls; uint8_t ct = b ? rt : lt;
        if (t == 1) { for (int i = 0; i < 4; i++) cs[i] ^= key.cw_seed[lv][i]; ct ^= b ? key.cw_t_right[lv] : key.cw_t_left[lv]; }
        for (int i = 0; i < 4; i++) sd[i] = cs[i]; t = ct;
    }
    for (int i = 0; i < 4; i++) out.seed[i] = sd[i]; out.t = t;
}

__device__ void dpf_eval_suffix(const DpfKeyGpu& key, uint32_t gidx, int sl, const DpfSeed& in, uint32_t om[4]) {
    uint32_t sd[4]; for (int i = 0; i < 4; i++) sd[i] = in.seed[i];
    uint8_t t = in.t; int db = key.domain_bits, rm = db - sl;
    for (int lv = 0; lv < rm; lv++) {
        uint32_t ls[4], rs[4]; uint8_t lt, rt;
        prg_expand_fast(sd, ls, rs, lt, rt);
        int al = sl + lv, b = (gidx >> (db - 1 - al)) & 1;
        uint32_t* cs = b ? rs : ls; uint8_t ct = b ? rt : lt;
        if (t == 1) { for (int i = 0; i < 4; i++) cs[i] ^= key.cw_seed[al][i]; ct ^= b ? key.cw_t_right[al] : key.cw_t_left[al]; }
        for (int i = 0; i < 4; i++) sd[i] = cs[i]; t = ct;
    }
    for (int i = 0; i < 4; i++) { om[i] = sd[i]; if (t == 1) om[i] ^= key.final_cw[i]; }
}

/**
 * v3 kernel: Process queries in groups to avoid shared memory overflow
 * - Shared memory: only seeds + small accumulator for MAX_SHARED_QUERIES queries
 * - Stream through pages, flushing to global memory
 */
template<int BATCH>
__device__ void fused_v3_tmpl(const uint4* db, const DpfKeyGpu* keys, uint4* out, uint4*, int np, int) {
    int tid = threadIdx.x, lane = tid % WARP_SIZE;
    int tile = blockIdx.x * SUBTREE_SIZE;
    int tile_end = min(tile + SUBTREE_SIZE, np);
    
    // Shared memory layout:
    // 1. Seeds for all queries: BATCH * 3 * sizeof(DpfSeed)
    // 2. Small accumulator for subset of queries
    extern __shared__ uint8_t s_mem[];
    DpfSeed* s_seeds = (DpfSeed*)s_mem;
    
    // Compute seeds for all queries (needed for all)
    if (tid < BATCH * 3) {
        int sl = (int)keys[tid].domain_bits - SUBTREE_BITS; if (sl < 0) sl = 0;
        dpf_eval_prefix(keys[tid], tile, sl, s_seeds[tid]);
    }
    __syncthreads();
    
    // Process queries in groups to avoid register pressure
    // Each group of warps handles one query subset
    int queries_per_iter = 3;  // Process 3 queries at a time (one Cuckoo set)
    int num_iters = (BATCH * 3 + queries_per_iter - 1) / queries_per_iter;
    
    for (int iter = 0; iter < num_iters; iter++) {
        int q_start = iter * queries_per_iter;
        int q_end = min(q_start + queries_per_iter, BATCH * 3);
        
        // Each thread accumulates for these queries in registers
        // 3 queries * 8 vectors = 24 uint4s = 384 bytes per thread
        uint4 acc[3][8];  // [query][vector]
        for (int q = 0; q < 3; q++)
            for (int v = 0; v < 8; v++)
                acc[q][v] = make_uint4(0,0,0,0);
        
        // All threads process all pages for this query subset
        for (int pg = tile; pg < tile_end; pg++) {
            // Compute masks for this page (all queries in parallel)
            uint4 masks[3];
            for (int qi = 0; qi < q_end - q_start; qi++) {
                int q = q_start + qi;
                uint32_t m[4]; int sl = (int)keys[q].domain_bits - SUBTREE_BITS; if (sl < 0) sl = 0;
                dpf_eval_suffix(keys[q], pg, sl, s_seeds[q], m);
                masks[qi] = make_uint4(m[0], m[1], m[2], m[3]);
            }
            
            const uint4* ptr = db + (unsigned long long)pg * VECS_PER_PAGE;
            for (int qi = 0; qi < q_end - q_start; qi++) {
                for (int v = 0; v < 8; v++) {
                    uint4 d = ptr[v * WARP_SIZE + lane];
                    acc[qi][v].x ^= (d.x & masks[qi].x);
                    acc[qi][v].y ^= (d.y & masks[qi].y);
                    acc[qi][v].z ^= (d.z & masks[qi].z);
                    acc[qi][v].w ^= (d.w & masks[qi].w);
                }
            }
        }
        
        // Warp reduction and write to global
        for (int off = 16; off > 0; off /= 2) {
            for (int qi = 0; qi < 3; qi++) {
                for (int v = 0; v < 8; v++) {
                    acc[qi][v].x ^= __shfl_xor_sync(0xFFFFFFFF, acc[qi][v].x, off);
                    acc[qi][v].y ^= __shfl_xor_sync(0xFFFFFFFF, acc[qi][v].y, off);
                    acc[qi][v].z ^= __shfl_xor_sync(0xFFFFFFFF, acc[qi][v].z, off);
                    acc[qi][v].w ^= __shfl_xor_sync(0xFFFFFFFF, acc[qi][v].w, off);
                }
            }
        }
        
        if (lane == 0) {
            for (int qi = 0; qi < q_end - q_start; qi++) {
                int q = q_start + qi;
                int qo = q * VECS_PER_PAGE;
                int wid = tid / WARP_SIZE;
                for (int v = 0; v < 8; v++) {
                    int ix = qo + wid * 8 + v;
                    atomicXor(&out[ix].x, acc[qi][v].x);
                    atomicXor(&out[ix].y, acc[qi][v].y);
                    atomicXor(&out[ix].z, acc[qi][v].z);
                    atomicXor(&out[ix].w, acc[qi][v].w);
                }
            }
        }
    }
}

extern "C" __global__ void fused_batch_pir_kernel_v3_1(const uint4* d, const DpfKeyGpu* k, uint4* o, uint4* v, int n, int b) { fused_v3_tmpl<1>(d,k,o,v,n,b); }
extern "C" __global__ void fused_batch_pir_kernel_v3_2(const uint4* d, const DpfKeyGpu* k, uint4* o, uint4* v, int n, int b) { fused_v3_tmpl<2>(d,k,o,v,n,b); }
extern "C" __global__ void fused_batch_pir_kernel_v3_4(const uint4* d, const DpfKeyGpu* k, uint4* o, uint4* v, int n, int b) { fused_v3_tmpl<4>(d,k,o,v,n,b); }
extern "C" __global__ void fused_batch_pir_kernel_v3_8(const uint4* d, const DpfKeyGpu* k, uint4* o, uint4* v, int n, int b) { fused_v3_tmpl<8>(d,k,o,v,n,b); }
extern "C" __global__ void fused_batch_pir_kernel_v3_16(const uint4* d, const DpfKeyGpu* k, uint4* o, uint4* v, int n, int b) { fused_v3_tmpl<16>(d,k,o,v,n,b); }
