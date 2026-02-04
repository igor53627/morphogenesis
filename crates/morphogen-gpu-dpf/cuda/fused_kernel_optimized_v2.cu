/**
 * Optimized v2 - Uses global memory accumulators instead of shared memory
 * to avoid overflow on large batches.
 */

#include <cstdint>
#include <cuda_runtime.h>

#define PAGE_SIZE_BYTES 4096
#define MAX_DOMAIN_BITS 25
#define SUBTREE_BITS 11
#define SUBTREE_SIZE 2048
#define VECS_PER_PAGE 256
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK 256
#define VECS_PER_THREAD 8

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

template<int BATCH>
__device__ void fused_v2_tmpl(const uint4* db, const DpfKeyGpu* keys, uint4* out, uint4*, int np, int) {
    int tid = threadIdx.x, wid = tid / WARP_SIZE, lane = tid % WARP_SIZE;
    int tile = blockIdx.x * SUBTREE_SIZE;
    
    __shared__ DpfSeed seeds[BATCH * 3];
    if (tid < BATCH * 3) {
        int sl = (int)keys[tid].domain_bits - SUBTREE_BITS; if (sl < 0) sl = 0;
        dpf_eval_prefix(keys[tid], tile, sl, seeds[tid]);
    }
    __syncthreads();

    // Each warp handles all pages for one query at a time
    int ppw = SUBTREE_SIZE / WARPS_PER_BLOCK;
    
    for (int q = 0; q < BATCH * 3; q++) {
        uint4 acc[VECS_PER_THREAD];
        for (int v = 0; v < VECS_PER_THREAD; v++) acc[v] = make_uint4(0,0,0,0);
        
        int ws = tile + wid * ppw;
        int we = min(ws + ppw, tile + SUBTREE_SIZE);
        we = min(we, np);
        
        for (int pg = ws; pg < we; pg++) {
            uint32_t m[4]; int sl = (int)keys[q].domain_bits - SUBTREE_BITS; if (sl < 0) sl = 0;
            dpf_eval_suffix(keys[q], pg, sl, seeds[q], m);
            uint4 mk = make_uint4(m[0], m[1], m[2], m[3]);
            const uint4* ptr = db + (unsigned long long)pg * VECS_PER_PAGE;
            for (int v = 0; v < VECS_PER_THREAD; v++) {
                uint4 d = ptr[v * WARP_SIZE + lane];
                acc[v].x ^= (d.x & mk.x); acc[v].y ^= (d.y & mk.y);
                acc[v].z ^= (d.z & mk.z); acc[v].w ^= (d.w & mk.w);
            }
        }
        
        // Warp reduction
        for (int off = 16; off > 0; off /= 2) {
            for (int v = 0; v < VECS_PER_THREAD; v++) {
                acc[v].x ^= __shfl_xor_sync(0xFFFFFFFF, acc[v].x, off);
                acc[v].y ^= __shfl_xor_sync(0xFFFFFFFF, acc[v].y, off);
                acc[v].z ^= __shfl_xor_sync(0xFFFFFFFF, acc[v].z, off);
                acc[v].w ^= __shfl_xor_sync(0xFFFFFFFF, acc[v].w, off);
            }
        }
        
        if (lane == 0) {
            int qo = q * VECS_PER_PAGE;
            for (int v = 0; v < VECS_PER_THREAD; v++) {
                int ix = qo + wid * VECS_PER_THREAD + v;
                atomicXor(&out[ix].x, acc[v].x); atomicXor(&out[ix].y, acc[v].y);
                atomicXor(&out[ix].z, acc[v].z); atomicXor(&out[ix].w, acc[v].w);
            }
        }
    }
}

extern "C" __global__ void fused_batch_pir_kernel_v2_1(const uint4* d, const DpfKeyGpu* k, uint4* o, uint4* v, int n, int b) { fused_v2_tmpl<1>(d,k,o,v,n,b); }
extern "C" __global__ void fused_batch_pir_kernel_v2_2(const uint4* d, const DpfKeyGpu* k, uint4* o, uint4* v, int n, int b) { fused_v2_tmpl<2>(d,k,o,v,n,b); }
extern "C" __global__ void fused_batch_pir_kernel_v2_4(const uint4* d, const DpfKeyGpu* k, uint4* o, uint4* v, int n, int b) { fused_v2_tmpl<4>(d,k,o,v,n,b); }
extern "C" __global__ void fused_batch_pir_kernel_v2_8(const uint4* d, const DpfKeyGpu* k, uint4* o, uint4* v, int n, int b) { fused_v2_tmpl<8>(d,k,o,v,n,b); }
extern "C" __global__ void fused_batch_pir_kernel_v2_16(const uint4* d, const DpfKeyGpu* k, uint4* o, uint4* v, int n, int b) { fused_v2_tmpl<16>(d,k,o,v,n,b); }
