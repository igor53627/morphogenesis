# Morphogenesis PIR - Project Kanban

## [DONE]

### Performance Optimization (Jan 18, 2026)
- [x] 8-row unroll optimization (28.5 GB/s single-threaded)
- [x] Parallel chunk processing with rayon (383 GB/s)
- [x] AVX-512 vectorized scan kernel
- [x] `DpfKey::eval_range_masks()` API for batch evaluation
- [x] `--parallel` flag for benchmark
- [x] Send/Sync impl for AlignedMatrix

### Real DPF Implementation (Jan 18, 2026)
- [x] AES-based PRF evaluation using VAES intrinsics
- [x] `AesDpfKey` struct with 128-bit key and correction word
- [x] Vectorized `eval_range_masks()` using `_mm512_aesenc_epi128`
- [x] 2-party DPF key generation (`generate_pair`)
- [x] Performance: 393 GB/s (vs 383 GB/s with DummyDpfKey)

### Core Infrastructure
- [x] AlignedMatrix (64-byte aligned allocation)
- [x] ChunkedMatrix (1GB chunk management)
- [x] DeltaBuffer (append-only update log)
- [x] EpochSnapshot structure
- [x] GlobalState with arc-swap
- [x] Profiling infrastructure

### Benchmarking
- [x] bench_scan binary with configurable rows/iterations
- [x] Warmup iterations for page faults
- [x] Flamegraph generation guide

### Core Protocol (Jan 18, 2026)
- [x] Client query generation (3 DPF keys per request)
- [x] Server response aggregation (XOR payloads)
- [x] Parallel Cuckoo addressing (3 simultaneous queries)
- [x] PIR2 fixture reader (102k entries from Sepolia)
- [x] E2E test: full client-server flow

### Network Layer (Jan 18, 2026)
- [x] HTTP query endpoint (POST /query with 3 DPF keys)
- [x] Health endpoint (GET /health)
- [x] Epoch metadata endpoint (GET /epoch)
- [x] Hex serialization for keys/payloads
- [x] WebSocket epoch streaming (GET /ws/epoch - push EpochMetadata)
- [x] WebSocket query endpoint (GET /ws/query - DPF keys in, responses out)

### Epoch Management (Jan 18, 2026)
- [x] Phase 0-6: Core implementation (dirty_chunks, COW merge, atomic switch, EpochHandle, background worker)
- [x] Phase 7-13: Post-impl review fixes (atomic drain, merge mutex, skip empty, spawn_blocking, error logging, Vec<bool>, chunk validation)

Key types: GlobalState, EpochManager, EpochHandle, MergeError

### Epoch Management - Oracle Review #2 (Jan 18, 2026)
Critical fixes for production readiness:

- [x] Phase 14: Restore drained entries on merge error (prevent data loss)
- [x] Phase 15: Remove/hide EpochSnapshot.delta (clarify single source of truth)
- [x] Phase 16: Replace panics with Result in DeltaBuffer::push
- [x] Phase 17: Validate row_size_bytes > 0 at EpochManager construction
- [x] Phase 18: Handle lock poisoning gracefully (return error, don't panic)

---

## [IN PROGRESS]

(none)

---

## [TODO]

### Epoch Management - Oracle Review #3 (Jan 18, 2026)
Critical correctness and production hardening:

- [x] Phase 19: Scan consistency fix (CRITICAL)
  - Race: scan reads GlobalState then pending independently
  - During try_advance: pending drained -> new snapshot stored
  - Concurrent scan sees old snapshot + empty pending = missing merged deltas
  - Fix: Double-check epoch_id loop in scan_consistent() and scan_consistent_parallel()

- [x] Phase 20: Stop swallowing lock poison errors
  - `unwrap_or_default()` in scan_delta, dirty_chunks, dirty_chunks_vec
  - Converts corruption into silent data loss (empty pending)
  - Fix: Added try_dirty_chunks, try_dirty_chunks_vec, try_scan_delta
  - Kept infallible versions as wrappers with expect() for convenience

- [x] Phase 21: Validate row bounds on push
  - Out-of-bounds row_idx only detected at merge time
  - Causes repeated merge failures (livelock-ish)
  - Fix: Added UpdateError enum and EpochManager::submit_update() with row bounds check
  - Added EpochManager::num_rows() helper

- [x] Phase 22: Early error in dirty_chunks for OOB
  - Currently silently ignores out-of-range chunks
  - But merge loop errors for same condition (inconsistent)
  - Fix: Changed try_dirty_chunks to propagate row_offset errors with `?`

- [x] Phase 23: Remove remaining expect()/panic paths
  - `build_next_snapshot().expect(...)` used in MorphogenServer::merge_epoch
  - Fix: Renamed to try_merge_epoch returning Result<u64, MergeError>
  - Kept infallible dirty_chunks/dirty_chunks_vec for test convenience only

### Epoch Management - Oracle Review #4 (Jan 18, 2026)
Critical correctness and production hardening:

- [ ] Phase 24: Pending epoch marker for scan linearizability (CRITICAL)
  - Current scan_consistent() epoch retry doesn't fully prevent races
  - Window exists where pending is drained but matrix epoch unchanged
  - Can cause missing updates or double-application
  - Fix: Add pending_epoch to DeltaBuffer, validate both epochs match in scan

- [x] Phase 25: Row/chunk alignment invariant (CRITICAL - potential UB)
  - Rows straddling chunk boundaries cause merge errors and AVX512 UB
  - Must enforce: chunk_size_bytes % row_size_bytes == 0
  - Must enforce: matrix_size_bytes % row_size_bytes == 0
  - Fix: Added ConfigError enum, ServerConfig::validate(), MorphogenServer::new returns Result

- [x] Phase 26: Backoff in scan_consistent() retry loop
  - Currently spins indefinitely under frequent merges
  - Fix: Added TooManyRetries error, scan_consistent_with_max_retries API
  - Backoff: spin_loop for first 10 attempts, yield_now after
  - Default max retries: 1000

- [ ] Phase 27: Max pending buffer size limit
  - Unbounded pending growth causes latency spikes / memory blowup
  - Fix: Add max pending length or byte budget, reject updates when full

### Core Protocol
- [ ] UBT Merkle proof generation



### Delta-PIR
- [ ] Integrate delta scan with main matrix scan
- [ ] Pending buffer for live updates
- [ ] Epoch ID in query response metadata

### Network Layer
- [ ] Query batch endpoint
- [ ] TLS configuration

### Helios Integration
- [ ] Payload format: `[AccountData | UBT_Proof]`
- [ ] Block header verification
- [ ] Epoch-to-block mapping

### Testing
- [ ] Correctness tests with known data patterns
- [ ] Multi-threaded stress tests
- [ ] Epoch transition under load
- [ ] Memory leak detection

### Production Hardening
- [ ] Error handling and recovery
- [ ] Logging and observability
- [ ] Configuration management
- [ ] Graceful shutdown

---

## [BACKLOG]

### Future Optimizations
- [ ] NUMA-aware memory allocation
- [ ] Huge pages (2MB/1GB)
- [ ] io_uring for async I/O
- [ ] Custom memory allocator

### Multi-Node Sharding
- [ ] Shard coordinator
- [ ] Cross-node query routing
- [ ] Result aggregation layer
- [ ] Failure recovery

### GPU-Accelerated PIR (Research)
- [ ] Port scan kernel to CUDA for HBM access
- [ ] H200: 4.8 TB/s bandwidth (vs 393 GB/s CPU) - potential 12x speedup
- [ ] B200: 8.0 TB/s bandwidth - potential 20x speedup
- [ ] Challenge: 75GB shard won't fit in single GPU (141GB H200, but need 2 for 2-party PIR)
- [ ] Explore multi-GPU sharding within single node

---

## Performance Targets

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Scan throughput | 140 GB/s | 393 GB/s | DONE (2.8x) |
| Privacy-Only latency (22GB) | <600ms | ~66ms | DONE (9.1x) |
| Trustless latency (175GB) | <600ms | ~439ms | DONE (1.4x) |
| Cuckoo load factor | >80% | 85% | DONE |
| Concurrent clients (Privacy-Only) | 1 | ~9 | DONE |
| Delta overhead | <0.5ms | TBD | PENDING |
| Memory overhead | <2x | 1.18x | DONE |

---

## References

- [Protocol Spec](morphogenesis_protocol.md) - PRD v3.2
- [Scientific Paper](morphogenesis_paper.md) - Epoch-Based Delta-PIR
- [Engineering Design](../morphogenesis_EDD.md) - Implementation details
- [Performance Report](PERFORMANCE.md) - Optimization findings
