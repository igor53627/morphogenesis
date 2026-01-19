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

### Epoch Management - Oracle Review #2-4 (Jan 18, 2026)
Critical fixes for production readiness:

- [x] Phase 14: Restore drained entries on merge error (prevent data loss)
- [x] Phase 15: Remove/hide EpochSnapshot.delta (clarify single source of truth)
- [x] Phase 16: Replace panics with Result in DeltaBuffer::push
- [x] Phase 17: Validate row_size_bytes > 0 at EpochManager construction
- [x] Phase 18: Handle lock poisoning gracefully (return error, don't panic)
- [x] Phase 19: Scan consistency fix (double-check epoch_id loop)
- [x] Phase 20: Stop swallowing lock poison errors (try_* variants)
- [x] Phase 21: Validate row bounds on push (UpdateError, submit_update())
- [x] Phase 22: Early error in dirty_chunks for OOB
- [x] Phase 23: Remove remaining expect()/panic paths (try_merge_epoch)
- [x] Phase 24: Pending epoch marker for scan linearizability
  - Added pending_epoch: AtomicU64 to DeltaBuffer
  - Added snapshot_with_epoch() and drain_for_epoch()
  - Updated scan_consistent to validate both matrix epoch AND pending_epoch match
- [x] Phase 25: Row/chunk alignment invariant (ConfigError, ServerConfig::validate())
- [x] Phase 26: Backoff in scan_consistent() retry loop (TooManyRetries, max_retries)
- [x] Phase 27: Max pending buffer size limit (DeltaError::BufferFull)
- [x] Phase 28: Fix snapshot_with_epoch() lock ordering (acquire lock first)
- [x] Phase 29: Reset pending_epoch on merge failure (restore_for_epoch)
- [x] Phase 30: Initialize pending_epoch from global epoch (new_with_epoch)

---

## [IN PROGRESS]

(none)

---

## [TODO]

### Epoch Management - Oracle Review #6 (Jan 18, 2026)
Non-critical hardening from post-fix review:

- [x] Phase 31: max_entries overflow on restore (MEDIUM)
  - After merge failure, restore_for_epoch can exceed max_entries limit
  - Scenario: buffer at max, drain, push max more, fail, restore = 2x max
  - Fix: restore_for_epoch now restores ALL data (no data loss) but returns BufferFull error
  - Epoch is always reset, entries always preserved, error signals overflow for logging

- [x] Phase 32: Multiple EpochManager risk (MEDIUM)
  - If 2+ EpochManagers share same GlobalState, per-manager merge_lock doesn't serialize
  - Can break pending_epoch == global_epoch invariant
  - Fix: Added has_manager: AtomicBool to GlobalState
  - EpochManager::new() uses try_acquire_manager(), returns ManagerAlreadyExists error
  - EpochManager::drop() calls release_manager() to allow new manager

- [x] Phase 33: Document pending_epoch invariants (LOW)
  - Added struct-level doc comment explaining concurrency invariant
  - Added field-level doc comment on pending_epoch explaining the invariant
  - Added doc comments + inline SAFETY comments to drain_for_epoch() and restore_for_epoch()
  - Added doc comment to snapshot_with_epoch() explaining atomicity guarantee

- [x] Phase 34: Add with_max_entries_and_epoch constructor (LOW)
  - Added DeltaBuffer::with_max_entries_and_epoch(row_size, max, epoch)
  - Allows setting both max_entries limit AND initial pending_epoch

### Epoch Management - Oracle Review #7 (Jan 18, 2026)
Hardening issues found in post-phase-34 review:

- [x] Phase 35: Validate entry sizes in restore paths (MEDIUM)
  - Added validate_entries() helper to check diff.len() == row_size_bytes
  - restore() validates before acquiring lock
  - restore_for_epoch() validates before acquiring lock (no side effects on error)

- [x] Phase 36: Fix restore_for_epoch BufferFull check on empty entries (LOW)
  - Removed early return when entries.is_empty()
  - Now computes total and checks overflow even with empty entries
  - Existing entries that exceed max are now properly reported

- [x] Phase 37: Prevent division-by-zero in try_dirty_chunks (MEDIUM)
  - Added chunk_size_bytes == 0 guard in try_dirty_chunks()
  - Added chunk_size_bytes == 0 guard in collect_dirty_chunks_from_entries()
  - Both now return MergeError::InvalidChunkSize instead of panicking

- [x] Phase 38: Deprecate panicking wrappers in epoch.rs (LOW)
  - Added #[deprecated] to dirty_chunks(), dirty_chunks_vec(), build_next_snapshot()
  - All recommend using try_* variants instead
  - Added #[allow(deprecated)] to test module to suppress warnings in tests

- [x] Phase 39: Verify EpochManager Drop resets has_manager (LOW)
  - Confirmed: Drop impl exists and calls release_manager()
  - Confirmed: test epoch_manager_allows_new_manager_after_drop covers this
  - GlobalState correctly allows new manager after previous one is dropped

### Epoch Management - Oracle Review #8 (Jan 18, 2026)
Final hardening pass - edge cases and failure modes:

- [x] Phase 40: Prevent epoch-id wraparound (HIGH)
  - Added MergeError::EpochOverflow variant
  - try_advance() now uses checked_add(1) and returns EpochOverflow on overflow

- [x] Phase 41: Handle rollback failure in try_advance (HIGH)
  - Added MergeError::RollbackFailed { merge_error, rollback_error }
  - try_advance now checks rollback result instead of ignoring with `let _ =`
  - On rollback failure, returns RollbackFailed with both error messages

- [x] Phase 42: Integer overflow guards in restore_for_epoch (MEDIUM)
  - Added DeltaError::EntryCountOverflow and UpdateError::EntryCountOverflow
  - restore_for_epoch uses checked_add BEFORE modifying any state
  - Returns EntryCountOverflow if total would exceed usize::MAX

- [x] Phase 43: Integer overflow guard in try_build_snapshot_from_entries (MEDIUM)
  - Added MergeError::OffsetOverflow { chunk_offset, len }
  - try_build_snapshot_from_entries uses checked_add for end calculation
  - Returns OffsetOverflow instead of panicking on usize overflow

- [x] Phase 44: Document pending_epoch() as non-linearizable (LOW)
  - Added "Warning: Not Linearizable" doc section to pending_epoch()
  - Explains the method reads without lock and may show torn view
  - Recommends snapshot_with_epoch() for consistent (epoch, entries) pair

- [x] Phase 45: Move deprecated wrappers behind #[cfg(test)] (LOW)
  - Added #[cfg(test)] to dirty_chunks(), dirty_chunks_vec(), build_next_snapshot()
  - These panicking wrappers are now only available in test builds
  - Production code cannot accidentally use them

### Epoch Management - Oracle Review #9 (Jan 18, 2026)
Final hardening from post-phase-45 review:

- [x] Phase 46: Treat BufferFull rollback as successful (HIGH)
  - restore_for_epoch returns BufferFull but data IS preserved
  - try_advance now distinguishes BufferFull (data safe) from true failures
  - Only LockPoisoned/EntryCountOverflow/SizeMismatch trigger RollbackFailed

- [x] Phase 47: Add overflow guards to restore() (MEDIUM)
  - restore() now has same checked_add overflow guard as restore_for_epoch
  - Returns EntryCountOverflow on usize overflow
  - Returns BufferFull when exceeds max_entries (but preserves data)

### Core Protocol
- [ ] UBT Merkle proof generation



### Delta-PIR Integration (Jan 18, 2026)
Wire up query handler to use scan_consistent with real DPF evaluation:

- [x] Phase 48: Add from_bytes/to_bytes to AesDpfKey
  - Key format: 16 bytes AES key + 8 bytes target + 1 byte correction_word = 25 bytes
  - Added DpfKeyError::InvalidLength, AES_DPF_KEY_SIZE constant

- [x] Phase 49: Add pending buffer and row_size to AppState
  - Added pending: Arc<DeltaBuffer> to AppState
  - Added row_size_bytes: usize to AppState
  - Updated all test fixtures to include new fields

- [x] Phase 50: Wire query_handler to use scan_consistent
  - Parse 3 hex keys into [AesDpfKey; 3] (rejects invalid 25-byte keys)
  - Call scan_consistent(global, pending, keys, row_size_bytes)
  - Returns real DPF-evaluated payloads
  - Also updated WebSocket query handler (handle_ws_query)

### Delta-PIR Hardening - Oracle Review #10 (Jan 18, 2026)
Post-integration review findings:

- [x] Phase 51: Fix u64->usize truncation in from_bytes (HIGH)
  - Added DpfKeyError::TargetTooLarge variant
  - from_bytes now uses usize::try_from() instead of `as usize`
  - On 32-bit targets, returns error for targets > u32::MAX

- [x] Phase 52: Remove unwrap() in from_bytes parsing (MEDIUM)
  - bytes[16..24].try_into() now uses .expect() with clear message
  - Safe because length is already validated

- [x] Phase 53: Fix WebSocket JSON injection (MEDIUM)
  - Added ws_error_json() helper using serde serialization
  - All error messages now properly JSON-escaped via WsQueryError struct

- [x] Phase 54: Remove panicking unwrap() in WS error paths (MEDIUM)
  - Added WS_INTERNAL_ERROR static fallback string
  - All serde_json::to_string() uses unwrap_or_else with fallback

- [x] Phase 55: Add request size limits (LOW)
  - Added MAX_REQUEST_BODY_SIZE constant (16KB)
  - Router now uses DefaultBodyLimit::max() layer

- [x] Phase 56: Return 503 for TooManyRetries (LOW)
  - query_handler and page_query_handler now return SERVICE_UNAVAILABLE (503)
  - LockPoisoned still returns INTERNAL_SERVER_ERROR (500)

### Oracle Review #11 - Post Phase 51-61 (Jan 19, 2026)
Security, correctness, and performance findings:

**Security (HIGH):**
- [x] Phase 57: Add WebSocket message size limit (HIGH)
  - Added MAX_WS_MESSAGE_BYTES = 16KB (same as HTTP body limit)
  - handle_ws_query checks text.len() before parsing
  - Returns "message too large" error and continues (doesn't close connection)
  - Test: ws_query_rejects_oversized_message

- [x] Phase 58: Add rate/concurrency limiting (HIGH)
  - Added MAX_CONCURRENT_SCANS = 32 constant
  - /query and /query/page now wrapped with ConcurrencyLimitLayer
  - Excess requests get 503 automatically
  - create_router_with_concurrency() for custom limits

**Correctness (MEDIUM):**
- [ ] Phase 59: Validate page matrix alignment (MEDIUM)
  - scan_main_matrix silently ignores remainder bytes
  - scan_pages_chunked ignores partial pages in chunk
  - Add assert/error at matrix construction if not page-aligned

- [ ] Phase 60: Expose page PIR params in /epoch (MEDIUM)
  - Client needs domain_bits, num_pages, prg_keys, rows_per_page
  - Current /epoch only returns row-level metadata
  - Add page_config fields when page_config.is_some()

- [ ] Phase 61g: Validate full PageDpfKey params (MEDIUM)
  - Only domain_bits checked, not PRG keys or page layout
  - Add params_id or prg_keys_hash validation
  - Prevent silent wrong answers from param mismatch

**Consistency (LOW-MEDIUM):**
- [ ] Phase 62: Add error codes to WS responses (LOW)
  - HTTP maps TooManyRetries→503, WS just returns string
  - Add { error: "...", code: "too_many_retries" } schema
  - Enables uniform client retry/backoff

- [ ] Phase 63: Remove remaining panic in scan_delta (LOW)
  - scan_delta still has .expect("lock poisoned")
  - Convert to try_scan_delta usage or return Result

- [ ] Phase 64: Add retry backoff in scan_consistent (LOW)
  - Currently spin-loops 1000 attempts under epoch churn
  - After ~50 attempts, add 1ms sleep to reduce CPU burn

**Performance (MEDIUM):**
- [ ] Phase 65: Pre-allocate page_refs in scan_pages_chunked (MEDIUM)
  - Currently builds Vec<&[u8]> of all pages per request
  - Add Vec::with_capacity(total_pages) at minimum
  - Better: refactor to iterator-based streaming scan

- [ ] Phase 66: Add checked_mul in client metadata (LOW)
  - num_pages * ROWS_PER_PAGE can overflow
  - Use checked_mul().ok_or(...)?

- [ ] Phase 67: Use chunk_size = PAGE_SIZE_BYTES (LOW)
  - Currently hardcoded 4096, could drift from PAGE_SIZE_BYTES
  - Derive from config to prevent mismatch

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

### Real DPF Implementation (Privacy Blocker)
Current AesDpfKey stores target in plaintext - servers can read queried index.
Need proper 2-server FSS/DPF where keys are computationally indistinguishable.

**Option A: fss-rs library (recommended)**
- Crate: https://crates.io/crates/fss-rs (v0.6.0)
- Implements Boyle-Gilboa-Ishai DPF with correction words
- Uses AES-NI PRG (Matyas-Meyer-Oseas construction)
- Key size: ~16 + 17*depth bytes (e.g., 305B for 100K rows, 492B for 250M rows)
- Full-domain eval via `full_eval()` method
- Supports AVX2/AVX-512 and ARM crypto extensions
- Overhead estimate: ~log(n) AES rounds per row evaluation

**Option B: Page-Level PIR (RECOMMENDED)**
- Use fss-rs DPF at page granularity (no privacy loss!)
- 16 rows/page at 256B/row = 4KB pages
- Domain: 368M accounts / 16 = 27M pages (vs 433M rows)
- Client downloads 4KB page, extracts target row locally
- Servers learn NOTHING about which page (proper 2-server DPF)
- Aligns with UBT Merkle tree page boundaries

**Tasks:**
- [x] Phase 57: Add fss-rs dependency and basic integration test
  - Added fss-rs v0.6 with "stable" feature
  - Tests verify DPF pair XOR to point function
  - Tests verify full_eval correctness (256 rows)
  - Added benchmark skeleton in benches/dpf_bench.rs
- [x] Phase 58: Benchmark fss-rs full_eval vs current AesDpfKey (COMPLETE)
  - Results show massive overhead: 57-428x slower than AesDpfKey
  - 256 rows: 165ns vs 70.8µs (428x)
  - 1K rows: 590ns vs 114µs (193x)
  - 4K rows: 2.35µs vs 228µs (97x)
  - 16K rows: 9.4µs vs 639µs (68x)
  - 64K rows: 37.7µs vs 2.14ms (57x)
  - fss-rs throughput: ~30 Melem/s vs AesDpfKey: ~1.7 Gelem/s
  - Conclusion: fss-rs too slow for full-domain eval at 250M rows
  - Estimated 250M latency: ~8 seconds (vs 143ms AesDpfKey)
  - Page/bucket PIR fallback required for production
- [x] Phase 59: Page-level PIR performance analysis (COMPLETE)
  - Benchmarked fss-rs at 1M domain: 31ms (vs AesDpfKey 620µs = 50x overhead)
  - Mainnet extrapolation (27M pages): ~840ms with fss-rs page-level
  - With half-tree optimization (1.5x): ~560ms
  - With GPU acceleration (15x): ~56ms
  - Conclusion: Page-level PIR viable for mainnet

- [x] Phase 60: Implement page-level PIR with fss-rs
  - Created page.rs module with PageDpfParams, PageDpfKey, generate_page_dpf_keys()
  - PageAddress struct for row→page+offset mapping
  - Supports 1-4 byte domains (1-32 bits) with proper validation
  - Uses DpfImpl with new_with_filter for non-power-of-8 domain sizes
  - Key insight: alpha must be left-shifted for new_with_filter (MSB encoding)
  - extract_row_from_page() and xor_pages() helpers
  - full_eval() now validates output size, returns Result
  - Tests pass for 8/10/16/20/25-bit domains + E2E PIR flow
  - 25-bit domain works (production: 27M pages for Ethereum mainnet)
  - True 2-server privacy: servers learn nothing about target page

- [x] Phase 60b: Oracle review #2 fixes
  - Fixed max_pages() for 32-bit targets (caps at usize::BITS)
  - Added boundary domain tests (2/7/9/15/17/24 bits with first/mid/last targets)
  - Added eval_and_accumulate() streaming API for server-side evaluation
  - Streaming API avoids separate O(N) DPF output allocation during page scan
  - 24 tests pass including new boundary and streaming tests

- [x] Phase 61a: Add PageDpfKey serialization (COMPLETE)
  - to_bytes()/from_bytes() with format: 66 + 17*domain_bits bytes
  - 25-bit domain (mainnet): 491 bytes per key
  - Roundtrip tests verify full_eval produces identical output
  - Created docs/PAGE_PIR_INTEGRATION.md with full migration plan
- [x] Phase 61b: Create page-level query types (PageQueryRequest/Response)
  - Added PageQueryRequest, PageQueryResponse structs
  - Added PagePirConfig to AppState (optional, for dual-mode support)
  - Updated all test fixtures with page_config: None
- [x] Phase 61c: Add page-level scan function (scan_pages_chunked)
  - scan_pages_chunked(): evaluates 3 PageDpfKeys over page matrix
  - scan_pages_consistent(): epoch-consistent version with retry loop
  - Uses eval_and_accumulate_chunked for O(chunk_size) memory
  - Test verifies 2-server XOR recovers correct target page
- [x] Phase 61d: Wire up page query handler (POST /query/page)
  - page_query_handler(): parses 3 PageDpfKeys, calls scan_pages_consistent
  - Returns 404 when page_config is None (page PIR disabled)
  - Validates domain_bits matches server config
  - Returns PageQueryResponse with epoch_id and 3 page payloads (4KB each)
- [x] Phase 61e: Update client library for page-level queries
  - Added page module to morphogen-client with PageEpochMetadata, PageQueryKeys
  - generate_page_query(): computes page addresses from CuckooAddresser, generates key pairs
  - aggregate_page_responses(): XORs server responses to recover plaintext pages
  - extract_rows_from_pages(): extracts target rows from pages using row_offset
  - Re-exports PageDpfParams, PageAddress, PAGE_SIZE_BYTES from morphogen-dpf
- [x] Phase 61f: Reorganize data as pages (migration)
  - No physical reorganization needed: pages are 16 consecutive rows (4KB each)
  - Current row-contiguous storage already supports page-level access
  - scan_pages_chunked() computes page boundaries within existing matrix chunks
  - Added PAGE_SIZE_BYTES constant (4096) for clarity

### Vendor fss-rs for Streaming Eval (Jan 19, 2026)
Goal: Eliminate O(N) DPF output allocation (528MB at 25-bit domain).
Oracle recommendation: Vendor for streaming only, defer half-tree until bottleneck measured.

**Phase 62: Vendor fss-rs**
- [x] Phase 62a: Copy fss-rs v0.6.0 source into crates/morphogen-dpf/vendor/fss-rs
- [x] Phase 62b: Update Cargo.toml to use vendored copy (path dependency)
- [x] Phase 62c: Verify existing tests still pass with vendored version (24/24 pass)
- [x] Phase 62d: Document vendoring rationale in vendor/fss-rs/VENDORING.md

**Phase 63: Implement streaming/chunked eval API**
- [x] Phase 63a: Add `eval_range(start_idx, &mut [G])` to DpfImpl (chunked eval)
  - Added eval_range() to Dpf trait and DpfImpl
  - Uses eval_range_layer() that only descends into overlapping subtrees
  - Supports rayon parallelism when multi-thread feature enabled
  - Tests verify correctness at boundaries and match full_eval output
- [x] Phase 63b-d: Add eval_and_accumulate_chunked() to PageDpfKey
  - Uses eval_range() to process pages in chunks
  - O(chunk_size) memory for DPF buffer instead of O(N)
  - Deprecated old eval_and_accumulate() with O(N) allocation
  - Tests verify chunked output matches full_eval for various chunk sizes
- [x] Phase 63e: Benchmark memory usage: before (528MB) vs after (<1MB)
  - At 16-bit domain (64K pages):
    - Full eval: 1024 KB buffer, 6.4ms, 10.2 Melem/s
    - Chunked (4096): 64 KB buffer, 9.3ms, 7.0 Melem/s
  - At 25-bit domain (27M pages for mainnet):
    - Full eval: 512 MB buffer
    - Chunked (4096): 64 KB buffer
    - Memory reduction: 8192x
  - Performance trade-off: ~1.45x slower for 16x less memory (at chunk=4096)

**Phase 64: Integrate streaming eval into page PIR**
- [ ] Phase 64a: Update PageDpfKey::eval_and_accumulate() to use eval_range
- [ ] Phase 64b: Process DB pages in chunks (e.g., 4096 pages at a time)
- [ ] Phase 64c: Parallel chunk processing with rayon
- [ ] Phase 64d: Benchmark end-to-end latency at 25-bit domain

**Phase 65: Measure bottleneck split**
- [ ] Phase 65a: Instrument time split: DPF eval vs DB scan vs XOR accumulate
- [ ] Phase 65b: Profile at mainnet scale (27M pages, 108GB data)
- [ ] Phase 65c: Document findings - is DPF eval >30% of total time?
- [ ] Phase 65d: Decision gate: proceed to half-tree only if DPF dominates

**Phase 66: Half-tree DPF (conditional)**
- [ ] Phase 66a: Implement HalfTreeDpfKey based on Guo et al. 2022
- [ ] Phase 66b: New key format with half-tree correction words
- [ ] Phase 66c: Correctness tests with deterministic test vectors
- [ ] Phase 66d: Benchmark 1.5x improvement claim
- [ ] Phase 66e: Integration with PageDpfParams (feature flag)

### DPF Optimization Research (Jan 19, 2026)
Literature review findings from Semantic Scholar search:

**Key Papers:**
1. **Half-Tree DPF** (Guo et al. 2022) - eprint.iacr.org/2022/1431
   - Halves AES calls for full-domain eval (1.5N vs 2N)
   - Halves communication and round complexity
   - Security: Random Permutation Model (vs PRG for standard)
   - Trade-off: Slightly stronger assumption, same key size
   
2. **Ternary-Tree DPF** (sachaservan/tri-dpf)
   - Ternary instead of binary tree = flatter tree = fewer levels
   - log3(N) vs log2(N) depth = ~1.58x fewer levels
   - Batched AES for better CPU utilization
   
3. **GPU-DPF** (Facebook Research) - arxiv.org/abs/2301.10904
   - Based on BGI 2014 DPF (same as fss-rs)
   - 200x speedup over single-threaded CPU, 15-20x over 32-thread
   - Memory-bounded tree traversal (optimal work + low memory)
   - Operator fusion: DPF eval + matrix multiply in single pass
   - V100: 923 DPFs/sec at 1M entries
   - Modal benchmark script created: modal_gpu_dpf_bench.py
   
4. **Programmable DPF** (Boyle et al. 2022) - eprint.iacr.org/2022/1060
   - Short offline key reusable for many queries
   - Only practical for small domains (poly-size)

**Mainnet Performance Projection (368M accounts, 27M pages):**

| Implementation | Latency | Notes |
|----------------|---------|-------|
| AesDpfKey (insecure) | ~250ms | Servers see target |
| fss-rs row-level | ~13.4s | Too slow |
| **fss-rs page-level** | **~840ms** | Viable! |
| + Half-tree (1.5x) | ~560ms | Under 600ms target |
| + GPU (15x) | ~56ms | Best case |

**Optimization Roadmap:**
- [ ] Phase 62: Prototype fss-rs page-level PIR
- [ ] Phase 63: Benchmark Facebook GPU-DPF on Modal (script ready)
- [ ] Phase 64: Implement half-tree optimization if needed (1.5x)
- [ ] Phase 65: GPU port for production (15-20x)

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
