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

- [ ] Phase 50: Wire query_handler to use scan_consistent
  - Parse 3 hex keys into [AesDpfKey; 3]
  - Call scan_consistent(global, pending, keys, row_size_bytes)
  - Return real payloads (not dummy vec![0u8; 256])

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
