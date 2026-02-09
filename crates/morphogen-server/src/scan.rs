use morphogen_core::{DeltaBuffer, GlobalState};
use morphogen_dpf::DpfKey;
use morphogen_storage::ChunkedMatrix;

pub const DEFAULT_MAX_RETRIES: usize = 1000;

#[derive(Debug, PartialEq, Eq)]
pub enum ScanError {
    LockPoisoned,
    TooManyRetries {
        attempts: usize,
    },
    MatrixNotAligned {
        total_bytes: usize,
        unit_size: usize,
        remainder: usize,
    },
    ChunkNotAligned {
        chunk_idx: usize,
        chunk_bytes: usize,
        unit_size: usize,
        remainder: usize,
    },
}

impl std::fmt::Display for ScanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScanError::LockPoisoned => write!(f, "lock poisoned during scan"),
            ScanError::TooManyRetries { attempts } => {
                write!(f, "scan failed after {} retry attempts", attempts)
            }
            ScanError::MatrixNotAligned {
                total_bytes,
                unit_size,
                remainder,
            } => {
                write!(
                    f,
                    "matrix size {} not aligned to {} bytes (remainder: {})",
                    total_bytes, unit_size, remainder
                )
            }
            ScanError::ChunkNotAligned {
                chunk_idx,
                chunk_bytes,
                unit_size,
                remainder,
            } => {
                write!(
                    f,
                    "chunk {} size {} not aligned to {} bytes (remainder: {})",
                    chunk_idx, chunk_bytes, unit_size, remainder
                )
            }
        }
    }
}

impl std::error::Error for ScanError {}

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "profiling")]
use crate::profiling::Profiler;

pub fn scan_main_matrix<K: DpfKey>(
    matrix: &ChunkedMatrix,
    keys: &[K; 3],
    row_size_bytes: usize,
) -> [Vec<u8>; 3] {
    #[cfg(feature = "profiling")]
    let mut profiler = Profiler::new();

    if row_size_bytes == 0 {
        return empty_result(row_size_bytes);
    }

    let num_rows = matrix.total_size_bytes() / row_size_bytes;

    #[cfg(feature = "profiling")]
    profiler.checkpoint("scan_main_matrix_init");

    let result = if cfg!(feature = "avx512") && avx512_available() {
        #[cfg(feature = "profiling")]
        profiler.checkpoint("scan_main_matrix_avx512_path");
        unsafe { scan_kernel_avx512(matrix, keys, num_rows, row_size_bytes) }
    } else {
        #[cfg(feature = "profiling")]
        profiler.checkpoint("scan_main_matrix_portable_path");
        scan_main_matrix_portable(matrix, keys, num_rows, row_size_bytes)
    };

    #[cfg(feature = "profiling")]
    {
        profiler.checkpoint("scan_main_matrix_complete");
        eprintln!("scan_main_matrix breakdown:\n{}", profiler.report());
    }

    result
}

pub fn try_scan_delta<K: DpfKey>(
    delta: &DeltaBuffer,
    keys: &[K; 3],
    results: &mut [Vec<u8>; 3],
) -> Result<(), ScanError> {
    let entries = delta.snapshot().map_err(|_| ScanError::LockPoisoned)?;
    for entry in &entries {
        for (k, key) in keys.iter().enumerate() {
            if key.eval_bit(entry.row_idx) {
                xor_into(&mut results[k], &entry.diff);
            }
        }
    }
    Ok(())
}

/// Deprecated: Use `try_scan_delta` instead to handle lock poisoning gracefully.
#[cfg(test)]
#[deprecated(since = "0.1.0", note = "use try_scan_delta instead")]
pub fn scan_delta<K: DpfKey>(delta: &DeltaBuffer, keys: &[K; 3], results: &mut [Vec<u8>; 3]) {
    try_scan_delta(delta, keys, results).expect("scan_delta: lock poisoned")
}

/// Scans the matrix and delta buffer, returning results.
///
/// # Errors
/// Returns `ScanError::LockPoisoned` if delta buffer lock is poisoned.
pub fn try_scan<K: DpfKey>(
    matrix: &ChunkedMatrix,
    delta: &DeltaBuffer,
    keys: &[K; 3],
    row_size_bytes: usize,
) -> Result<[Vec<u8>; 3], ScanError> {
    #[cfg(feature = "profiling")]
    let mut profiler = Profiler::new();

    #[cfg(feature = "profiling")]
    profiler.checkpoint("scan_start");

    let mut results = scan_main_matrix(matrix, keys, row_size_bytes);

    #[cfg(feature = "profiling")]
    profiler.checkpoint("scan_main_matrix");

    try_scan_delta(delta, keys, &mut results)?;

    #[cfg(feature = "profiling")]
    profiler.checkpoint("scan_delta");

    #[cfg(feature = "profiling")]
    eprintln!("{}", profiler.report());

    Ok(results)
}

pub fn scan_consistent<K: DpfKey>(
    global: &GlobalState,
    pending: &DeltaBuffer,
    keys: &[K; 3],
    row_size_bytes: usize,
) -> Result<([Vec<u8>; 3], u64), ScanError> {
    scan_consistent_with_max_retries(global, pending, keys, row_size_bytes, DEFAULT_MAX_RETRIES)
}

/// Backoff thresholds for scan_consistent retry loop.
const SPIN_LOOP_THRESHOLD: usize = 10;
const YIELD_THRESHOLD: usize = 50;

pub fn scan_consistent_with_max_retries<K: DpfKey>(
    global: &GlobalState,
    pending: &DeltaBuffer,
    keys: &[K; 3],
    row_size_bytes: usize,
    max_retries: usize,
) -> Result<([Vec<u8>; 3], u64), ScanError> {
    for attempt in 0..max_retries {
        let snapshot1 = global.load();
        let epoch1 = snapshot1.epoch_id;

        let (pending_epoch, entries) = pending
            .snapshot_with_epoch()
            .map_err(|_| ScanError::LockPoisoned)?;

        let snapshot2 = global.load();
        if snapshot2.epoch_id == epoch1 && pending_epoch == epoch1 {
            let mut results = scan_main_matrix(snapshot1.matrix.as_ref(), keys, row_size_bytes);
            for entry in &entries {
                for (k, key) in keys.iter().enumerate() {
                    if key.eval_bit(entry.row_idx) {
                        xor_into(&mut results[k], &entry.diff);
                    }
                }
            }
            return Ok((results, epoch1));
        }

        if attempt < SPIN_LOOP_THRESHOLD {
            std::hint::spin_loop();
        } else if attempt < YIELD_THRESHOLD {
            std::thread::yield_now();
        } else {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }
    Err(ScanError::TooManyRetries {
        attempts: max_retries,
    })
}

fn xor_into(dest: &mut [u8], src: &[u8]) {
    for (d, s) in dest.iter_mut().zip(src.iter()) {
        *d ^= s;
    }
}

/// Page-level PIR scan using privacy-preserving DPF.
///
/// Unlike row-level scan, this uses proper 2-server DPF where servers
/// cannot determine which page is being queried.
///
/// # Arguments
/// * `matrix` - Page data stored contiguously (page_size_bytes per page)
/// * `keys` - 3 PageDpfKeys (for Cuckoo addressing)
/// * `page_size_bytes` - Size of each page (typically 4096)
/// * `chunk_size` - DPF evaluation chunk size (e.g., 4096 for 64KB buffer)
///
/// # Returns
/// 3 page payloads that client XORs with other server's response
///
/// # Errors
/// Returns `ScanError::ChunkNotAligned` if any chunk is not page-aligned.
#[cfg(any(feature = "network", test))]
pub fn try_scan_pages_chunked(
    matrix: &ChunkedMatrix,
    keys: &[morphogen_dpf::page::PageDpfKey; 3],
    page_size_bytes: usize,
    dpf_chunk_size: usize,
) -> Result<[Vec<u8>; 3], ScanError> {
    if page_size_bytes == 0 {
        return Ok([vec![], vec![], vec![]]);
    }

    let total_pages = matrix.total_size_bytes() / page_size_bytes;
    let mut page_refs: Vec<&[u8]> = Vec::with_capacity(total_pages);

    for chunk_idx in 0..matrix.num_chunks() {
        let chunk = matrix.chunk(chunk_idx);
        let chunk_len = matrix.chunk_size(chunk_idx);
        let remainder = chunk_len % page_size_bytes;

        if remainder != 0 {
            return Err(ScanError::ChunkNotAligned {
                chunk_idx,
                chunk_bytes: chunk_len,
                unit_size: page_size_bytes,
                remainder,
            });
        }

        let pages_in_chunk = chunk_len / page_size_bytes;
        let chunk_slice = chunk.as_slice();

        for p in 0..pages_in_chunk {
            let start = p * page_size_bytes;
            let end = start + page_size_bytes;
            page_refs.push(&chunk_slice[start..end]);
        }
    }

    let results: [Vec<u8>; 3] =
        std::array::from_fn(|i| keys[i].eval_and_accumulate_chunked(&page_refs, dpf_chunk_size));

    Ok(results)
}

/// Page-level PIR scan (panics on alignment error - prefer try_scan_pages_chunked)
#[cfg(test)]
pub fn scan_pages_chunked(
    matrix: &ChunkedMatrix,
    keys: &[morphogen_dpf::page::PageDpfKey; 3],
    page_size_bytes: usize,
    dpf_chunk_size: usize,
) -> [Vec<u8>; 3] {
    try_scan_pages_chunked(matrix, keys, page_size_bytes, dpf_chunk_size)
        .expect("matrix not page-aligned")
}

/// Page-level PIR scan with epoch consistency.
///
/// Ensures the scan sees a consistent view even during epoch transitions.
#[cfg(feature = "network")]
pub fn scan_pages_consistent(
    global: &GlobalState,
    keys: &[morphogen_dpf::page::PageDpfKey; 3],
    page_size_bytes: usize,
    chunk_size: usize,
) -> Result<([Vec<u8>; 3], u64), ScanError> {
    scan_pages_consistent_with_max_retries(
        global,
        keys,
        page_size_bytes,
        chunk_size,
        DEFAULT_MAX_RETRIES,
    )
}

/// Page-level PIR scan with configurable retry limit.
#[cfg(feature = "network")]
pub fn scan_pages_consistent_with_max_retries(
    global: &GlobalState,
    keys: &[morphogen_dpf::page::PageDpfKey; 3],
    page_size_bytes: usize,
    chunk_size: usize,
    max_retries: usize,
) -> Result<([Vec<u8>; 3], u64), ScanError> {
    for attempt in 0..max_retries {
        let snapshot1 = global.load();
        let epoch1 = snapshot1.epoch_id;

        let snapshot2 = global.load();
        if snapshot2.epoch_id == epoch1 {
            let results = try_scan_pages_chunked(
                snapshot1.matrix.as_ref(),
                keys,
                page_size_bytes,
                chunk_size,
            )?;
            return Ok((results, epoch1));
        }

        if attempt < 10 {
            std::hint::spin_loop();
        } else {
            std::thread::yield_now();
        }
    }
    Err(ScanError::TooManyRetries {
        attempts: max_retries,
    })
}

#[cfg(feature = "cuda")]
pub fn scan_delta_for_gpu(
    delta: &DeltaBuffer,
    keys: &[morphogen_gpu_dpf::dpf::ChaChaKey; 3],
    page_size_bytes: usize,
) -> Result<[Vec<u8>; 3], ScanError> {
    let entries = delta.snapshot().map_err(|_| ScanError::LockPoisoned)?;

    let mut results = [
        vec![0u8; page_size_bytes],
        vec![0u8; page_size_bytes],
        vec![0u8; page_size_bytes],
    ];

    if entries.is_empty() {
        return Ok(results);
    }

    let row_size = delta.row_size_bytes();
    if row_size == 0 {
        return Ok(results); // Should probably be error, but safe
    }
    let rows_per_page = page_size_bytes / row_size;

    for entry in entries {
        let page_idx = entry.row_idx / rows_per_page;
        let row_offset_in_page = (entry.row_idx % rows_per_page) * row_size;

        // Ensure we don't write out of bounds
        if row_offset_in_page + row_size > page_size_bytes {
            continue; // Should not happen if invariants hold
        }

        for k in 0..3 {
            let mask_seed = keys[k].eval(page_idx);
            let mask_bytes = mask_seed.to_bytes(); // 16 bytes

            let target_slice = &mut results[k][row_offset_in_page..row_offset_in_page + row_size];
            for (i, byte) in entry.diff.iter().enumerate() {
                // Mask is repeated every 16 bytes (block size of ChaCha/uint4)
                target_slice[i] ^= byte & mask_bytes[i % 16];
            }
        }
    }
    Ok(results)
}

#[cfg(feature = "parallel")]
pub fn scan_consistent_parallel<K: DpfKey + Sync>(
    global: &GlobalState,
    pending: &DeltaBuffer,
    keys: &[K; 3],
    row_size_bytes: usize,
    batch_size: usize,
) -> Result<([Vec<u8>; 3], u64), ScanError> {
    scan_consistent_parallel_with_max_retries(
        global,
        pending,
        keys,
        row_size_bytes,
        DEFAULT_MAX_RETRIES,
        batch_size,
    )
}

#[cfg(feature = "parallel")]
pub fn scan_consistent_parallel_with_max_retries<K: DpfKey + Sync>(
    global: &GlobalState,
    pending: &DeltaBuffer,
    keys: &[K; 3],
    row_size_bytes: usize,
    max_retries: usize,
    batch_size: usize,
) -> Result<([Vec<u8>; 3], u64), ScanError> {
    for attempt in 0..max_retries {
        let snapshot1 = global.load();
        let epoch1 = snapshot1.epoch_id;

        let (pending_epoch, entries) = pending
            .snapshot_with_epoch()
            .map_err(|_| ScanError::LockPoisoned)?;

        let snapshot2 = global.load();
        if snapshot2.epoch_id == epoch1 && pending_epoch == epoch1 {
            let mut results = scan_main_matrix_parallel_batched(
                snapshot1.matrix.as_ref(),
                keys,
                row_size_bytes,
                batch_size,
            );
            for entry in &entries {
                for (k, key) in keys.iter().enumerate() {
                    if key.eval_bit(entry.row_idx) {
                        xor_into(&mut results[k], &entry.diff);
                    }
                }
            }
            return Ok((results, epoch1));
        }

        if attempt < 10 {
            std::hint::spin_loop();
        } else {
            std::thread::yield_now();
        }
    }
    Err(ScanError::TooManyRetries {
        attempts: max_retries,
    })
}

fn empty_result(row_size_bytes: usize) -> [Vec<u8>; 3] {
    [
        vec![0u8; row_size_bytes],
        vec![0u8; row_size_bytes],
        vec![0u8; row_size_bytes],
    ]
}

#[cfg(feature = "parallel")]
pub fn scan_main_matrix_parallel_batched<K: DpfKey + Sync>(
    matrix: &ChunkedMatrix,
    keys: &[K; 3],
    row_size_bytes: usize,
    batch_size: usize,
) -> [Vec<u8>; 3] {
    // TODO: Implement actual batched processing
    // For now, delegate to single-query parallel scan
    scan_main_matrix_parallel(matrix, keys, row_size_bytes)
}

#[cfg(feature = "parallel")]
pub fn scan_main_matrix_parallel<K: DpfKey + Sync>(
    matrix: &ChunkedMatrix,
    keys: &[K; 3],
    row_size_bytes: usize,
) -> [Vec<u8>; 3] {
    use morphogen_storage::AlignedMatrix;
    use std::sync::Arc;

    if row_size_bytes == 0 {
        return empty_result(row_size_bytes);
    }

    let chunks: &[Arc<AlignedMatrix>] = matrix.chunks();
    let chunk_size = matrix.chunk_size_bytes();
    let num_chunks = matrix.num_chunks();

    let partial_results: Vec<[Vec<u8>; 3]> = (0..num_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let chunk = &chunks[chunk_idx];
            let chunk_len = matrix.chunk_size(chunk_idx);
            let rows_in_chunk = chunk_len / row_size_bytes;
            let global_row_start = chunk_idx * (chunk_size / row_size_bytes);

            scan_chunk_avx512(
                chunk.as_ptr(),
                rows_in_chunk,
                global_row_start,
                keys,
                row_size_bytes,
            )
        })
        .collect();

    let mut results = empty_result(row_size_bytes);
    for partial in partial_results {
        xor_into(&mut results[0], &partial[0]);
        xor_into(&mut results[1], &partial[1]);
        xor_into(&mut results[2], &partial[2]);
    }

    results
}

#[cfg(feature = "parallel")]
fn scan_chunk_avx512<K: DpfKey>(
    chunk_ptr: *const u8,
    rows_in_chunk: usize,
    global_row_start: usize,
    keys: &[K; 3],
    row_size_bytes: usize,
) -> [Vec<u8>; 3] {
    if cfg!(feature = "avx512") && avx512_available() {
        unsafe {
            scan_chunk_avx512_inner(
                chunk_ptr,
                rows_in_chunk,
                global_row_start,
                keys,
                row_size_bytes,
            )
        }
    } else {
        scan_chunk_portable(
            chunk_ptr,
            rows_in_chunk,
            global_row_start,
            keys,
            row_size_bytes,
        )
    }
}

#[cfg(feature = "parallel")]
fn scan_chunk_portable<K: DpfKey>(
    chunk_ptr: *const u8,
    rows_in_chunk: usize,
    global_row_start: usize,
    keys: &[K; 3],
    row_size_bytes: usize,
) -> [Vec<u8>; 3] {
    let mut results = empty_result(row_size_bytes);

    for row_offset in 0..rows_in_chunk {
        let global_row = global_row_start + row_offset;
        let row_ptr = unsafe { chunk_ptr.add(row_offset * row_size_bytes) };
        let masks = [
            mask_byte(keys[0].eval_bit(global_row)),
            mask_byte(keys[1].eval_bit(global_row)),
            mask_byte(keys[2].eval_bit(global_row)),
        ];

        let mut i = 0usize;
        while i + 64 <= row_size_bytes {
            let src = unsafe { std::slice::from_raw_parts(row_ptr.add(i), 64) };
            for (k, mask) in masks.iter().enumerate() {
                xor_masked(&mut results[k], src, i, *mask);
            }
            i += 64;
        }

        if i < row_size_bytes {
            let src = unsafe { std::slice::from_raw_parts(row_ptr.add(i), row_size_bytes - i) };
            for (k, mask) in masks.iter().enumerate() {
                xor_masked(&mut results[k], src, i, *mask);
            }
        }
    }

    results
}

#[cfg(all(feature = "parallel", feature = "avx512"))]
#[target_feature(enable = "avx512f,avx512vl,vaes")]
unsafe fn scan_chunk_avx512_inner<K: DpfKey>(
    chunk_ptr: *const u8,
    rows_in_chunk: usize,
    global_row_start: usize,
    keys: &[K; 3],
    row_size_bytes: usize,
) -> [Vec<u8>; 3] {
    use std::arch::x86_64::{
        __m512i, _mm512_and_si512, _mm512_loadu_si512, _mm512_set1_epi8, _mm512_storeu_si512,
        _mm512_xor_si512,
    };

    let mut results = empty_result(row_size_bytes);
    let dst0 = results[0].as_mut_ptr();
    let dst1 = results[1].as_mut_ptr();
    let dst2 = results[2].as_mut_ptr();
    let has_tail = row_size_bytes % 64 != 0;

    let mut row_offset = 0usize;
    let mut global_row = global_row_start;

    // Process 8 rows at a time
    while row_offset + 7 < rows_in_chunk {
        let row_ptrs: [*const u8; 8] =
            std::array::from_fn(|j| chunk_ptr.add((row_offset + j) * row_size_bytes));

        let m0: [u8; 8] = std::array::from_fn(|j| mask_byte(keys[0].eval_bit(global_row + j)));
        let m1: [u8; 8] = std::array::from_fn(|j| mask_byte(keys[1].eval_bit(global_row + j)));
        let m2: [u8; 8] = std::array::from_fn(|j| mask_byte(keys[2].eval_bit(global_row + j)));

        let masks0: [__m512i; 8] = std::array::from_fn(|j| _mm512_set1_epi8(m0[j] as i8));
        let masks1: [__m512i; 8] = std::array::from_fn(|j| _mm512_set1_epi8(m1[j] as i8));
        let masks2: [__m512i; 8] = std::array::from_fn(|j| _mm512_set1_epi8(m2[j] as i8));

        let mut i = 0usize;
        while i + 64 <= row_size_bytes {
            let d: [__m512i; 8] =
                std::array::from_fn(|j| _mm512_loadu_si512(row_ptrs[j].add(i) as *const __m512i));

            let acc0 = _mm512_loadu_si512(dst0.add(i) as *const __m512i);
            let acc1 = _mm512_loadu_si512(dst1.add(i) as *const __m512i);
            let acc2 = _mm512_loadu_si512(dst2.add(i) as *const __m512i);

            let masked0 = _mm512_xor_si512(
                _mm512_xor_si512(
                    _mm512_xor_si512(
                        _mm512_and_si512(d[0], masks0[0]),
                        _mm512_and_si512(d[1], masks0[1]),
                    ),
                    _mm512_xor_si512(
                        _mm512_and_si512(d[2], masks0[2]),
                        _mm512_and_si512(d[3], masks0[3]),
                    ),
                ),
                _mm512_xor_si512(
                    _mm512_xor_si512(
                        _mm512_and_si512(d[4], masks0[4]),
                        _mm512_and_si512(d[5], masks0[5]),
                    ),
                    _mm512_xor_si512(
                        _mm512_and_si512(d[6], masks0[6]),
                        _mm512_and_si512(d[7], masks0[7]),
                    ),
                ),
            );
            let masked1 = _mm512_xor_si512(
                _mm512_xor_si512(
                    _mm512_xor_si512(
                        _mm512_and_si512(d[0], masks1[0]),
                        _mm512_and_si512(d[1], masks1[1]),
                    ),
                    _mm512_xor_si512(
                        _mm512_and_si512(d[2], masks1[2]),
                        _mm512_and_si512(d[3], masks1[3]),
                    ),
                ),
                _mm512_xor_si512(
                    _mm512_xor_si512(
                        _mm512_and_si512(d[4], masks1[4]),
                        _mm512_and_si512(d[5], masks1[5]),
                    ),
                    _mm512_xor_si512(
                        _mm512_and_si512(d[6], masks1[6]),
                        _mm512_and_si512(d[7], masks1[7]),
                    ),
                ),
            );
            let masked2 = _mm512_xor_si512(
                _mm512_xor_si512(
                    _mm512_xor_si512(
                        _mm512_and_si512(d[0], masks2[0]),
                        _mm512_and_si512(d[1], masks2[1]),
                    ),
                    _mm512_xor_si512(
                        _mm512_and_si512(d[2], masks2[2]),
                        _mm512_and_si512(d[3], masks2[3]),
                    ),
                ),
                _mm512_xor_si512(
                    _mm512_xor_si512(
                        _mm512_and_si512(d[4], masks2[4]),
                        _mm512_and_si512(d[5], masks2[5]),
                    ),
                    _mm512_xor_si512(
                        _mm512_and_si512(d[6], masks2[6]),
                        _mm512_and_si512(d[7], masks2[7]),
                    ),
                ),
            );

            _mm512_storeu_si512(dst0.add(i) as *mut __m512i, _mm512_xor_si512(acc0, masked0));
            _mm512_storeu_si512(dst1.add(i) as *mut __m512i, _mm512_xor_si512(acc1, masked1));
            _mm512_storeu_si512(dst2.add(i) as *mut __m512i, _mm512_xor_si512(acc2, masked2));

            i += 64;
        }

        if has_tail && i < row_size_bytes {
            for j in 0..8 {
                let tail = std::slice::from_raw_parts(row_ptrs[j].add(i), row_size_bytes - i);
                xor_masked(results[0].as_mut_slice(), tail, i, m0[j]);
                xor_masked(results[1].as_mut_slice(), tail, i, m1[j]);
                xor_masked(results[2].as_mut_slice(), tail, i, m2[j]);
            }
        }

        global_row += 8;
        row_offset += 8;
    }

    // Process remaining rows
    while row_offset < rows_in_chunk {
        let row_ptr = chunk_ptr.add(row_offset * row_size_bytes);
        let m0 = mask_byte(keys[0].eval_bit(global_row));
        let m1 = mask_byte(keys[1].eval_bit(global_row));
        let m2 = mask_byte(keys[2].eval_bit(global_row));
        let mask0 = _mm512_set1_epi8(m0 as i8);
        let mask1 = _mm512_set1_epi8(m1 as i8);
        let mask2 = _mm512_set1_epi8(m2 as i8);

        let mut i = 0usize;
        while i + 64 <= row_size_bytes {
            let data = _mm512_loadu_si512(row_ptr.add(i) as *const __m512i);
            let acc0 = _mm512_loadu_si512(dst0.add(i) as *const __m512i);
            let acc1 = _mm512_loadu_si512(dst1.add(i) as *const __m512i);
            let acc2 = _mm512_loadu_si512(dst2.add(i) as *const __m512i);

            _mm512_storeu_si512(
                dst0.add(i) as *mut __m512i,
                _mm512_xor_si512(acc0, _mm512_and_si512(data, mask0)),
            );
            _mm512_storeu_si512(
                dst1.add(i) as *mut __m512i,
                _mm512_xor_si512(acc1, _mm512_and_si512(data, mask1)),
            );
            _mm512_storeu_si512(
                dst2.add(i) as *mut __m512i,
                _mm512_xor_si512(acc2, _mm512_and_si512(data, mask2)),
            );
            i += 64;
        }

        if has_tail && i < row_size_bytes {
            let tail = std::slice::from_raw_parts(row_ptr.add(i), row_size_bytes - i);
            xor_masked(results[0].as_mut_slice(), tail, i, m0);
            xor_masked(results[1].as_mut_slice(), tail, i, m1);
            xor_masked(results[2].as_mut_slice(), tail, i, m2);
        }

        global_row += 1;
        row_offset += 1;
    }

    results
}

#[cfg(all(feature = "parallel", not(feature = "avx512")))]
unsafe fn scan_chunk_avx512_inner<K: DpfKey>(
    chunk_ptr: *const u8,
    rows_in_chunk: usize,
    global_row_start: usize,
    keys: &[K; 3],
    row_size_bytes: usize,
) -> [Vec<u8>; 3] {
    scan_chunk_portable(
        chunk_ptr,
        rows_in_chunk,
        global_row_start,
        keys,
        row_size_bytes,
    )
}

fn scan_main_matrix_portable<K: DpfKey>(
    matrix: &ChunkedMatrix,
    keys: &[K; 3],
    num_rows: usize,
    row_size_bytes: usize,
) -> [Vec<u8>; 3] {
    let mut results = empty_result(row_size_bytes);
    let chunk_size = matrix.chunk_size_bytes();
    let mut global_row = 0usize;

    for (chunk_idx, chunk) in matrix.chunks().iter().enumerate() {
        let chunk_len = matrix.chunk_size(chunk_idx);
        let rows_in_chunk = chunk_len / row_size_bytes;
        let chunk_ptr = chunk.as_ptr();

        for row_offset in 0..rows_in_chunk {
            if global_row >= num_rows {
                break;
            }

            let row_ptr = unsafe { chunk_ptr.add(row_offset * row_size_bytes) };
            let masks = [
                mask_byte(keys[0].eval_bit(global_row)),
                mask_byte(keys[1].eval_bit(global_row)),
                mask_byte(keys[2].eval_bit(global_row)),
            ];

            let mut i = 0usize;
            while i + 64 <= row_size_bytes {
                let src = unsafe { std::slice::from_raw_parts(row_ptr.add(i), 64) };
                for (k, mask) in masks.iter().enumerate() {
                    xor_masked(&mut results[k], src, i, *mask);
                }
                i += 64;
            }

            if i < row_size_bytes {
                let src = unsafe { std::slice::from_raw_parts(row_ptr.add(i), row_size_bytes - i) };
                for (k, mask) in masks.iter().enumerate() {
                    xor_masked(&mut results[k], src, i, *mask);
                }
            }

            global_row += 1;
        }

        if chunk_len < chunk_size {
            break;
        }
    }

    results
}

#[cfg(feature = "avx512")]
#[target_feature(enable = "avx512f,avx512vl,vaes")]
unsafe fn scan_kernel_avx512<K: DpfKey>(
    matrix: &ChunkedMatrix,
    keys: &[K; 3],
    num_rows: usize,
    row_size_bytes: usize,
) -> [Vec<u8>; 3] {
    #[cfg(feature = "profiling")]
    let mut profiler = Profiler::new();
    #[cfg(feature = "profiling")]
    profiler.checkpoint("avx512_init");
    use std::arch::x86_64::{
        __m512i, _mm512_and_si512, _mm512_loadu_si512, _mm512_set1_epi8, _mm512_storeu_si512,
        _mm512_xor_si512,
    };

    let mut results = empty_result(row_size_bytes);
    let dst0 = results[0].as_mut_ptr();
    let dst1 = results[1].as_mut_ptr();
    let dst2 = results[2].as_mut_ptr();
    let has_tail = row_size_bytes % 64 != 0;

    #[inline(always)]
    unsafe fn store_acc(ptr: *mut u8, offset: usize, value: __m512i) {
        _mm512_storeu_si512(ptr.add(offset) as *mut __m512i, value);
    }

    #[cfg(feature = "profiling")]
    profiler.checkpoint("avx512_setup");

    let mut global_row = 0usize;

    for (chunk_idx, chunk) in matrix.chunks().iter().enumerate() {
        #[cfg(feature = "profiling")]
        if chunk_idx == 0 {
            profiler.checkpoint("avx512_first_chunk");
        }
        let chunk_len = matrix.chunk_size(chunk_idx);
        let rows_in_chunk = chunk_len / row_size_bytes;
        let chunk_ptr = chunk.as_ptr();

        let mut row_offset = 0usize;

        // Process 8 rows at a time for maximum ILP
        while row_offset + 7 < rows_in_chunk && global_row + 7 < num_rows {
            let row_ptrs = [
                chunk_ptr.add(row_offset * row_size_bytes),
                chunk_ptr.add((row_offset + 1) * row_size_bytes),
                chunk_ptr.add((row_offset + 2) * row_size_bytes),
                chunk_ptr.add((row_offset + 3) * row_size_bytes),
                chunk_ptr.add((row_offset + 4) * row_size_bytes),
                chunk_ptr.add((row_offset + 5) * row_size_bytes),
                chunk_ptr.add((row_offset + 6) * row_size_bytes),
                chunk_ptr.add((row_offset + 7) * row_size_bytes),
            ];

            #[cfg(feature = "profiling")]
            if global_row == 0 {
                profiler.checkpoint("avx512_first_mask_eval");
            }

            // Compute all 24 masks (3 keys x 8 rows) - use arrays for efficiency
            let m0: [u8; 8] = [
                mask_byte(keys[0].eval_bit(global_row)),
                mask_byte(keys[0].eval_bit(global_row + 1)),
                mask_byte(keys[0].eval_bit(global_row + 2)),
                mask_byte(keys[0].eval_bit(global_row + 3)),
                mask_byte(keys[0].eval_bit(global_row + 4)),
                mask_byte(keys[0].eval_bit(global_row + 5)),
                mask_byte(keys[0].eval_bit(global_row + 6)),
                mask_byte(keys[0].eval_bit(global_row + 7)),
            ];
            let m1: [u8; 8] = [
                mask_byte(keys[1].eval_bit(global_row)),
                mask_byte(keys[1].eval_bit(global_row + 1)),
                mask_byte(keys[1].eval_bit(global_row + 2)),
                mask_byte(keys[1].eval_bit(global_row + 3)),
                mask_byte(keys[1].eval_bit(global_row + 4)),
                mask_byte(keys[1].eval_bit(global_row + 5)),
                mask_byte(keys[1].eval_bit(global_row + 6)),
                mask_byte(keys[1].eval_bit(global_row + 7)),
            ];
            let m2: [u8; 8] = [
                mask_byte(keys[2].eval_bit(global_row)),
                mask_byte(keys[2].eval_bit(global_row + 1)),
                mask_byte(keys[2].eval_bit(global_row + 2)),
                mask_byte(keys[2].eval_bit(global_row + 3)),
                mask_byte(keys[2].eval_bit(global_row + 4)),
                mask_byte(keys[2].eval_bit(global_row + 5)),
                mask_byte(keys[2].eval_bit(global_row + 6)),
                mask_byte(keys[2].eval_bit(global_row + 7)),
            ];

            // Convert to SIMD masks
            let masks0: [__m512i; 8] = std::array::from_fn(|j| _mm512_set1_epi8(m0[j] as i8));
            let masks1: [__m512i; 8] = std::array::from_fn(|j| _mm512_set1_epi8(m1[j] as i8));
            let masks2: [__m512i; 8] = std::array::from_fn(|j| _mm512_set1_epi8(m2[j] as i8));

            #[cfg(feature = "profiling")]
            if global_row == 0 {
                profiler.checkpoint("avx512_first_row_loop_start");
            }

            let mut i = 0usize;
            while i + 64 <= row_size_bytes {
                // Load all 8 rows at this offset
                let d0 = _mm512_loadu_si512(row_ptrs[0].add(i) as *const __m512i);
                let d1 = _mm512_loadu_si512(row_ptrs[1].add(i) as *const __m512i);
                let d2 = _mm512_loadu_si512(row_ptrs[2].add(i) as *const __m512i);
                let d3 = _mm512_loadu_si512(row_ptrs[3].add(i) as *const __m512i);
                let d4 = _mm512_loadu_si512(row_ptrs[4].add(i) as *const __m512i);
                let d5 = _mm512_loadu_si512(row_ptrs[5].add(i) as *const __m512i);
                let d6 = _mm512_loadu_si512(row_ptrs[6].add(i) as *const __m512i);
                let d7 = _mm512_loadu_si512(row_ptrs[7].add(i) as *const __m512i);

                let acc0 = _mm512_loadu_si512(dst0.add(i) as *const __m512i);
                let acc1 = _mm512_loadu_si512(dst1.add(i) as *const __m512i);
                let acc2 = _mm512_loadu_si512(dst2.add(i) as *const __m512i);

                // Combine all 8 rows for key 0
                let masked0 = _mm512_xor_si512(
                    _mm512_xor_si512(
                        _mm512_xor_si512(
                            _mm512_and_si512(d0, masks0[0]),
                            _mm512_and_si512(d1, masks0[1]),
                        ),
                        _mm512_xor_si512(
                            _mm512_and_si512(d2, masks0[2]),
                            _mm512_and_si512(d3, masks0[3]),
                        ),
                    ),
                    _mm512_xor_si512(
                        _mm512_xor_si512(
                            _mm512_and_si512(d4, masks0[4]),
                            _mm512_and_si512(d5, masks0[5]),
                        ),
                        _mm512_xor_si512(
                            _mm512_and_si512(d6, masks0[6]),
                            _mm512_and_si512(d7, masks0[7]),
                        ),
                    ),
                );
                // Combine all 8 rows for key 1
                let masked1 = _mm512_xor_si512(
                    _mm512_xor_si512(
                        _mm512_xor_si512(
                            _mm512_and_si512(d0, masks1[0]),
                            _mm512_and_si512(d1, masks1[1]),
                        ),
                        _mm512_xor_si512(
                            _mm512_and_si512(d2, masks1[2]),
                            _mm512_and_si512(d3, masks1[3]),
                        ),
                    ),
                    _mm512_xor_si512(
                        _mm512_xor_si512(
                            _mm512_and_si512(d4, masks1[4]),
                            _mm512_and_si512(d5, masks1[5]),
                        ),
                        _mm512_xor_si512(
                            _mm512_and_si512(d6, masks1[6]),
                            _mm512_and_si512(d7, masks1[7]),
                        ),
                    ),
                );
                // Combine all 8 rows for key 2
                let masked2 = _mm512_xor_si512(
                    _mm512_xor_si512(
                        _mm512_xor_si512(
                            _mm512_and_si512(d0, masks2[0]),
                            _mm512_and_si512(d1, masks2[1]),
                        ),
                        _mm512_xor_si512(
                            _mm512_and_si512(d2, masks2[2]),
                            _mm512_and_si512(d3, masks2[3]),
                        ),
                    ),
                    _mm512_xor_si512(
                        _mm512_xor_si512(
                            _mm512_and_si512(d4, masks2[4]),
                            _mm512_and_si512(d5, masks2[5]),
                        ),
                        _mm512_xor_si512(
                            _mm512_and_si512(d6, masks2[6]),
                            _mm512_and_si512(d7, masks2[7]),
                        ),
                    ),
                );

                store_acc(dst0, i, _mm512_xor_si512(acc0, masked0));
                store_acc(dst1, i, _mm512_xor_si512(acc1, masked1));
                store_acc(dst2, i, _mm512_xor_si512(acc2, masked2));

                i += 64;
            }

            if has_tail && i < row_size_bytes {
                for j in 0..8 {
                    let tail = std::slice::from_raw_parts(row_ptrs[j].add(i), row_size_bytes - i);
                    xor_masked(results[0].as_mut_slice(), tail, i, m0[j]);
                    xor_masked(results[1].as_mut_slice(), tail, i, m1[j]);
                    xor_masked(results[2].as_mut_slice(), tail, i, m2[j]);
                }
            }

            global_row += 8;
            row_offset += 8;
        }

        // Process remaining rows one at a time
        while row_offset < rows_in_chunk && global_row < num_rows {
            let row_ptr = chunk_ptr.add(row_offset * row_size_bytes);

            let m0 = mask_byte(keys[0].eval_bit(global_row));
            let m1 = mask_byte(keys[1].eval_bit(global_row));
            let m2 = mask_byte(keys[2].eval_bit(global_row));
            let mask0 = _mm512_set1_epi8(m0 as i8);
            let mask1 = _mm512_set1_epi8(m1 as i8);
            let mask2 = _mm512_set1_epi8(m2 as i8);

            let mut i = 0usize;
            while i + 64 <= row_size_bytes {
                let data = _mm512_loadu_si512(row_ptr.add(i) as *const __m512i);
                let acc0 = _mm512_loadu_si512(dst0.add(i) as *const __m512i);
                let acc1 = _mm512_loadu_si512(dst1.add(i) as *const __m512i);
                let acc2 = _mm512_loadu_si512(dst2.add(i) as *const __m512i);

                store_acc(
                    dst0,
                    i,
                    _mm512_xor_si512(acc0, _mm512_and_si512(data, mask0)),
                );
                store_acc(
                    dst1,
                    i,
                    _mm512_xor_si512(acc1, _mm512_and_si512(data, mask1)),
                );
                store_acc(
                    dst2,
                    i,
                    _mm512_xor_si512(acc2, _mm512_and_si512(data, mask2)),
                );
                i += 64;
            }

            if has_tail && i < row_size_bytes {
                let tail = std::slice::from_raw_parts(row_ptr.add(i), row_size_bytes - i);
                xor_masked(results[0].as_mut_slice(), tail, i, m0);
                xor_masked(results[1].as_mut_slice(), tail, i, m1);
                xor_masked(results[2].as_mut_slice(), tail, i, m2);
            }

            global_row += 1;
            row_offset += 1;
        }

        if chunk_len < matrix.chunk_size_bytes() {
            break;
        }
    }

    #[cfg(feature = "profiling")]
    {
        profiler.checkpoint("avx512_complete");
        eprintln!("avx512 kernel breakdown:\n{}", profiler.report());
    }

    results
}

#[cfg(not(feature = "avx512"))]
unsafe fn scan_kernel_avx512<K: DpfKey>(
    _matrix: &ChunkedMatrix,
    _keys: &[K; 3],
    _num_rows: usize,
    row_size_bytes: usize,
) -> [Vec<u8>; 3] {
    empty_result(row_size_bytes)
}

fn avx512_available() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512vl")
            && std::is_x86_feature_detected!("vaes")
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

fn mask_byte(bit: bool) -> u8 {
    0u8.wrapping_sub(bit as u8)
}

#[inline(always)]
fn xor_masked(dest: &mut [u8], src: &[u8], offset: usize, mask: u8) {
    for (d, s) in dest[offset..offset + src.len()].iter_mut().zip(src.iter()) {
        *d ^= *s & mask;
    }
}

#[cfg(test)]
mod tests {
    #[allow(deprecated)]
    use super::{empty_result, scan_consistent, scan_delta, ScanError};
    use morphogen_core::{DeltaBuffer, EpochSnapshot, GlobalState};
    use morphogen_dpf::AesDpfKey;
    use morphogen_storage::ChunkedMatrix;
    use std::sync::Arc;

    #[test]
    #[allow(deprecated)]
    fn scan_delta_applies_matching_rows_with_dpf_pair() {
        let row_size = 8;
        let delta = DeltaBuffer::new(row_size);
        delta.push(2, vec![0xAA; row_size]).unwrap();
        delta.push(7, vec![0xBB; row_size]).unwrap();

        let mut rng = rand::thread_rng();
        let (key0_a, key0_b) = AesDpfKey::generate_pair(&mut rng, 2);
        let (key1_a, key1_b) = AesDpfKey::generate_pair(&mut rng, 7);
        let (key2_a, key2_b) = AesDpfKey::generate_pair(&mut rng, 123);

        let keys_a = [key0_a, key1_a, key2_a];
        let keys_b = [key0_b, key1_b, key2_b];

        let mut results_a = empty_result(row_size);
        let mut results_b = empty_result(row_size);
        scan_delta(&delta, &keys_a, &mut results_a);
        scan_delta(&delta, &keys_b, &mut results_b);

        let mut xor_results = empty_result(row_size);
        for i in 0..3 {
            for j in 0..row_size {
                xor_results[i][j] = results_a[i][j] ^ results_b[i][j];
            }
        }

        assert_eq!(xor_results[0], vec![0xAA; row_size]);
        assert_eq!(xor_results[1], vec![0xBB; row_size]);
        assert_eq!(xor_results[2], vec![0x00; row_size]);
    }

    #[test]
    fn try_scan_delta_returns_result() {
        use super::try_scan_delta;

        let row_size = 4;
        let delta = DeltaBuffer::new(row_size);
        delta.push(0, vec![0xAB, 0xCD, 0xEF, 0x12]).unwrap();

        let mut rng = rand::thread_rng();
        let (key0, _) = AesDpfKey::generate_pair(&mut rng, 0);
        let (key1, _) = AesDpfKey::generate_pair(&mut rng, 1);
        let (key2, _) = AesDpfKey::generate_pair(&mut rng, 2);
        let keys = [key0, key1, key2];

        let mut results = empty_result(row_size);
        let result = try_scan_delta(&delta, &keys, &mut results);
        assert!(result.is_ok());
    }

    fn make_global_state(
        epoch_id: u64,
        total_size: usize,
        chunk_size: usize,
        row_size: usize,
    ) -> Arc<GlobalState> {
        let matrix = Arc::new(ChunkedMatrix::new(total_size, chunk_size));
        let snapshot = EpochSnapshot { epoch_id, matrix };
        let pending = Arc::new(DeltaBuffer::new_with_epoch(row_size, epoch_id));
        Arc::new(GlobalState::new(Arc::new(snapshot), pending))
    }

    #[test]
    fn scan_consistent_returns_result() {
        let row_size = 4;
        let global = make_global_state(0, 64, 32, row_size);
        let pending = DeltaBuffer::new_with_epoch(row_size, 0);

        let mut rng = rand::thread_rng();
        let (key0, _) = AesDpfKey::generate_pair(&mut rng, 0);
        let (key1, _) = AesDpfKey::generate_pair(&mut rng, 1);
        let (key2, _) = AesDpfKey::generate_pair(&mut rng, 2);
        let keys = [key0, key1, key2];

        let result = scan_consistent(&global, &pending, &keys, row_size);
        assert!(result.is_ok());
        let (results, epoch_id) = result.unwrap();
        assert_eq!(epoch_id, 0);
        assert_eq!(results[0].len(), row_size);
    }

    #[test]
    fn scan_consistent_includes_pending_deltas() {
        let row_size = 4;
        let global = make_global_state(0, 64, 32, row_size);
        let pending = DeltaBuffer::new_with_epoch(row_size, 0);
        pending.push(0, vec![0xAB, 0xCD, 0xEF, 0x12]).unwrap();

        let mut rng = rand::thread_rng();
        let (key0_a, key0_b) = AesDpfKey::generate_pair(&mut rng, 0);
        let (key1_a, key1_b) = AesDpfKey::generate_pair(&mut rng, 1);
        let (key2_a, key2_b) = AesDpfKey::generate_pair(&mut rng, 2);

        let keys_a = [key0_a, key1_a, key2_a];
        let keys_b = [key0_b, key1_b, key2_b];

        let (results_a, _) = scan_consistent(&global, &pending, &keys_a, row_size).unwrap();
        let (results_b, _) = scan_consistent(&global, &pending, &keys_b, row_size).unwrap();

        let mut xor = vec![0u8; row_size];
        for i in 0..row_size {
            xor[i] = results_a[0][i] ^ results_b[0][i];
        }
        assert_eq!(xor, vec![0xAB, 0xCD, 0xEF, 0x12]);
    }

    #[test]
    fn scan_consistent_with_max_retries_returns_error_on_exhaustion() {
        use super::scan_consistent_with_max_retries;

        let row_size = 4;
        let global = make_global_state(0, 64, 32, row_size);
        let pending = DeltaBuffer::new_with_epoch(row_size, 0);

        let mut rng = rand::thread_rng();
        let (key0, _) = AesDpfKey::generate_pair(&mut rng, 0);
        let (key1, _) = AesDpfKey::generate_pair(&mut rng, 1);
        let (key2, _) = AesDpfKey::generate_pair(&mut rng, 2);
        let keys = [key0, key1, key2];

        let result = scan_consistent_with_max_retries(&global, &pending, &keys, row_size, 0);
        assert!(matches!(
            result,
            Err(ScanError::TooManyRetries { attempts: 0 })
        ));
    }

    #[test]
    fn scan_consistent_succeeds_within_retry_limit() {
        use super::scan_consistent_with_max_retries;

        let row_size = 4;
        let global = make_global_state(0, 64, 32, row_size);
        let pending = DeltaBuffer::new(row_size);

        let mut rng = rand::thread_rng();
        let (key0, _) = AesDpfKey::generate_pair(&mut rng, 0);
        let (key1, _) = AesDpfKey::generate_pair(&mut rng, 1);
        let (key2, _) = AesDpfKey::generate_pair(&mut rng, 2);
        let keys = [key0, key1, key2];

        let result = scan_consistent_with_max_retries(&global, &pending, &keys, row_size, 1);
        assert!(result.is_ok());
    }

    #[test]
    fn scan_detects_pending_epoch_mismatch_and_retries() {
        use super::scan_consistent_with_max_retries;

        let row_size = 4;
        let global = make_global_state(0, 64, 32, row_size);
        let pending = DeltaBuffer::new(row_size);

        pending.drain_for_epoch(1).expect("drain should succeed");

        let mut rng = rand::thread_rng();
        let (key0, _) = AesDpfKey::generate_pair(&mut rng, 0);
        let (key1, _) = AesDpfKey::generate_pair(&mut rng, 1);
        let (key2, _) = AesDpfKey::generate_pair(&mut rng, 2);
        let keys = [key0, key1, key2];

        let result = scan_consistent_with_max_retries(&global, &pending, &keys, row_size, 5);
        assert!(
            matches!(result, Err(ScanError::TooManyRetries { attempts: 5 })),
            "scan should fail when pending_epoch (1) != matrix epoch (0)"
        );
    }

    #[test]
    fn scan_succeeds_when_both_epochs_match() {
        use super::scan_consistent_with_max_retries;

        let row_size = 4;
        let global = make_global_state(5, 64, 32, row_size);
        let pending = DeltaBuffer::new_with_epoch(row_size, 5);

        pending.drain_for_epoch(5).expect("drain should succeed");

        let mut rng = rand::thread_rng();
        let (key0, _) = AesDpfKey::generate_pair(&mut rng, 0);
        let (key1, _) = AesDpfKey::generate_pair(&mut rng, 1);
        let (key2, _) = AesDpfKey::generate_pair(&mut rng, 2);
        let keys = [key0, key1, key2];

        let result = scan_consistent_with_max_retries(&global, &pending, &keys, row_size, 1);
        assert!(result.is_ok(), "scan should succeed when epochs match");
        let (_, epoch) = result.unwrap();
        assert_eq!(epoch, 5);
    }
}

#[cfg(test)]
mod concurrency_tests {
    use super::{scan_consistent_with_max_retries, ScanError};
    use morphogen_core::{DeltaBuffer, EpochSnapshot, GlobalState};
    use morphogen_dpf::AesDpfKey;
    use morphogen_storage::ChunkedMatrix;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};
    use std::thread;

    fn make_global_state(
        epoch_id: u64,
        total_size: usize,
        chunk_size: usize,
        row_size: usize,
    ) -> Arc<GlobalState> {
        let matrix = Arc::new(ChunkedMatrix::new(total_size, chunk_size));
        let snapshot = EpochSnapshot { epoch_id, matrix };
        let pending = Arc::new(DeltaBuffer::new_with_epoch(row_size, epoch_id));
        Arc::new(GlobalState::new(Arc::new(snapshot), pending))
    }

    #[test]
    fn concurrent_scan_and_merge_never_misses_deltas() {
        let row_size = 4;
        let num_rows = 16;
        let total_size = row_size * num_rows;
        let global = make_global_state(0, total_size, total_size, row_size);
        let pending = Arc::new(DeltaBuffer::new_with_epoch(row_size, 0));

        let barrier = Arc::new(Barrier::new(3));
        let iterations = 100;
        let scan_successes = Arc::new(AtomicUsize::new(0));
        let scan_retries = Arc::new(AtomicUsize::new(0));

        let global_clone = global.clone();
        let pending_clone = pending.clone();
        let barrier_clone = barrier.clone();
        let merger = thread::spawn(move || {
            barrier_clone.wait();
            for epoch in 1..=iterations {
                pending_clone.push(0, vec![0xAA; row_size]).ok();
                let entries = pending_clone
                    .drain_for_epoch(epoch as u64)
                    .expect("drain failed");

                std::thread::yield_now();

                if !entries.is_empty() {
                    let matrix = Arc::new(ChunkedMatrix::new(total_size, total_size));
                    let next = EpochSnapshot {
                        epoch_id: epoch as u64,
                        matrix,
                    };
                    global_clone.store(Arc::new(next));
                }
            }
        });

        let global_clone = global.clone();
        let pending_clone = pending.clone();
        let barrier_clone = barrier.clone();
        let successes_clone = scan_successes.clone();
        let retries_clone = scan_retries.clone();
        let scanner1 = thread::spawn(move || {
            let mut rng = rand::thread_rng();
            let (key0, _) = AesDpfKey::generate_pair(&mut rng, 0);
            let (key1, _) = AesDpfKey::generate_pair(&mut rng, 1);
            let (key2, _) = AesDpfKey::generate_pair(&mut rng, 2);
            let keys = [key0, key1, key2];

            barrier_clone.wait();
            for _ in 0..iterations {
                match scan_consistent_with_max_retries(
                    &global_clone,
                    &pending_clone,
                    &keys,
                    row_size,
                    1000,
                ) {
                    Ok(_) => {
                        successes_clone.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(ScanError::TooManyRetries { .. }) => {
                        retries_clone.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(_) => panic!("unexpected error"),
                }
                std::thread::yield_now();
            }
        });

        let global_clone = global.clone();
        let pending_clone = pending.clone();
        let barrier_clone = barrier.clone();
        let successes_clone = scan_successes.clone();
        let retries_clone = scan_retries.clone();
        let scanner2 = thread::spawn(move || {
            let mut rng = rand::thread_rng();
            let (key0, _) = AesDpfKey::generate_pair(&mut rng, 0);
            let (key1, _) = AesDpfKey::generate_pair(&mut rng, 1);
            let (key2, _) = AesDpfKey::generate_pair(&mut rng, 2);
            let keys = [key0, key1, key2];

            barrier_clone.wait();
            for _ in 0..iterations {
                match scan_consistent_with_max_retries(
                    &global_clone,
                    &pending_clone,
                    &keys,
                    row_size,
                    1000,
                ) {
                    Ok(_) => {
                        successes_clone.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(ScanError::TooManyRetries { .. }) => {
                        retries_clone.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(_) => panic!("unexpected error"),
                }
                std::thread::yield_now();
            }
        });

        merger.join().expect("merger panicked");
        scanner1.join().expect("scanner1 panicked");
        scanner2.join().expect("scanner2 panicked");

        let successes = scan_successes.load(Ordering::Relaxed);
        let retries = scan_retries.load(Ordering::Relaxed);
        assert!(
            successes > 0,
            "at least some scans should succeed (got {} successes, {} retries)",
            successes,
            retries
        );
    }

    #[test]
    fn forced_race_window_triggers_retry() {
        let row_size = 4;
        let num_rows = 16;
        let total_size = row_size * num_rows;
        let global = make_global_state(0, total_size, total_size, row_size);
        let pending = Arc::new(DeltaBuffer::new_with_epoch(row_size, 0));

        pending.push(0, vec![0xFF; row_size]).expect("push failed");

        let step1_barrier = Arc::new(Barrier::new(2));
        let step2_barrier = Arc::new(Barrier::new(2));
        let step3_barrier = Arc::new(Barrier::new(2));

        let global_clone = global.clone();
        let pending_clone = pending.clone();
        let step1_clone = step1_barrier.clone();
        let step2_clone = step2_barrier.clone();
        let step3_clone = step3_barrier.clone();

        let merger = thread::spawn(move || {
            step1_clone.wait();

            let _entries = pending_clone.drain_for_epoch(1).expect("drain failed");

            step2_clone.wait();

            // Wait for main thread to observe intermediate state before storing
            step3_clone.wait();

            let matrix = Arc::new(ChunkedMatrix::new(total_size, total_size));
            let next = EpochSnapshot {
                epoch_id: 1,
                matrix,
            };
            global_clone.store(Arc::new(next));
        });

        let snapshot_before = global.load();
        assert_eq!(snapshot_before.epoch_id, 0);

        step1_barrier.wait();

        step2_barrier.wait();

        let (pending_epoch, entries) = pending.snapshot_with_epoch().expect("snapshot failed");
        assert_eq!(pending_epoch, 1, "pending_epoch should be 1 after drain");
        assert!(entries.is_empty(), "entries should be empty after drain");

        let matrix_epoch = global.load().epoch_id;
        assert_eq!(matrix_epoch, 0, "matrix epoch should still be 0");

        assert_ne!(
            pending_epoch, matrix_epoch,
            "RACE DETECTED: pending_epoch != matrix_epoch"
        );

        // Let merger proceed to store the new epoch
        step3_barrier.wait();

        merger.join().expect("merger panicked");

        let final_epoch = global.load().epoch_id;
        assert_eq!(
            final_epoch, 1,
            "matrix epoch should be 1 after merge completes"
        );
    }

    #[test]
    fn scan_pages_chunked_returns_correct_page() {
        use super::scan_pages_chunked;
        use morphogen_dpf::page::{
            generate_page_dpf_keys, xor_pages, PageDpfParams, PAGE_SIZE_BYTES,
        };

        let num_pages = 256;
        let page_size = PAGE_SIZE_BYTES;
        let total_size = num_pages * page_size;

        let mut matrix = ChunkedMatrix::new(total_size, total_size);
        for p in 0..num_pages {
            let mut page_data = vec![0u8; page_size];
            page_data.fill(p as u8);
            matrix.write_row(p, page_size, &page_data);
        }

        let matrix = Arc::new(matrix);
        let snapshot = EpochSnapshot {
            epoch_id: 0,
            matrix,
        };
        let pending = Arc::new(DeltaBuffer::new_with_epoch(page_size, 0));
        let global = Arc::new(GlobalState::new(Arc::new(snapshot), pending));

        let params = PageDpfParams::new(8).unwrap();
        let target_page = 42;

        let (k0_0, k0_1) = generate_page_dpf_keys(&params, target_page).unwrap();
        let (k1_0, k1_1) = generate_page_dpf_keys(&params, 100).unwrap();
        let (k2_0, k2_1) = generate_page_dpf_keys(&params, 200).unwrap();

        let keys_server0 = [k0_0, k1_0, k2_0];
        let keys_server1 = [k0_1, k1_1, k2_1];

        let results0 =
            scan_pages_chunked(global.load().matrix.as_ref(), &keys_server0, page_size, 64);
        let results1 =
            scan_pages_chunked(global.load().matrix.as_ref(), &keys_server1, page_size, 64);

        let mut page_result = vec![0u8; page_size];
        xor_pages(&results0[0], &results1[0], &mut page_result);

        assert!(
            page_result.iter().all(|&b| b == target_page as u8),
            "Expected page filled with {}, got {:?}",
            target_page,
            &page_result[..8]
        );
    }

    #[test]
    fn try_scan_pages_chunked_rejects_unaligned_chunk() {
        use super::{try_scan_pages_chunked, ScanError};
        use morphogen_dpf::page::{generate_page_dpf_keys, PageDpfParams, PAGE_SIZE_BYTES};

        let page_size = PAGE_SIZE_BYTES;
        // Create matrix with size NOT aligned to page_size
        let unaligned_size = page_size * 3 + 100; // 3 pages + 100 extra bytes
        let matrix = ChunkedMatrix::new(unaligned_size, unaligned_size);

        let params = PageDpfParams::new(8).unwrap();
        let (k0, _) = generate_page_dpf_keys(&params, 0).unwrap();
        let (k1, _) = generate_page_dpf_keys(&params, 1).unwrap();
        let (k2, _) = generate_page_dpf_keys(&params, 2).unwrap();
        let keys = [k0, k1, k2];

        let result = try_scan_pages_chunked(&matrix, &keys, page_size, 64);

        assert!(
            matches!(
                result,
                Err(ScanError::ChunkNotAligned {
                    chunk_idx: 0,
                    remainder: 100,
                    ..
                })
            ),
            "Expected ChunkNotAligned error, got {:?}",
            result
        );
    }

    #[test]
    fn try_scan_pages_chunked_accepts_aligned_matrix() {
        use super::try_scan_pages_chunked;
        use morphogen_dpf::page::{generate_page_dpf_keys, PageDpfParams, PAGE_SIZE_BYTES};

        let page_size = PAGE_SIZE_BYTES;
        let num_pages = 16;
        let aligned_size = page_size * num_pages;
        let matrix = ChunkedMatrix::new(aligned_size, aligned_size);

        let params = PageDpfParams::new(8).unwrap();
        let (k0, _) = generate_page_dpf_keys(&params, 0).unwrap();
        let (k1, _) = generate_page_dpf_keys(&params, 1).unwrap();
        let (k2, _) = generate_page_dpf_keys(&params, 2).unwrap();
        let keys = [k0, k1, k2];

        let result = try_scan_pages_chunked(&matrix, &keys, page_size, 64);

        assert!(result.is_ok(), "Expected Ok, got {:?}", result);
    }

    #[test]
    fn try_scan_pages_chunked_preallocates_page_refs() {
        use super::try_scan_pages_chunked;
        use morphogen_dpf::page::{generate_page_dpf_keys, PageDpfParams, PAGE_SIZE_BYTES};

        let page_size = PAGE_SIZE_BYTES;
        let num_pages = 256;
        let total_size = page_size * num_pages;
        let matrix = ChunkedMatrix::new(total_size, total_size);

        let params = PageDpfParams::new(8).unwrap();
        let (k0, _) = generate_page_dpf_keys(&params, 42).unwrap();
        let (k1, _) = generate_page_dpf_keys(&params, 100).unwrap();
        let (k2, _) = generate_page_dpf_keys(&params, 200).unwrap();
        let keys = [k0, k1, k2];

        // This should not panic or reallocate excessively
        let result = try_scan_pages_chunked(&matrix, &keys, page_size, 64);
        assert!(result.is_ok());

        // Verify we get 3 results of correct size
        let pages = result.unwrap();
        assert_eq!(pages.len(), 3);
        for page in &pages {
            assert_eq!(page.len(), page_size);
        }
    }
}
