//! Page-level PIR using fss-rs DPF for true 2-server privacy.
//!
//! Instead of evaluating DPF at row granularity (where servers would learn the target),
//! we evaluate at page granularity. Each page contains multiple rows (e.g., 16 rows × 256B = 4KB).
//! The client receives the full page and extracts the target row locally.
//!
//! This provides computational privacy: servers cannot distinguish which page is being queried.
//!
//! # Security Model
//! - Privacy requires non-collusion between the two servers.
//! - Uses replicated DB model: both servers hold identical page data.
//! - BGI-style DPF with correction words (fss-rs implementation).

#[cfg(feature = "fss")]
use fss_rs::dpf::{Dpf, DpfImpl, PointFn};
#[cfg(feature = "fss")]
use fss_rs::group::byte::ByteGroup;
#[cfg(feature = "fss")]
use fss_rs::prg::Aes128MatyasMeyerOseasPrg;
#[cfg(feature = "fss")]
use fss_rs::Share;

pub const ROWS_PER_PAGE: usize = 16;
pub const ROW_SIZE_BYTES: usize = 256;
pub const PAGE_SIZE_BYTES: usize = ROWS_PER_PAGE * ROW_SIZE_BYTES; // 4096 = 4KB

/// Timing breakdown for DPF evaluation and accumulation.
#[derive(Debug, Clone)]
pub struct EvalTiming {
    /// Time spent evaluating the DPF (in nanoseconds)
    pub dpf_eval_ns: u64,
    /// Time spent XOR-accumulating pages (in nanoseconds)
    pub xor_accumulate_ns: u64,
    /// The resulting response page
    pub response: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageAddress {
    pub page_index: usize,
    pub row_offset: usize,
}

impl PageAddress {
    pub fn from_row_index(row_index: usize) -> Self {
        Self {
            page_index: row_index / ROWS_PER_PAGE,
            row_offset: row_index % ROWS_PER_PAGE,
        }
    }

    pub fn to_row_index(&self) -> usize {
        self.page_index * ROWS_PER_PAGE + self.row_offset
    }
}

#[derive(Debug)]
pub enum PageDpfError {
    InvalidKeyLength {
        expected: usize,
        actual: usize,
    },
    PageIndexTooLarge {
        page_index: usize,
        max_pages: usize,
    },
    InvalidDomainBits {
        domain_bits: usize,
        reason: &'static str,
    },
    OutputSizeMismatch {
        expected: usize,
        actual: usize,
    },
    FssError(String),
    InvalidKeyFormat(&'static str),
    CorrectionWordCountMismatch {
        expected: usize,
        actual: usize,
    },
}

impl std::fmt::Display for PageDpfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidKeyLength { expected, actual } => {
                write!(
                    f,
                    "Invalid key length: expected {}, got {}",
                    expected, actual
                )
            }
            Self::PageIndexTooLarge {
                page_index,
                max_pages,
            } => {
                write!(
                    f,
                    "Page index {} exceeds max pages {}",
                    page_index, max_pages
                )
            }
            Self::InvalidDomainBits {
                domain_bits,
                reason,
            } => {
                write!(f, "Invalid domain_bits {}: {}", domain_bits, reason)
            }
            Self::OutputSizeMismatch { expected, actual } => {
                write!(
                    f,
                    "Output size mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            Self::FssError(msg) => write!(f, "FSS error: {}", msg),
            Self::InvalidKeyFormat(reason) => {
                write!(f, "Invalid key format: {}", reason)
            }
            Self::CorrectionWordCountMismatch { expected, actual } => {
                write!(
                    f,
                    "Correction word count mismatch: expected {}, got {}",
                    expected, actual
                )
            }
        }
    }
}

impl std::error::Error for PageDpfError {}

#[cfg(feature = "fss")]
#[derive(Clone)]
pub struct PageDpfParams {
    pub prg_keys: [[u8; 16]; 2],
    pub domain_bits: usize,
}

#[cfg(feature = "fss")]
impl PageDpfParams {
    pub fn new(domain_bits: usize) -> Result<Self, PageDpfError> {
        if domain_bits == 0 {
            return Err(PageDpfError::InvalidDomainBits {
                domain_bits,
                reason: "must be > 0",
            });
        }
        let max_bits = (usize::BITS as usize).min(32);
        if domain_bits > max_bits {
            return Err(PageDpfError::InvalidDomainBits {
                domain_bits,
                reason: "exceeds platform usize width or 32-bit limit",
            });
        }
        Ok(Self {
            prg_keys: rand::random(),
            domain_bits,
        })
    }

    #[cfg(test)]
    pub fn new_unchecked(domain_bits: usize) -> Self {
        Self {
            prg_keys: rand::random(),
            domain_bits,
        }
    }

    pub fn max_pages(&self) -> usize {
        1usize
            .checked_shl(self.domain_bits as u32)
            .unwrap_or(usize::MAX)
    }

    pub fn input_bytes(&self) -> usize {
        match self.domain_bits {
            1..=8 => 1,
            9..=16 => 2,
            17..=24 => 3,
            _ => 4,
        }
    }

    fn create_dpf_1byte(&self) -> DpfImpl<1, 16, Aes128MatyasMeyerOseasPrg<16, 1, 2>> {
        let prg =
            Aes128MatyasMeyerOseasPrg::<16, 1, 2>::new(&std::array::from_fn(|i| &self.prg_keys[i]));
        if self.domain_bits == 8 {
            DpfImpl::<1, 16, _>::new(prg)
        } else {
            DpfImpl::<1, 16, _>::new_with_filter(prg, self.domain_bits)
        }
    }

    fn create_dpf_2byte(&self) -> DpfImpl<2, 16, Aes128MatyasMeyerOseasPrg<16, 1, 2>> {
        let prg =
            Aes128MatyasMeyerOseasPrg::<16, 1, 2>::new(&std::array::from_fn(|i| &self.prg_keys[i]));
        if self.domain_bits == 16 {
            DpfImpl::<2, 16, _>::new(prg)
        } else {
            DpfImpl::<2, 16, _>::new_with_filter(prg, self.domain_bits)
        }
    }

    fn create_dpf_3byte(&self) -> DpfImpl<3, 16, Aes128MatyasMeyerOseasPrg<16, 1, 2>> {
        let prg =
            Aes128MatyasMeyerOseasPrg::<16, 1, 2>::new(&std::array::from_fn(|i| &self.prg_keys[i]));
        if self.domain_bits == 24 {
            DpfImpl::<3, 16, _>::new(prg)
        } else {
            DpfImpl::<3, 16, _>::new_with_filter(prg, self.domain_bits)
        }
    }

    fn create_dpf_4byte(&self) -> DpfImpl<4, 16, Aes128MatyasMeyerOseasPrg<16, 1, 2>> {
        let prg =
            Aes128MatyasMeyerOseasPrg::<16, 1, 2>::new(&std::array::from_fn(|i| &self.prg_keys[i]));
        if self.domain_bits == 32 {
            DpfImpl::<4, 16, _>::new(prg)
        } else {
            DpfImpl::<4, 16, _>::new_with_filter(prg, self.domain_bits)
        }
    }

    fn encode_alpha_1byte(&self, target_page: usize) -> [u8; 1] {
        let shift = 8 - self.domain_bits;
        let shifted = target_page << shift;
        [shifted as u8]
    }

    fn encode_alpha_2byte(&self, target_page: usize) -> [u8; 2] {
        let shift = 16 - self.domain_bits;
        let shifted = target_page << shift;
        [(shifted >> 8) as u8, shifted as u8]
    }

    fn encode_alpha_3byte(&self, target_page: usize) -> [u8; 3] {
        let shift = 24 - self.domain_bits;
        let shifted = target_page << shift;
        [(shifted >> 16) as u8, (shifted >> 8) as u8, shifted as u8]
    }

    fn encode_alpha_4byte(&self, target_page: usize) -> [u8; 4] {
        let shift = 32 - self.domain_bits;
        let shifted = target_page << shift;
        [
            (shifted >> 24) as u8,
            (shifted >> 16) as u8,
            (shifted >> 8) as u8,
            shifted as u8,
        ]
    }
}

#[cfg(feature = "fss")]
pub fn generate_page_dpf_keys(
    params: &PageDpfParams,
    target_page: usize,
) -> Result<(PageDpfKey, PageDpfKey), PageDpfError> {
    let max_pages = params.max_pages();
    if target_page >= max_pages {
        return Err(PageDpfError::PageIndexTooLarge {
            page_index: target_page,
            max_pages,
        });
    }

    let s0s: [[u8; 16]; 2] = rand::random();
    let beta = ByteGroup([0xFF; 16]);

    let (cws, cw_np1, s0s_out) = match params.input_bytes() {
        1 => {
            let dpf = params.create_dpf_1byte();
            let alpha = params.encode_alpha_1byte(target_page);
            let point_fn = PointFn { alpha, beta };
            let share = dpf.gen(&point_fn, [&s0s[0], &s0s[1]]);
            (share.cws, share.cw_np1, share.s0s)
        }
        2 => {
            let dpf = params.create_dpf_2byte();
            let alpha = params.encode_alpha_2byte(target_page);
            let point_fn = PointFn { alpha, beta };
            let share = dpf.gen(&point_fn, [&s0s[0], &s0s[1]]);
            (share.cws, share.cw_np1, share.s0s)
        }
        3 => {
            let dpf = params.create_dpf_3byte();
            let alpha = params.encode_alpha_3byte(target_page);
            let point_fn = PointFn { alpha, beta };
            let share = dpf.gen(&point_fn, [&s0s[0], &s0s[1]]);
            (share.cws, share.cw_np1, share.s0s)
        }
        _ => {
            let dpf = params.create_dpf_4byte();
            let alpha = params.encode_alpha_4byte(target_page);
            let point_fn = PointFn { alpha, beta };
            let share = dpf.gen(&point_fn, [&s0s[0], &s0s[1]]);
            (share.cws, share.cw_np1, share.s0s)
        }
    };

    let k0 = PageDpfKey {
        share: Share {
            s0s: vec![s0s_out[0]],
            cws: cws.clone(),
            cw_np1: cw_np1.clone(),
        },
        is_party1: false,
        params: params.clone(),
    };

    let k1 = PageDpfKey {
        share: Share {
            s0s: vec![s0s_out[1]],
            cws,
            cw_np1,
        },
        is_party1: true,
        params: params.clone(),
    };

    Ok((k0, k1))
}

#[cfg(feature = "fss")]
pub struct PageDpfKey {
    share: Share<16, ByteGroup<16>>,
    is_party1: bool,
    params: PageDpfParams,
}

#[cfg(feature = "fss")]
impl PageDpfKey {
    pub fn is_party1(&self) -> bool {
        self.is_party1
    }

    pub fn domain_bits(&self) -> usize {
        self.params.domain_bits
    }

    /// Returns the PRG keys embedded in this DPF key.
    ///
    /// Servers should validate these match their configuration to prevent
    /// silent incorrect results from parameter mismatch.
    pub fn prg_keys(&self) -> &[[u8; 16]; 2] {
        &self.params.prg_keys
    }

    pub fn full_eval(&self, output: &mut [ByteGroup<16>]) -> Result<(), PageDpfError> {
        let expected = self.params.max_pages();
        if output.len() != expected {
            return Err(PageDpfError::OutputSizeMismatch {
                expected,
                actual: output.len(),
            });
        }

        let mut refs: Vec<&mut ByteGroup<16>> = output.iter_mut().collect();
        match self.params.input_bytes() {
            1 => {
                let dpf = self.params.create_dpf_1byte();
                dpf.full_eval(self.is_party1, &self.share, &mut refs);
            }
            2 => {
                let dpf = self.params.create_dpf_2byte();
                dpf.full_eval(self.is_party1, &self.share, &mut refs);
            }
            3 => {
                let dpf = self.params.create_dpf_3byte();
                dpf.full_eval(self.is_party1, &self.share, &mut refs);
            }
            _ => {
                let dpf = self.params.create_dpf_4byte();
                dpf.full_eval(self.is_party1, &self.share, &mut refs);
            }
        }
        Ok(())
    }

    /// Streaming evaluation: evaluates DPF and accumulates masked pages in one pass.
    /// Avoids O(N) memory allocation for DPF outputs.
    ///
    /// Returns a response page where only the target page's data survives the masking.
    #[deprecated(note = "use eval_and_accumulate_chunked for O(chunk_size) memory instead of O(N)")]
    pub fn eval_and_accumulate<'a>(&self, pages: impl Iterator<Item = &'a [u8]>) -> Vec<u8> {
        use fss_rs::group::Group;

        let num_pages = self.params.max_pages();
        let mut response = vec![0u8; PAGE_SIZE_BYTES];

        let mut dpf_output = vec![ByteGroup::zero(); num_pages];
        let mut refs: Vec<&mut ByteGroup<16>> = dpf_output.iter_mut().collect();

        match self.params.input_bytes() {
            1 => {
                let dpf = self.params.create_dpf_1byte();
                dpf.full_eval(self.is_party1, &self.share, &mut refs);
            }
            2 => {
                let dpf = self.params.create_dpf_2byte();
                dpf.full_eval(self.is_party1, &self.share, &mut refs);
            }
            3 => {
                let dpf = self.params.create_dpf_3byte();
                dpf.full_eval(self.is_party1, &self.share, &mut refs);
            }
            _ => {
                let dpf = self.params.create_dpf_4byte();
                dpf.full_eval(self.is_party1, &self.share, &mut refs);
            }
        }

        for (page_idx, page_data) in pages.enumerate() {
            if page_idx >= num_pages {
                break;
            }
            let mask = dpf_output[page_idx].0[0];
            if mask != 0 {
                xor_page_masked(&mut response, page_data, mask);
            }
        }

        response
    }

    /// Chunked streaming evaluation: evaluates DPF in chunks and accumulates masked pages.
    /// Uses O(chunk_size) memory instead of O(N) for the DPF output buffer.
    ///
    /// # Arguments
    /// * `pages` - Slice of page data references (indexed by page number)
    /// * `chunk_size` - Number of DPF evaluations per chunk (e.g., 4096)
    ///
    /// Returns a response page where only the target page's data survives the masking.
    pub fn eval_and_accumulate_chunked(&self, pages: &[&[u8]], chunk_size: usize) -> Vec<u8> {
        use fss_rs::dpf::Dpf;
        use fss_rs::group::Group;

        let num_pages = self.params.max_pages().min(pages.len());
        let mut response = vec![0u8; PAGE_SIZE_BYTES];
        let chunk_size = chunk_size.max(1);

        let mut dpf_chunk = vec![ByteGroup::zero(); chunk_size];

        let mut chunk_start = 0;
        while chunk_start < num_pages {
            let chunk_end = (chunk_start + chunk_size).min(num_pages);
            let this_chunk_len = chunk_end - chunk_start;

            let output = &mut dpf_chunk[..this_chunk_len];
            for g in output.iter_mut() {
                *g = ByteGroup::zero();
            }

            match self.params.input_bytes() {
                1 => {
                    let dpf = self.params.create_dpf_1byte();
                    dpf.eval_range(self.is_party1, &self.share, chunk_start, output);
                }
                2 => {
                    let dpf = self.params.create_dpf_2byte();
                    dpf.eval_range(self.is_party1, &self.share, chunk_start, output);
                }
                3 => {
                    let dpf = self.params.create_dpf_3byte();
                    dpf.eval_range(self.is_party1, &self.share, chunk_start, output);
                }
                _ => {
                    let dpf = self.params.create_dpf_4byte();
                    dpf.eval_range(self.is_party1, &self.share, chunk_start, output);
                }
            }

            for (i, dpf_val) in output.iter().enumerate() {
                let page_idx = chunk_start + i;
                let mask = dpf_val.0[0];
                if mask != 0 {
                    let page_data = pages[page_idx];
                    xor_page_masked(&mut response, page_data, mask);
                }
            }

            chunk_start = chunk_end;
        }

        response
    }
}

#[inline(always)]
fn xor_page_masked(response: &mut [u8], page_data: &[u8], mask: u8) {
    let len = page_data.len().min(PAGE_SIZE_BYTES).min(response.len());
    for i in 0..len {
        response[i] ^= page_data[i] & mask;
    }
}

#[cfg(feature = "fss")]
impl PageDpfKey {
    /// Instrumented version of eval_and_accumulate_chunked that returns timing breakdown.
    ///
    /// Use this to measure bottleneck split between DPF evaluation and XOR accumulation.
    pub fn eval_and_accumulate_chunked_timed(
        &self,
        pages: &[&[u8]],
        chunk_size: usize,
    ) -> EvalTiming {
        use fss_rs::dpf::Dpf;
        use fss_rs::group::Group;
        use std::time::Instant;

        let num_pages = self.params.max_pages().min(pages.len());
        let mut response = vec![0u8; PAGE_SIZE_BYTES];
        let chunk_size = chunk_size.max(1);

        let mut dpf_chunk = vec![ByteGroup::zero(); chunk_size];
        let mut total_dpf_ns: u64 = 0;
        let mut total_xor_ns: u64 = 0;

        let mut chunk_start = 0;
        while chunk_start < num_pages {
            let chunk_end = (chunk_start + chunk_size).min(num_pages);
            let this_chunk_len = chunk_end - chunk_start;

            let output = &mut dpf_chunk[..this_chunk_len];
            for g in output.iter_mut() {
                *g = ByteGroup::zero();
            }

            let dpf_start = Instant::now();
            match self.params.input_bytes() {
                1 => {
                    let dpf = self.params.create_dpf_1byte();
                    dpf.eval_range(self.is_party1, &self.share, chunk_start, output);
                }
                2 => {
                    let dpf = self.params.create_dpf_2byte();
                    dpf.eval_range(self.is_party1, &self.share, chunk_start, output);
                }
                3 => {
                    let dpf = self.params.create_dpf_3byte();
                    dpf.eval_range(self.is_party1, &self.share, chunk_start, output);
                }
                _ => {
                    let dpf = self.params.create_dpf_4byte();
                    dpf.eval_range(self.is_party1, &self.share, chunk_start, output);
                }
            }
            total_dpf_ns += dpf_start.elapsed().as_nanos() as u64;

            let xor_start = Instant::now();
            for (i, dpf_val) in output.iter().enumerate() {
                let page_idx = chunk_start + i;
                let mask = dpf_val.0[0];
                if mask != 0 {
                    let page_data = pages[page_idx];
                    xor_page_masked(&mut response, page_data, mask);
                }
            }
            total_xor_ns += xor_start.elapsed().as_nanos() as u64;

            chunk_start = chunk_end;
        }

        EvalTiming {
            dpf_eval_ns: total_dpf_ns,
            xor_accumulate_ns: total_xor_ns,
            response,
        }
    }

    /// Serialize the key to bytes.
    ///
    /// Format:
    /// - 1 byte: domain_bits
    /// - 32 bytes: prg_keys (2 × 16 bytes)
    /// - 1 byte: is_party1 (0 or 1)
    /// - 16 bytes: s0 seed
    /// - 16 bytes: cw_np1 (ByteGroup)
    /// - For each of domain_bits correction words:
    ///   - 16 bytes: s
    ///   - 1 byte: (tl as u8) | ((tr as u8) << 1)
    ///
    /// Total size: 66 + 17 * domain_bits bytes
    /// For 25-bit domain: 66 + 17 * 25 = 491 bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let n = self.params.domain_bits;
        let size = 66 + 17 * n;
        let mut out = Vec::with_capacity(size);

        out.push(n as u8);

        out.extend_from_slice(&self.params.prg_keys[0]);
        out.extend_from_slice(&self.params.prg_keys[1]);

        out.push(self.is_party1 as u8);

        out.extend_from_slice(&self.share.s0s[0]);

        out.extend_from_slice(&self.share.cw_np1.0);

        for cw in &self.share.cws {
            out.extend_from_slice(&cw.s);
            let flags = (cw.tl as u8) | ((cw.tr as u8) << 1);
            out.push(flags);
        }

        out
    }

    /// Deserialize a key from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, PageDpfError> {
        use fss_rs::group::Group;
        use fss_rs::Cw;

        if bytes.is_empty() {
            return Err(PageDpfError::InvalidKeyFormat("empty input"));
        }

        let domain_bits = bytes[0] as usize;
        if domain_bits == 0 || domain_bits > 32 {
            return Err(PageDpfError::InvalidDomainBits {
                domain_bits,
                reason: "must be 1-32",
            });
        }

        let expected_size = 66 + 17 * domain_bits;
        if bytes.len() != expected_size {
            return Err(PageDpfError::InvalidKeyLength {
                expected: expected_size,
                actual: bytes.len(),
            });
        }

        let mut pos = 1;

        let mut prg_keys = [[0u8; 16]; 2];
        prg_keys[0].copy_from_slice(&bytes[pos..pos + 16]);
        pos += 16;
        prg_keys[1].copy_from_slice(&bytes[pos..pos + 16]);
        pos += 16;

        let is_party1 = bytes[pos] != 0;
        pos += 1;

        let mut s0 = [0u8; 16];
        s0.copy_from_slice(&bytes[pos..pos + 16]);
        pos += 16;

        let mut cw_np1_bytes = [0u8; 16];
        cw_np1_bytes.copy_from_slice(&bytes[pos..pos + 16]);
        let cw_np1 = ByteGroup(cw_np1_bytes);
        pos += 16;

        let mut cws = Vec::with_capacity(domain_bits);
        for _ in 0..domain_bits {
            let mut s = [0u8; 16];
            s.copy_from_slice(&bytes[pos..pos + 16]);
            pos += 16;

            let flags = bytes[pos];
            pos += 1;

            let tl = (flags & 1) != 0;
            let tr = (flags & 2) != 0;

            cws.push(Cw {
                s,
                v: ByteGroup::zero(),
                tl,
                tr,
            });
        }

        let params = PageDpfParams {
            prg_keys,
            domain_bits,
        };

        Ok(Self {
            share: Share {
                s0s: vec![s0],
                cws,
                cw_np1,
            },
            is_party1,
            params,
        })
    }

    /// Returns the serialized key size for a given domain_bits.
    pub fn serialized_size(domain_bits: usize) -> usize {
        66 + 17 * domain_bits
    }
}

pub fn extract_row_from_page(page: &[u8], row_offset: usize) -> Option<&[u8]> {
    if row_offset >= ROWS_PER_PAGE {
        return None;
    }
    let start = row_offset * ROW_SIZE_BYTES;
    let end = start + ROW_SIZE_BYTES;
    if end > page.len() {
        return None;
    }
    Some(&page[start..end])
}

pub fn xor_pages(page0: &[u8], page1: &[u8], output: &mut [u8]) {
    assert_eq!(page0.len(), page1.len());
    assert_eq!(page0.len(), output.len());
    for i in 0..output.len() {
        output[i] = page0[i] ^ page1[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_address_roundtrip() {
        for row in [0, 1, 15, 16, 17, 100, 1000, 65535] {
            let addr = PageAddress::from_row_index(row);
            assert_eq!(addr.to_row_index(), row);
        }
    }

    #[test]
    fn page_address_calculation() {
        let addr = PageAddress::from_row_index(42);
        assert_eq!(addr.page_index, 2);
        assert_eq!(addr.row_offset, 10);

        let addr = PageAddress::from_row_index(0);
        assert_eq!(addr.page_index, 0);
        assert_eq!(addr.row_offset, 0);

        let addr = PageAddress::from_row_index(15);
        assert_eq!(addr.page_index, 0);
        assert_eq!(addr.row_offset, 15);

        let addr = PageAddress::from_row_index(16);
        assert_eq!(addr.page_index, 1);
        assert_eq!(addr.row_offset, 0);
    }

    #[test]
    fn extract_row_from_page_works() {
        let mut page = vec![0u8; PAGE_SIZE_BYTES];
        for i in 0..ROWS_PER_PAGE {
            page[i * ROW_SIZE_BYTES] = i as u8;
        }

        for i in 0..ROWS_PER_PAGE {
            let row = extract_row_from_page(&page, i).unwrap();
            assert_eq!(row.len(), ROW_SIZE_BYTES);
            assert_eq!(row[0], i as u8);
        }

        assert!(extract_row_from_page(&page, ROWS_PER_PAGE).is_none());
    }

    #[test]
    fn xor_pages_works() {
        let page0 = vec![0xAA; PAGE_SIZE_BYTES];
        let page1 = vec![0x55; PAGE_SIZE_BYTES];
        let mut result = vec![0u8; PAGE_SIZE_BYTES];

        xor_pages(&page0, &page1, &mut result);

        assert!(result.iter().all(|&b| b == 0xFF));
    }
}

#[cfg(all(test, feature = "fss"))]
mod fss_page_tests {
    use super::*;
    use fss_rs::group::Group;

    #[test]
    fn page_dpf_params_creation() {
        let params = PageDpfParams::new(10).unwrap();
        assert_eq!(params.max_pages(), 1024);
        assert_eq!(params.domain_bits, 10);
    }

    #[test]
    fn page_dpf_params_rejects_invalid() {
        assert!(PageDpfParams::new(0).is_err());
        assert!(PageDpfParams::new(33).is_err());
    }

    #[test]
    fn page_dpf_key_generation() {
        let params = PageDpfParams::new(8).unwrap();
        let (k0, k1) = generate_page_dpf_keys(&params, 42).unwrap();

        assert!(!k0.is_party1());
        assert!(k1.is_party1());
        assert_eq!(k0.domain_bits(), 8);
        assert_eq!(k1.domain_bits(), 8);
    }

    #[test]
    fn page_dpf_key_generation_rejects_out_of_range() {
        let params = PageDpfParams::new(8).unwrap();
        let result = generate_page_dpf_keys(&params, 300);
        assert!(matches!(
            result,
            Err(PageDpfError::PageIndexTooLarge { .. })
        ));
    }

    #[test]
    fn page_dpf_key_serialization_roundtrip() {
        let params = PageDpfParams::new(10).unwrap();
        let (k0, k1) = generate_page_dpf_keys(&params, 500).unwrap();

        let bytes0 = k0.to_bytes();
        let bytes1 = k1.to_bytes();

        assert_eq!(bytes0.len(), PageDpfKey::serialized_size(10));
        assert_eq!(bytes1.len(), PageDpfKey::serialized_size(10));
        assert_eq!(bytes0.len(), 66 + 17 * 10); // 236 bytes

        let restored0 = PageDpfKey::from_bytes(&bytes0).unwrap();
        let restored1 = PageDpfKey::from_bytes(&bytes1).unwrap();

        assert_eq!(restored0.is_party1(), k0.is_party1());
        assert_eq!(restored1.is_party1(), k1.is_party1());
        assert_eq!(restored0.domain_bits(), k0.domain_bits());
        assert_eq!(restored1.domain_bits(), k1.domain_bits());

        let num_pages = params.max_pages();
        let mut orig0 = vec![ByteGroup::zero(); num_pages];
        let mut orig1 = vec![ByteGroup::zero(); num_pages];
        let mut rest0 = vec![ByteGroup::zero(); num_pages];
        let mut rest1 = vec![ByteGroup::zero(); num_pages];

        k0.full_eval(&mut orig0).unwrap();
        k1.full_eval(&mut orig1).unwrap();
        restored0.full_eval(&mut rest0).unwrap();
        restored1.full_eval(&mut rest1).unwrap();

        assert_eq!(orig0, rest0, "party0 eval mismatch after roundtrip");
        assert_eq!(orig1, rest1, "party1 eval mismatch after roundtrip");
    }

    #[test]
    fn page_dpf_key_serialization_25bit() {
        let params = PageDpfParams::new(25).unwrap();
        let (k0, _) = generate_page_dpf_keys(&params, 1_000_000).unwrap();

        let bytes = k0.to_bytes();
        assert_eq!(bytes.len(), 66 + 17 * 25); // 491 bytes

        let restored = PageDpfKey::from_bytes(&bytes).unwrap();
        assert_eq!(restored.domain_bits(), 25);
        assert!(!restored.is_party1());
    }

    #[test]
    fn page_dpf_key_from_bytes_rejects_invalid() {
        assert!(matches!(
            PageDpfKey::from_bytes(&[]),
            Err(PageDpfError::InvalidKeyFormat(_))
        ));

        assert!(matches!(
            PageDpfKey::from_bytes(&[0]),
            Err(PageDpfError::InvalidDomainBits { .. })
        ));

        assert!(matches!(
            PageDpfKey::from_bytes(&[33]),
            Err(PageDpfError::InvalidDomainBits { .. })
        ));

        let mut short = vec![8u8]; // domain_bits = 8
        short.extend_from_slice(&[0u8; 50]); // too short
        assert!(matches!(
            PageDpfKey::from_bytes(&short),
            Err(PageDpfError::InvalidKeyLength { .. })
        ));
    }

    #[test]
    fn page_dpf_full_eval_correctness_8bit() {
        let params = PageDpfParams::new(8).unwrap();
        let target_page = 42;
        let (k0, k1) = generate_page_dpf_keys(&params, target_page).unwrap();

        let mut output0 = vec![ByteGroup::zero(); 256];
        let mut output1 = vec![ByteGroup::zero(); 256];

        k0.full_eval(&mut output0).unwrap();
        k1.full_eval(&mut output1).unwrap();

        for i in 0..256 {
            let mut xor = [0u8; 16];
            for (j, byte) in xor.iter_mut().enumerate() {
                *byte = output0[i].0[j] ^ output1[i].0[j];
            }
            if i == target_page {
                assert_eq!(xor, [0xFF; 16], "XOR at target page {} should be 0xFF", i);
            } else {
                assert_eq!(
                    xor, [0x00; 16],
                    "XOR at non-target page {} should be 0x00",
                    i
                );
            }
        }
    }

    #[test]
    fn page_dpf_full_eval_correctness_10bit() {
        let params = PageDpfParams::new(10).unwrap();
        let target_page = 500;
        let (k0, k1) = generate_page_dpf_keys(&params, target_page).unwrap();

        let num_pages = params.max_pages();
        let mut output0 = vec![ByteGroup::zero(); num_pages];
        let mut output1 = vec![ByteGroup::zero(); num_pages];

        k0.full_eval(&mut output0).unwrap();
        k1.full_eval(&mut output1).unwrap();

        for i in [0, 1, 100, target_page, 1000] {
            let mut xor = [0u8; 16];
            for (j, byte) in xor.iter_mut().enumerate() {
                *byte = output0[i].0[j] ^ output1[i].0[j];
            }
            if i == target_page {
                assert_eq!(xor, [0xFF; 16], "XOR at target page {} should be 0xFF", i);
            } else {
                assert_eq!(
                    xor, [0x00; 16],
                    "XOR at non-target page {} should be 0x00",
                    i
                );
            }
        }
    }

    #[test]
    fn page_dpf_full_eval_correctness_16bit() {
        let params = PageDpfParams::new(16).unwrap();
        let target_page = 500;
        let (k0, k1) = generate_page_dpf_keys(&params, target_page).unwrap();

        let num_pages = params.max_pages();
        let mut output0 = vec![ByteGroup::zero(); num_pages];
        let mut output1 = vec![ByteGroup::zero(); num_pages];

        k0.full_eval(&mut output0).unwrap();
        k1.full_eval(&mut output1).unwrap();

        for i in [0, 1, 100, target_page, 1000, 65535] {
            let mut xor = [0u8; 16];
            for (j, byte) in xor.iter_mut().enumerate() {
                *byte = output0[i].0[j] ^ output1[i].0[j];
            }
            if i == target_page {
                assert_eq!(xor, [0xFF; 16], "XOR at target page {} should be 0xFF", i);
            } else {
                assert_eq!(
                    xor, [0x00; 16],
                    "XOR at non-target page {} should be 0x00",
                    i
                );
            }
        }
    }

    #[test]
    fn page_dpf_full_eval_correctness_20bit() {
        let params = PageDpfParams::new(20).unwrap();
        let target_page = 500_000;
        let (k0, k1) = generate_page_dpf_keys(&params, target_page).unwrap();

        let num_pages = params.max_pages();
        let mut output0 = vec![ByteGroup::zero(); num_pages];
        let mut output1 = vec![ByteGroup::zero(); num_pages];

        k0.full_eval(&mut output0).unwrap();
        k1.full_eval(&mut output1).unwrap();

        for i in [0, 1, 100, target_page, 1_000_000] {
            let mut xor = [0u8; 16];
            for (j, byte) in xor.iter_mut().enumerate() {
                *byte = output0[i].0[j] ^ output1[i].0[j];
            }
            if i == target_page {
                assert_eq!(xor, [0xFF; 16], "XOR at target page {} should be 0xFF", i);
            } else {
                assert_eq!(
                    xor, [0x00; 16],
                    "XOR at non-target page {} should be 0x00",
                    i
                );
            }
        }
    }

    #[test]
    fn page_dpf_full_eval_correctness_25bit() {
        let params = PageDpfParams::new(25).unwrap();
        let target_page = 27_000_000;
        let (k0, k1) = generate_page_dpf_keys(&params, target_page).unwrap();

        let num_pages = params.max_pages();
        let mut output0 = vec![ByteGroup::zero(); num_pages];
        let mut output1 = vec![ByteGroup::zero(); num_pages];

        k0.full_eval(&mut output0).unwrap();
        k1.full_eval(&mut output1).unwrap();

        for i in [0, 1, 100, target_page, 30_000_000] {
            let mut xor = [0u8; 16];
            for (j, byte) in xor.iter_mut().enumerate() {
                *byte = output0[i].0[j] ^ output1[i].0[j];
            }
            if i == target_page {
                assert_eq!(xor, [0xFF; 16], "XOR at target page {} should be 0xFF", i);
            } else {
                assert_eq!(
                    xor, [0x00; 16],
                    "XOR at non-target page {} should be 0x00",
                    i
                );
            }
        }
    }

    #[test]
    fn page_dpf_full_eval_rejects_wrong_output_size() {
        let params = PageDpfParams::new(8).unwrap();
        let (k0, _) = generate_page_dpf_keys(&params, 42).unwrap();

        let mut output = vec![ByteGroup::zero(); 100];
        let result = k0.full_eval(&mut output);
        assert!(matches!(
            result,
            Err(PageDpfError::OutputSizeMismatch { .. })
        ));
    }

    #[test]
    fn page_dpf_boundary_domain_sizes() {
        for domain_bits in [2, 7, 9, 15, 17, 24] {
            let params = PageDpfParams::new(domain_bits).unwrap();
            let max = params.max_pages();

            let target_first = 0;
            let target_last = max - 1;
            let target_mid = max / 2;

            for target in [target_first, target_mid, target_last] {
                let (k0, k1) = generate_page_dpf_keys(&params, target).unwrap();

                let mut output0 = vec![ByteGroup::zero(); max];
                let mut output1 = vec![ByteGroup::zero(); max];

                k0.full_eval(&mut output0).unwrap();
                k1.full_eval(&mut output1).unwrap();

                let mut xor_target = [0u8; 16];
                for (j, byte) in xor_target.iter_mut().enumerate() {
                    *byte = output0[target].0[j] ^ output1[target].0[j];
                }
                assert_eq!(
                    xor_target, [0xFF; 16],
                    "domain_bits={}, target={}: XOR should be 0xFF",
                    domain_bits, target
                );

                let check_idx = if target == 0 { 1 } else { 0 };
                let mut xor_other = [0u8; 16];
                for (j, byte) in xor_other.iter_mut().enumerate() {
                    *byte = output0[check_idx].0[j] ^ output1[check_idx].0[j];
                }
                assert_eq!(
                    xor_other, [0x00; 16],
                    "domain_bits={}, non-target={}: XOR should be 0x00",
                    domain_bits, check_idx
                );
            }
        }
    }

    #[test]
    #[allow(deprecated)]
    fn page_dpf_streaming_eval_correctness() {
        let params = PageDpfParams::new(8).unwrap();
        let target_page = 42;
        let (k0, k1) = generate_page_dpf_keys(&params, target_page).unwrap();

        let num_pages = params.max_pages();
        let mut matrix = vec![[0u8; PAGE_SIZE_BYTES]; num_pages];
        for (p, page) in matrix.iter_mut().enumerate() {
            page.fill(p as u8);
        }

        let response0 = k0.eval_and_accumulate(matrix.iter().map(|p| p.as_slice()));
        let response1 = k1.eval_and_accumulate(matrix.iter().map(|p| p.as_slice()));

        let mut result = vec![0u8; PAGE_SIZE_BYTES];
        xor_pages(&response0, &response1, &mut result);

        assert!(
            result.iter().all(|&b| b == target_page as u8),
            "Expected page filled with {}, got {:?}",
            target_page,
            &result[..8]
        );
    }

    #[test]
    fn page_dpf_chunked_eval_correctness() {
        let params = PageDpfParams::new(8).unwrap();
        let target_page = 42;
        let (k0, k1) = generate_page_dpf_keys(&params, target_page).unwrap();

        let num_pages = params.max_pages();
        let mut matrix = vec![[0u8; PAGE_SIZE_BYTES]; num_pages];
        for (p, page) in matrix.iter_mut().enumerate() {
            page.fill(p as u8);
        }

        let pages: Vec<&[u8]> = matrix.iter().map(|p| p.as_slice()).collect();

        let response0 = k0.eval_and_accumulate_chunked(&pages, 32);
        let response1 = k1.eval_and_accumulate_chunked(&pages, 32);

        let mut result = vec![0u8; PAGE_SIZE_BYTES];
        xor_pages(&response0, &response1, &mut result);

        assert!(
            result.iter().all(|&b| b == target_page as u8),
            "Expected page filled with {}, got {:?}",
            target_page,
            &result[..8]
        );
    }

    #[test]
    #[allow(deprecated)]
    fn page_dpf_chunked_matches_full_eval() {
        let params = PageDpfParams::new(10).unwrap();
        let target_page = 500;
        let (k0, k1) = generate_page_dpf_keys(&params, target_page).unwrap();

        let num_pages = params.max_pages();
        let mut matrix = vec![[0u8; PAGE_SIZE_BYTES]; num_pages];
        for (p, page) in matrix.iter_mut().enumerate() {
            for (i, byte) in page.iter_mut().enumerate() {
                *byte = ((p * PAGE_SIZE_BYTES + i) % 256) as u8;
            }
        }

        let pages: Vec<&[u8]> = matrix.iter().map(|p| p.as_slice()).collect();

        let response0_full = k0.eval_and_accumulate(pages.iter().copied());
        let response1_full = k1.eval_and_accumulate(pages.iter().copied());

        for chunk_size in [1, 16, 64, 256, 1024] {
            let response0_chunked = k0.eval_and_accumulate_chunked(&pages, chunk_size);
            let response1_chunked = k1.eval_and_accumulate_chunked(&pages, chunk_size);

            assert_eq!(
                response0_full, response0_chunked,
                "chunk_size={}: party0 mismatch",
                chunk_size
            );
            assert_eq!(
                response1_full, response1_chunked,
                "chunk_size={}: party1 mismatch",
                chunk_size
            );
        }
    }

    #[test]
    fn page_dpf_eval_timing_breakdown_16bit() {
        let params = PageDpfParams::new(16).unwrap();
        let num_pages = params.max_pages(); // 65536
        let target_page = 12345;
        let (k0, _) = generate_page_dpf_keys(&params, target_page).unwrap();

        let mut pages_data = vec![vec![0u8; PAGE_SIZE_BYTES]; num_pages];
        for (i, page) in pages_data.iter_mut().enumerate() {
            page[0] = (i & 0xFF) as u8;
            page[1] = ((i >> 8) & 0xFF) as u8;
        }
        let page_refs: Vec<&[u8]> = pages_data.iter().map(|p| p.as_slice()).collect();

        let timing = k0.eval_and_accumulate_chunked_timed(&page_refs, 4096);

        assert!(timing.dpf_eval_ns > 0, "DPF eval time should be > 0");
        assert!(
            timing.xor_accumulate_ns > 0,
            "XOR accumulate time should be > 0"
        );
        assert_eq!(timing.response.len(), PAGE_SIZE_BYTES);

        let total = timing.dpf_eval_ns + timing.xor_accumulate_ns;
        let dpf_pct = (timing.dpf_eval_ns as f64 / total as f64) * 100.0;
        println!(
            "16-bit domain (64K pages, 256MB): DPF={:.1}% ({:.2}ms), XOR={:.1}% ({:.2}ms), Total={:.2}ms",
            dpf_pct,
            timing.dpf_eval_ns as f64 / 1_000_000.0,
            100.0 - dpf_pct,
            timing.xor_accumulate_ns as f64 / 1_000_000.0,
            total as f64 / 1_000_000.0
        );
    }

    #[test]
    #[ignore] // Takes ~4GB memory and ~1 second
    fn page_dpf_eval_timing_breakdown_20bit() {
        let params = PageDpfParams::new(20).unwrap();
        let num_pages = params.max_pages(); // 1M pages = 4GB
        let target_page = 500_000;
        let (k0, _) = generate_page_dpf_keys(&params, target_page).unwrap();

        let mut pages_data = vec![vec![0u8; PAGE_SIZE_BYTES]; num_pages];
        for (i, page) in pages_data.iter_mut().enumerate() {
            page[0] = (i & 0xFF) as u8;
            page[1] = ((i >> 8) & 0xFF) as u8;
        }
        let page_refs: Vec<&[u8]> = pages_data.iter().map(|p| p.as_slice()).collect();

        let timing = k0.eval_and_accumulate_chunked_timed(&page_refs, 4096);

        assert!(timing.dpf_eval_ns > 0, "DPF eval time should be > 0");
        assert!(
            timing.xor_accumulate_ns > 0,
            "XOR accumulate time should be > 0"
        );
        assert_eq!(timing.response.len(), PAGE_SIZE_BYTES);

        let total = timing.dpf_eval_ns + timing.xor_accumulate_ns;
        let dpf_pct = (timing.dpf_eval_ns as f64 / total as f64) * 100.0;
        println!(
            "20-bit domain (1M pages, 4GB): DPF={:.1}% ({:.2}ms), XOR={:.1}% ({:.2}ms), Total={:.2}ms",
            dpf_pct,
            timing.dpf_eval_ns as f64 / 1_000_000.0,
            100.0 - dpf_pct,
            timing.xor_accumulate_ns as f64 / 1_000_000.0,
            total as f64 / 1_000_000.0
        );
    }

    #[test]
    fn page_dpf_e2e_pir_flow() {
        let params = PageDpfParams::new(8).unwrap();
        let num_pages = 256;

        let mut matrix0 = vec![vec![0u8; PAGE_SIZE_BYTES]; num_pages];
        let mut matrix1 = vec![vec![0u8; PAGE_SIZE_BYTES]; num_pages];
        for p in 0..num_pages {
            for r in 0..ROWS_PER_PAGE {
                let row_value = ((p * ROWS_PER_PAGE + r) % 256) as u8;
                for b in 0..ROW_SIZE_BYTES {
                    matrix0[p][r * ROW_SIZE_BYTES + b] = row_value;
                    matrix1[p][r * ROW_SIZE_BYTES + b] = row_value;
                }
            }
        }

        let target_row = 678;
        let addr = PageAddress::from_row_index(target_row);

        let (k0, k1) = generate_page_dpf_keys(&params, addr.page_index).unwrap();

        let mut dpf_output0 = vec![ByteGroup::zero(); num_pages];
        let mut dpf_output1 = vec![ByteGroup::zero(); num_pages];
        k0.full_eval(&mut dpf_output0).unwrap();
        k1.full_eval(&mut dpf_output1).unwrap();

        let mut response0 = vec![0u8; PAGE_SIZE_BYTES];
        let mut response1 = vec![0u8; PAGE_SIZE_BYTES];
        for p in 0..num_pages {
            let mask0 = dpf_output0[p].0[0];
            let mask1 = dpf_output1[p].0[0];
            for b in 0..PAGE_SIZE_BYTES {
                response0[b] ^= matrix0[p][b] & mask0;
                response1[b] ^= matrix1[p][b] & mask1;
            }
        }

        let mut result_page = vec![0u8; PAGE_SIZE_BYTES];
        xor_pages(&response0, &response1, &mut result_page);

        let result_row = extract_row_from_page(&result_page, addr.row_offset).unwrap();

        let expected_value = (target_row % 256) as u8;
        assert!(
            result_row.iter().all(|&b| b == expected_value),
            "Expected row to contain {}, got {:?}",
            expected_value,
            &result_row[..8]
        );
    }
}
