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
        if domain_bits > 32 {
            return Err(PageDpfError::InvalidDomainBits {
                domain_bits,
                reason: "must be <= 32",
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
            for j in 0..16 {
                xor[j] = output0[i].0[j] ^ output1[i].0[j];
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
            for j in 0..16 {
                xor[j] = output0[i].0[j] ^ output1[i].0[j];
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
            for j in 0..16 {
                xor[j] = output0[i].0[j] ^ output1[i].0[j];
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
            for j in 0..16 {
                xor[j] = output0[i].0[j] ^ output1[i].0[j];
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
            for j in 0..16 {
                xor[j] = output0[i].0[j] ^ output1[i].0[j];
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
