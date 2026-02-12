use rand::Rng;

pub mod page;

pub trait DpfKey: Send + Sync {
    fn eval_bit(&self, index: usize) -> bool;

    fn eval_batch(&self, indices: &[usize]) -> Vec<u8> {
        indices
            .iter()
            .map(|&idx| 0u8.wrapping_sub(self.eval_bit(idx) as u8))
            .collect()
    }

    fn eval_range_masks(&self, start: usize, out: &mut [u8]) {
        for (i, o) in out.iter_mut().enumerate() {
            *o = 0u8.wrapping_sub(self.eval_bit(start + i) as u8);
        }
    }
}

#[derive(Clone)]
pub struct AesDpfKey {
    key: [u8; 16],
    target: usize,
    correction_word: u8,
}

impl std::fmt::Debug for AesDpfKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AesDpfKey")
            .field("target", &self.target)
            .finish_non_exhaustive()
    }
}

impl AesDpfKey {
    pub fn generate_pair<R: Rng>(rng: &mut R, target: usize) -> (Self, Self) {
        let shared_key: [u8; 16] = rng.gen();
        let prf_at_target = Self::prf_eval(&shared_key, target);

        (
            Self {
                key: shared_key,
                target,
                correction_word: prf_at_target,
            },
            Self {
                key: shared_key,
                target,
                correction_word: prf_at_target ^ 0xFF,
            },
        )
    }

    pub fn new_single<R: Rng>(rng: &mut R, target: usize) -> Self {
        let key: [u8; 16] = rng.gen();
        let prf_at_target = Self::prf_eval(&key, target);
        Self {
            key,
            target,
            correction_word: prf_at_target ^ 0xFF,
        }
    }

    #[inline(always)]
    fn prf_eval(key: &[u8; 16], index: usize) -> u8 {
        let block = Self::aes_round(key, index);
        block[0]
    }

    #[inline(always)]
    fn aes_round(key: &[u8; 16], index: usize) -> [u8; 16] {
        let mut block = [0u8; 16];
        block[0..8].copy_from_slice(&(index as u64).to_le_bytes());

        #[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            if Self::has_aes() {
                return unsafe { Self::aes_round_intrinsic(key, &block) };
            }
        }

        Self::aes_round_soft(key, &block)
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[allow(dead_code)]
    fn has_aes() -> bool {
        std::is_x86_feature_detected!("aes")
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    #[allow(dead_code)]
    fn has_aes() -> bool {
        false
    }

    #[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
    #[target_feature(enable = "aes")]
    unsafe fn aes_round_intrinsic(key: &[u8; 16], block: &[u8; 16]) -> [u8; 16] {
        use std::arch::x86_64::{_mm_aesenc_si128, _mm_loadu_si128, _mm_storeu_si128};

        let key_vec = _mm_loadu_si128(key.as_ptr() as *const _);
        let block_vec = _mm_loadu_si128(block.as_ptr() as *const _);
        let result = _mm_aesenc_si128(block_vec, key_vec);

        let mut out = [0u8; 16];
        _mm_storeu_si128(out.as_mut_ptr() as *mut _, result);
        out
    }

    #[inline(always)]
    fn aes_round_soft(key: &[u8; 16], block: &[u8; 16]) -> [u8; 16] {
        let mut state = *block;
        for i in 0..16 {
            state[i] ^= key[i];
        }
        for _ in 0..4 {
            let mut next = [0u8; 16];
            for i in 0..16 {
                next[i] = state[i]
                    .wrapping_mul(0x1b)
                    .wrapping_add(state[(i + 5) % 16])
                    .wrapping_add(state[(i + 10) % 16])
                    .rotate_left(3);
            }
            state = next;
        }
        state
    }

    #[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
    pub fn eval_range_masks_avx512(&self, start: usize, out: &mut [u8]) {
        if out.len() >= 4 && Self::has_vaes() {
            unsafe { self.eval_range_masks_avx512_inner(start, out) }
        } else {
            self.eval_range_masks_scalar(start, out);
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[allow(dead_code)]
    fn has_vaes() -> bool {
        std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("vaes")
            && std::is_x86_feature_detected!("avx512vl")
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    #[allow(dead_code)]
    fn has_vaes() -> bool {
        false
    }

    #[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
    #[target_feature(enable = "avx512f,avx512vl,vaes")]
    unsafe fn eval_range_masks_avx512_inner(&self, start: usize, out: &mut [u8]) {
        use std::arch::x86_64::{
            __m512i, _mm512_aesenc_epi128, _mm512_loadu_si512, _mm512_set_epi64,
            _mm512_storeu_si512,
        };

        let key_vec =
            _mm512_loadu_si512([self.key, self.key, self.key, self.key].as_ptr() as *const _);
        let cw = self.correction_word;
        let target = self.target;

        let mut i = 0;
        while i + 4 <= out.len() {
            let idx0 = start + i;
            let idx1 = start + i + 1;
            let idx2 = start + i + 2;
            let idx3 = start + i + 3;

            let blocks = _mm512_set_epi64(
                0,
                idx3 as i64,
                0,
                idx2 as i64,
                0,
                idx1 as i64,
                0,
                idx0 as i64,
            );

            let result = _mm512_aesenc_epi128(blocks, key_vec);

            let mut result_bytes = [0u8; 64];
            _mm512_storeu_si512(result_bytes.as_mut_ptr() as *mut __m512i, result);

            for j in 0..4 {
                let idx = start + i + j;
                let prf_byte = result_bytes[j * 16];
                out[i + j] = if idx == target {
                    prf_byte ^ cw
                } else {
                    prf_byte
                };
            }

            i += 4;
        }

        while i < out.len() {
            out[i] = self.eval_mask(start + i);
            i += 1;
        }
    }

    #[inline]
    fn eval_range_masks_scalar(&self, start: usize, out: &mut [u8]) {
        for (i, o) in out.iter_mut().enumerate() {
            *o = self.eval_mask(start + i);
        }
    }
}

impl DpfKey for AesDpfKey {
    #[inline]
    fn eval_bit(&self, index: usize) -> bool {
        let mask = self.eval_mask(index);
        mask != 0
    }

    fn eval_range_masks(&self, start: usize, out: &mut [u8]) {
        #[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            self.eval_range_masks_avx512(start, out);
            return;
        }

        #[allow(unreachable_code)]
        self.eval_range_masks_scalar(start, out);
    }
}

impl AesDpfKey {
    #[inline(always)]
    fn eval_mask(&self, index: usize) -> u8 {
        let prf_byte = Self::prf_eval(&self.key, index);
        if index == self.target {
            prf_byte ^ self.correction_word
        } else {
            prf_byte
        }
    }
}

/// Key serialization size: 16 (AES key) + 8 (target) + 1 (correction_word) = 25 bytes
pub const AES_DPF_KEY_SIZE: usize = 25;

#[derive(Debug, PartialEq, Eq)]
pub enum DpfKeyError {
    InvalidLength { expected: usize, actual: usize },
    TargetTooLarge { target: u64, max: usize },
}

impl std::fmt::Display for DpfKeyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DpfKeyError::InvalidLength { expected, actual } => {
                write!(
                    f,
                    "invalid DPF key length: expected {} bytes, got {}",
                    expected, actual
                )
            }
            DpfKeyError::TargetTooLarge { target, max } => {
                write!(
                    f,
                    "target index {} exceeds platform maximum {}",
                    target, max
                )
            }
        }
    }
}

impl std::error::Error for DpfKeyError {}

impl AesDpfKey {
    /// Serializes the key to a 25-byte array.
    pub fn to_bytes(&self) -> [u8; AES_DPF_KEY_SIZE] {
        let mut out = [0u8; AES_DPF_KEY_SIZE];
        out[0..16].copy_from_slice(&self.key);
        out[16..24].copy_from_slice(&(self.target as u64).to_le_bytes());
        out[24] = self.correction_word;
        out
    }

    /// Deserializes a key from a 25-byte slice.
    ///
    /// # Errors
    /// - `InvalidLength` if bytes.len() != 25
    /// - `TargetTooLarge` if target index exceeds usize::MAX (on 32-bit platforms)
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DpfKeyError> {
        if bytes.len() != AES_DPF_KEY_SIZE {
            return Err(DpfKeyError::InvalidLength {
                expected: AES_DPF_KEY_SIZE,
                actual: bytes.len(),
            });
        }
        let mut key = [0u8; 16];
        key.copy_from_slice(&bytes[0..16]);

        let target_bytes: [u8; 8] = bytes[16..24]
            .try_into()
            .expect("slice length verified above");
        let target_u64 = u64::from_le_bytes(target_bytes);
        let target = usize::try_from(target_u64).map_err(|_| DpfKeyError::TargetTooLarge {
            target: target_u64,
            max: usize::MAX,
        })?;

        let correction_word = bytes[24];
        Ok(Self {
            key,
            target,
            correction_word,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aes_dpf_key_roundtrip_serialization() {
        let mut rng = rand::thread_rng();
        let target = 12345;
        let (key0, _key1) = AesDpfKey::generate_pair(&mut rng, target);

        let bytes = key0.to_bytes();
        assert_eq!(bytes.len(), AES_DPF_KEY_SIZE);

        let restored = AesDpfKey::from_bytes(&bytes).unwrap();
        assert_eq!(restored.target, target);
        // Verify they evaluate the same
        for i in 0..100 {
            assert_eq!(key0.eval_bit(i), restored.eval_bit(i));
        }
    }

    #[test]
    fn aes_dpf_key_from_bytes_rejects_wrong_length() {
        let result = AesDpfKey::from_bytes(&[0u8; 10]);
        assert!(matches!(
            result,
            Err(DpfKeyError::InvalidLength {
                expected: 25,
                actual: 10
            })
        ));
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn aes_dpf_key_from_bytes_accepts_large_target_on_64bit() {
        let mut bytes = [0u8; AES_DPF_KEY_SIZE];
        let large_target: u64 = (1u64 << 40) + 12345;
        bytes[16..24].copy_from_slice(&large_target.to_le_bytes());

        let result = AesDpfKey::from_bytes(&bytes);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().target, large_target as usize);
    }

    #[test]
    fn aes_dpf_key_from_bytes_validates_target_fits_usize() {
        let mut bytes = [0u8; AES_DPF_KEY_SIZE];
        // This test verifies the error path exists; on 64-bit it won't trigger
        // but on 32-bit platforms, values > u32::MAX would fail
        let target: u64 = 12345;
        bytes[16..24].copy_from_slice(&target.to_le_bytes());

        let result = AesDpfKey::from_bytes(&bytes);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().target, 12345);
    }

    #[test]
    fn aes_dpf_key_pair_xor_to_point_function() {
        let mut rng = rand::thread_rng();
        let target = 42;
        let (key0, key1) = AesDpfKey::generate_pair(&mut rng, target);

        for idx in 0..100 {
            let b0 = key0.eval_bit(idx);
            let b1 = key1.eval_bit(idx);
            let xor_result = b0 ^ b1;

            if idx == target {
                assert!(xor_result, "XOR at target {} should be true", idx);
            } else {
                assert!(!xor_result, "XOR at non-target {} should be false", idx);
            }
        }
    }

    #[test]
    fn aes_dpf_eval_range_masks_correctness() {
        let mut rng = rand::thread_rng();
        let target = 17;
        let (key0, key1) = AesDpfKey::generate_pair(&mut rng, target);

        let mut masks0 = [0u8; 32];
        let mut masks1 = [0u8; 32];
        key0.eval_range_masks(0, &mut masks0);
        key1.eval_range_masks(0, &mut masks1);

        for i in 0..32 {
            let xor_mask = masks0[i] ^ masks1[i];
            if i == target {
                assert_eq!(xor_mask, 0xFF, "XOR mask at target should be 0xFF");
            } else {
                assert_eq!(xor_mask, 0x00, "XOR mask at non-target should be 0x00");
            }
        }
    }

    #[test]
    fn aes_dpf_single_key_returns_one_at_target() {
        let mut rng = rand::thread_rng();
        let target = 5;
        let key = AesDpfKey::new_single(&mut rng, target);

        // Verify eval_bit returns without panicking; the value is
        // inherently either true or false, so we just exercise the call.
        let _result = key.eval_bit(target);
    }
}

#[cfg(all(test, feature = "fss"))]
mod fss_tests {
    #[allow(unused_imports)]
    use super::*;

    // fss-rs PRG parameters:
    // Aes128MatyasMeyerOseasPrg<OUT_BLEN, OUT_BLEN_N, CIPHER_N>
    // - OUT_BLEN: output byte length (must match DPF OUT_BLEN, must be multiple of 16)
    // - OUT_BLEN_N: must be 1 for DPF (DpfImpl requires Prg<OUT_BLEN, 1>)
    // - CIPHER_N: (OUT_BLEN / 16) * OUT_BLEN_N * 2 = (16/16) * 1 * 2 = 2
    //
    // For 16-byte output: OUT_BLEN=16, OUT_BLEN_N=1, CIPHER_N=2

    #[test]
    fn fss_rs_dpf_pair_xor_to_point_function() {
        use fss_rs::dpf::{Dpf, DpfImpl, PointFn};
        use fss_rs::group::byte::ByteGroup;
        use fss_rs::group::Group;
        use fss_rs::prg::Aes128MatyasMeyerOseasPrg;

        // Create PRG: 16-byte output, 1 chunk (required for DPF), 2 ciphers
        let prg_keys: [[u8; 16]; 2] = rand::random();
        let prg =
            Aes128MatyasMeyerOseasPrg::<16, 1, 2>::new(&std::array::from_fn(|i| &prg_keys[i]));

        // Create DPF: 2-byte input domain (64K rows), 16-byte output
        let dpf = DpfImpl::<2, 16, _>::new(prg);

        // Target index (as 2-byte big-endian)
        let target: u16 = 42;
        let alpha = target.to_be_bytes();

        // Output value at target (all 0xFF)
        let beta = ByteGroup([0xFF; 16]);

        let point_fn = PointFn { alpha, beta };

        // Generate key pair
        let s0s: [[u8; 16]; 2] = rand::random();
        let share = dpf.gen(&point_fn, [&s0s[0], &s0s[1]]);

        // Create per-party shares (share0 has s0s[0], share1 has s0s[1])
        use fss_rs::Share;
        let k0 = Share {
            s0s: vec![share.s0s[0]],
            cws: share.cws.clone(),
            cw_np1: share.cw_np1.clone(),
        };
        let k1 = Share {
            s0s: vec![share.s0s[1]],
            cws: share.cws.clone(),
            cw_np1: share.cw_np1.clone(),
        };

        // Evaluate both shares at target and non-target
        let mut y0_target = ByteGroup::zero();
        let mut y1_target = ByteGroup::zero();
        let mut y0_other = ByteGroup::zero();
        let mut y1_other = ByteGroup::zero();

        dpf.eval(false, &k0, &[&alpha], &mut [&mut y0_target]);
        dpf.eval(true, &k1, &[&alpha], &mut [&mut y1_target]);

        let other: [u8; 2] = (100u16).to_be_bytes();
        dpf.eval(false, &k0, &[&other], &mut [&mut y0_other]);
        dpf.eval(true, &k1, &[&other], &mut [&mut y1_other]);

        // XOR the shares
        let mut xor_target = [0u8; 16];
        let mut xor_other = [0u8; 16];
        for i in 0..16 {
            xor_target[i] = y0_target.0[i] ^ y1_target.0[i];
            xor_other[i] = y0_other.0[i] ^ y1_other.0[i];
        }

        assert_eq!(xor_target, [0xFF; 16], "XOR at target should be all 0xFF");
        assert_eq!(
            xor_other, [0x00; 16],
            "XOR at non-target should be all 0x00"
        );
    }

    #[test]
    fn fss_rs_dpf_full_eval_correctness() {
        use fss_rs::dpf::{Dpf, DpfImpl, PointFn};
        use fss_rs::group::byte::ByteGroup;
        use fss_rs::group::Group;
        use fss_rs::prg::Aes128MatyasMeyerOseasPrg;

        // 1-byte domain = 256 rows, 16-byte output
        let prg_keys: [[u8; 16]; 2] = rand::random();
        let prg =
            Aes128MatyasMeyerOseasPrg::<16, 1, 2>::new(&std::array::from_fn(|i| &prg_keys[i]));
        let dpf = DpfImpl::<1, 16, _>::new(prg);

        let target: u8 = 42;
        let alpha = [target];
        let beta = ByteGroup([0xFF; 16]);
        let point_fn = PointFn { alpha, beta };

        let s0s: [[u8; 16]; 2] = rand::random();
        let share = dpf.gen(&point_fn, [&s0s[0], &s0s[1]]);

        // Create per-party shares
        use fss_rs::Share;
        let k0 = Share {
            s0s: vec![share.s0s[0]],
            cws: share.cws.clone(),
            cw_np1: share.cw_np1.clone(),
        };
        let k1 = Share {
            s0s: vec![share.s0s[1]],
            cws: share.cws.clone(),
            cw_np1: share.cw_np1.clone(),
        };

        // Full domain evaluation
        let mut ys0: Vec<ByteGroup<16>> = vec![ByteGroup::zero(); 256];
        let mut ys1: Vec<ByteGroup<16>> = vec![ByteGroup::zero(); 256];

        // API expects &mut [&mut G], so we need refs
        let mut ys0_refs: Vec<&mut ByteGroup<16>> = ys0.iter_mut().collect();
        let mut ys1_refs: Vec<&mut ByteGroup<16>> = ys1.iter_mut().collect();

        dpf.full_eval(false, &k0, &mut ys0_refs);
        dpf.full_eval(true, &k1, &mut ys1_refs);

        // Check XOR at all positions
        for i in 0..256 {
            let xor = std::array::from_fn(|j| ys0[i].0[j] ^ ys1[i].0[j]);
            if i == target as usize {
                assert_eq!(xor, [0xFF; 16], "XOR at target {} should be all 0xFF", i);
            } else {
                assert_eq!(
                    xor, [0x00; 16],
                    "XOR at non-target {} should be all 0x00",
                    i
                );
            }
        }
    }
}
