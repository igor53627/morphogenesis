//! ChaCha8-based DPF for GPU acceleration.
//!
//! This module provides a DPF implementation compatible with both CPU and GPU.
//! The CPU implementation serves as a reference for testing GPU correctness.

use crate::chacha_prg::{ChaCha8Prg, Seed128};

/// Error types for GPU DPF operations.
#[derive(Debug, Clone)]
pub enum GpuDpfError {
    InvalidDomainBits { bits: usize, reason: &'static str },
    PageIndexTooLarge { index: usize, max: usize },
    InvalidKeyLength { expected: usize, actual: usize },
    InvalidKeyFormat(&'static str),
}

impl std::fmt::Display for GpuDpfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDomainBits { bits, reason } => {
                write!(f, "Invalid domain bits {}: {}", bits, reason)
            }
            Self::PageIndexTooLarge { index, max } => {
                write!(f, "Page index {} exceeds max {}", index, max)
            }
            Self::InvalidKeyLength { expected, actual } => {
                write!(f, "Invalid key length: expected {}, got {}", expected, actual)
            }
            Self::InvalidKeyFormat(reason) => {
                write!(f, "Invalid key format: {}", reason)
            }
        }
    }
}

impl std::error::Error for GpuDpfError {}

/// Correction word for one level of the DPF tree.
#[derive(Clone, Copy, Debug)]
pub struct CorrectionWord {
    pub seed_cw: Seed128,
    pub t_cw_left: u8,
    pub t_cw_right: u8,
}

impl CorrectionWord {
    pub const ZERO: Self = Self {
        seed_cw: Seed128::ZERO,
        t_cw_left: 0,
        t_cw_right: 0,
    };
}

/// Parameters for ChaCha-based DPF.
#[derive(Clone, Debug)]
pub struct ChaChaParams {
    pub domain_bits: usize,
}

impl ChaChaParams {
    pub fn new(domain_bits: usize) -> Result<Self, GpuDpfError> {
        if domain_bits == 0 {
            return Err(GpuDpfError::InvalidDomainBits {
                bits: domain_bits,
                reason: "must be > 0",
            });
        }
        if domain_bits > 32 {
            return Err(GpuDpfError::InvalidDomainBits {
                bits: domain_bits,
                reason: "must be <= 32",
            });
        }
        Ok(Self { domain_bits })
    }

    pub fn max_pages(&self) -> usize {
        1usize
            .checked_shl(self.domain_bits as u32)
            .unwrap_or(usize::MAX)
    }
}

/// DPF key for one party (0 or 1).
///
/// Contains the root seed, control bit, and correction words for each level.
#[derive(Clone, Debug)]
pub struct ChaChaKey {
    pub root_seed: Seed128,
    pub root_t: u8,
    pub domain_bits: usize,
    pub correction_words: Vec<CorrectionWord>,
    pub final_cw: Seed128,
    pub party: u8, // 0 or 1
}

impl ChaChaKey {
    /// Evaluate the DPF at a single point.
    pub fn eval(&self, index: usize) -> Seed128 {
        let mut seed = self.root_seed;
        let mut t = self.root_t;

        for level in 0..self.domain_bits {
            // Expand current node
            let out = ChaCha8Prg::expand(&seed);

            // Get direction bit (MSB first)
            let bit = (index >> (self.domain_bits - 1 - level)) & 1;

            // Select child based on direction
            let (mut child_seed, mut child_t) = if bit == 0 {
                (out.left_seed, out.left_t)
            } else {
                (out.right_seed, out.right_t)
            };

            // Apply correction word if t == 1
            if t == 1 {
                let cw = &self.correction_words[level];
                child_seed = child_seed.xor(&cw.seed_cw);
                if bit == 0 {
                    child_t ^= cw.t_cw_left;
                } else {
                    child_t ^= cw.t_cw_right;
                }
            }

            seed = child_seed;
            t = child_t;
        }

        // Apply final correction word if t == 1
        if t == 1 {
            seed = seed.xor(&self.final_cw);
        }

        seed
    }

    /// Full domain evaluation (CPU reference implementation).
    ///
    /// Optimized to O(N) using GGM tree expansion.
    pub fn full_eval(&self, output: &mut [Seed128]) -> Result<(), GpuDpfError> {
        let num_pages = self.max_pages();
        if output.len() != num_pages {
            return Err(GpuDpfError::InvalidKeyLength {
                expected: num_pages,
                actual: output.len(),
            });
        }

        self.expand_recursive(0, self.root_seed, self.root_t, output);
        Ok(())
    }

    /// Evaluate a subtree of the DPF tree.
    ///
    /// The range [start..start + output.len()] must be a valid aligned subtree
    /// (i.e., output.len() is a power of 2 and start is a multiple of output.len()).
    pub fn eval_subtree(&self, start: usize, output: &mut [Seed128]) -> Result<(), GpuDpfError> {
        let subtree_size = output.len();
        if !subtree_size.is_power_of_two() {
            return Err(GpuDpfError::InvalidKeyFormat("Subtree size must be power of 2"));
        }
        if start % subtree_size != 0 {
            return Err(GpuDpfError::InvalidKeyFormat("Start index must be aligned to subtree size"));
        }

        // Find the seed and t at the root of this subtree
        let mut seed = self.root_seed;
        let mut t = self.root_t;
        let subtree_depth = subtree_size.trailing_zeros() as usize;
        let root_level = self.domain_bits.saturating_sub(subtree_depth);

        for level in 0..root_level {
            let out = ChaCha8Prg::expand(&seed);
            let bit = (start >> (self.domain_bits - 1 - level)) & 1;
            let (mut next_seed, mut next_t) = if bit == 0 {
                (out.left_seed, out.left_t)
            } else {
                (out.right_seed, out.right_t)
            };

            if t == 1 {
                let cw = &self.correction_words[level];
                next_seed = next_seed.xor(&cw.seed_cw);
                next_t ^= if bit == 0 { cw.t_cw_left } else { cw.t_cw_right };
            }
            seed = next_seed;
            t = next_t;
        }

        // Expand the subtree
        self.expand_recursive(root_level, seed, t, output);
        Ok(())
    }

    fn expand_recursive(
        &self,
        level: usize,
        seed: Seed128,
        t: u8,
        output: &mut [Seed128],
    ) {
        if level == self.domain_bits {
            // Leaf reached
            let mut final_seed = seed;
            if t == 1 {
                final_seed = final_seed.xor(&self.final_cw);
            }
            output[0] = final_seed;
            return;
        }

        // Expand current node
        let out = ChaCha8Prg::expand(&seed);
        let cw = &self.correction_words[level];

        // Left child
        let mut left_seed = out.left_seed;
        let mut left_t = out.left_t;
        if t == 1 {
            left_seed = left_seed.xor(&cw.seed_cw);
            left_t ^= cw.t_cw_left;
        }

        // Right child
        let mut right_seed = out.right_seed;
        let mut right_t = out.right_t;
        if t == 1 {
            right_seed = right_seed.xor(&cw.seed_cw);
            right_t ^= cw.t_cw_right;
        }

        let half = output.len() / 2;
        let (left_out, right_out) = output.split_at_mut(half);

        self.expand_recursive(level + 1, left_seed, left_t, left_out);
        self.expand_recursive(level + 1, right_seed, right_t, right_out);
    }

    pub fn max_pages(&self) -> usize {
        1usize
            .checked_shl(self.domain_bits as u32)
            .unwrap_or(usize::MAX)
    }

    pub fn is_party1(&self) -> bool {
        self.party == 1
    }

    /// Serialize the key to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(64 + self.correction_words.len() * 18);
        bytes.extend_from_slice(&self.root_seed.to_bytes());
        bytes.push(self.root_t);
        bytes.push(self.party);
        bytes.push(self.domain_bits as u8);
        
        for cw in &self.correction_words {
            bytes.extend_from_slice(&cw.seed_cw.to_bytes());
            bytes.push(cw.t_cw_left);
            bytes.push(cw.t_cw_right);
        }
        
        bytes.extend_from_slice(&self.final_cw.to_bytes());
        bytes
    }

    /// Deserialize the key from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, GpuDpfError> {
        if bytes.len() < 19 + 16 {
            return Err(GpuDpfError::InvalidKeyFormat("Byte array too short"));
        }

        let root_seed = Seed128::from_bytes(bytes[0..16].try_into().unwrap());
        let root_t = bytes[16];
        let party = bytes[17];
        let domain_bits = bytes[18] as usize;

        let expected_len = 19 + domain_bits * 18 + 16;
        if bytes.len() != expected_len {
            return Err(GpuDpfError::InvalidKeyFormat("Invalid byte array length"));
        }

        let mut correction_words = Vec::with_capacity(domain_bits);
        let mut offset = 19;
        for _ in 0..domain_bits {
            let seed_cw = Seed128::from_bytes(bytes[offset..offset + 16].try_into().unwrap());
            let t_cw_left = bytes[offset + 16];
            let t_cw_right = bytes[offset + 17];
            correction_words.push(CorrectionWord {
                seed_cw,
                t_cw_left,
                t_cw_right,
            });
            offset += 18;
        }

        let final_cw = Seed128::from_bytes(bytes[offset..offset + 16].try_into().unwrap());

        Ok(Self {
            root_seed,
            root_t,
            party,
            domain_bits,
            correction_words,
            final_cw,
        })
    }
}

/// Generate a pair of DPF keys for a target page index.
///
/// Returns (key0, key1) where key0 XOR key1 = point function at target.
pub fn generate_chacha_dpf_keys(
    params: &ChaChaParams,
    target: usize,
) -> Result<(ChaChaKey, ChaChaKey), GpuDpfError> {
    let max = params.max_pages();
    if target >= max {
        return Err(GpuDpfError::PageIndexTooLarge { index: target, max });
    }

    // Generate random root seeds
    let seed0 = Seed128::random();
    let seed1 = Seed128::random();

    // Initial control bits: t0=0, t1=1 (differ at root)
    let t0: u8 = 0;
    let t1: u8 = 1;

    let mut s0 = seed0;
    let mut s1 = seed1;
    let mut tt0 = t0;
    let mut tt1 = t1;

    let mut correction_words = Vec::with_capacity(params.domain_bits);

    for level in 0..params.domain_bits {
        // Expand both parties' current nodes
        let out0 = ChaCha8Prg::expand(&s0);
        let out1 = ChaCha8Prg::expand(&s1);

        // Get direction for this level (MSB first)
        let bit = (target >> (params.domain_bits - 1 - level)) & 1;

        // Compute correction word to make "lose" path outputs equal
        // and "keep" path outputs differ with correct control bit
        let (keep_seed0, keep_t0, lose_seed0, lose_t0) = if bit == 0 {
            (out0.left_seed, out0.left_t, out0.right_seed, out0.right_t)
        } else {
            (out0.right_seed, out0.right_t, out0.left_seed, out0.left_t)
        };
        let (keep_seed1, keep_t1, lose_seed1, lose_t1) = if bit == 0 {
            (out1.left_seed, out1.left_t, out1.right_seed, out1.right_t)
        } else {
            (out1.right_seed, out1.right_t, out1.left_seed, out1.left_t)
        };

        // CW seed: XOR of "lose" children (to make them equal after correction)
        let seed_cw = lose_seed0.xor(&lose_seed1);

        // CW control bits: designed to make outputs agree on lose path, differ on keep path
        let t_cw_lose = lose_t0 ^ lose_t1 ^ tt0 ^ tt1 ^ 1;
        let t_cw_keep = keep_t0 ^ keep_t1 ^ tt0 ^ tt1;

        let (t_cw_left, t_cw_right) = if bit == 0 {
            (t_cw_keep, t_cw_lose)
        } else {
            (t_cw_lose, t_cw_keep)
        };

        correction_words.push(CorrectionWord {
            seed_cw,
            t_cw_left,
            t_cw_right,
        });

        // Update states for next level
        // Party 0: apply CW if t0 == 1
        let mut next_s0 = keep_seed0;
        let mut next_t0 = keep_t0;
        if tt0 == 1 {
            next_s0 = next_s0.xor(&seed_cw);
            next_t0 ^= t_cw_keep;
        }

        // Party 1: apply CW if t1 == 1
        let mut next_s1 = keep_seed1;
        let mut next_t1 = keep_t1;
        if tt1 == 1 {
            next_s1 = next_s1.xor(&seed_cw);
            next_t1 ^= t_cw_keep;
        }

        s0 = next_s0;
        s1 = next_s1;
        tt0 = next_t0;
        tt1 = next_t1;
    }

    // Final correction word: make XOR at target be all-1s (0xFF bytes)
    let final_xor = s0.xor(&s1);
    let all_ones = Seed128::new([0xFFFFFFFF; 4]);
    let final_cw = final_xor.xor(&all_ones);

    Ok((
        ChaChaKey {
            root_seed: seed0,
            root_t: t0,
            domain_bits: params.domain_bits,
            correction_words: correction_words.clone(),
            final_cw,
            party: 0,
        },
        ChaChaKey {
            root_seed: seed1,
            root_t: t1,
            domain_bits: params.domain_bits,
            correction_words,
            final_cw,
            party: 1,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_creation() {
        let params = ChaChaParams::new(10).unwrap();
        assert_eq!(params.domain_bits, 10);
        assert_eq!(params.max_pages(), 1024);
    }

    #[test]
    fn params_rejects_zero() {
        assert!(ChaChaParams::new(0).is_err());
    }

    #[test]
    fn params_rejects_too_large() {
        assert!(ChaChaParams::new(33).is_err());
    }

    #[test]
    fn key_generation_basic() {
        let params = ChaChaParams::new(8).unwrap();
        let (k0, k1) = generate_chacha_dpf_keys(&params, 42).unwrap();

        assert!(!k0.is_party1());
        assert!(k1.is_party1());
        assert_eq!(k0.domain_bits, 8);
        assert_eq!(k1.domain_bits, 8);
    }

    #[test]
    fn key_generation_rejects_out_of_range() {
        let params = ChaChaParams::new(8).unwrap();
        let result = generate_chacha_dpf_keys(&params, 300);
        assert!(matches!(result, Err(GpuDpfError::PageIndexTooLarge { .. })));
    }

    #[test]
    fn dpf_correctness_8bit() {
        let params = ChaChaParams::new(8).unwrap();
        let target = 42;
        let (k0, k1) = generate_chacha_dpf_keys(&params, target).unwrap();

        // Check all points
        for i in 0..256 {
            let out0 = k0.eval(i);
            let out1 = k1.eval(i);
            let xor = out0.xor(&out1);

            if i == target {
                // XOR should be all-1s at target
                assert_eq!(
                    xor,
                    Seed128::new([0xFFFFFFFF; 4]),
                    "XOR at target {} should be 0xFF",
                    i
                );
            } else {
                // XOR should be all-0s elsewhere
                assert_eq!(xor, Seed128::ZERO, "XOR at non-target {} should be 0x00", i);
            }
        }
    }

    #[test]
    fn dpf_correctness_10bit() {
        let params = ChaChaParams::new(10).unwrap();
        let target = 500;
        let (k0, k1) = generate_chacha_dpf_keys(&params, target).unwrap();

        // Spot check
        for i in [0, 1, 100, target, 1000, 1023] {
            let out0 = k0.eval(i);
            let out1 = k1.eval(i);
            let xor = out0.xor(&out1);

            if i == target {
                assert_eq!(xor, Seed128::new([0xFFFFFFFF; 4]));
            } else {
                assert_eq!(xor, Seed128::ZERO);
            }
        }
    }

    #[test]
    fn dpf_correctness_16bit() {
        let params = ChaChaParams::new(16).unwrap();
        let target = 32000;
        let (k0, k1) = generate_chacha_dpf_keys(&params, target).unwrap();

        // Spot check edges and target
        for i in [0, 1, target, 65534, 65535] {
            let out0 = k0.eval(i);
            let out1 = k1.eval(i);
            let xor = out0.xor(&out1);

            if i == target {
                assert_eq!(xor, Seed128::new([0xFFFFFFFF; 4]));
            } else {
                assert_eq!(xor, Seed128::ZERO);
            }
        }
    }

    #[test]
    fn full_eval_matches_point_eval() {
        let params = ChaChaParams::new(8).unwrap();
        let (k0, _) = generate_chacha_dpf_keys(&params, 100).unwrap();

        let mut full = vec![Seed128::ZERO; 256];
        k0.full_eval(&mut full).unwrap();

        for i in 0..256 {
            let point = k0.eval(i);
            assert_eq!(full[i], point, "Mismatch at index {}", i);
        }
    }

    #[test]
    fn dpf_boundary_cases() {
        // Test first and last indices
        for bits in [4, 8, 12, 16] {
            let params = ChaChaParams::new(bits).unwrap();
            let max = params.max_pages();

            for target in [0, max / 2, max - 1] {
                let (k0, k1) = generate_chacha_dpf_keys(&params, target).unwrap();

                let out0 = k0.eval(target);
                let out1 = k1.eval(target);
                let xor = out0.xor(&out1);

                assert_eq!(
                    xor,
                    Seed128::new([0xFFFFFFFF; 4]),
                    "Failed at domain_bits={}, target={}",
                    bits,
                    target
                );

                // Check a non-target
                let other = (target + 1) % max;
                let out0 = k0.eval(other);
                let out1 = k1.eval(other);
                let xor = out0.xor(&out1);
                assert_eq!(xor, Seed128::ZERO);
            }
        }
    }

    #[test]
    fn serialization_roundtrip() {
        let params = ChaChaParams::new(10).unwrap();
        let (k0, k1) = generate_chacha_dpf_keys(&params, 42).unwrap();

        let b0 = k0.to_bytes();
        let b1 = k1.to_bytes();

        let k0_rec = ChaChaKey::from_bytes(&b0).unwrap();
        let k1_rec = ChaChaKey::from_bytes(&b1).unwrap();

        assert_eq!(k0.root_seed, k0_rec.root_seed);
        assert_eq!(k0.root_t, k0_rec.root_t);
        assert_eq!(k0.party, k0_rec.party);
        assert_eq!(k0.domain_bits, k0_rec.domain_bits);
        assert_eq!(k0.correction_words.len(), k0_rec.correction_words.len());
        assert_eq!(k0.final_cw, k0_rec.final_cw);

        for i in 0..1024 {
            assert_eq!(k0.eval(i), k0_rec.eval(i));
            assert_eq!(k1.eval(i), k1_rec.eval(i));
        }
    }
}
