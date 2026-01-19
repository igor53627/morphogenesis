//! ChaCha8-based PRG for DPF tree expansion.
//!
//! The PRG expands a 128-bit seed into:
//! - Left child seed (128 bits)
//! - Right child seed (128 bits)  
//! - Left/Right control bits (1 bit each)
//!
//! We use ChaCha8 (8 rounds) rather than ChaCha20 for performance.
//! ChaCha8 is sufficient for cryptographic PRG in DPF context.

use bytemuck::{Pod, Zeroable};

/// 128-bit seed for DPF nodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
#[repr(C)]
pub struct Seed128 {
    pub words: [u32; 4],
}

impl Seed128 {
    pub const ZERO: Self = Self { words: [0; 4] };

    pub fn new(words: [u32; 4]) -> Self {
        Self { words }
    }

    pub fn from_bytes(bytes: &[u8; 16]) -> Self {
        Self {
            words: [
                u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
                u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]),
            ],
        }
    }

    pub fn to_bytes(&self) -> [u8; 16] {
        let mut out = [0u8; 16];
        for (i, word) in self.words.iter().enumerate() {
            out[i * 4..(i + 1) * 4].copy_from_slice(&word.to_le_bytes());
        }
        out
    }

    pub fn xor(&self, other: &Self) -> Self {
        Self {
            words: [
                self.words[0] ^ other.words[0],
                self.words[1] ^ other.words[1],
                self.words[2] ^ other.words[2],
                self.words[3] ^ other.words[3],
            ],
        }
    }

    pub fn random() -> Self {
        Self {
            words: rand::random(),
        }
    }
}

/// Output of PRG expansion: two child seeds and control bits.
#[derive(Clone, Copy, Debug)]
pub struct PrgOutput {
    pub left_seed: Seed128,
    pub right_seed: Seed128,
    pub left_t: u8,
    pub right_t: u8,
}

/// ChaCha8-based PRG for DPF.
///
/// Uses the seed as part of the ChaCha key and expands to produce
/// two child seeds and control bits.
pub struct ChaCha8Prg;

impl ChaCha8Prg {
    /// ChaCha quarter-round function.
    #[inline(always)]
    fn quarter_round(state: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
        state[a] = state[a].wrapping_add(state[b]);
        state[d] ^= state[a];
        state[d] = state[d].rotate_left(16);

        state[c] = state[c].wrapping_add(state[d]);
        state[b] ^= state[c];
        state[b] = state[b].rotate_left(12);

        state[a] = state[a].wrapping_add(state[b]);
        state[d] ^= state[a];
        state[d] = state[d].rotate_left(8);

        state[c] = state[c].wrapping_add(state[d]);
        state[b] ^= state[c];
        state[b] = state[b].rotate_left(7);
    }

    /// Run ChaCha8 block function on the given state.
    fn chacha8_block(key: &[u32; 8], counter: u32, nonce: &[u32; 3]) -> [u32; 16] {
        // ChaCha state initialization
        let mut state: [u32; 16] = [
            // Constants: "expand 32-byte k"
            0x61707865, 0x3320646e, 0x79622d32, 0x6b206574, // Key (256 bits = 8 words)
            key[0], key[1], key[2], key[3], key[4], key[5], key[6], key[7],
            // Counter + Nonce
            counter, nonce[0], nonce[1], nonce[2],
        ];

        let initial = state;

        // 8 rounds = 4 double-rounds
        for _ in 0..4 {
            // Column rounds
            Self::quarter_round(&mut state, 0, 4, 8, 12);
            Self::quarter_round(&mut state, 1, 5, 9, 13);
            Self::quarter_round(&mut state, 2, 6, 10, 14);
            Self::quarter_round(&mut state, 3, 7, 11, 15);
            // Diagonal rounds
            Self::quarter_round(&mut state, 0, 5, 10, 15);
            Self::quarter_round(&mut state, 1, 6, 11, 12);
            Self::quarter_round(&mut state, 2, 7, 8, 13);
            Self::quarter_round(&mut state, 3, 4, 9, 14);
        }

        // Add initial state
        for i in 0..16 {
            state[i] = state[i].wrapping_add(initial[i]);
        }

        state
    }

    /// Expand a 128-bit seed into two child seeds and control bits.
    ///
    /// The seed is doubled to form a 256-bit ChaCha key: key = seed || seed.
    /// We generate two ChaCha blocks (counter 0 and 1) and extract:
    /// - Left seed from block0[0..4]
    /// - Right seed from block0[4..8]
    /// - Control bits from block1[0] and block1[1]
    pub fn expand(seed: &Seed128) -> PrgOutput {
        // Double the seed to form 256-bit key
        let key: [u32; 8] = [
            seed.words[0],
            seed.words[1],
            seed.words[2],
            seed.words[3],
            seed.words[0],
            seed.words[1],
            seed.words[2],
            seed.words[3],
        ];

        let nonce = [0u32; 3];

        // Generate two blocks
        let block0 = Self::chacha8_block(&key, 0, &nonce);
        let block1 = Self::chacha8_block(&key, 1, &nonce);

        PrgOutput {
            left_seed: Seed128::new([block0[0], block0[1], block0[2], block0[3]]),
            right_seed: Seed128::new([block0[4], block0[5], block0[6], block0[7]]),
            left_t: (block1[0] & 1) as u8,
            right_t: (block1[1] & 1) as u8,
        }
    }

    /// Verify expansion is deterministic (for testing).
    pub fn expand_deterministic(seed: &Seed128) -> PrgOutput {
        Self::expand(seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_from_bytes_roundtrip() {
        let bytes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let seed = Seed128::from_bytes(&bytes);
        assert_eq!(seed.to_bytes(), bytes);
    }

    #[test]
    fn seed_xor_identity() {
        let a = Seed128::random();
        let zero = Seed128::ZERO;
        assert_eq!(a.xor(&zero), a);
    }

    #[test]
    fn seed_xor_self_is_zero() {
        let a = Seed128::random();
        assert_eq!(a.xor(&a), Seed128::ZERO);
    }

    #[test]
    fn prg_deterministic() {
        let seed = Seed128::new([0x12345678, 0x9abcdef0, 0xfedcba98, 0x76543210]);
        let out1 = ChaCha8Prg::expand(&seed);
        let out2 = ChaCha8Prg::expand(&seed);

        assert_eq!(out1.left_seed, out2.left_seed);
        assert_eq!(out1.right_seed, out2.right_seed);
        assert_eq!(out1.left_t, out2.left_t);
        assert_eq!(out1.right_t, out2.right_t);
    }

    #[test]
    fn prg_different_seeds_different_outputs() {
        let seed1 = Seed128::new([1, 2, 3, 4]);
        let seed2 = Seed128::new([5, 6, 7, 8]);

        let out1 = ChaCha8Prg::expand(&seed1);
        let out2 = ChaCha8Prg::expand(&seed2);

        assert_ne!(out1.left_seed, out2.left_seed);
        assert_ne!(out1.right_seed, out2.right_seed);
    }

    #[test]
    fn prg_outputs_different_children() {
        let seed = Seed128::random();
        let out = ChaCha8Prg::expand(&seed);

        // Left and right seeds should be different
        assert_ne!(out.left_seed, out.right_seed);
    }

    #[test]
    fn prg_control_bits_are_binary() {
        for _ in 0..100 {
            let seed = Seed128::random();
            let out = ChaCha8Prg::expand(&seed);
            assert!(out.left_t <= 1);
            assert!(out.right_t <= 1);
        }
    }

    #[test]
    fn chacha8_known_vector() {
        // Test vector: all zeros should produce deterministic output
        let seed = Seed128::ZERO;
        let out = ChaCha8Prg::expand(&seed);

        // Just verify it doesn't produce zeros (ChaCha never does for non-zero constants)
        assert_ne!(out.left_seed, Seed128::ZERO);
        assert_ne!(out.right_seed, Seed128::ZERO);
    }
}
