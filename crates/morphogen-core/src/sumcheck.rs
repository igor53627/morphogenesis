use binius_field::{BinaryField128b, Field};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// A round polynomial in the Sum-Check protocol.
/// For a product of two multilinear polynomials, the round polynomial is quadratic.
/// g(X) = a*X^2 + b*X + c
/// In binary fields, we can represent this by its evaluations at 3 points: 0, 1, and some alpha.
/// We store these as u128 for easy serialization.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RoundPolynomial {
    pub evals: [u128; 3],
}

/// A Sum-Check proof for the inner product of two vectors.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SumCheckProof {
    pub round_polynomials: Vec<RoundPolynomial>,
    pub sum: u128,
}

pub struct SumCheckProver {
    /// The database vector (multilinear extension values)
    db: Vec<BinaryField128b>,
    /// The query vector (multilinear extension values)
    query: Vec<BinaryField128b>,
    /// Number of variables (domain bits)
    num_vars: usize,
}

impl SumCheckProver {
    pub fn new(db: Vec<BinaryField128b>, query: Vec<BinaryField128b>) -> Self {
        let n = db.len();
        assert_eq!(n, query.len());
        assert!(n.is_power_of_two());
        let num_vars = n.trailing_zeros() as usize;

        Self {
            db,
            query,
            num_vars,
        }
    }

    /// Prove the sum-check protocol.
    pub fn prove(&self, challenges: &[BinaryField128b]) -> SumCheckProof {
        assert_eq!(challenges.len(), self.num_vars);

        let mut round_polynomials = Vec::with_capacity(self.num_vars);

        // Current partial evaluations
        let mut current_db = self.db.clone();
        let mut current_query = self.query.clone();

        // Initial sum (Parallelized)
        let sum_f: BinaryField128b = current_db
            .par_iter()
            .zip(current_query.par_iter())
            .map(|(d, q)| *d * *q)
            .reduce(|| BinaryField128b::ZERO, |a, b| a + b);

        for (round, &r) in challenges.iter().enumerate() {
            let n = 1 << (self.num_vars - round - 1);

            // Calculate g(0), g(1), g(alpha) in parallel
            let (g_0, g_1, g_alpha) = (0..n)
                .into_par_iter()
                .map(|i| {
                    let d0 = current_db[i];
                    let d1 = current_db[i + n];
                    let q0 = current_query[i];
                    let q1 = current_query[i + n];

                    let term_0 = d0 * q0;
                    let term_1 = d1 * q1;

                    let alpha = BinaryField128b::new(2);
                    let one_plus_alpha = BinaryField128b::ONE + alpha;

                    let d_alpha = one_plus_alpha * d0 + alpha * d1;
                    let q_alpha = one_plus_alpha * q0 + alpha * q1;
                    let term_alpha = d_alpha * q_alpha;

                    (term_0, term_1, term_alpha)
                })
                .reduce(
                    || {
                        (
                            BinaryField128b::ZERO,
                            BinaryField128b::ZERO,
                            BinaryField128b::ZERO,
                        )
                    },
                    |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
                );

            round_polynomials.push(RoundPolynomial {
                evals: [g_0.val(), g_1.val(), g_alpha.val()],
            });

            // Update current_db and current_query (Parallelized folding)
            let one_plus_r = BinaryField128b::ONE + r;

            // We fold in-place into the first half of the vectors
            // Using par_iter_mut for the first half and reading from the second half is tricky in Rust borrow checker
            // So we split at n
            let (db_left, db_right) = current_db.split_at_mut(n);
            let (q_left, q_right) = current_query.split_at_mut(n);

            db_left
                .par_iter_mut()
                .zip(db_right.par_iter())
                .zip(q_left.par_iter_mut())
                .zip(q_right.par_iter())
                .for_each(|(((d_l, d_r), q_l), q_r)| {
                    *d_l = one_plus_r * *d_l + r * *d_r;
                    *q_l = one_plus_r * *q_l + r * *q_r;
                });

            // Truncate to the new size n
            current_db.truncate(n);
            current_query.truncate(n);
        }

        SumCheckProof {
            round_polynomials,
            sum: sum_f.val(),
        }
    }
}

pub struct SumCheckVerifier;

impl SumCheckVerifier {
    pub fn verify(
        num_vars: usize,
        sum: BinaryField128b,
        proof: &SumCheckProof,
        challenges: &[BinaryField128b],
    ) -> bool {
        if proof.round_polynomials.len() != num_vars || challenges.len() != num_vars {
            return false;
        }

        let mut expected_sum = sum;
        let alpha = BinaryField128b::new(2);

        for (i, poly) in proof.round_polynomials.iter().enumerate() {
            let evals = [
                BinaryField128b::new(poly.evals[0]),
                BinaryField128b::new(poly.evals[1]),
                BinaryField128b::new(poly.evals[2]),
            ];

            // 1. Check consistency with previous round
            if evals[0] + evals[1] != expected_sum {
                return false;
            }

            // 2. Interpolate
            let r = challenges[i];
            let inv_alpha = alpha.invert().unwrap();
            let inv_one_plus_alpha = (BinaryField128b::ONE + alpha).invert().unwrap();

            let l0 = (r + BinaryField128b::ONE) * (r + alpha) * inv_alpha;
            let l1 = r * (r + alpha) * inv_one_plus_alpha;
            let l_alpha = r * (r + BinaryField128b::ONE) * inv_alpha * inv_one_plus_alpha;

            expected_sum = l0 * evals[0] + l1 * evals[1] + l_alpha * evals[2];
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sumcheck_basic() {
        let n = 16;
        let mut db = vec![BinaryField128b::ZERO; n];
        let mut query = vec![BinaryField128b::ZERO; n];

        for i in 0..n {
            db[i] = BinaryField128b::new(i as u128);
            query[i] = BinaryField128b::new((i + 1) as u128);
        }

        let prover = SumCheckProver::new(db, query);
        let challenges: Vec<_> = (0..4)
            .map(|i| BinaryField128b::new(i as u128 + 100))
            .collect();

        let proof = prover.prove(&challenges);

        let is_valid =
            SumCheckVerifier::verify(4, BinaryField128b::new(proof.sum), &proof, &challenges);
        assert!(is_valid);
    }
}
