// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2023 Yulong Ming (myl7)

//! See [`Dpf`].

use bitvec::prelude::*;
#[cfg(feature = "multi-thread")]
use rayon::prelude::*;

use crate::group::Group;
use crate::utils::{xor, xor_inplace};
pub use crate::PointFn;
use crate::{Cw, Prg, Share};

/// API of distributed point functions (DPFs).
///
/// `PointFn` used here means `$f(x) = \beta$` iff. `$x = \alpha$`, otherwise `$f(x) = 0$`.
///
/// - See [`PointFn`] for `IN_BLEN` and `OUT_BLEN`.
/// - See [`DpfImpl`] for the implementation.
pub trait Dpf<const IN_BLEN: usize, const OUT_BLEN: usize, G>
where
    G: Group<OUT_BLEN>,
{
    /// `s0s` is `$s^{(0)}_0$` and `$s^{(0)}_1$` which should be randomly sampled.
    fn gen(
        &self,
        f: &PointFn<IN_BLEN, OUT_BLEN, G>,
        s0s: [&[u8; OUT_BLEN]; 2],
    ) -> Share<OUT_BLEN, G>;

    /// `b` is the party. `false` is 0 and `true` is 1.
    fn eval(&self, b: bool, k: &Share<OUT_BLEN, G>, xs: &[&[u8; IN_BLEN]], ys: &mut [&mut G]);

    /// Full domain eval.
    /// See [`Dpf::eval`] for `b`.
    /// The corresponding `xs` to `ys` is the big endian representation of `0..=u*::MAX`.
    fn full_eval(&self, b: bool, k: &Share<OUT_BLEN, G>, ys: &mut [&mut G]);

    /// Evaluate a contiguous range of the domain.
    ///
    /// Evaluates indices `start_idx..start_idx + ys.len()` and writes results to `ys`.
    /// This enables chunked evaluation without allocating the full domain.
    ///
    /// # Arguments
    /// * `b` - party bit (false = party 0, true = party 1)
    /// * `k` - the DPF key share
    /// * `start_idx` - starting index in the domain
    /// * `ys` - output buffer (length determines how many indices to evaluate)
    fn eval_range(&self, b: bool, k: &Share<OUT_BLEN, G>, start_idx: usize, ys: &mut [G]);
}

/// Implementation of [`Dpf`].
pub struct DpfImpl<const IN_BLEN: usize, const OUT_BLEN: usize, P>
where
    P: Prg<OUT_BLEN, 1>,
{
    prg: P,
    filter_bitn: usize,
}

impl<const IN_BLEN: usize, const OUT_BLEN: usize, P> DpfImpl<IN_BLEN, OUT_BLEN, P>
where
    P: Prg<OUT_BLEN, 1>,
{
    pub fn new(prg: P) -> Self {
        Self {
            prg,
            filter_bitn: 8 * IN_BLEN,
        }
    }

    pub fn new_with_filter(prg: P, filter_bitn: usize) -> Self {
        assert!(filter_bitn <= 8 * IN_BLEN && filter_bitn > 1);
        Self { prg, filter_bitn }
    }
}

const IDX_L: usize = 0;
const IDX_R: usize = 1;

impl<const IN_BLEN: usize, const OUT_BLEN: usize, P, G> Dpf<IN_BLEN, OUT_BLEN, G>
    for DpfImpl<IN_BLEN, OUT_BLEN, P>
where
    P: Prg<OUT_BLEN, 1>,
    G: Group<OUT_BLEN>,
{
    fn gen(
        &self,
        f: &PointFn<IN_BLEN, OUT_BLEN, G>,
        s0s: [&[u8; OUT_BLEN]; 2],
    ) -> Share<OUT_BLEN, G> {
        // The bit size of `$\alpha$`.
        let n = self.filter_bitn;
        // Set `$s^{(1)}_0$` and `$s^{(1)}_1$`.
        let mut ss_prev = [s0s[0].to_owned(), s0s[1].to_owned()];
        // Set `$t^{(0)}_0$` and `$t^{(0)}_1$`.
        let mut ts_prev = [false, true];
        let mut cws = Vec::<Cw<OUT_BLEN, G>>::with_capacity(n);
        for i in 0..n {
            // MSB is required since we index from high to low in arrays.
            let alpha_i = f.alpha.view_bits::<Msb0>()[i];
            let [([s0l], t0l), ([s0r], t0r)] = self.prg.gen(&ss_prev[0]);
            let [([s1l], t1l), ([s1r], t1r)] = self.prg.gen(&ss_prev[1]);
            let (keep, lose) = if alpha_i {
                (IDX_R, IDX_L)
            } else {
                (IDX_L, IDX_R)
            };
            let s_cw = xor(&[[&s0l, &s0r][lose], [&s1l, &s1r][lose]]);
            let tl_cw = t0l ^ t1l ^ alpha_i ^ true;
            let tr_cw = t0r ^ t1r ^ alpha_i;
            let cw = Cw {
                s: s_cw,
                v: G::zero(),
                tl: tl_cw,
                tr: tr_cw,
            };
            cws.push(cw);
            ss_prev = [
                xor(&[
                    [&s0l, &s0r][keep],
                    if ts_prev[0] { &s_cw } else { &[0; OUT_BLEN] },
                ]),
                xor(&[
                    [&s1l, &s1r][keep],
                    if ts_prev[1] { &s_cw } else { &[0; OUT_BLEN] },
                ]),
            ];
            ts_prev = [
                [t0l, t0r][keep] ^ (ts_prev[0] & [tl_cw, tr_cw][keep]),
                [t1l, t1r][keep] ^ (ts_prev[1] & [tl_cw, tr_cw][keep]),
            ];
        }
        let cw_np1 =
            (f.beta.clone() + -Into::<G>::into(ss_prev[0]) + ss_prev[1].into()).neg_if(ts_prev[1]);
        Share {
            s0s: vec![s0s[0].to_owned(), s0s[1].to_owned()],
            cws,
            cw_np1,
        }
    }

    fn eval(&self, b: bool, k: &Share<OUT_BLEN, G>, xs: &[&[u8; IN_BLEN]], ys: &mut [&mut G]) {
        #[cfg(feature = "multi-thread")]
        self.eval_mt(b, k, xs, ys);
        #[cfg(not(feature = "multi-thread"))]
        self.eval_st(b, k, xs, ys);
    }

    fn full_eval(&self, b: bool, k: &Share<OUT_BLEN, G>, ys: &mut [&mut G]) {
        let n = k.cws.len();
        assert_eq!(n, self.filter_bitn);

        let s = k.s0s[0];
        let t = b;
        self.full_eval_layer(b, k, ys, 0, (s, t));
    }

    fn eval_range(&self, b: bool, k: &Share<OUT_BLEN, G>, start_idx: usize, ys: &mut [G]) {
        let n = k.cws.len();
        assert_eq!(n, self.filter_bitn);

        if ys.is_empty() {
            return;
        }

        let domain_size = 1usize
            .checked_shl(self.filter_bitn as u32)
            .expect("eval_range: domain_size overflow (filter_bitn too large)");
        let end_idx = start_idx
            .checked_add(ys.len())
            .expect("eval_range: end_idx overflow");
        assert!(
            end_idx <= domain_size,
            "eval_range: end_idx {} exceeds domain_size {}",
            end_idx,
            domain_size
        );

        let s = k.s0s[0];
        let t = b;
        self.eval_range_layer(b, k, start_idx, end_idx, ys, 0, 0, domain_size, (s, t));
    }
}

impl<const IN_BLEN: usize, const OUT_BLEN: usize, P> DpfImpl<IN_BLEN, OUT_BLEN, P>
where
    P: Prg<OUT_BLEN, 1>,
{
    /// Eval with single-threading.
    /// See [`Dpf::eval`].
    pub fn eval_st<G>(
        &self,
        b: bool,
        k: &Share<OUT_BLEN, G>,
        xs: &[&[u8; IN_BLEN]],
        ys: &mut [&mut G],
    ) where
        G: Group<OUT_BLEN>,
    {
        xs.iter()
            .zip(ys.iter_mut())
            .for_each(|(x, y)| self.eval_point(b, k, x, y));
    }

    #[cfg(feature = "multi-thread")]
    /// Eval with multi-threading.
    /// See [`Dpf::eval`].
    pub fn eval_mt<G>(
        &self,
        b: bool,
        k: &Share<OUT_BLEN, G>,
        xs: &[&[u8; IN_BLEN]],
        ys: &mut [&mut G],
    ) where
        G: Group<OUT_BLEN>,
    {
        xs.par_iter()
            .zip(ys.par_iter_mut())
            .for_each(|(x, y)| self.eval_point(b, k, x, y));
    }

    fn full_eval_layer<G>(
        &self,
        b: bool,
        k: &Share<OUT_BLEN, G>,
        ys: &mut [&mut G],
        layer_i: usize,
        (s, t): ([u8; OUT_BLEN], bool),
    ) where
        G: Group<OUT_BLEN>,
    {
        assert_eq!(ys.len(), 1 << (self.filter_bitn - layer_i));
        if ys.len() == 1 {
            *ys[0] = (Into::<G>::into(s) + if t { k.cw_np1.clone() } else { G::zero() }).neg_if(b);
            return;
        }

        let cw = &k.cws[layer_i];
        let [([mut sl], mut tl), ([mut sr], mut tr)] = self.prg.gen(&s);
        xor_inplace(&mut sl, &[if t { &cw.s } else { &[0; OUT_BLEN] }]);
        xor_inplace(&mut sr, &[if t { &cw.s } else { &[0; OUT_BLEN] }]);
        tl ^= t & cw.tl;
        tr ^= t & cw.tr;

        let (ys_l, ys_r) = ys.split_at_mut(ys.len() / 2);
        #[cfg(feature = "multi-thread")]
        rayon::join(
            || self.full_eval_layer(b, k, ys_l, layer_i + 1, (sl, tl)),
            || self.full_eval_layer(b, k, ys_r, layer_i + 1, (sr, tr)),
        );
        #[cfg(not(feature = "multi-thread"))]
        {
            self.full_eval_layer(b, k, ys_l, layer_i + 1, (sl, tl));
            self.full_eval_layer(b, k, ys_r, layer_i + 1, (sr, tr));
        }
    }

    /// Evaluate a range of the domain by descending only into relevant subtrees.
    ///
    /// # Arguments
    /// * `range_start`, `range_end` - the requested range [range_start, range_end)
    /// * `ys` - output buffer, written at offset (current subtree start - range_start)
    /// * `subtree_start`, `subtree_end` - the domain range covered by this subtree
    #[allow(clippy::too_many_arguments)]
    fn eval_range_layer<G>(
        &self,
        b: bool,
        k: &Share<OUT_BLEN, G>,
        range_start: usize,
        range_end: usize,
        ys: &mut [G],
        layer_i: usize,
        subtree_start: usize,
        subtree_end: usize,
        (s, t): ([u8; OUT_BLEN], bool),
    ) where
        G: Group<OUT_BLEN>,
    {
        debug_assert!(
            range_start < range_end,
            "empty range passed to eval_range_layer"
        );
        debug_assert_eq!(
            ys.len(),
            range_end - range_start,
            "ys.len() must equal range size"
        );

        let subtree_size = subtree_end - subtree_start;
        if subtree_size == 1 {
            debug_assert_eq!(ys.len(), 1, "leaf node but ys.len() != 1");
            ys[0] = (Into::<G>::into(s) + if t { k.cw_np1.clone() } else { G::zero() }).neg_if(b);
            return;
        }

        let cw = &k.cws[layer_i];
        let [([mut sl], mut tl), ([mut sr], mut tr)] = self.prg.gen(&s);
        xor_inplace(&mut sl, &[if t { &cw.s } else { &[0; OUT_BLEN] }]);
        xor_inplace(&mut sr, &[if t { &cw.s } else { &[0; OUT_BLEN] }]);
        tl ^= t & cw.tl;
        tr ^= t & cw.tr;

        let mid = subtree_start + subtree_size / 2;

        let left_start = range_start.max(subtree_start);
        let left_end = range_end.min(mid);
        let left_overlaps = left_start < left_end;

        let right_start = range_start.max(mid);
        let right_end = range_end.min(subtree_end);
        let right_overlaps = right_start < right_end;

        match (left_overlaps, right_overlaps) {
            (true, true) => {
                let left_count = left_end - left_start;
                #[cfg(feature = "multi-thread")]
                {
                    let (ys_l, ys_r) = ys.split_at_mut(left_count);
                    rayon::join(
                        || {
                            self.eval_range_layer(
                                b,
                                k,
                                left_start,
                                left_end,
                                ys_l,
                                layer_i + 1,
                                subtree_start,
                                mid,
                                (sl, tl),
                            )
                        },
                        || {
                            self.eval_range_layer(
                                b,
                                k,
                                right_start,
                                right_end,
                                ys_r,
                                layer_i + 1,
                                mid,
                                subtree_end,
                                (sr, tr),
                            )
                        },
                    );
                }
                #[cfg(not(feature = "multi-thread"))]
                {
                    let (ys_l, ys_r) = ys.split_at_mut(left_count);
                    self.eval_range_layer(
                        b,
                        k,
                        left_start,
                        left_end,
                        ys_l,
                        layer_i + 1,
                        subtree_start,
                        mid,
                        (sl, tl),
                    );
                    self.eval_range_layer(
                        b,
                        k,
                        right_start,
                        right_end,
                        ys_r,
                        layer_i + 1,
                        mid,
                        subtree_end,
                        (sr, tr),
                    );
                }
            }
            (true, false) => {
                self.eval_range_layer(
                    b,
                    k,
                    left_start,
                    left_end,
                    ys,
                    layer_i + 1,
                    subtree_start,
                    mid,
                    (sl, tl),
                );
            }
            (false, true) => {
                self.eval_range_layer(
                    b,
                    k,
                    right_start,
                    right_end,
                    ys,
                    layer_i + 1,
                    mid,
                    subtree_end,
                    (sr, tr),
                );
            }
            (false, false) => {}
        }
    }

    pub fn eval_point<G>(&self, b: bool, k: &Share<OUT_BLEN, G>, x: &[u8; IN_BLEN], y: &mut G)
    where
        G: Group<OUT_BLEN>,
    {
        let n = k.cws.len();
        assert_eq!(n, self.filter_bitn);
        let v = y;

        let mut s_prev = k.s0s[0];
        let mut t_prev = b;
        for i in 0..n {
            let cw = &k.cws[i];
            let [([mut sl], mut tl), ([mut sr], mut tr)] = self.prg.gen(&s_prev);
            xor_inplace(&mut sl, &[if t_prev { &cw.s } else { &[0; OUT_BLEN] }]);
            xor_inplace(&mut sr, &[if t_prev { &cw.s } else { &[0; OUT_BLEN] }]);
            tl ^= t_prev & cw.tl;
            tr ^= t_prev & cw.tr;
            if x.view_bits::<Msb0>()[i] {
                s_prev = sr;
                t_prev = tr;
            } else {
                s_prev = sl;
                t_prev = tl;
            }
        }
        *v =
            (Into::<G>::into(s_prev) + if t_prev { k.cw_np1.clone() } else { G::zero() }).neg_if(b);
    }
}

#[cfg(all(test, feature = "prg"))]
mod tests {
    use arbtest::arbtest;
    use std::iter;

    use super::*;
    use crate::group::byte::ByteGroup;
    use crate::prg::Aes128MatyasMeyerOseasPrg;

    type GroupImpl = ByteGroup<16>;
    type PrgImpl = Aes128MatyasMeyerOseasPrg<16, 1, 2>;
    type DpfImplImpl = DpfImpl<2, 16, PrgImpl>;

    #[test]
    fn test_correctness() {
        arbtest(|u| {
            let keys: [[u8; 16]; 2] = u.arbitrary()?;
            let prg = PrgImpl::new(&std::array::from_fn(|i| &keys[i]));
            let filter_bitn = u.arbitrary::<usize>()? % 15 + 2; // 2..=16
            let dpf = DpfImplImpl::new_with_filter(prg, filter_bitn);
            let s0s: [[u8; 16]; 2] = u.arbitrary()?;
            let alpha_i: u16 = u.arbitrary::<u16>()? >> (16 - filter_bitn);
            let alpha: [u8; 2] = (alpha_i << (16 - filter_bitn)).to_be_bytes();
            let beta: [u8; 16] = u.arbitrary()?;
            let f = PointFn {
                alpha,
                beta: beta.into(),
            };
            let k = dpf.gen(&f, [&s0s[0], &s0s[1]]);
            let mut k0 = k.clone();
            k0.s0s = vec![k0.s0s[0]];
            let mut k1 = k;
            k1.s0s = vec![k1.s0s[1]];

            let xs: Vec<_> = (0u16..=u16::MAX >> (16 - filter_bitn))
                .map(|i| (i << (16 - filter_bitn)).to_be_bytes())
                .collect();
            assert_eq!(xs.len(), 1 << filter_bitn);
            let xs_lt_num = alpha_i;
            let xs_gt_num = (u16::MAX >> (16 - filter_bitn)) - alpha_i;
            let ys_expected: Vec<_> = iter::repeat_n(beta.into(), xs_lt_num as usize)
                .chain([beta.into()])
                .chain(iter::repeat_n(GroupImpl::zero(), xs_gt_num as usize))
                .collect();

            let mut ys0 = vec![GroupImpl::zero(); xs.len()];
            let mut ys1 = vec![GroupImpl::zero(); xs.len()];
            dpf.eval(
                false,
                &k0,
                &xs.iter().collect::<Vec<_>>(),
                &mut ys0.iter_mut().collect::<Vec<_>>(),
            );
            ys0.iter().for_each(|y0| {
                assert_ne!(*y0, GroupImpl::zero());
                assert_ne!(*y0, [0xff; 16].into());
            });
            dpf.eval(
                true,
                &k1,
                &xs.iter().collect::<Vec<_>>(),
                &mut ys1.iter_mut().collect::<Vec<_>>(),
            );
            ys1.iter().for_each(|y1| {
                assert_ne!(*y1, GroupImpl::zero());
                assert_ne!(*y1, [0xff; 16].into());
            });
            let ys: Vec<_> = ys0
                .iter()
                .zip(ys1.iter())
                .map(|(y0, y1)| y0.clone() + y1.clone())
                .collect();
            assert_ys_eq(&ys, &ys_expected, &xs, &alpha);

            let mut ys0_full_eval = vec![ByteGroup::zero(); 1 << filter_bitn];
            dpf.full_eval(
                false,
                &k0,
                &mut ys0_full_eval.iter_mut().collect::<Vec<_>>(),
            );
            assert_ys_eq(&ys0_full_eval, &ys0, &xs, &alpha);
            let mut ys1_full_eval = vec![ByteGroup::zero(); 1 << filter_bitn];
            dpf.full_eval(true, &k1, &mut ys1_full_eval.iter_mut().collect::<Vec<_>>());
            assert_ys_eq(&ys1_full_eval, &ys1, &xs, &alpha);

            Ok(())
        });
    }

    fn assert_ys_eq(ys: &[GroupImpl], ys_expected: &[GroupImpl], xs: &[[u8; 2]], alpha: &[u8; 2]) {
        let alpha_int = u16::from_be_bytes(*alpha);
        for (i, (x, (y, y_expected))) in
            xs.iter().zip(ys.iter().zip(ys_expected.iter())).enumerate()
        {
            let x_int = u16::from_be_bytes(*x);
            let cmp = if x_int < alpha_int {
                "<"
            } else if x_int > alpha_int {
                ">"
            } else {
                "="
            };
            assert_eq!(y, y_expected, "where i={}, x={:?}, x{}alpha", i, *x, cmp);
        }
    }

    #[test]
    fn test_eval_range_matches_full_eval() {
        let prg_keys: [[u8; 16]; 2] = rand::random();
        let prg = PrgImpl::new(&std::array::from_fn(|i| &prg_keys[i]));
        let dpf = DpfImpl::<1, 16, _>::new(prg);

        let target: u8 = 42;
        let alpha = [target];
        let beta = ByteGroup([0xFF; 16]);
        let f = PointFn { alpha, beta };

        let s0s: [[u8; 16]; 2] = rand::random();
        let share = dpf.gen(&f, [&s0s[0], &s0s[1]]);
        let mut k0 = share.clone();
        k0.s0s = vec![k0.s0s[0]];

        let mut full_output = vec![ByteGroup::zero(); 256];
        dpf.full_eval(false, &k0, &mut full_output.iter_mut().collect::<Vec<_>>());

        let mut range_output = vec![ByteGroup::zero(); 64];
        dpf.eval_range(false, &k0, 32, &mut range_output);

        for i in 0..64 {
            assert_eq!(
                range_output[i],
                full_output[32 + i],
                "mismatch at index {} (domain idx {})",
                i,
                32 + i
            );
        }

        let mut single = vec![ByteGroup::zero(); 1];
        dpf.eval_range(false, &k0, target as usize, &mut single);
        assert_eq!(single[0], full_output[target as usize]);
    }

    #[test]
    fn test_eval_range_at_boundaries() {
        let prg_keys: [[u8; 16]; 2] = rand::random();
        let prg = PrgImpl::new(&std::array::from_fn(|i| &prg_keys[i]));
        let dpf = DpfImpl::<1, 16, _>::new(prg);

        let target: u8 = 200;
        let alpha = [target];
        let beta = ByteGroup([0xAB; 16]);
        let f = PointFn { alpha, beta };

        let s0s: [[u8; 16]; 2] = rand::random();
        let share = dpf.gen(&f, [&s0s[0], &s0s[1]]);
        let mut k0 = share.clone();
        k0.s0s = vec![k0.s0s[0]];

        let mut full_output = vec![ByteGroup::zero(); 256];
        dpf.full_eval(false, &k0, &mut full_output.iter_mut().collect::<Vec<_>>());

        let mut first_chunk = vec![ByteGroup::zero(); 16];
        dpf.eval_range(false, &k0, 0, &mut first_chunk);
        for i in 0..16 {
            assert_eq!(
                first_chunk[i], full_output[i],
                "first chunk mismatch at {}",
                i
            );
        }

        let mut last_chunk = vec![ByteGroup::zero(); 16];
        dpf.eval_range(false, &k0, 240, &mut last_chunk);
        for i in 0..16 {
            assert_eq!(
                last_chunk[i],
                full_output[240 + i],
                "last chunk mismatch at {}",
                i
            );
        }

        let mut entire = vec![ByteGroup::zero(); 256];
        dpf.eval_range(false, &k0, 0, &mut entire);
        for i in 0..256 {
            assert_eq!(entire[i], full_output[i], "entire domain mismatch at {}", i);
        }
    }
}
