import unittest

from numpy.random import uniform, randint
from numpy import arange, argsort, array, zeros_like

from mirror.search import *
from mirror.util import AMINO_MASS_MONO

def dummy_gaps(size, mass_alphabet):
    positions = sorted(uniform(0, 1000, size = size))
    gap_types = randint(0, len(mass_alphabet), size = size)
    mz = []
    for (pos, gap_idx) in zip(positions, gap_types):
        mz.append(pos)
        mz.append(pos + mass_alphabet[gap_idx])
    mz = array(mz)
    mz_sorter = argsort(mz)
    sorter_inv = zeros_like(mz_sorter)
    for i in range(2 * size):
        sorter_inv[mz_sorter[i]] = i
    mz = mz[mz_sorter]
    gap_indexer_arrs = [sorter_inv[2 * i: 2 * i + 2] for i in range(size)]
    gap_indices = [tuple(g) for g in gap_indexer_arrs]
    gap_mz = [mz[g] for g in gap_indexer_arrs]
    return mz, (gap_indices, gap_mz)

def _dummy_pivots(size, gap_mz, offsets):
    mz = []
    for offset, gap in zip(offsets,gap_mz):
        paired_gap = gap + offset
        mz.extend(gap)
        mz.extend(paired_gap)
    mz = array(mz)
    mz_sorter = argsort(mz)
    sorter_inv = zeros_like(mz_sorter)
    for i in range(4 * size):
        sorter_inv[mz_sorter[i]] = i
    mz = mz[mz_sorter]
    pivot_indexer_arrs = [sorter_inv[4 * i: 4 * i + 4] for i in range(size)]
    pivot_indices = [tuple(p) for p in pivot_indexer_arrs]
    pivot_mz = [mz[p] for p in pivot_indexer_arrs]
    return mz, [Pivot(x[:2],x[2:],i[:2],i[2:]) for (x,i) in zip(pivot_mz, pivot_indices)]

def dummy_disjoint_pivots(size, mass_alphabet):
    gap_mz = dummy_gaps(size, mass_alphabet)[1][1]
    offsets = uniform(2 * max(mass_alphabet), 500, size = size)
    mz, pivots = _dummy_pivots(size, gap_mz, offsets)
    assert all(PivotDisjointConstraint(0).is_ordered(p.pair_a, p.pair_b) for p in pivots)
    return mz, pivots

def dummy_overlap_pivots(size, mass_alphabet):
    gap_mz = dummy_gaps(size, mass_alphabet)[1][1]
    offsets = [uniform(min(mass_alphabet) / 2, 0.9 * (gap[1] - gap[0])) for gap in gap_mz]
    mz, pivots = _dummy_pivots(size, gap_mz, offsets)
    assert all(PivotOverlapConstraint(0).is_ordered(p.pair_a, p.pair_b) for p in pivots)
    return mz, pivots

class Test030_Search(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print(f"\n{cls.__name__}")

    def test_gap_constraints(self):
        # gap target looks for a particular gap
        pos_pair = (1,3.5)
        neg_pair = (1,10)
        neg_pair2 = (1,1.5)
        target_gap = 2.45
        target_threshold = 0.1
        gt = GapTargetConstraint(target_gap, target_threshold)
        self.assertTrue(
            gt.match(gt.evaluate(pos_pair))
        )
        self.assertFalse(
            gt.match(gt.evaluate(neg_pair))
        )
        self.assertFalse(
            gt.match(gt.evaluate(neg_pair2))
        )
        # gap range looks for gaps in an interval
        pos_pair = (1,3.5)
        pos_pair2 = (1,10)
        pos_pair3 = (1,7)
        neg_pair = (1,15)
        neg_pair2 = (1,1.5)
        gaps = [2.45,8.9]
        range_threshold = 0.1
        gr = GapRangeConstraint(gaps, range_threshold)
        self.assertTrue(
            gt.match(gt.evaluate(pos_pair))
        )
        self.assertTrue(
            gr.match(gr.evaluate(pos_pair2))
        )
        self.assertTrue(
            gr.match(gr.evaluate(pos_pair3))
        )
        self.assertFalse(
            gr.match(gr.evaluate(neg_pair))
        )
        self.assertFalse(
            gr.match(gr.evaluate(neg_pair2))
        )

    def test_gap_target(self,
        size = 100, 
        tolerance = 0.1, 
        mass_alphabet = AMINO_MASS_MONO
    ):
        for gap in mass_alphabet:
            mz, out = dummy_gaps(size, [gap])
            gt = GapTargetConstraint(gap, tolerance)
            real_out = find_gaps(mz, gt)
            out_idx = set(out[0])
            real_out_idx = set(real_out[0])
            self.assertTrue(out_idx.issubset(real_out_idx))

    def test_gap_range(self, 
        size = 100, 
        tolerance = 0.1, 
        mass_alphabet = AMINO_MASS_MONO
    ):
        mz, out = dummy_gaps(size, mass_alphabet)
        gr = GapRangeConstraint(mass_alphabet, tolerance)
        real_out = find_gaps(mz, gr)
        out_idx = set(out[0])
        real_out_idx = set(real_out[0])
        self.assertTrue(out_idx.issubset(real_out_idx))

    def test_pivot_constraints(self):
        so = ((1,3),(2,4))
        sd = ((1,2),(3,4))
        # disjoint pivots
        pd = PivotDisjointConstraint(0)
        self.assertTrue(
            pd.match(pd.evaluate(sd))
        )
        self.assertFalse(
            pd.match(pd.evaluate(so))
        )
        # overlapping pivots
        po = PivotOverlapConstraint(0)
        self.assertTrue(
            po.match(po.evaluate(so))
        )
        self.assertFalse(
            po.match(po.evaluate(sd))
        )

    def _pivots_inclusion(self, pivots_a, pivots_b):
        # check if all pivots_a are in pivots_b
        pivots_a = set(tuple(p.data) for p in pivots_a)
        pivots_b = set(tuple(p.data) for p in pivots_b)
        return pivots_a.issubset(pivots_b)

    def _test_pivot(self,
        mz,
        mass_alphabet,
        gap_tolerance,
        pivot_constraint,
        synth_pivots
    ):
        # use gap range
        gr_indices, gr_mz = find_gaps(mz, GapRangeConstraint(mass_alphabet, gap_tolerance))
        gr_pivots = find_pivots(gr_indices, gr_mz, pivot_constraint)
        self.assertTrue(
            self._pivots_inclusion(synth_pivots, gr_pivots)
        )

        # use gap targeting
        gt_pivots = []
        for mass in mass_alphabet:
            gt_indices, gt_mz = find_gaps(mz, GapTargetConstraint(mass, gap_tolerance))
            gt_pivots.extend(
                find_pivots(gt_indices, gt_mz, pivot_constraint))
        self.assertTrue(
            self._pivots_inclusion(synth_pivots, gt_pivots)
        )

    def test_pivot_disjoint(self,
        size = 100, 
        gap_tolerance = 0.1, 
        pivot_tolerance = 0.1, 
        mass_alphabet = AMINO_MASS_MONO
    ):
        mz, synth_pivots = dummy_disjoint_pivots(size, mass_alphabet)
        self._test_pivot(mz, mass_alphabet, gap_tolerance, PivotDisjointConstraint(pivot_tolerance), synth_pivots)
        
    def test_pivot_overlap(self,
        size = 100, 
        gap_tolerance = 0.1, 
        pivot_tolerance = 0.1, 
        mass_alphabet = AMINO_MASS_MONO
    ):
        mz, synth_pivots = dummy_overlap_pivots(size, mass_alphabet)
        self._test_pivot(mz, mass_alphabet, gap_tolerance, PivotOverlapConstraint(pivot_tolerance), synth_pivots)

