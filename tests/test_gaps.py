import unittest

import numpy as np
from mirror import TargetGroup, TargetSpace, GapResult, collapse_second_order_list

TRUE_GAPS = [np.array([10,11]),np.array([50,51]),np.array([103,104])]
TRUE_CHARGES = [np.array([1,1]),np.array([1,2]),np.array([1,3])]
PEAKS = np.array(collapse_second_order_list([g / c for (g, c) in zip(TRUE_GAPS, TRUE_CHARGES)]))
CHARGE_STATES = [1,2,3]
RESIDUES = ['A']
MASSES = [[1]]
MODIFIERS = [[]]
LOSSES = [[]]
TARGET_GROUPS = [TargetGroup(res, mass, mods, loss) for (res, mass, mods, loss) in zip(RESIDUES, MASSES, MODIFIERS, LOSSES)]
TOLERANCE = 0

class TestGaps(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print(f"\n{cls.__name__}")
    
    def test_gaps(self):
        target_space = TargetSpace(TARGET_GROUPS, TOLERANCE, CHARGE_STATES)
        gaps = target_space.find_gaps(PEAKS)[0]
        found_gaps = [PEAKS[ind] * z for (ind, z) in zip(gaps.indices(), gaps.charge_state_pairs())]
        found_charges = [z for z in gaps.charge_state_pairs()]
        assert (np.array(found_gaps) == np.array(TRUE_GAPS)).all()
        assert (np.array(found_charges) == np.array(TRUE_CHARGES)).all()

if __name__ == "__main__":
    unittest.main()