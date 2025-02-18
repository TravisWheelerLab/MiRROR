import unittest

import numpy as np
from mirror import TargetSpace, GapResult, collapse_second_order_list

TRUE_GAPS = [np.array([10,11]),np.array([50,51/2]),np.array([103,104/3])]
TRUE_CHARGES = [np.array([1,1]),np.array([1,2]),np.array([1,3])]
PEAKS = np.array(collapse_second_order_list([g * c for (g, c) in zip(TRUE_GAPS, TRUE_CHARGES)]))
TARGET_GROUPS = [[1]]
CHARGE_STATES = [1,2,3]
RESIDUES = ['A']
TOLERANCE = 0

class TestGaps(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print(f"\n{cls.__name__}")
    
    def test_gaps(self):
        target_space = TargetSpace(TARGET_GROUPS, RESIDUES, TOLERANCE, CHARGE_STATES)
        gaps = target_space.find_gaps(PEAKS)[0]
        print("true gaps", TRUE_GAPS)
        print("true charges", TRUE_CHARGES)
        print("found gaps", [PEAKS[ind] for ind in gaps.indices()])
        print("found charges", [z for z in gaps.charge_state_pairs()])
        
if __name__ == "__main__":
    unittest.main()