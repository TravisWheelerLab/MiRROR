from typing import Iterator
import unittest
import itertools as it

import mirror.util as util

class TestUtil(unittest.TestCase):

    @staticmethod
    def _powerset(x: Iterator) -> Iterator:
        """given a finite iterator `x`, return all subsets of list(x)."""
        s = list(x)
        return it.chain.from_iterable(it.combinations(s, r) for r in range(len(s) + 1))


    def test_disjoint_pairs(self):
        # generate all subsets [1,2,3,4,5] and find  all disjoint pairs of those subsets.
        x = list(self._powerset(range(10)))
        observed_disjoint_pairs = [(x[i],x[j]) for (i,j) in util.disjoint_pairs(x, x)]
        
        # check that all pairs are disjoint
        self.assertTrue(all(set(xj).isdisjoint(xi) for (xi,xj) in observed_disjoint_pairs))

        # check that every disjoint pair was found
        n = len(x)
        expected_disjoint_pairs = [(x[i],x[j]) for i in range(n) for j in range(i + 1, n) if set(x[i]).isdisjoint(x[j])]
        self.assertEqual(expected_disjoint_pairs,observed_disjoint_pairs)
