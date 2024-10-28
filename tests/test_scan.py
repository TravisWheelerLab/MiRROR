import unittest

from numpy.random import uniform
from time import time

from mirror.scan import ScanConstraint, constrained_pair_scan
from mirror.util import add_tqdm

class DummyConstraint(ScanConstraint):
	def __init__(self, threshold):
		self.threshold = threshold
	def evaluate(self, state):
		return state[1] - state[0]
	def stop(self, val):
		return val > self.threshold
	def match(self, val):
		return val <= self.threshold

def dummy_data(n):
	return sorted(uniform(size=n))

def _naiive_pair_scan(arr, threshold):
	n = len(arr)
	for i in range(n):
		for j in range(i + 1, n):
			if arr[j] - arr[i] <= threshold:
				yield (i,j)

class TestScan(unittest.TestCase):

	def naiive_pair_scan(self, arr, threshold):
		return list(_naiive_pair_scan(arr, threshold))

	def constrained_pair_scan(self, arr, threshold):
		return list(constrained_pair_scan(arr, DummyConstraint(threshold)))

	def test_scan(self, tries = 10, size = 10000, threshold = 0.02):
		print(f"\ntest_scan tries={tries} size={size} threshold={threshold}")
		t_elap_n = 0
		t_elap_c = 0
		for _ in add_tqdm(range(tries),tries):
			# generate a sorted list of values between 0 and 1
			x = dummy_data(size)
			# naiive search with the difference condition
			t_init_n = time()
			naiive_results = self.naiive_pair_scan(x, threshold)
			t_elap_n += time() - t_init_n
			# constrained search with the same condition
			t_init_c = time()
			constrained_results = self.constrained_pair_scan(x, threshold)
			t_elap_c += time() - t_init_c
			# check equality
			self.assertEqual(naiive_results,constrained_results)
		print(f"\tnaiive scan time (avg)\t{t_elap_n / tries}\n\tconstr. scan time (avg)\t{t_elap_c / tries}")

if __name__ == '__main__':
	unittest.main()
