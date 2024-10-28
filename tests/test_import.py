import unittest

class TestImports(unittest.TestCase):

	def test_util(self):
		from mirror import util
	
	def test_pivot(self):
		from mirror import pivot

	def test_scan(self):
		from mirror import scan

	def test_search(self):
		from mirror import search

	def test_paired_paths(self):
		from mirror import paired_paths

	def test_sequence(self):
		from mirror import sequence

if __name__ == '__main__':
	unittest.main()
