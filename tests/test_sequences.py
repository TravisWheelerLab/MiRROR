import unittest

from mirror.sequences.suffix_array import SuffixArray
class TestSuffixArray(unittest.TestCase):
    
    def _get_suffix_array(self):
        return SuffixArray.create(
            path_to_fasta = "./data/sequences/test.fasta",
            path_to_suffix_array = "./data/output/test.sufr")
    
    def test_create(self):
        self._get_suffix_array()
    
    def test_count(self):
        queries = ["HVV", "ILE", "TPP"]
        counts = [2,2,9]
        self.assertEqual(
            counts,
            self._get_suffix_array().count(queries))
    
    def test_bisect(self):
        suffix_array = self._get_suffix_array()
        full_query = "ILEKL"
        counts = [64,10,2,1,0]
        prefix_result = None
        for i in range(len(full_query)):
            query = full_query[:i + 1]
            bisect_result = suffix_array.bisect([query], prefix_result)[0]
            count_result = suffix_array.count([query])[0]
            self.assertEqual(
                count_result,
                bisect_result.count)
            self.assertEqual(
                counts[i],
                bisect_result.count)
            if not(prefix_result is None):
                self.assertLessEqual(bisect_result.count, prefix_result.count)
            prefix_result = bisect_result