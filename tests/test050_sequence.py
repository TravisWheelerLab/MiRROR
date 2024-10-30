import unittest
from mirror.pivot import Pivot

EXAMPLE_1 = {
    "peptide" : 'SIAGGLQTIGR',
    "spectrum" : [175.11895291, 201.12336998, 232.14041701, 272.16048414, 329.18194823, 345.22448136, 386.20341233, 446.27216058, 499.28747668, 574.33073884, 627.34605493, 687.41480319, 728.39373416, 744.43626729, 801.45773138, 841.47779851, 872.49484554, 898.4992626 , 985.57890989],
    "pivot" : Pivot((446.27216058327105, 574.330738838471),
                    (499.2874766789711, 627.3460549341711),
                    (7, 9),
                    (8, 10)),
    "asc_edges" : [(9, 11), (11, 13), (11, 14), (10, 12), (12, 15), (13, 14), (13, 15), (13, 16), (14, 16), (14, 17), (15, 17), (16, 18)],
    "desc_edges": [(2, 0), (3, 0), (3, 1), (4, 1), (4, 2), (4, 3), (5, 2), (6, 3), (6, 4), (7, 5), (8, 6)]
}

class Test050_Sequence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(f"\n{cls.__name__}")
    
    def test_spectrum_graphs(self):
        pass
    
    def test_partial_sequences(self):
        pass
    
    def test_candidate_sequences(self):
        # not yet implemented
        pass