import unittest
from time import time

from mirror.util import log 
from mirror.io import load_spectrum_from_mzML

TEST_DATA = "./tests/data/BY_04_1.mzML"

class Test001_IO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(f"\n{cls.__name__}")
    
    def test_io(self, path_to_data = TEST_DATA):
        log("loading spectra")
        load_t = time()
        exp = load_spectrum_from_mzML(path_to_data)
        load_t = time() - load_t
        print(load_t, "s")