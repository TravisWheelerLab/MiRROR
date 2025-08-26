import unittest
from math import sqrt
from time import time
from tqdm import tqdm
from dataclasses import asdict

from mirror.spectra.types import BenchmarkPeakList
from mirror.presets import MONO_ANNOTATION_PARAMS, AVG_ANNOTATION_PARAMS, VALIDATION_PEPTIDES
from mirror.annotation import AnnotationParams, AnnotationResult, annotate

from tests.test_spectra import VALIDATION_SIMS

class TestAnnotation(unittest.TestCase):

    def test(self,
        params = MONO_ANNOTATION_PARAMS,
        dir = "./data/output/annotation/"):
        print(params)
        for (i, (peptide, mode, charges, sim_bpl)) in enumerate(VALIDATION_SIMS):
            print(i, peptide, mode, charges)
            print(sim_bpl.mz)
            anno = annotate(sim_bpl, MONO_ANNOTATION_PARAMS)
            fpath = dir + f"{i}_{peptide}_{mode}_{charges}.ann"
            anno.write(fpath)
            anno2 = AnnotationResult.read(fpath)
            print(anno._profile)
            try:
                for (a, a2) in zip(anno.pairs, anno2.pairs):
                    self.assertEqual(asdict(a), asdict(a2))
                for (a, a2) in zip(anno.pivots, anno2.pivots):
                    self.assertEqual(asdict(a), asdict(a2))
                for (a, a2) in zip(anno.left_boundaries, anno2.left_boundaries):
                    self.assertEqual(asdict(a), asdict(a2))
                for (x, x2) in zip(anno.right_boundaries, anno2.right_boundaries):
                    for (a, a2) in zip(x, x2):
                        self.assertEqual(asdict(a), asdict(a2))
            except Exception as e:
                print(a)
                print(a2)
                raise e
            self.assertEqual(anno.pivot_index, anno2.pivot_index)
