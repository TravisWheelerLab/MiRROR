import unittest
from math import sqrt
from time import time
from tqdm import tqdm
from dataclasses import asdict

from mirror.spectra.types import BenchmarkPeakList
from mirror.presets import MONO_ANNOTATION_PARAMS, AVG_ANNOTATION_PARAMS, VALIDATION_PEPTIDES
from mirror.annotation import AnnotationParams, AnnotationResult, annotate

from tests.test_spectra import VALIDATION_SIMS

VALIDATION_ANNOTATION_DIR = "./data/output/annotation/"
VALIDATION_ANNOTATION_FILES = [VALIDATION_ANNOTATION_DIR + f"{i}_{peptide}_{mode}_{charges}.ann"
    for (i, (peptide, mode, charges, _)) in enumerate(VALIDATION_SIMS)]

class TestAnnotation(unittest.TestCase):

    def test_serialization(self,
        params = MONO_ANNOTATION_PARAMS,
        dir = VALIDATION_ANNOTATION_DIR,
        sims = VALIDATION_ANNOTATION_FILES
    ):
        print(params)
        for (i, ((peptide, mode, charges, sim_bpl), fpath)) in enumerate(zip(VALIDATION_SIMS, VALIDATION_ANNOTATION_FILES)):
            print(i, peptide, mode, charges)
            print(sim_bpl.mz)
            anno = annotate(sim_bpl, MONO_ANNOTATION_PARAMS)
            print(anno._profile)
            anno.write(fpath)
            anno2 = AnnotationResult.read(fpath)
            try:
                for (a, a2) in zip(anno._pairs, anno2._pairs):
                    self.assertEqual(asdict(a), asdict(a2))
                for (a, a2) in zip(anno._pivots, anno2._pivots):
                    self.assertEqual(asdict(a), asdict(a2))
                for (a, a2) in zip(anno._left_boundaries, anno2._left_boundaries):
                    self.assertEqual(asdict(a), asdict(a2))
                for (x, x2) in zip(anno._right_boundaries, anno2._right_boundaries):
                    for (a, a2) in zip(x, x2):
                        self.assertEqual(asdict(a), asdict(a2))
            except Exception as e:
                print(a)
                print(a2)
                raise e
            self.assertEqual([x.tolist() for x in anno._pivot_clusters], [x.tolist() for x in anno2._pivot_clusters])
