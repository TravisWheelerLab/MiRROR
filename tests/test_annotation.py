import unittest
from math import sqrt
from time import time
from tqdm import tqdm

from mirror.spectra.types import BenchmarkPeakList
from mirror.presets import MONO_ANNOTATION_PARAMS, AVG_ANNOTATION_PARAMS, VALIDATION_PEPTIDES
from mirror.annotation import AnnotationParams, AnnotationResult, annotate

class TestAnnotation(unittest.TestCase):

    def test(self):
        pass

    def validate(self):
        for simulation_params in [SIMPLE_SIMULATION_PARAMS, COMPLEX_SIMULATION_PARAMS]:
            for peptide in VALIDATION_PEPTIDES:
                peaks = BenchmarkPeakList.from_simulation(peptide, simulation_params)
                true_annotation = AnnotationResult.from_benchmark(peaks)
                mono_annotation = annotate(peaks, MONO_ANNOTATION_PARAMS)
                mono_quality = true_annotation.assess(mono_annotation)
                avg_annotation = annotate(peaks, AVG_ANNOTATION_PARAMS)
                avg_quality = true_annotation.assess(avg_annotation)
                print(f"peptide:\n\t{peptide}\nsim:\n\t{simulation_params}\ntrue annotation\n\t{true_annotation}\nmono annotation\n\t{mono_annotation}\navg annotation\n\t{avg_annotation}")
                
    def benchmark(self):
        pass
