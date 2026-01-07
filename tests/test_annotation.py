import itertools as it

from mrror.annotation import AnnotationParams, AnnotationResult, annotate
from mrror.annotated_peaks import AnnotatedPeaks
from mrror.util import load_config

from .shared import TEST_PEAKS, ANNO_CFG
    
def test_annotate():
    params = AnnotationParams.from_config(ANNO_CFG)
    mz_errs = []
    for i, peaks in enumerate(TEST_PEAKS):
        res = annotate(peaks, params)
        peak_annotation = AnnotatedPeaks.from_fragment_masses(res.fragment_masses, params)
        comparison = peaks.compare(peak_annotation)
        mz_errs.append(comparison.mz_err)
    assert sum(mz_errs) == 0
