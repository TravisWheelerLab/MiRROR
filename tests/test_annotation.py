import itertools as it
import pprint

from mrror.annotation import AnnotationParams, AnnotationResult, annotate
from mrror.evaluation.labeled_peaks import AnnotationLabeledPeaks
from mrror.util import load_config

from .shared import TEST_PEAKS, ANNO_CFG
    
def test_annotate():
    params = AnnotationParams.from_config(ANNO_CFG)
    mz_errs = []
    charge_errs = []
    loss_errs = []
    for i, peaks in enumerate(TEST_PEAKS):
        print(i)
        pprint.pprint(peaks)
        res = annotate(peaks, params)
        pprint.pprint(res.pairs)
        peak_annotation = AnnotationLabeledPeaks.from_fragment_labels(
            res.fragment_labels, params.target_masses)
        pprint.pprint(peak_annotation)
        comparison = peaks.compare(peak_annotation)
        print(peaks.tabulate())
        print(peak_annotation.tabulate())
        pprint.pprint(comparison)
        mz_errs.append(comparison.mz_err)
        charge_errs.append(len([x for x in comparison.charge_aln if tuple(x) == (-1,-1)]) / len(comparison.charge_aln))
        loss_errs.append(len([x for x in comparison.loss_aln if tuple(x) == (-1,-1)]) / len(comparison.loss_aln))
        print()
    assert sum(mz_errs) == 0
    print(charge_errs)
    print(loss_errs)
