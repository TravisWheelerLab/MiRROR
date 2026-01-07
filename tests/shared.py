from mrror.annotated_peaks import DEFAULT_PARAM, COMPLEX_PARAM, AnnotatedPeaks
from mrror.spectra.types import AugmentedPeaks
from mrror.util import load_config

import numpy as np

TEST_PEPTIDES = [
    "PEPTIDE",
    "NREQSTK",
    "AEEHANR",
    "GNAGGLHHHR",
    "HHVLHHQTVDK",
    "HHSTIPQK",
    "FTHQHKPDER",
    "CEACPKPGTHAHK",
    "HHTIAHYK",
    "KPGVHQPQR",
    "AAHLAAHEAAK",
    "GHSCYRPR",
    "HHNIIR",
    "HLAEHEVK",
    "HGLTNTASHTR",
    "INPDNHNEK",
    "HGATVVNHVK",
    "HLNGHGSPPATNSSHR",
    "HASNIHVEK",
    "ELHVHPK",
]

CONFIG_DIR = "/home/user/Projects/MiRROR/params"
CONFIG_NAME = "config"

CFG = load_config(CONFIG_DIR, CONFIG_NAME)
ANNO_CFG = CFG.annotation

TEST_PEAKS = [
    AnnotatedPeaks.from_simulation(peptide, param, num_charges)
    for peptide in TEST_PEPTIDES
    for param in (DEFAULT_PARAM, COMPLEX_PARAM)
    for num_charges in (1,3)
]

AUG_PEAKS = [
    AugmentedPeaks.from_peaks(x, charges=np.array([1,2,3]))
    for x in TEST_PEAKS
]

def _assert_maxmin_tolerance(queries, targets, tolerance):
    max_min_err = -np.inf
    max_min_idx = -1
    max_min_tgt = -1
    queries = list(set(queries))
    target = list(set(targets))
    for (i,x) in enumerate(queries):
        min_err = np.inf
        min_tgt = -1
        for (j,y) in enumerate(targets):
            dif = abs(x - y)
            if dif < min_err:
                min_err = dif
                min_tgt = j
        if min_err > max_min_err:
            max_min_err = min_err
            max_min_idx = i
            max_min_tgt = min_tgt
    assert max_min_err < tolerance

def _assert_positive(arrs: list[list]):
    assert np.concat(arrs).min() > 0
