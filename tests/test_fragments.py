from sys import argv

from mrror.util import HYDROGEN_MASS
from mrror.spectra.types import AugmentedPeaks
from mrror.fragments.types import TargetMasses, FragmentStateSpace, ResidueStateSpace
from mrror.fragments.masses import construct_pair_target_masses, construct_boundary_target_masses
from mrror.fragments.search import find_pairs, find_pivots, find_boundaries

from .shared import ANNO_CFG, TEST_PEPTIDES, TEST_PEAKS, AUG_PEAKS, DEFAULT_PARAM, COMPLEX_PARAM, _assert_maxmin_tolerance


import pytest
import numpy as np
from tabulate import tabulate
from hydra import compose, initialize
from omegaconf import OmegaConf

def test_state_space_constructors():
    print(FragmentStateSpace.trivial())
    
    print(FragmentStateSpace.from_config_to_pairs(
        ANNO_CFG))

    print(FragmentStateSpace.from_config_to_boundaries(
        ANNO_CFG))

    print(FragmentStateSpace.from_config_to_boundaries(
        ANNO_CFG, reflect=True))

    print(ResidueStateSpace.from_config(
        ANNO_CFG))
    
    print(ResidueStateSpace.from_config(
        ANNO_CFG, reflect=True))

def test_pair_target_masses():
    residue_space = ResidueStateSpace.from_config(
        ANNO_CFG)
    fragment_space = FragmentStateSpace.from_config_to_pairs(
        ANNO_CFG)
    targets = construct_pair_target_masses(residue_space, fragment_space)
    tolerance = 0.01
    for peaks in TEST_PEAKS:
        _assert_maxmin_tolerance(
            peaks.decharged_pair_masses(),
            targets.target_masses,
            tolerance,
        )

def test_lower_boundary_target_masses():
    residue_space = ResidueStateSpace.from_config(
        ANNO_CFG)
    fragment_space = FragmentStateSpace.from_config_to_boundaries(
        ANNO_CFG)
    targets = construct_boundary_target_masses(residue_space, fragment_space)
    print(targets)
    tolerance = 0.01
    for peaks in TEST_PEAKS:
        print(peaks.tabulate())
        _assert_maxmin_tolerance(
            peaks.decharged_lower_boundary_masses(),
            targets.target_masses,
            tolerance,
        )

def test_reflected_upper_boundary_target_masses():
    residue_space = ResidueStateSpace.from_config(
        ANNO_CFG, reflect=True)
    fragment_space = FragmentStateSpace.from_config_to_boundaries(
        ANNO_CFG, reflect=True)
    targets = construct_boundary_target_masses(residue_space, fragment_space)
    print(targets)
    tolerance = 0.015
    for peaks in TEST_PEAKS:
        print(peaks.tabulate())
        _assert_maxmin_tolerance(
            2 * peaks.pivot - peaks.decharged_upper_boundary_masses(),
            targets.target_masses,
            tolerance,
        )

def test_search_pairs():
    residue_space = ResidueStateSpace.from_config(
        ANNO_CFG)
    fragment_space = FragmentStateSpace.from_config_to_pairs(
        ANNO_CFG)
    targets = construct_pair_target_masses(residue_space, fragment_space)
    tolerance = 0.01
    for (anno_peaks, aug_peaks) in zip(TEST_PEAKS, AUG_PEAKS):
        print(anno_peaks.tabulate())
        print(aug_peaks)
        true_pairs = [(x,y) for (x,y) in anno_peaks.pairs()]
        res = find_pairs(aug_peaks, tolerance, targets, -1)
        observed_pairs = [(x,y) for (x,y) in res.indices.tolist()]
        missed_pairs = list(set(true_pairs).difference(observed_pairs))
        print("missed pairs",len(missed_pairs),missed_pairs)
        print(anno_peaks.tabulate_pair_indices(missed_pairs))
        assert len(missed_pairs) == 0

def test_search_pivots():
    residue_space = ResidueStateSpace.from_config(
        ANNO_CFG)
    fragment_space = FragmentStateSpace.from_config_to_pairs(
        ANNO_CFG)
    targets = construct_pair_target_masses(residue_space, fragment_space)
    tolerance = 0.01
    sym_tolerance = 0.015
    score_factor = 0.33
    for (anno_peaks, aug_peaks) in zip(TEST_PEAKS, AUG_PEAKS):
        print(anno_peaks.tabulate())
        print("true pivot",anno_peaks.pivot)
        pairs = find_pairs(aug_peaks, tolerance, targets, -1)
        pivots = find_pivots(aug_peaks, pairs, tolerance, sym_tolerance, score_factor)
        print(pivots.cluster_points)
        assert any(abs(x - anno_peaks.pivot) < tolerance for x in pivots.cluster_points)

def test_search_boundaries_lower():
    residue_space = ResidueStateSpace.from_config(
        ANNO_CFG)
    fragment_space = FragmentStateSpace.from_config_to_boundaries(
        ANNO_CFG)
    targets = construct_boundary_target_masses(residue_space, fragment_space)
    tolerance = 0.01
    for (anno_peaks, aug_peaks) in zip(TEST_PEAKS, AUG_PEAKS):
        true_boundaries = anno_peaks.lower_boundaries().tolist()
        print(anno_peaks.tabulate())
        print("true boundaries",true_boundaries)

        res = find_boundaries(aug_peaks, tolerance, targets, -1)
        observed_boundaries = res.index.tolist()
        print("observed boundaries", observed_boundaries)

        missed_boundaries = list(set(true_boundaries).difference(observed_boundaries))
        print("missed boundaries", missed_boundaries)

        print(res)
        print(aug_peaks)
        print(targets)
        assert len(missed_boundaries) == 0

def test_search_boundaries_upper_reflected():
    residue_space = ResidueStateSpace.from_config(
        ANNO_CFG, reflect=True)
    fragment_space = FragmentStateSpace.from_config_to_boundaries(
        ANNO_CFG, reflect=True)
    targets = construct_boundary_target_masses(residue_space, fragment_space)
    tolerance = 0.015
    for anno_peaks in TEST_PEAKS:
        true_boundaries = sorted(anno_peaks.upper_boundaries().tolist())
        print(anno_peaks.tabulate())
        print("true boundaries",true_boundaries)

        aug_peaks = AugmentedPeaks.from_peaks(
            anno_peaks,
            charges=np.array([1,2,3]),
            pivot_point=anno_peaks.pivot,
        )
        res = find_boundaries(aug_peaks, tolerance, targets, -1)
        observed_boundaries = sorted(res.index.tolist())
        print("observed boundaries", observed_boundaries)

        missed_boundaries = list(set(true_boundaries).difference(observed_boundaries))
        print("missed boundaries", missed_boundaries)

        assert len(missed_boundaries) == 0
