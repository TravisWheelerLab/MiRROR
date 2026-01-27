from sys import argv

from mrror.util import HYDROGEN_MASS
from mrror.spectra.types import AugmentedPeaks
from mrror.fragments.types import TargetMasses, MultiResidueTargetMasses,FragmentStateSpace, ResidueStateSpace
from mrror.fragments.masses import construct_pair_target_masses, construct_boundary_target_masses, cluster_combinations_by_mass, combine_target_masses
from mrror.fragments.search import find_pairs, find_pivots, find_boundaries
from mrror.sequences.suffix_array import TrivialSuffixArray
from mrror.sequences.queries import generate_unordered_combinations

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

def test_target_masses():
    residue_space = ResidueStateSpace.from_config(
        ANNO_CFG)
    fragment_space = FragmentStateSpace.from_config_to_pairs(
        ANNO_CFG)
    boundary_fragment_space = FragmentStateSpace.from_config_to_boundaries(
        ANNO_CFG)
    reflected_residue_space = ResidueStateSpace.from_config(
        ANNO_CFG, reflect=True)
    reflected_boundary_fragment_space = FragmentStateSpace.from_config_to_boundaries(
        ANNO_CFG, reflect=True)
    pair_targets = construct_pair_target_masses(residue_space, fragment_space)
    boundary_targets = construct_boundary_target_masses(residue_space, boundary_fragment_space)
    reflected_boundary_targets = construct_boundary_target_masses(reflected_residue_space, reflected_boundary_fragment_space)
    pair_tolerance = 0.01
    boundary_tolerance = 0.01
    reflected_boundary_tolerance = 0.015
    for peaks in TEST_PEAKS:
        print(peaks.peptide)
        _assert_maxmin_tolerance(
            peaks.decharged_pair_masses(),
            pair_targets.target_masses,
            pair_tolerance,
        )
        _assert_maxmin_tolerance(
            peaks.decharged_lower_boundary_masses(),
            boundary_targets.target_masses,
            boundary_tolerance,
        )
        _assert_maxmin_tolerance(
            2 * peaks.pivot - peaks.decharged_upper_boundary_masses(),
            reflected_boundary_targets.target_masses,
            reflected_boundary_tolerance,
        )

def test_multiresidue_target_masses():
    residue_space = ResidueStateSpace.from_config(
        ANNO_CFG)
    fragment_space = FragmentStateSpace.from_config_to_pairs(
        ANNO_CFG)
    boundary_fragment_space = FragmentStateSpace.from_config_to_boundaries(
        ANNO_CFG)
    reflected_residue_space = ResidueStateSpace.from_config(
        ANNO_CFG, reflect=True)
    reflected_boundary_fragment_space = FragmentStateSpace.from_config_to_boundaries(
        ANNO_CFG, reflect=True)
    pair_targets = construct_pair_target_masses(residue_space, fragment_space)
    boundary_targets = construct_boundary_target_masses(residue_space, boundary_fragment_space)
    reflected_boundary_targets = construct_boundary_target_masses(reflected_residue_space, reflected_boundary_fragment_space)
    pair_tolerance = 0.01
    boundary_tolerance = 0.01
    reflected_boundary_tolerance = 0.015

    for k in range(2, 3 + 1):
        operand = [pair_targets,] * (k - 1)
        multi_pair_targets = combine_target_masses(
            [pair_targets,] + operand)
        multi_boundary_targets = combine_target_masses(
            [boundary_targets,] + operand)
        multi_reflected_boundary_targets = combine_target_masses(
            [reflected_boundary_targets,] + operand) # does this need to be a reflected pair target space? hm
        # construct multi-residue target spaces

        tolerance = 0.015 * k
        for peaks in TEST_PEAKS:
            print(peaks.tabulate())
            _assert_maxmin_tolerance(
                peaks.decharged_pair_masses(),
                pair_targets.target_masses,
                tolerance,
            )
            _assert_maxmin_tolerance(
                peaks.decharged_lower_boundary_masses(),
                boundary_targets.target_masses,
                boundary_tolerance,
            )
            _assert_maxmin_tolerance(
                2 * peaks.pivot - peaks.decharged_upper_boundary_masses(),
                reflected_boundary_targets.target_masses,
                reflected_boundary_tolerance,
            )
        

def test_search():
    residue_space = ResidueStateSpace.from_config(
        ANNO_CFG)
    fragment_space = FragmentStateSpace.from_config_to_pairs(
        ANNO_CFG)
    boundary_fragment_space = FragmentStateSpace.from_config_to_boundaries(
        ANNO_CFG)
    reflected_residue_space = ResidueStateSpace.from_config(
        ANNO_CFG, reflect=True)
    reflected_boundary_fragment_space = FragmentStateSpace.from_config_to_boundaries(
        ANNO_CFG, reflect=True)
    pair_targets = construct_pair_target_masses(residue_space, fragment_space)
    boundary_targets = construct_boundary_target_masses(residue_space, boundary_fragment_space)
    reflected_boundary_targets = construct_boundary_target_masses(reflected_residue_space, reflected_boundary_fragment_space)
    pair_tolerance = 0.01
    boundary_tolerance = 0.01
    reflected_boundary_tolerance = 0.015

    for (anno_peaks, aug_peaks) in zip(TEST_PEAKS, AUG_PEAKS):
        print(anno_peaks.tabulate())
        true_pairs = [(x,y) for (x,y) in anno_peaks.pairs()]
        res = find_pairs(aug_peaks, pair_tolerance, pair_targets, -1)
        observed_pairs = [(x,y) for (x,y) in res.indices.tolist()]
        missed_pairs = list(set(true_pairs).difference(observed_pairs))
        assert len(missed_pairs) == 0
        # peak search

        true_boundaries = anno_peaks.lower_boundaries().tolist()
        res = find_boundaries(aug_peaks, boundary_tolerance, boundary_targets, -1)
        observed_boundaries = res.index.tolist()
        missed_boundaries = list(set(true_boundaries).difference(observed_boundaries))
        assert len(missed_boundaries) == 0
        # boundary search

        true_boundaries = sorted(anno_peaks.upper_boundaries().tolist())
        aug_peaks = AugmentedPeaks.from_peaks(
            anno_peaks,
            charges=np.array([1,2,3]),
            pivot_point=anno_peaks.pivot,
        )
        res = find_boundaries(aug_peaks, reflected_boundary_tolerance, reflected_boundary_targets, -1)
        observed_boundaries = sorted(res.index.tolist())
        missed_boundaries = list(set(true_boundaries).difference(observed_boundaries))
        assert len(missed_boundaries) == 0
        # reflected boundary search

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
