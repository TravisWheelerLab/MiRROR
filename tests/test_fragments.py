import unittest
from typing import Iterator, Any
from math import sqrt
from time import time
from tqdm import tqdm
import itertools as it
from bisect import bisect_left, bisect_right

from mirror.spectra.simulation import simulate_simple_peaks, simulate_complex_peaks, simulate_pivot
from mirror.spectra.types import PeakList, BenchmarkPeakList
from mirror.io import read_mzlib
from mirror.presets import VALIDATION_PEPTIDES, MONO_ANNOTATION_PARAMS, AVG_ANNOTATION_PARAMS, BOUNDARY_ANNOTATION_PARAMS
from mirror.annotation import AnnotationParams
import mirror.util as util

import numpy as np

from mirror.fragments import FragmentStateSpace, FragmentState, ResidueStateSpace, ResidueState, AbstractFragmentSolver, BisectFragmentSolver
class TestSolvers(unittest.TestCase):

    @staticmethod
    def _dummy_fragment_space():
        return FragmentStateSpace(
            loss_masses = [0, 22/70, sqrt(2)/2],
            loss_symbols = ["", "π/10", "√2/2"],
            charges = [1,2,3])

    def test01_fragment_state_space(self):
        """Verify that the FragmentStateSpace can be constructed and used to create a well-formed FragmentState."""
        state_space = self._dummy_fragment_space()
        state = FragmentState.from_index(
            peak_idx = -1,
            fragment_mass = 5 - (22/7),
            loss_id = 1,
            charge = 1,
            state_space = state_space)
        self.assertEqual(
            state.fragment_mass - state.loss_mass,
            state.fragment_mass - state_space.loss_masses[state.loss_id])

    @staticmethod
    def _dummy_residue_space():
        return ResidueStateSpace(
            amino_masses = [2., 3., 5., 7.],
            amino_symbols = [f"p_{i}" for i in range(1,5)],
            modification_masses = [0, 1/2, 1/3, 1/5, 1/7],
            modification_symbols = [''] + [f"q_{i}" for i in range(1,5)],
            applicable_modifications = [[0,1],[0,2],[0,3],[0,4]],
            max_num_modifications = 3)

    def test02_residue_state_space(self):
        """Verify that the ResidueStateSpace can be constructed and used to create a well-formed ResidueState."""
        state_space = self._dummy_residue_space()
        state = ResidueState.from_index(
            residue_mass = 3 + 1/3,
            amino_id = 1,
            modification_id = 1,
            state_space = state_space)
        self.assertEqual(
            state.residue_mass,
            state.amino_mass + state.modification_mass)
        self.assertEqual(
            state.residue_mass,
            state_space.amino_masses[state.amino_id] + state_space.modification_masses[state_space.applicable_modifications[state.amino_id][state.modification_id]])

    def test03_solver_utils(self):
        """Verify that the static utility methods of AbstractFragmentSolver perform, indepedently of any implementation."""
        # augment peaks with charge states
        # augment target masses with losses and conditional modification
        # generate fragment states
        fragment_space = self._dummy_fragment_space()
        fragment_states = list(AbstractFragmentSolver._generate_fragment_states(
            loss_indices = [0,1,2],
            peak_idx = 0,
            fragment_mass = 10.,
            charge = 1,
            state_space = fragment_space))
#        print(fragment_states)
        # generate residue states
        residue_space = self._dummy_residue_space()
        # filter out degenrate solutions

    @staticmethod
    def _simulate_fragment_masses(
        fragment_space: FragmentStateSpace,
        residue_space: ResidueStateSpace,
        amino_ids: list[int],
        modification_ids: list[int],
        loss_ids: list[int],
        charges: list[int],
    ) -> tuple[list[float],list[[float]]]:
        amino_masses = residue_space.amino_masses
        modification_masses = residue_space.modification_masses
        applicable_modifications = residue_space.applicable_modifications
        loss_masses = fragment_space.loss_masses
        residue_masses = [(amino_masses[i] + modification_masses[applicable_modifications[i][j]] - loss_masses[k]) for (i,j,k) in zip(amino_ids,modification_ids,loss_ids)]
        fragment_masses = [sum(residue_masses[:i + 1]) / charges[i] for i in range(len(residue_masses))]
        return residue_masses, fragment_masses

    @classmethod
    def _simulate_states(cls,
        fragment_space: FragmentStateSpace,
        residue_space: ResidueStateSpace,
        amino_ids: list[int],
        modification_ids: list[int],
        loss_ids: list[int],
        charges: list[int],
    ) -> tuple[list[float],list[tuple[FragmentState,FragmentState,ResidueState]]]:
        residue_masses, fragment_masses = cls._simulate_fragment_masses(fragment_space, residue_space, amino_ids, modification_ids, loss_ids, charges)
        states = []
        for i in range(1, len(amino_ids)):
            left_fragment = FragmentState.from_index(
                peak_idx = i,
                fragment_mass = fragment_masses[i - 1],
                loss_id = loss_ids[i - 1],
                charge = charges[i - 1],
                state_space = fragment_space)
            right_fragment = FragmentState.from_index(
                peak_idx = i + 1,
                fragment_mass = fragment_masses[i],
                loss_id = loss_ids[i],
                charge = charges[i],
                state_space = fragment_space)
            residue = ResidueState.from_index(
                residue_mass = residue_masses[i],
                amino_id = amino_ids[i],
                modification_id = modification_ids[i],
                state_space = residue_space)
            states.append((left_fragment, right_fragment, residue))
        return fragment_masses, states

    @staticmethod
    def _repr_states(states):
        return '\n\n\t'.join(map(lambda x: '\n\t'.join(map(str, x)), states))

    def test04_bisect_solver(self):
        """Verify that the BisectFragmentSolver can be constructed and resolve queries on a dummy state space."""
        # construct the dummy fragments
        state_space = (self._dummy_fragment_space(), self._dummy_fragment_space(), self._dummy_residue_space())
        print(state_space)
        #state_space[0].charges = [1]
        #state_space[1].charges = [1]
        amino_ids = [
            0,2,1,0,]
        modification_ids = [
            1,1,0,0,]
        loss_ids = [
            0,1,1,0,]
        charges = [
            1,1,2,1,]
        fragment_masses, true_states = self._simulate_states(
            fragment_space = state_space[1],
            residue_space = state_space[2],
            amino_ids = amino_ids,
            modification_ids = modification_ids,
            loss_ids = loss_ids,
            charges = charges)
        true_states_str = self._repr_states(true_states)
        print(true_states_str)
        # construct the solver
        solver = BisectFragmentSolver.from_state_space(
            mz = fragment_masses,
            tolerance = 0.01,
            state_space = state_space)
        # does it correctly annotate the fragment masses?
        self.assertEqual(solver.n_reference(),solver.n_query())
        matches = [False for _ in true_states]
        for i in range(solver.n_reference()):
            peak_i, charge_i = solver.set_reference(i)
            print(i, peak_i, charge_i)
            print(matches[peak_i - 1])
            for j in range(i + 1, solver.n_query()):
                peak_j, charge_j, _ = solver.set_query(j)
                if (peak_i != 0) and (peak_j == peak_i + 1):
                    true_state = true_states[peak_i - 1]
                    true_left_fragment, true_right_fragment, true_residue = true_state
                    solution_states = list(solver.get_solutions())
#                    print(f"true state:\n\t{self._repr_states([true_state])}")
#                    print(f"solution states[{peak_i},{peak_j}]:\n\t{self._repr_states(solution_states)}\n")
                    for (soln_left_fragment, soln_right_fragment, soln_residue) in solution_states:
                        left_match = soln_left_fragment == true_left_fragment
                        left_occl = soln_left_fragment.loss_id == 0 and true_left_fragment.loss_id != 0
                        left_charge_match = soln_left_fragment.charge == true_left_fragment.charge
                        right_match = soln_right_fragment == true_right_fragment
                        residue_match = soln_residue == true_residue
                        print(f"---\nleft match {left_match} (\n\t\tleft occl? {left_occl}\n\t\tleft charge match? {left_charge_match})\nright match {right_match}\nresidue match {residue_match}")
                        print(soln_residue,true_residue)
                        matches[peak_i - 1] |= (left_match or (left_occl and left_charge_match)) and right_match and residue_match
            print(matches[peak_i - 1])
        self.assertTrue(all(matches))

from tests.test_spectra import VALIDATION_SIMS

from mirror.fragments.pairs import PairedFragments, find_pairs
class TestPairs(unittest.TestCase):

    @staticmethod
    def _validate_pairs(
        observed_pairs: list[PairedFragments],
        bpl: BenchmarkPeakList,
    ) -> bool:
        observed_data = sorted([
            (*sorted(p.peak_indices()), p.amino_symbol()) for p in observed_pairs if p.amino_symbol() in bpl.peptide])
        # print("observed:\n", observed_data)
        index_pairs = list(it.pairwise(range(1, len(bpl.peptide))))
        true_data_sets = [list(bpl.get_pairs(l, r)) for (l, r) in index_pairs]
        coverage = [0 for _ in index_pairs]
        for (i, true_data) in enumerate(true_data_sets):
            # print("true data ",i)
            if len(true_data) == 0:
                coverage[i] == 1
            else:
                for (left_peak_idx, right_peak_idx, amino_sym) in true_data:
                    query = (*sorted([left_peak_idx, right_peak_idx]), amino_sym)
                    if query in observed_data:
                        coverage[i] += 1
                        # print(f"\tquery {query} -> hit")
                    # else:
                        # print(f"\tquery {query} -> miss")
        return all([x > 0 for x in coverage]), coverage

    def _test_sims(self,
        sims: tuple[str,str,int,BenchmarkPeakList],
        params: AnnotationParams,
    ):
        # check that fragment and residue states can be recovered across a variety of peptides and spectrum complexities.
        for (peptide, mode, charges, sim_bpl) in VALIDATION_SIMS:
            observed_pairs = list(find_pairs(
                peaks = sim_bpl,
                tolerance = params.fragment_search_tolerance,
                residue_space = params.residue_state_space,
                fragment_space = params.fragment_state_space))
            validation, coverage = self._validate_pairs(observed_pairs, sim_bpl)
            print(f"{peptide}, {mode}, {charges}, {coverage}")
            self.assertTrue(validation)

    def test_sims(self):
        """Run the find_pairs function on a set of simulated spectra."""
        self._test_sims(VALIDATION_SIMS, MONO_ANNOTATION_PARAMS)
        # self._test_sims(VALIDATION_SIMS, AVG_ANNOTATION_PARAMS)

from mirror.fragments.pivots import AbstractPivot, find_overlap_pivots, find_virtual_pivots
from mirror.annotation import reindex_by_fragment_masses
from time import time
class TestPivots(unittest.TestCase):

    def _test_sims(self, params: AnnotationParams = MONO_ANNOTATION_PARAMS, reindex: bool = False):
        """Run the find_pivots function on a set of simulated spectra."""
        pair_times = []
        pivot_times = []
        virtual_times = []
        overlap_hits = 0
        virtual_hits = 0
        for (i, (peptide, mode, charges, sim_bpl)) in enumerate(VALIDATION_SIMS):
            true_pivot = simulate_pivot(sim_bpl.peptide)

            pair_time = time()
            pairs = list(find_pairs(
                peaks = sim_bpl,
                tolerance = params.fragment_search_tolerance,
                residue_space = params.residue_state_space,
                fragment_space = params.fragment_state_space))
            pair_times.append(time() - pair_time)
            
            if reindex:
                pairs, spectrum = reindex_by_fragment_masses(
                    pairs = pairs,
                    fragment_state_space = params.fragment_state_space)
            else:
                spectrum = sim_bpl.mz

            pivot_time = time()
            overlap_pivots = list(find_overlap_pivots(spectrum, pairs, params.fragment_search_tolerance * 2))
            pivot_times.append(time() - pivot_time)
            overlap_pivot_points = [p.get_pivot_point() for p in overlap_pivots]            
            op = sorted(overlap_pivot_points)
            tolerance = params.fragment_search_tolerance * 2
            op_l = bisect_left(op, true_pivot - tolerance)
            op_r = bisect_right(op, true_pivot + tolerance)
            op_hit = op_l < op_r
            overlap_hits += op_hit

            virtual_time = time()
            virtual_pivots = list(find_virtual_pivots(pairs, params.fragment_search_tolerance * 2))
            virtual_times.append(time() - virtual_time)
            virtual_pivot_points = [p.get_pivot_point() for p in virtual_pivots]

            vt = sorted(virtual_pivot_points)
            vt_l = bisect_left(vt, true_pivot - tolerance)
            vt_r = bisect_right(vt, true_pivot + tolerance)
            vt_hit = vt_l < vt_r
            virtual_hits += vt_hit

            self.assertTrue(op_hit or vt_hit)
            print(f"{i}", flush=True, end=' ')
        print(sum(pair_times), sum(pivot_times), sum(virtual_times), overlap_hits, virtual_hits)

    def test_sims(self):
        self._test_sims()

    def test_sims_reindex(self):
        self._test_sims(reindex=True)
            
from mirror.util import measure_mirror_symmetry
from mirror.fragments.pivots import VirtualPivot
from mirror.fragments.boundaries import BoundaryFragment, ReflectedBoundaryFragment, find_left_boundaries, find_right_boundaries, rescore_pivots
class TestBoundaries(unittest.TestCase):

    def test_left_boundaries(self, params = BOUNDARY_ANNOTATION_PARAMS):
        """Run the find_left_boundaries function on simulated spectra."""
        print(params)
        for (i, (peptide, mode, charges, sim_bpl)) in enumerate(VALIDATION_SIMS):
            left_boundaries = list(find_left_boundaries(
                peaks = sim_bpl,
                tolerance = 0.1,#params.fragment_search_tolerance,
                residue_state_space = params.residue_state_space,
                fragment_state_space = params.fragment_state_space))
            print(i, peptide, mode, charges)
            expected_boundaries = list(sim_bpl.get_left_boundaries())
            observed_boundaries = [(lb.fragment.peak_idx, lb.residue.amino_symbol) for lb in left_boundaries]
            print("observed:",observed_boundaries)
            for (idx,res) in expected_boundaries:
                print(res, idx, round(sim_bpl[idx], 4), end='\t')
                if (idx,res) in observed_boundaries:
                    print('●')
                else:
                    print('◌')
            # input()

    def test_right_boundaries(self, params = BOUNDARY_ANNOTATION_PARAMS):
        """Run the find_right_boundaries function on simulated spectra."""
        print(params)
        for (i, (peptide, mode, charges, sim_bpl)) in enumerate(VALIDATION_SIMS):
            sim_pivot = VirtualPivot(
                pivot_point = simulate_pivot(peptide),
                frequency = 1)
            right_boundaries = find_right_boundaries(
                pivots = [sim_pivot],
                peaks = sim_bpl,
                tolerance = 0.1,
                residue_state_space = params.residue_state_space,
                fragment_state_space = params.fragment_state_space)[0] # collapsing the list b/c there is only one pivot about which to create right boundaries.
            print(i, peptide, mode, charges)
            print(f"pivot = {sim_pivot.pivot_point}")
            expected_boundaries = list(sim_bpl.get_right_boundaries())
            observed_boundaries = [(rb.fragment.peak_idx, rb.residue.amino_symbol) for rb in right_boundaries]
            for (idx,res) in expected_boundaries:
                print(res, idx, round(sim_bpl[idx], 4), end='\t')
                if (idx,res) in observed_boundaries:
                    print('●')
                else:
                    print('◌')
            # input()

    def test_mirror_symmetry(self):
        c = 0.3
        n_signal = 10
        n_noise = 1000
        n_trials = 100
        for tolerance in (1e-1,1e-5,1e-10):
            for _ in range(n_trials):
                left = np.random.uniform(size=n_signal) * c
                right = 2 * c - left
                noise = np.random.uniform(size=n_noise)
                signal = np.array(sorted(left.tolist() + right.tolist()))
                aggregate = np.array(sorted(noise.tolist() + signal.tolist()))
                self.assertTrue(
                    all(left - (2 * c - right) < tolerance))
                self.assertEqual(
                    measure_mirror_symmetry(signal, c, tolerance),
                    2 * n_signal)
                self.assertGreaterEqual(
                    measure_mirror_symmetry(aggregate, c, tolerance),
                    2 * n_signal)

    def test_rescore_pivots(self):
        """Run the rescoring / filtering function on simulated spectra; verify that good pivots are not being discarded."""
        pass
