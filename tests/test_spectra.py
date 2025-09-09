from time import time
from tqdm import tqdm
from random import shuffle
from typing import Iterable, Iterator
import itertools as it

import mirror.util as util
from mirror.io import *
from mirror.spectra.types import *
from mirror.presets import VALIDATION_PEPTIDES, MONO_RESIDUE_SPACE, AMINO_SYMBOLS, AMINO_MONO_MASSES

import numpy as np

import unittest

def _simulate() -> list[tuple[str,str,int,BenchmarkPeakList]]:
    return [(
            peptide, 
            mode, 
            charges, 
            BenchmarkPeakList.from_simulation(peptide, mode, charges)) 
        for peptide in VALIDATION_PEPTIDES 
        for (mode, charges) in (("simple", 1), ("simple", 3), ("complex", 3))]

VALIDATION_SIMS = _simulate()

class TestSpectraTypes(unittest.TestCase):

    @staticmethod
    def read_test_9species_mzlib():
        return read_mzlib("data/spectra/Apis-mellifera.mzlib.txt")
    
    @staticmethod
    def read_test_9species_mgf():
        return read_mgf("data/spectra/Apis-mellifera.mgf")

    def test_benchmark_from_simulation(self):
        lookup_amino_mass = {symbol: mass for (symbol, mass) in zip(MONO_RESIDUE_SPACE.amino_symbols, MONO_RESIDUE_SPACE.amino_masses)}
        for peptide in VALIDATION_PEPTIDES:
            for mode, charges in (("simple", 1), ("simple", 3), ("complex", 3)):
                sim_bpl = BenchmarkPeakList.from_simulation(peptide, mode, charges)

    def test_BenchmarkPeakList_charges(self):
        for (peptide, mode, charges, sim_bpl) in VALIDATION_SIMS:
            # verify charges
            if charges > 1:
                self.assertTrue(any(charge > 1 for charge in sim_bpl.charge))

    def test_BenchmarkPeakList_losses(self):
        for (peptide, mode, charges, sim_bpl) in VALIDATION_SIMS:
            # verify losses
            if mode == "simple":
                self.assertTrue(all(loss == '' for loss in sim_bpl.losses))
            elif mode == "complex":
                self.assertTrue(any(loss != '' for loss in sim_bpl.losses))
            # in the simplest case, verify that the pairwise annotation isn't doing anything insane.
                
    def test_BenchmarkPeakList_pairs(self):
        lookup_amino_mass = {amino_symbol: amino_mass for (amino_mass, amino_symbol) in zip(AMINO_MONO_MASSES, AMINO_SYMBOLS)}
        for (peptide, mode, charges, sim_bpl) in VALIDATION_SIMS:
            if mode == "simple" and charges == 1:
                pair_indices = list(it.pairwise(range(len(peptide) + 1)))
                for (l, r) in pair_indices:
                    pairs = list(sim_bpl.get_pairs(l, r))
                    difs = [abs(sim_bpl[y] - sim_bpl[x]) for (x, y, _) in pairs]
                    target_sym = peptide[l:r]
                    target_dif = lookup_amino_mass[target_sym]
                    self.assertTrue(all([abs(difs[0] - d) < 0.01 for d in difs]))
                    self.assertTrue(all([abs(target_dif - d) < 0.01 for d in difs]))

from mirror.spectra.registration import Registration, _register
from mirror.spectra.simulation import simulate_simple_peaks, simulate_complex_peaks, simulate_noise
from mirror.presets import AMINO_SYMBOLS
class TestSpectraRegistration(unittest.TestCase):

    def test_Registration(self):
        """Test Registration construction and scoring."""
        mz = np.arange(10)
        intensity = np.empty(10)
        intensity[0::2] = np.ones(5)
        intensity[1::2] = np.ones(5) / 2
        peaks = PeakList(mz = mz, intensity = intensity)
        matching1 = (0., [(i,i) for i in range(0,10,2)])
        matching2 = (0., [(i,i) for i in range(1,10,2)])
        reg1 = Registration.from_fastdtw(0, peaks, peaks, matching1)
        reg2 = Registration.from_fastdtw(1, peaks, peaks, matching2)
        self.assertGreater(reg1.score(), reg2.score())

    def test__register_noise(self):
        """Test _register on noisy inputs."""
        sc = []
        cs = []
        sn = []
        cn = []
        for radius in range(1, 10):
            sc.append([])
            cs.append([])
            sn.append([])
            cn.append([])
            for (i, peptide) in enumerate(VALIDATION_PEPTIDES):
                simple = simulate_simple_peaks(peptide)
                complex = simulate_complex_peaks(peptide)
                sc[-1].append(_register(simple, complex, radius)[0])
                cs[-1].append(_register(complex, simple, radius)[0])
                for _ in range(10):
                    simple_noisy = simulate_noise(simple)
                    complex_noisy = simulate_noise(complex)
                    sn[-1].append(_register(simple_noisy, simple, radius)[0])
                    cn[-1].append(_register(complex_noisy, complex, radius)[0])
        print("simple | complex", [np.mean(x).round(4).tolist() for x in sc])
        print("complex | simple", [np.mean(x).round(4).tolist() for x in cs])
        print("simple | noisy simple", [np.mean(x).round(4).tolist() for x in sn])
        print("complex | noisy complex", [np.mean(x).round(4).tolist() for x in cn])

    def test__register_edits(self):
        """Test _register on inputs of peptides with short edit distance"""
        ins = [] 
        dels = []
        muts = []
        for n in range(1, 10):
            ins.append([])
            dels.append([])
            muts.append([])
            for (i, peptide) in enumerate(VALIDATION_PEPTIDES):
                if n >= len(peptide) - 2:
                    continue
                ref_peaks = simulate_simple_peaks(peptide)
                peptide_locs = np.arange(len(peptide))
                for _ in range(10):
                    mut_peptide = list(peptide)
                    del_peptide = list(peptide)
                    ins_peptide = list(peptide)
                    for loc in np.random.choice(peptide_locs, n, replace=False):
                        char = peptide[loc]
                        other_chars = [x for x in AMINO_SYMBOLS if x != char]
                        mut_peptide[loc] = np.random.choice(other_chars)
                        del_peptide[loc] = None
                        ins_peptide.insert(loc, np.random.choice(AMINO_SYMBOLS))
                    ins_peptide = ''.join(ins_peptide)
                    mut_peptide = ''.join(mut_peptide)
                    del_peptide = ''.join(x for x in del_peptide if x is not None)
                    for (query_pep, result_arrs) in [(mut_peptide, muts),(del_peptide, dels), (ins_peptide, ins)]:
                        query_peaks = simulate_simple_peaks(query_pep)
                        result_arrs[-1].append(_register(query_peaks, ref_peaks, 1)[0])
        print("| dist", np.arange(1,10).tolist())
        print("| ins", [np.mean(x).round(4).tolist() for x in ins])
        print("| dels", [np.mean(x).round(4).tolist() for x in dels])
        print("| muts", [np.mean(x).round(4).tolist() for x in muts])
