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

class TestSpectra(unittest.TestCase):

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
