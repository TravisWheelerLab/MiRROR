import dataclasses
import itertools as it
from typing import Iterable, Iterator

from .suffix_array import SuffixArray, BisectResult
from ..fragments.types import ResidueStateSpace

import numpy as np

def generate_unordered_combinations(
    residues: np.ndarray,       # [str; n]
    length: int,            
    suffix_array: SuffixArray,
) -> list[np.ndarray]:          # [[int; k]; _]
    indices = np.arange(residues.size)
    ordered_tuples = it.product(indices, repeat=length)
    comb = set()
    # setup.

    ordered_residues = [''.join(residues[list[x]]) for x in ordered_tuples]
    occurrences = suffix_array.count(ordered_tuples)
    # filter by suffix array.
    # TODO: replace w call to mass query engine.

    for (tup, count) in zip(ordered_tuples, occurrences):
        if count > 0:
            key = tuple(sorted(tup))
            comb.add(key)
    # project ordered tuples onto unordered combinations.

    return [np.array(x) for x in comb]

def _all_kmers(
    k: int,
    masses: np.ndarray,
    symbols: np.ndarray,
    applicable_modifications: list[list[int]],
) -> tuple[np.ndarray,np.ndarray,list[np.ndarray]]:
    """arg 'k' is the int kmer length, arg 'masses' is a 1d np array of n floats. arg 'symbols' is a 1d np array of n strings. returns a list of n^k (float, str) tuples, where each float is a sum of 'k' elements in 'masses', and each str is a concatenation of 'k' elements in 'symbols'."""
    n = len(masses)
    ind = list(range(n))
    kmer_ind = [list(i) for i in it.product(*[ind] * k)]
    kmer_masses = np.array([sum(masses[i]) for i in kmer_ind])
    kmer_symbols = np.array([''.join(symbols[i].tolist()) for i in kmer_ind])
    kmer_mods = [np.array(list(set(x for j in i for x in applicable_modifications[j]))) for i in kmer_ind]
    return (
        kmer_masses,
        kmer_symbols,
        kmer_mods,
    )

def all_kmers(
    residue_space: ResidueStateSpace,
    k: int,
) -> tuple[np.ndarray,np.ndarray,list[np.ndarray]]:
    return _all_kmers(
        k,
        residue_space.amino_masses,
        residue_space.amino_symbols,
        residue_space.applicable_modifications,
    )

@dataclasses.dataclass(slots=True)
class PeptideMassQueryEngine:
    residue_space: ResidueStateSpace
    suffix_array: SuffixArray
    max_iter: int

    def step(self,
        sequences: list[str],
        masses: list[float],
        states: list[BisectResult],
    ) -> tuple[list[str],list[float],list[BisectResult]]:
        amino_symbols = self.residue_space.amino_symbols
        amino_masses = self.residue_space.amino_masses
        num_amino = len(amino_masses)
        result_sequences = []
        result_masses = []
        result_states = []
        for (seq, mass, state) in zip(sequences, masses, states):
            res = self.suffix_array.bisect(
                queries = amino_symbols,
                prefix_result = state,
            )
            unpacked_res = [
                (seq + amino_symbols[i], mass + amino_masses[i], res[i])
                for i in range(num_amino) if res[i].count > 0]
            if len(unpacked_res) > 0:
                res_seq, res_mass, res_states = zip(*unpacked_res)
                result_sequences.extend(res_seq)
                result_masses.extend(res_mass)
                result_states.extend(res_states)
        return np.array(result_sequences), np.array(result_masses), np.array(result_states)

    def _query(
        self,
        partition,
    ) -> tuple[np.ndarray,np.ndarray]:
        # initialize
        sequences = [""]
        masses = [0.]
        states = [None]
        # search
        solution_masses = []
        solution_sequences = []
        for i in range(self.max_iter):
            ## exit when there is nothing left to do
            if len(sequences) == 0:
                break
            ## step engine to get current iteration
            new_sequences, new_masses, new_states = self.step(sequences, masses, states)
            ## partition the current iteration into solved, in progress, and discard.
            solved, inpr, _ = partition(zip(new_masses, new_sequences))
            ## store the solutions
            solution_masses.extend(new_masses[solved])
            solution_sequences.extend(new_sequences[solved])
            ## prepare the in progress for another iteration.
            sequences = new_sequences[inpr]
            masses = new_masses[inpr]
            states = new_states[inpr]
        return np.array(solution_masses), np.array(solution_sequences)
    
    def query_mass(
        self,
        q_mass: float,
        tolerance: float,
    ) -> tuple[np.ndarray,np.ndarray]:
        min_residue_mass = min(self.residue_space.amino_masses)
        def mass_partition(
            data: Iterator[tuple[float, str]],
        ) -> tuple[list[int],list[int],list[int]]:
            discard = []
            solved = []
            inpr = []
            for (i, (mass, seq)) in enumerate(data):
                dif = q_mass - mass
                if abs(dif) < tolerance:
                    solved.append(i)
                elif dif < min_residue_mass - tolerance:
                    discard.append(i)
                else:
                    inpr.append(i)
            return (
                solved,
                inpr,
                discard
            )
        return self._query(
            mass_partition,
        )
    
    def query_kmers(
        self,
        k: int,
    ) -> tuple[np.ndarray,np.ndarray]:
        def length_partition(
            data: Iterator[tuple[float, str]],
        ) -> tuple[list[int],list[int],list[int]]:
            discard = []
            solved = []
            inpr = []
            for (i, (mass, seq)) in enumerate(data):
                if len(seq) == k:
                    solved.append(i)
                else:
                    inpr.append(i)
            return (
                solved,
                inpr,
                discard
            )
        return self._query(
            length_partition,
        )
