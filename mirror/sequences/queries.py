from typing import Self, Any, Iterator, Iterable
import dataclasses

import numpy as np

from .suffix_array import SuffixArray, BisectResult
from ..fragments import FragmentStateSpace, ResidueStateSpace

@dataclasses.dataclass(slots=True)
class PeptideMassQueryEngine:
    """Structures used to generate peptides for mass and length queries."""
    fragment_space: FragmentStateSpace
    residue_space: ResidueStateSpace
    suffix_array: SuffixArray
    max_num_losses: int
    max_num_modifications: int
    max_iter: int

    def step(self,
        sequences: list[str],
        masses: list[float],
        states: list[BisectResult],
    ) -> tuple[list[str],list[float],list[BisectResult]]:
        """For each sequence s with length denoted n, find all sequences (s1 ... sn) * (c) of length n + 1 in the suffix array."""
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

    def augment(self,
        sequences: list[str],
    ):
        """Generate augmentations of a collection of sequences, which describe all possible loss and modification states."""
        # TODO
        return None

def _query(
    engine: PeptideMassQueryEngine,
    partition,
) -> Iterable[tuple[float,str]]:
    # initialize
    sequences = [""]
    masses = [0.]
    states = [None]
    # search
    solutions = []
    for i in range(engine.max_iter):
        ## break when there is nothing left to do
        if len(sequences) == 0:
            break
        ## step engine to get current iteration
        new_sequences, new_masses, new_states = engine.step(sequences, masses, states)
        ## partition the current iteration into solved, in progress, and discard.
        solved, inpr, _ = partition(zip(new_masses, new_sequences))
        ## store the solutions
        solutions.extend(zip(new_masses[solved], new_sequences[solved]))
        ## prepare the in progress for another iteration.
        sequences = new_sequences[inpr]
        masses = new_masses[inpr]
        states = new_states[inpr]
    return solutions

def query_mass(
    engine: PeptideMassQueryEngine,
    q_mass: float,
    tolerance: float,
) -> Iterable[tuple[float,str]]:
    min_residue_mass = min(engine.residue_space.amino_masses)
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
    return _query(
        engine,
        mass_partition,
    )

def query_kmers(
    engine: PeptideMassQueryEngine,
    k: int,
) -> Iterable[tuple[float,str]]:
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
    return _query(
        engine,
        length_partition,
    )
