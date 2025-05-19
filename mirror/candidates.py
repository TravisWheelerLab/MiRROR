#

import networkx as nx
import numpy as np
from editdistance import eval as edit_distance

from .affixes import Affix, AffixPair
from .spectral_alignment import score_sequence_to_spectrum
from .util import mask_ambiguous_residues, residue_lookup, disjoint_pairs, sequence_mass
from .pivots import Pivot
from .graphs.graph_utils import path_to_edges, unzip_dual_path, extend_truncated_paths, GraphPair

#=============================================================================#

class Candidate:
    def __init__(self,
        spectrum: np.ndarray,
        affix_pair: AffixPair,
        boundary_chrs: tuple[chr, chr],
        pivot_res: chr,
    ):
        self._affixes = affix_pair
        self._boundary = boundary_chrs
        self._pivot_res = pivot_res
        self._called_affix_a = list(self._affixes[0].call())
        self._called_affix_b = list(self._affixes[1].call())
        forward_seq = self._called_affix_a + ([pivot_res] if pivot_res != "" else []) + reverse_called_sequence(self._called_affix_b)
        backward_seq = reverse_called_sequence(forward_seq)
        self._sequences = (
            apply_boundary_residues(forward_seq, *boundary_chrs), 
            apply_boundary_residues(backward_seq, *boundary_chrs))

    def edit_distance(self, peptide, verbose = False):
        target = construct_target(peptide)
        dist = [edit_distance(query, target) for query in self._sequences]
        optimizer = np.argmin(dist)
        optimum = dist[optimizer]
        if verbose:
            print()
            print(target)
            print(self._sequences[optimizer])
            print(optimum)
        return optimum, optimizer
    
    def characterize_errors(self, peptide):
        optimum, optimizer = self.edit_distance(peptide)
        if optimum == 0:
            return []
        else:
            query = self._sequences[optimizer]
            target = construct_target(peptide)
            errors = []
            if query[0] != target[0]:
                errors.append("first")
            if query[-1] != target[-1]:
                errors.append("last")
            if query[1:-2] != target[1:-2]:
                errors.append("mid")
            # how to detect a pivot error?
            return errors
    
    def sequences(self):
        return (
            ' '.join(self._sequences[0]),
            ' '.join(self._sequences[1])
        )
    
    def mass(self):
        return sequence_mass(self._sequences[0])
    
    def call(self):
        return self.sequences()[self.edit_distance()[1]]
    
    def __eq__(self, other):
        if isinstance(other, Candidate):
            return (
            (self._boundary == other._boundary) and
            (self._called_affix_a == other._called_affix_a) and
            (self._pivot_res == other._pivot_res) and
            (self._called_affix_b == other._called_affix_b) and
            (self.sequences() == other.sequences()))
        return False
    
    def __repr__(self):
        return f"""Candidate(
    boundary = {self._boundary}
    affix_a = {self._called_affix_a}
    pivot = {self._pivot_res}
    affix_b = {self._called_affix_b}
    sequences = {self.sequences()}
)"""

#=============================================================================#

def create_candidates(
    augmented_spectrum: np.ndarray,
    spectrum_graphs,
    affix_pair: AffixPair,
    boundary_chrs,
    pivot_res,
) -> list[Candidate]:
    # enumerates the candidates that can be constructed from a pair of affixes
    return list(_enumerate_candidates(augmented_spectrum, affix_pair, spectrum_graphs, boundary_chrs, pivot_res))

def _enumerate_candidates(
    aug_spectrum,
    affix_pair: AffixPair,
    spectrum_graphs: GraphPair,
    boundary_chrs,
    pivot_res,
) -> list[Candidate]:
    yield Candidate(
        aug_spectrum,
        affix_pair,
        boundary_chrs,
        pivot_res
    )

#=============================================================================#

def filter_candidate_sequences(
    original_spectrum: np.ndarray,
    candidate_sequences: np.ndarray,
    alignment_threshold: float,
    alignment_parameters = None
) -> np.ndarray:
    # admits a candidate sequence if its synthetic spectrum aligns to the original spectrum.
    aligner = lambda seq: score_sequence_to_spectrum(seq, original_spectrum, *alignment_parameters)
    candidate_scores = np.vectorize(aligner)(candidate_sequences)
    return candidate_sequences[candidate_scores > alignment_threshold]

#=============================================================================#

def _call_sequence_from_path(spectrum, dual_path):
    for half_path in unzip_dual_path(dual_path):
        edges = path_to_edges(half_path)
        get_res = lambda e: residue_lookup(spectrum[max(e)] - spectrum[min(e)])
        yield list(map(get_res, edges))

def call_sequence_from_path(spectrum, dual_path):
    seq1, seq2 = _call_sequence_from_path(spectrum, dual_path)
    sequence = []
    for (res1, res2) in zip(seq1, seq2):
        if res1 == 'X' and res2 != 'X':
            sequence.append(res2)
        elif res1 != 'X' and res2 == 'X':
            sequence.append(res1) 
        elif res1 == res2:
            sequence.append(res1)
        else:
            sequence.append(f"{res1}/{res2}")
    return sequence

def reverse_called_sequence(sequence):
    return sequence[::-1]

def apply_boundary_residues(seq, initial_b, terminal_y):
    return [initial_b, *seq, terminal_y]

def construct_target(peptide):
    return [mask_ambiguous_residues(r) for r in peptide]