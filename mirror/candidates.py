#

import networkx as nx
import numpy as np
from editdistance import eval as edit_distance

from .affixes import Affix, AffixPair
from .spectral_alignment import score_sequence_to_spectrum
from .util import mask_ambiguous_residues, residue_lookup, disjoint_pairs
from .pivots import Pivot
from .graph_utils import path_to_edges, unzip_dual_path, extend_truncated_paths, GraphPair

#=============================================================================#

class Candidate:
    def __init__(self,
        spectrum: np.ndarray,
        affix_a: list[tuple[int,int]],
        affix_b: list[tuple[int,int]],
        boundary_chrs: tuple[chr, chr],
        pivot_res: chr,
    ):
        called_affix_a = call_sequence_from_path(spectrum, affix_a)
        called_affix_b = call_sequence_from_path(spectrum, affix_b)
        forward_seq = called_affix_a + ([pivot_res] if pivot_res != "" else []) + reverse_called_sequence(called_affix_b)
        backward_seq = reverse_called_sequence(forward_seq)
        self._path_affixes = (
            affix_a,
            affix_b
        )
        self._boundary = boundary_chrs
        self._pivot_res = pivot_res
        self._affixes = (
            called_affix_a,
            called_affix_b
        )
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
    
    def __repr__(self):
        return ' | '.join(self.sequences())

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
    afx_a, afx_b = affix_pair
    path_a = afx_a.path()
    path_b = afx_b.path()
    extension_a = [path_a] + list(extend_truncated_paths([path_a], *spectrum_graphs))
    extension_b = [path_b] + list(extend_truncated_paths([path_b], *spectrum_graphs))
    for ext_afx_a in extension_a:
        for ext_afx_b in extension_b:
            end_a = ext_afx_a[-1]
            end_b = ext_afx_b[-1]
            # if one of the affixes wasn't extended,
            # or if they don't end at the same position,
            # we want to use the pivot residue.
            yield Candidate(
                aug_spectrum, 
                ext_afx_a, 
                ext_afx_b, 
                boundary_chrs,
                pivot_res)
            yield Candidate(
                aug_spectrum, 
                ext_afx_a, 
                ext_afx_b, 
                boundary_chrs,
                '')
            if ((-1 in end_a) and (-1 in end_b)) and (end_a == end_b):
                # otherwise, if the extensions overlap, 
                # they already contain the pivot residue;
                # we don't want to repeat it.
                for overlap in range(1, min(len(ext_afx_a),len(ext_afx_b))):
                    if ext_afx_a[-overlap] != ext_afx_b[-overlap]:
                        overlap -= 1
                        break
                if overlap > 0:
                    yield Candidate(
                        aug_spectrum, 
                        ext_afx_a[:-overlap], 
                        ext_afx_b, 
                        boundary_chrs,
                        '')
                    yield Candidate(
                        aug_spectrum, 
                        ext_afx_a, 
                        ext_afx_b[:-overlap], 
                        boundary_chrs,
                        '')

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