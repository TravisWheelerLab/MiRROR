from .util import mask_ambiguous_residues, residue_lookup, disjoint_pairs
from .pivot import Pivot
from .graph_util import path_to_edges, unzip_paired_path, extend_truncated_paths, find_edge_disjoint_paired_paths

import numpy as np
import networkx as nx
from editdistance import eval as edit_distance

def _call_sequence_from_path(spectrum, paired_path):
    for half_path in unzip_paired_path(paired_path):
        edges = path_to_edges(half_path)
        get_res = lambda e: residue_lookup(spectrum[max(e)] - spectrum[min(e)])
        yield list(map(get_res, edges))

def call_sequence_from_path(spectrum, paired_path):
    seq1, seq2 = _call_sequence_from_path(spectrum, paired_path)
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

class Candidate:
    def __init__(self,
        spectrum: np.ndarray,
        affix_a: list[tuple[int,int]],
        affix_b: list[tuple[int,int]],
        boundary: tuple[chr, chr],
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
        self._boundary = boundary
        self._affixes = (
            called_affix_a,
            called_affix_b
        )
        self._sequences = (
            apply_boundary_residues(forward_seq, *boundary), 
            apply_boundary_residues(backward_seq, *boundary))

    def edit_distance(self, peptide):
        target = construct_target(peptide)
        dist = [edit_distance(query, target) for query in self._sequences]
        optimizer = np.argmin(dist)
        optimum = dist[optimizer]
        return optimum, optimizer
    
    def sequences(self):
        return (
            ' '.join(self._sequences[0]),
            ' '.join(self._sequences[1])
        )
    
    def __repr__(self):
        return '\n'.join(self.sequences())

def construct_candidates(
    spectrum: np.ndarray,
    aug_spectrum: np.ndarray,
    pivot: Pivot,
    spectrum_graphs: tuple[nx.DiGraph, nx.DiGraph],
    path_affixes: list[list[tuple[int,int]]],
    disjoint_pairing_mode = "table",
    verbose = False,
):
    boundary = (pivot.initial_b_ion(spectrum)[1], pivot.terminal_y_ion(spectrum)[1])
    pivot_res = residue_lookup(pivot.gap())
    disjoint_afx_ind = list(find_edge_disjoint_paired_paths(path_affixes, mode=disjoint_pairing_mode))
    for (i, j) in disjoint_afx_ind:
        afx_a = path_affixes[i]
        afx_b = path_affixes[j]
        extension_a = [afx_a] + list(extend_truncated_paths([afx_a], *spectrum_graphs))
        extension_b = [afx_b] + list(extend_truncated_paths([afx_b], *spectrum_graphs))
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
                    boundary,
                    pivot_res)
                yield Candidate(
                    aug_spectrum, 
                    ext_afx_a, 
                    ext_afx_b, 
                    boundary,
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
                            boundary,
                            '')
                        yield Candidate(
                            aug_spectrum, 
                            ext_afx_a, 
                            ext_afx_b[:-overlap], 
                            boundary,
                            '')
                    #if overlap == 1:
                    #    yield Candidate(
                    #        aug_spectrum, 
                    #        ext_afx_a, 
                    #        ext_afx_b, 
                    #        boundary,
                    #        '')
                    #else:
                    #    yield Candidate(
                    #        aug_spectrum, 
                    #        ext_afx_a[:-overlap], 
                    #        ext_afx_b, 
                    #        boundary,
                    #        '')
                    #    yield Candidate(
                    #        aug_spectrum, 
                    #        ext_afx_a, 
                    #        ext_afx_b[:-overlap], 
                    #        boundary,
                    #        '')