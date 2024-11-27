import itertools
from time import time
from copy import copy

import numpy as np
import networkx as nx

import mirror
from mirror import io, util, gap, pivot, boundary, graph_util, spectrum_graph, sequence

from networkx.drawing.nx_agraph import graphviz_layout, to_agraph

def all_edges(graph):
    return sorted([(i, j) for (i, j) in graph.edges if i != -1 and j != -1])

class TestData:

    def __init__(self, peptide):
        self.peptide = peptide
        self.spectrum = mirror.util.list_mz(mirror.util.generate_default_fragment_spectrum(peptide))
        self.gaps = None
        self.pivots = None
        self.pivot_mode = "-"
        self.viable_pivots = None
        self.sym_spectrum = None
        self.graphs = {}
        self.paths = {}
        self.candidates = {}

    def expected_mirror_symmetry(self):
        n_peaks = len(self.spectrum)
        return n_peaks - 1
    
    def set_gaps(self, gaps):
        self.gaps = gaps
    
    def set_pivots(self, pivots):
        self.pivots = pivots

    def set_pivot_mode(self, pivot_mode):
        self.pivot_mode = pivot_mode

    def num_pivots(self):
        if self.viable_pivots == None:
            return 0
        else:
            return len(self.viable_pivots)
    
    def set_viable_pivots(self, viable_pivots):
        self.viable_pivots = viable_pivots

    def set_symmetric_spectrum(self, spectrum):
        self.sym_spectrum = spectrum
    
    def set_graphs(self, pivot_no, asc_graph, desc_graph):
        self.graphs[pivot_no] = (asc_graph, desc_graph)
    
    def set_paths(self, pivot_no, paths):
        self.paths[pivot_no] = paths
    
    def set_candidates(self, pivot_no, candidates):
        self.candidates[pivot_no] = candidates

    def _draw_graph(self, graph, title):
        graph.remove_node(-1)
        graph.graph['graph'] = {'rankdir':'LR'}
        graph.graph['node'] = {'shape':'circle'}
        graph.graph['edges'] = {'arrowsize':'4.0'}
        A = to_agraph(graph)
        for (i,j) in graph.edges:
            truncated_weight = round(graph[i][j][mirror.spectrum_graph.GAP_KEY],4)
            res = mirror.util.residue_lookup(truncated_weight)
            A.get_edge(i,j).attr['label'] = f"{res} [ {str(truncated_weight)} ]" 
        A.layout('dot')
        A.draw(title)

    def draw_graphs(self, pivot_no, name_gen):
        asc, desc = copy(self.graphs[pivot_no])
        self._draw_graph(asc, name_gen("asc"))
        self._draw_graph(desc, name_gen("desc"))

def generate_test_data(k):
    return TestData(mirror.util.generate_random_tryptic_peptide(k))

def print_graph(D: nx.DiGraph):
    for n in sorted(D.nodes):
        print(f"[{n}] : {D[n]}")

def eval_gap(data: TestData):
    # find gaps
    gap_inds = [mirror.gap.find_gaps(data.spectrum, mirror.gap.GapTargetConstraint(amino_mass, mirror.util.GAP_TOLERANCE))
                for amino_mass in mirror.util.AMINO_MASS_MONO]
    gap_ind = list(itertools.chain.from_iterable(gap_inds))
    data.set_gaps(gap_ind)
    return gap_ind

def eval_pivot(data: TestData):
    #pivots = mirror.pivot.find_pivots(
    #    data.spectrum,
    #    data.gaps, 
    #    mirror.pivot.PivotOverlapConstraint(mirror.util.INTERGAP_TOLERANCE))
    mode, pivots = mirror.pivot.locate_pivot_point(data.spectrum, data.gaps, mirror.util.INTERGAP_TOLERANCE)
    data.set_pivot_mode(mode)
    data.set_pivots(pivots)
    return pivots

def eval_viable_pivots(data: TestData):
    pivot_residues = [mirror.util.residue_lookup(pivot.gap()) for pivot in data.pivots]
    pivot_scores = [mirror.util.count_mirror_symmetries(data.spectrum, pivot.center()) for pivot in data.pivots]
    pivot_errors = [mirror.util.mass_error(pivot.gap()) for pivot in data.pivots]
    viable_pivots = [(pivot,residue,score,error) for (pivot,residue,score,error) in zip(data.pivots,pivot_residues,pivot_scores,pivot_errors) 
                     if residue != 'X' and score > (data.expected_mirror_symmetry() / 2)]
    data.set_viable_pivots(viable_pivots)
    return viable_pivots

def eval_old_graphs(data: TestData):
    graphs = []
    for pivot_no, (pivot,residue,score,error) in enumerate(data.viable_pivots):
        asc_graph, desc_graph = mirror.spectrum_graph.construct_spectrum_graphs_from_spectrum(data.spectrum, pivot)
        data.set_graphs(pivot_no, asc_graph, desc_graph)
        graphs.append((asc_graph, desc_graph))
    return graphs
        
def eval_graphs(data: TestData):
    graphs = []
    for pivot_no, (pivot,residue,score,error) in enumerate(data.viable_pivots):
        asc_graph, desc_graph = mirror.spectrum_graph.construct_spectrum_graphs(data.spectrum, data.gaps, pivot)
        data.set_graphs(pivot_no, asc_graph, desc_graph)
        graphs.append((asc_graph, desc_graph))
    return graphs
        
def eval_new_graphs(data: TestData):
    graphs = []
    for pivot_no, (pivot,residue,score,error) in enumerate(data.viable_pivots):
        sym_spectrum, pivot = mirror.boundary.create_symmetric_boundary(data.spectrum, pivot)
        data.set_symmetric_spectrum(sym_spectrum)
        gap_inds = [mirror.gap.find_gaps(sym_spectrum, mirror.gap.GapTargetConstraint(amino_mass, mirror.util.GAP_TOLERANCE))
                    for amino_mass in mirror.util.AMINO_MASS_MONO]
        gap_ind = list(itertools.chain.from_iterable(gap_inds))
        asc_graph, desc_graph = mirror.spectrum_graph.construct_spectrum_graphs(sym_spectrum, gap_ind, pivot)
        data.set_graphs(pivot_no, asc_graph, desc_graph)
        graphs.append((asc_graph, desc_graph))
    return graphs

def eval_paths(data: TestData):
    paths = []
    for i in range(data.num_pivots()):
        asc_graph, desc_graph = data.graphs[i]

        paired_paths = mirror.graph_util.all_weighted_paired_simple_paths(
            asc_graph, 
            desc_graph, 
            mirror.spectrum_graph.GAP_KEY, 
            mirror.spectrum_graph.GAP_COMPARATOR)
        
        extended_paired_paths = list(mirror.graph_util.extend_truncated_paths(paired_paths, asc_graph, desc_graph))

        data.set_paths(i, extended_paired_paths)
        paths.append(extended_paired_paths)
    return paths

def eval_candidates(data: TestData):
    candidates = []
    for pivot_no, (pivot,residue,score,error) in enumerate(data.viable_pivots):
        paths = data.paths[pivot_no]

        disjoint_ind = mirror.graph_util.find_edge_disjoint_paired_paths(paths)

        for i,j in disjoint_ind:
            pass

def _call_sequence_from_path(spectrum, paired_path):
    for half_path in mirror.graph_util.unzip_paired_path(paired_path):
        edges = mirror.graph_util.path_to_edges(half_path)
        get_res = lambda e: mirror.util.residue_lookup(spectrum[max(e)] - spectrum[min(e)])
        yield list(map(get_res, edges))

def call_sequence_from_path(spectrum, paired_path):
    seq1, seq2 = _call_sequence_from_path(spectrum, paired_path)
    sequence = ''
    for (res1, res2) in zip(seq1, seq2):
        if res1 == res2 or (res1 == 'X' or res2 == 'X'):
            sequence += res1
        else:
            sequence += f"{res1}/{res2}"
        sequence += ' '
    return sequence.replace('X', 'x').strip()

def check_match(target, candidate):
    return "\t[ ✔️ ]" if candidate == target else ""

def reverse_called_sequence(sequence):
    return ' '.join(sequence.split(' ')[::-1])

def run(data):
    gaps = eval_gap(data)
    eval_pivot(data)
    pivots = eval_viable_pivots(data)
    graphs = eval_new_graphs(data)
    paths = eval_paths(data)
    target = [mirror.util.mask_ambiguous_residues(r) for r in data.peptide[1:-1]]
    target = ' '.join(target)
    matches = []
    n_pivots = len(pivots)
    for idx in range(n_pivots):
        pivot_res = pivots[idx][1]
        pathspace = paths[idx]
        n_paths = len(pathspace)
        residue_sequences = [call_sequence_from_path(data.sym_spectrum, p) for p in pathspace]
        for i in range(n_paths):
            for j in range(i + 1, n_paths):
                candidate = reverse_called_sequence(residue_sequences[i]) + f" {pivot_res} " + residue_sequences[j]
                alt_candidate = reverse_called_sequence(candidate)
                if target == candidate:
                    matches.append(candidate)
                if target == alt_candidate:
                    matches.append(alt_candidate)
    return matches, data

if __name__ == '__main__':
    import sys
    N = int(sys.argv[1])
    for k in [7, 10, 20, 30, 40, 50]:
        print(f"length {k}")
        misses = []
        crashes = []
        for _ in mirror.util.add_tqdm(range(N)):
            data = generate_test_data(k)
            try:
                matches, _ = run(generate_test_data(k))
                if len(matches) == 0:
                    misses.append(data)
            except Exception as e:
                eval_gap(data)
                eval_pivot(data)
                crashes.append(data)
        miss_peptides = [data.peptide for data in misses]
        crash_peptides = [data.peptide for data in crashes]
        n_misses = len(misses)
        n_crashes = len(crashes)
        n_overlap_crashes = len([data for data in crashes if data.pivot_mode == "o"])
        n_disjoint_crashes = len([data for data in crashes if data.pivot_mode == "d"])
        mirror.io.save_strings_to_fasta(f"misses/misses_{k}_{N}.fasta", miss_peptides, lambda i: f"miss_{i}")
        mirror.io.save_strings_to_fasta(f"misses/crashes_{k}_{N}.fasta", crash_peptides, lambda i: f"crash_{i}")
        print("misses:", n_misses / N, (n_misses, N))
        print("crashes:", n_crashes / N, (n_crashes, N))
        if any(crashes):
            print(f"crash pivot modes:\no {n_overlap_crashes}\nd {n_disjoint_crashes}")
        