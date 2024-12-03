import itertools
from time import time
from copy import copy
import sys, os

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
        self.sym_spectrum = {}
        self.graphs = {}
        self.paths = {}
        self.candidates = {}
        self.exp_sym_time = 0.0

    def set_exp_sym_time(self, time):
        self.exp_sym_time = time

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

    def set_symmetric_spectrum(self, pivot_no, spectrum):
        self.sym_spectrum[pivot_no] = spectrum
    
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

def eval_viable_pivots(data: TestData):
    t_init = time()
    symmetry_threshold = 2 * mirror.util.expected_num_mirror_symmetries(data.spectrum)
    t_elap = time() - t_init
    viable_pivots = mirror.pivot.construct_viable_pivots(data.spectrum, symmetry_threshold, data.gaps)
    data.set_viable_pivots(viable_pivots)
    data.set_exp_sym_time(t_elap)
    return viable_pivots

def eval_old_graphs(data: TestData):
    graphs = []
    for pivot_no, pivot in enumerate(data.viable_pivots):
        asc_graph, desc_graph = mirror.spectrum_graph.construct_spectrum_graphs_from_spectrum(data.spectrum, pivot)
        data.set_graphs(pivot_no, asc_graph, desc_graph)
        graphs.append((asc_graph, desc_graph))
    return graphs
        
def eval_graphs(data: TestData):
    graphs = []
    for pivot_no, pivot in enumerate(data.viable_pivots):
        asc_graph, desc_graph = mirror.spectrum_graph.construct_spectrum_graphs(data.spectrum, data.gaps, pivot)
        data.set_graphs(pivot_no, asc_graph, desc_graph)
        graphs.append((asc_graph, desc_graph))
    return graphs
        
def eval_new_graphs(data: TestData):
    graphs = []
    for pivot_no, pivot in enumerate(data.viable_pivots):
        sym_spectrum, pivot = mirror.boundary.create_symmetric_boundary(data.spectrum, pivot)
        data.set_symmetric_spectrum(pivot_no, sym_spectrum)
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
    pivots = data.viable_pivots
    paths = data.paths
    target = construct_target(data.peptide)
    matches = []
    candidates = []
    n_pivots = len(pivots)
    for idx in range(n_pivots):
        pivot = pivots[idx]
        pivot_res = mirror.util.residue_lookup(pivots[idx].gap())
        pathspace = paths[idx]
        n_paths = len(pathspace)
        residue_affixes = [call_sequence_from_path(data.sym_spectrum[idx], p) for p in pathspace]
        for i in range(n_paths):
            for j in range(i + 1, n_paths):
                initial_b = pivot.initial_b_ion(data.spectrum)[1]
                terminal_y = pivot.terminal_y_ion(data.spectrum)[1]
                called_seq = reverse_called_sequence(residue_affixes[i]) + f" {pivot_res} " + residue_affixes[j]
                candidate = apply_boundary_residues(called_seq, initial_b, terminal_y)
                alt_candidate = apply_boundary_residues(reverse_called_sequence(called_seq), initial_b, terminal_y)
                candidates.append(candidate)
                candidates.append(alt_candidate)
                if target == candidate:
                    matches.append(candidate)
                if target == alt_candidate:
                    matches.append(alt_candidate)
    return candidates, matches

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

def apply_boundary_residues(candidate, initial_b, terminal_y):
    return initial_b + ' ' + candidate + ' ' + terminal_y

def construct_target(peptide):
    target = [mirror.util.mask_ambiguous_residues(r) for r in peptide]
    target = ' '.join(target)
    return target

def time_op(op, arg, table, name):
    t_init = time()
    val = op(arg)
    table[name] = time() - t_init
    return val

def run(data):
    times = np.zeros(5)
    gaps = time_op(eval_gap, data, times, 0)
    pivots = time_op(eval_viable_pivots, data, times, 1)
    graphs = time_op(eval_new_graphs, data, times, 2)
    paths = time_op(eval_paths, data, times, 3)
    candidates, matches = time_op(eval_candidates, data, times, 4)
    return candidates, matches, data, times

if __name__ == '__main__':
    import sys
    N = int(sys.argv[1])
    times_overall = np.zeros(5)
    peptide_lengths = [7, 10, 20, 30, 40, 50]
    max_time_peptides = []
    for k in peptide_lengths:
        local_max_time = 0
        local_max_peptide = ""
        local_max_table = []
        times_k = np.zeros(5)
        print(f"length {k}")
        misses = []
        crashes = []
        for _ in mirror.util.add_tqdm(range(N)):
            data = generate_test_data(k)
            try:
                __, matches, data2, times_individual = run(generate_test_data(k))
                times_k += times_individual
                if sum(times_individual) > local_max_time:
                    local_max_time = sum(times_individual)
                    local_max_peptide = data2.peptide
                    local_max_table = times_individual
                if len(matches) == 0:
                    misses.append(data)
            except KeyboardInterrupt:
                print('Interrupted')
                print(data.peptide)
                try:
                    sys.exit(130)
                except SystemExit:
                    os._exit(130)
            except Exception as e:
                raise e
                eval_gap(data)
                eval_viable_pivots(data)
                append(data)
                print(e)
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
        print("observed times\t\t", times_k, sum(times_k))
        print("normalized times\t", times_k / sum(times_k))
        print("max", (local_max_time,local_max_peptide,local_max_table / sum(local_max_table)))
        times_overall += times_k
    print("overall observed times\t\t", times_overall, sum(times_overall))
    print("overall normalized times\t", times_overall / sum(times_overall))