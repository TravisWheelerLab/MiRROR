import itertools
from time import time
from copy import copy
import sys, os

import numpy as np
import networkx as nx

import mirror
from mirror import io, util, gap, pivot, boundary, graph_util, spectrum_graph, sequence

from networkx.drawing.nx_agraph import graphviz_layout, to_agraph

from editdistance import eval as edit_distance

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
        self.candidates = []
        self.matches = []
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
    
    def set_candidates(self, candidates):
        self.candidates = candidates
    
    def set_matches(self, matches):
        self.matches = matches

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

    def draw_graphs(self, pivot_no, name_gen = None):
        if name_gen == None:
            name_gen = lambda x: f"drawings/{self.peptide}_p{pivot_no}_{x}.png"
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
        n_affixes = len(residue_affixes)
        for i in range(n_affixes):
            for j in range(i + 1, n_affixes):
                initial_b = pivot.initial_b_ion(data.spectrum)[1]
                terminal_y = pivot.terminal_y_ion(data.spectrum)[1]
                called_seq = residue_affixes[i] + f" {pivot_res} " + reverse_called_sequence(residue_affixes[j])
                candidate = apply_boundary_residues(called_seq, initial_b, terminal_y)
                alt_candidate = apply_boundary_residues(reverse_called_sequence(called_seq), initial_b, terminal_y)
                candidates.append(candidate)
                candidates.append(alt_candidate)
                if target == candidate:
                    matches.append(candidate)
                if target == alt_candidate:
                    matches.append(alt_candidate)
    data.set_candidates(candidates)
    data.set_matches(matches)
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
        if res1 == 'X' and res2 != 'X':
            sequence += res2
        elif res1 != 'X' and res2 == 'X':
            sequence = res1 
        elif res1 == res2:
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

def measure_error(data: TestData):
    target = construct_target(data.peptide).replace(' ', '').replace("I/L", 'J')
    if len(data.matches) > 0:
        return 0
    elif len(data.candidates) > 0:
        candidates = [x.replace(' ', '').replace("I/L", 'J') for x in data.candidates]
        candidates.sort(key = lambda x: edit_distance(target, x))
        return edit_distance(target, candidates[0])
    else:
        return len(data.peptide)

def run_on_seqs(seqs, path_miss, path_crash):
    N = len(seqs)
    local_max_time = 0
    local_max_peptide = ""
    local_max_table = np.zeros(5)
    times_k = np.zeros(5)
    misses = []
    crashes = []
    for i in mirror.util.add_tqdm(range(N)):
        pep = seqs[i]
        data = TestData(pep)
        try:
            __, matches, data2, times_individual = run(data)
            times_k += times_individual
            if sum(times_individual) > local_max_time:
                local_max_time = sum(times_individual)
                local_max_peptide = pep
                local_max_table = times_individual
            if len(matches) == 0:
                misses.append(data)
        except KeyboardInterrupt:
            print('Interrupted')
            print(pep)
            try:
                sys.exit(130)
            except SystemExit:
                os._exit(130)
        except Exception as e:
            crashes.append(data)
    miss_peptides = [data.peptide for data in misses]
    crash_peptides = [data.peptide for data in crashes]
    n_misses = len(misses)
    n_crashes = len(crashes)
    if path_miss != None and n_misses > 0:
        mirror.io.save_strings_to_fasta(path_miss, miss_peptides, lambda i: f"miss_{i}")
    if path_crash != None and n_crashes > 0:
        mirror.io.save_strings_to_fasta(path_crash, crash_peptides, lambda i: f"crash_{i}")
    print("misses:", n_misses / N, (n_misses, N))
    print("crashes:", n_crashes / N, (n_crashes, N))
    print("observed times\t\t", times_k, sum(times_k))
    print("normalized times\t", times_k / sum(times_k))
    print("max", (local_max_time,local_max_peptide,local_max_table / sum(local_max_table)))
    return times_k, misses, crashes

def plot_hist(x: list[int], width = 60):
    left = min(x)
    right = max(x)
    vals = list(range(left, right + 1))
    counts = [0 for _ in range(left, right + 1)]
    for pt in x:
        counts[pt - left] += 1
    
    lo = min(counts)
    hi = max(counts)
    rng = hi - lo
    print('-' * (width + 20))
    for i in range(len(counts)):
        frac = (counts[i] - lo) / rng
        bars = int(width * frac)
        if counts[i] > 0:
            bars = max(bars, 1)
        print(f"{vals[i]}\t|" + "o" * bars + f"  ({counts[i]})")
    print('-' * (width + 20))

if __name__ == '__main__':
    import sys
    mode = sys.argv[1]
    if mode == "random":
        N = int(sys.argv[2])
        peptide_lengths = list(map(int, sys.argv[3].split(',')))
        record = sys.argv[4] == "True"
        times_overall = np.zeros(5)
        print(f"record {record}")
        print(f"trials {N}")
        for k in peptide_lengths:
            print(f"length {k}")
            seqs = [mirror.util.generate_random_tryptic_peptide(k) for _ in range(N)]
            times_k, _, __ = run_on_seqs(
                seqs, 
                f"misses/misses_{k}_{N}.fasta" if record else None, 
                f"misses/crashes_{k}_{N}.fasta" if record else None)
            times_overall += times_k
        print("overall observed times\t\t", times_overall, sum(times_overall))
        print("overall normalized times\t", times_overall / sum(times_overall))
    elif mode == "fasta":
        fasta_path = sys.argv[2]
        seqs = mirror.io.load_fasta_as_strings(fasta_path)
        _,  misses, crashes = run_on_seqs(seqs, None, None)
        miss_errors = [measure_error(miss) for miss in misses]
        num_pivots = 0
        num_virtual = 0
        for data in misses:
            if any(type(pivot) == mirror.pivot.VirtualPivot for pivot in data.viable_pivots):
                num_virtual += 1
            else:
                num_pivots += 1
        print("\nedit distance of errors:")
        plot_hist(miss_errors)
        print(f"min dist:\t{min(miss_errors)}")
        print(f"max dist:\t{max(miss_errors)}")
        print(f"avg dist:\t{np.mean(miss_errors)}")
        print(f"pivot types\nPivot:\t\t{num_pivots}\nVirtualPivot:\t{num_virtual}")