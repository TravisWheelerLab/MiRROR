import itertools
from time import time
from copy import copy
import sys, os

import numpy as np
import networkx as nx

import mirror
from mirror import io, util, gap, pivot, boundary, graph_util, spectrum_graph, candidate

from networkx.drawing.nx_agraph import graphviz_layout, to_agraph

from editdistance import eval as edit_distance

from Bio import Align

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
        
def eval_graphs(data: TestData):
    graphs = []
    for pivot_no, pivot in enumerate(data.viable_pivots):
        # identify boundary y and b ions, augment spectrum to include mirrored boundaries.
        sym_spectrum, pivot = mirror.boundary.create_symmetric_boundary(data.spectrum, pivot)
        data.set_symmetric_spectrum(pivot_no, sym_spectrum)
        # reconstruct gap set over the augmented spectrum
        gap_inds = [mirror.gap.find_gaps(sym_spectrum, mirror.gap.GapTargetConstraint(amino_mass, mirror.util.GAP_TOLERANCE))
                    for amino_mass in mirror.util.AMINO_MASS_MONO]
        gap_ind = list(itertools.chain.from_iterable(gap_inds))
        # filter out bad gaps
        negative_gaps = pivot.negative_index_pairs()
        gap_ind = [(i,j) for (i,j) in gap_ind if (i,j) not in negative_gaps]
        # 
        asc_graph, desc_graph = mirror.spectrum_graph.construct_spectrum_graphs(sym_spectrum, gap_ind, pivot)
        data.set_graphs(pivot_no, asc_graph, desc_graph)
        graphs.append((asc_graph, desc_graph))
    return graphs

def eval_paths(data: TestData):
    paths = []
    for i in range(data.num_pivots()):
        asc_graph, desc_graph = data.graphs[i]

        paired_paths = list(mirror.graph_util.all_weighted_paired_simple_paths(
            asc_graph, 
            desc_graph, 
            mirror.spectrum_graph.GAP_KEY, 
            mirror.spectrum_graph.GAP_COMPARATOR))
        
        data.set_paths(i, paired_paths)
        paths.append(paired_paths)
    return paths

def eval_candidates(data: TestData):
    peptide = data.peptide
    spectrum = data.spectrum
    pivots = data.viable_pivots
    matches = []
    all_candidates = []
    n_pivots = len(pivots)
    for idx in range(n_pivots):
        pivot = pivots[idx]
        affixes = data.paths[idx]
        augmented_spectrum = data.sym_spectrum[idx]
        graphs = data.graphs[idx]
        candidates = mirror.candidate.construct_candidates(spectrum, augmented_spectrum, pivot, graphs, affixes)
        for cand in candidates:
            cand_seqs = cand.sequences()
            optimum, optimizer = cand.edit_distance(peptide)
            all_candidates.append(cand)
            if optimum == 0:
                matches.append(cand_seqs[optimizer])    
    data.set_candidates(all_candidates)
    data.set_matches(matches)
    return all_candidates, matches

def time_op(op, arg, table, name):
    t_init = time()
    val = op(arg)
    table[name] = time() - t_init
    return val

def run(data):
    times = np.zeros(5)
    gaps = time_op(eval_gap, data, times, 0)
    pivots = time_op(eval_viable_pivots, data, times, 1)
    graphs = time_op(eval_graphs, data, times, 2)
    paths = time_op(eval_paths, data, times, 3)
    candidates, matches = time_op(eval_candidates, data, times, 4)
    return candidates, matches, data, times

def measure_error_distance(data: TestData):
    target = data.peptide
    if len(data.matches) > 0:
        return 0
    elif len(data.candidates) > 0:
        candidate_errs = [candidate.edit_distance(target)[0] for candidate in data.candidates]
        return min(candidate_errs)
    else:
        return len(data.peptide)

def measure_error_distribution(data: TestData):
    overall_err = measure_error_distance(data)
    n = len(data.peptide)
    if overall_err == n:
        return [np.ones(n)]
    else:
        results = [(cand,cand.edit_distance(data.peptide)) for cand in data.candidates]
        best_seqs = [cand.sequences()[minimizer].replace("I/L", 'J').replace(' ', '')
            for (cand,(min_err, minimizer)) in results if min_err == overall_err]
        aligner = Align.PairwiseAligner()
        alignment_indices = [aligner.align(data.peptide, seq)[0].indices[0,:] for seq in best_seqs]
        error_vectors = [np.array([1 if v == -1 else 0 for v in ind]) for ind in alignment_indices]
        return error_vectors

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
            raise e
            crashes.append(data)
    miss_peptides = [data.peptide for data in misses]
    crash_peptides = [data.peptide for data in crashes]
    n_misses = len(misses)
    n_crashes = len(crashes)
    if path_miss != None and n_misses > 0:
        mirror.io.save_strings_to_fasta(path_miss, miss_peptides, lambda i: f"miss_{i}")
    if path_crash != None and n_crashes > 0:
        mirror.io.save_strings_to_fasta(path_crash, crash_peptides, lambda i: f"crash_{i}")
    print("| misses:", n_misses / N, (n_misses, N))
    print("| crashes:", n_crashes / N, (n_crashes, N))
    print(f"| observed times\t{times_k} {sum(times_k)}")
    print(f"| normalized times\t{times_k / sum(times_k)}")
    print(f"| max individual time:\n\t| elapsed:\t{local_max_time}\n\t| normalized:\t{local_max_table / sum(local_max_table)}\n\t| peptide:\t{local_max_peptide}")
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
            print(f"\nlength {k}")
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
        miss_distances = [measure_error_distance(data) for data in misses]
        num_pivots = 0
        num_virtual = 0
        for data in misses:
            if any(type(pivot) == mirror.pivot.VirtualPivot for pivot in data.viable_pivots):
                num_virtual += 1
            else:
                num_pivots += 1
        print("\nedit distance of errors:")
        plot_hist(miss_distances)
        print(f"min dist:\t{min(miss_distances)}")
        print(f"max dist:\t{max(miss_distances)}")
        print(f"avg dist:\t{np.mean(miss_distances)}")
        print(f"pivot types\nPivot:\t\t{num_pivots}\nVirtualPivot:\t{num_virtual}")
        for data in misses:
            input()
            target_str = ' '.join(candidate.construct_target(data.peptide))
            error_description = []
            error_description.append(target_str)
            error_description.append('-' * len(target_str))
            if len(data.candidates) == 0:
                error_description.append("no match!")
            for pivot_no, cand in enumerate(data.candidates):
                data.draw_graphs(pivot_no, name_gen = lambda x: f"current_{x}.png")
                best_query = cand.sequences()[cand.edit_distance(data.peptide)[1]]
                error_description.append(best_query)
                pivot = data.viable_pivots[pivot_no]
                pivot_res = util.residue_lookup(pivot.gap())
                def offset_gaps(pivot, spectrum):
                    true_gap = pivot.gap()
                    inds_a, inds_b = pivot.index_pairs()
                    offset_pairs = [
                        (inds_a[0], inds_b[0]),
                        (inds_a[1], inds_b[1]),
                        (inds_a[1], inds_b[0]),
                        (inds_a[0], inds_b[1])]
                    offset_gaps = [spectrum[max(e)] - spectrum[min(e)] for e in offset_pairs]
                    offset_res = [util.residue_lookup(g) for g in offset_gaps]
                    return offset_gaps, offset_res
                off_gaps, off_res = offset_gaps(pivot, data.sym_spectrum[pivot_no])
                offsets = list(zip(off_res, off_gaps))
                pivot_type = str(type(pivot))
                boundary = cand._boundary
                affixes = cand._affixes
                error_description.append(f"\tboundary: {boundary}\n\taffixes: {affixes}\n\tpivot: {pivot_res} {pivot_type}\n\toffset: {offsets}")
                if len(data.candidates) > 1:
                    input("(next candidate)")
            error_desc_str = '\n'.join(error_description)
            print(error_desc_str)