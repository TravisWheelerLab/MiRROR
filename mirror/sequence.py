from .util import residue_lookup, collapse_second_order_list, reflect, AMINOS, AMINO_MASS_MONO, ION_OFFSET_LOOKUP, TOLERANCE
from .scan import ScanConstraint, constrained_pair_scan
from .pivot import Pivot
from .spectrum_graphs import * 

from networkx import DiGraph

import itertools
        
def recover_initial_residue(
    mz: float,
    pivot: float,
    amino_names = AMINOS,
    amino_masses = AMINO_MASS_MONO,
    ion_offset = ION_OFFSET_LOOKUP
):
    residue_mass = reflect(mz, pivot) - ion_offset['b']
    return residue_lookup(residue_mass, amino_names, amino_masses)

def recover_final_residue(
    mz: float,
    amino_names = AMINOS,
    amino_masses = AMINO_MASS_MONO,
    ion_offset = ION_OFFSET_LOOKUP
):
    residue_mass = mz - ion_offset['y']
    return residue_lookup(residue_mass, amino_names, amino_masses)

class PartialSequence:

    def __init__(self,
        desc_path,
        asc_path,
        desc_graph,
        asc_graph,
        gap_key,
        res_key
    ):
        assert len(desc_path) == len(asc_path)
        self._len = len(asc_path)
        self._desc_path = desc_path
        self._asc_path = asc_path
        self._desc_graph = desc_graph
        self._asc_graph = asc_graph
        self._gap_key = gap_key
        self._res_key = res_key
    
    def __len__(self):
        return self._len
    
    def get_edges(self):
        di = self.desc_path
        ai = self.asc_path
        return [((di[i], di[i + 1]), (ai[i], ai[i + 1])) for i in range(len(self) - 1)]
    
    def is_disjoint(self, other):
        return len(set(self.get_edges()) & set(other.get_edges())) == 0
    
    def get_residues(self):
        desc_residues = list(map(self._weight_transform, get_weights(self._desc_graph, self._desc_path, self._gap_key)))
        
    def __repr__(self):
        #precision = 3
        #round_gaps = lambda gaps: list(map(lambda x: round(x, precision), gaps))
        #round_pairs = lambda pairs: list(map(lambda x: (round(x[0], precision),round(x[1], precision)), pairs))
        #make_str = lambda idcs, gaps, pairs: f"path\t{idcs}\ngaps\t{round_gaps(gaps)}\npairs\t{round_pairs(pairs)}"
        #desc = make_str(self.desc_indices, self.desc_gaps, self.desc_pairs)
        #asc = make_str(self.asc_indices, self.asc_gaps, self.asc_pairs)
        #return f"\n{self.residues}\nd {desc}\na {asc}\n"

def construct_partial_sequences(
    asc_graph: DiGraph,
    desc_graph: DiGraph,
    tolerance = TOLERANCE,
    amino_names = AMINOS,
    amino_masses = AMINO_MASS_MONO,
    gap_key = GAP_KEY,
    res_key = RES_KEY
):
    #asc_graph, desc_graph = construct_spectrum_graphs(spectrum, pivot)
    asc_sources = get_sources(asc_graph)
    asc_sinks = get_sinks(asc_graph)
    desc_sources = get_sources(desc_graph)
    desc_sinks = get_sinks(desc_graph)
    # iterate all the shared path spaces for all combinations of source nodes
    # in the product space
    mirror_paths_iterator = itertools.chain.from_iterable(
        weighted_paired_simple_paths(
            desc_graph, d_source, set(desc_sinks),
            asc_graph, a_source, set(asc_sinks),
            res_key) 
        for d_source in desc_sources for a_source in asc_sources
    )
    # iterate paths; use path weights to generate amino sequences
    for mirror_path_pair in mirror_paths_iterator:
        desc_path, asc_path = list(zip(*mirror_path_pair))
        yield PartialSequence(desc_path, asc_path, desc_graph, asc_path, gap_key, res_key)
    
class JoinedSequence:

    def __init__(self, sequence_a: PartialSequence, sequence_b: PartialSequence, pivot: Pivot):
        self.sequence_a = sequence_a
        self.sequence_b = sequence_b
        self.pivot = pivot

    def call(self):
        # TODO
        # recover initial and final residues
        # using initial and final residues, order the partial sequence pair
        # reverse and concatenate residue strings around pivot char
        pivot = [residue_lookup(self.pivot.gap())]
        seq_a = self.sequence_a.residues
        seq_b = self.sequence_b.residues
        return [
            seq_a + pivot + seq_b[::-1],
            seq_b + pivot + seq_a[::-1],
            seq_a[::-1] + pivot + seq_b,
            seq_b[::-1] + pivot + seq_a,
        ]
    
    def __repr__(self):
        calls = self.call()
        call_abr, call_bar, call_arb, call_bra = map(lambda x: ' '.join(x), calls)
        return f"ab̅:\t{call_abr}\nba̅:\t{call_bar}\na̅b:\t{call_arb}\nb̅a:\t{call_bra}\n"


def construct_candidates(partial_seqs: list[PartialSequence], pivot: Pivot):
    # if the size of partial_seqs becomes unruly, implement the methods from ../2024_1101-disjoint_pairs.
    n = len(partial_seqs)
    return [JoinedSequence(partial_seqs[i], partial_seqs[j], pivot) for i in range(n) for j in range(i + 1, n) if partial_seqs[i].is_disjoint(partial_seqs[j])]