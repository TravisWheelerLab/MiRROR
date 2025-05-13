from typing import Iterator, Any
from itertools import chain
from .graph_types import DiGraph, DAG, StrongProductDAG
from .align_types import CostModel
from networkx import Graph, is_bipartite

class FragmentIntersectionGraph(Graph):
    """A bipartite graph over aligned fragments.."""

    def __init__(self,
        alignments: Iterator[tuple[float, list[tuple[int, int]]]],
        product_graph: StrongProductDAG,
        cost_model: CostModel,
        weight_key = "score",
    ):
        n = 0
        fragment_index = dict()
        first_fragments = dict()
        second_fragments = dict()
        fragment_pairs = list()
        for score, aligned_fragment in alignments:
            fragment1, fragment2 = zip(*aligned_fragment)
            # index the first fragment
            key1 = (1, tuple(fragment1))
            if key1 not in fragment_index:
                fragment_index[key1] = n
                first_fragments[n] = fragment1
                n += 1
            index1 = fragment_index[key1]
            # index the second fragment
            key2 = (2, tuple(fragment2))
            if key2 not in fragment_index:
                fragment_index[key2] = n
                second_fragments[n] = fragment2
                n += 1
            index2 = fragment_index[key2]
            # create a weighted edge between the fragment indices
            fragment_pairs.append((index1, index2, {weight_key : score}))
        self._fragment_index = fragment_index
        self._first_fragments = first_fragments
        self._second_fragments = second_fragments
        super(FragmentIntersectionGraph, self).__init__(incoming_graph_data = fragment_pairs)
    
    def get_first_fragment(self, i: int):
        return self._first_fragments[i]

    def get_second_fragment(self, i: int):
        return self._second_fragments[i]


class FragmentPairGraph(DAG):
    """A directed acyclic dag over pairs of aligned fragments."""

