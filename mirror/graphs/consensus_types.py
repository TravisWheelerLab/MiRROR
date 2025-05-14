from typing import Iterator, Any
from itertools import chain, pairwise
from .graph_types import DiGraph, DAG, StrongProductDAG
from .align_types import AlignedPath, CostModel
from networkx import Graph, is_bipartite, connected_components

class FragmentIntersectionGraph(Graph):
    """A bipartite graph over aligned fragments."""

    def __init__(self,
        alignments: Iterator[tuple[float, list[tuple[int, int]]]],
        index_key = "index"
    ):
        n = 0
        fragment_index = dict()
        first_fragments = dict()
        second_fragments = dict()
        fragment_pairs = list()
        for alignment_index, aligned_path in enumerate(alignments):
            fragment1, fragment2 = aligned_path.fragments()
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
            fragment_pairs.append((index1, index2, {index_key : (alignment_index, index1, index2)}))
        
        # private fields
        self._alignments = alignments
        self._fragment_index = fragment_index
        self._first_fragments = first_fragments
        self._second_fragments = second_fragments
        
        # public fields
        self.index_key = index_key
        
        super(FragmentIntersectionGraph, self).__init__(incoming_graph_data = fragment_pairs)
    
    def get_first_fragment(self, i: int):
        return self._first_fragments[i]

    def get_second_fragment(self, i: int):
        return self._second_fragments[i]
    
    def get_alignment(self, i: int):
        return self._alignments[i]

class FragmentPairGraph(DAG):
    """A directed acyclic graph formed over a component of a FragmentIntersectionGraph
    by taking its edges as nodes, and its length-two paths as edges. Ordered according
    to the interval order of alignments."""

    def __init__(self,
        fragment_itx: FragmentIntersectionGraph,
        component: set[int],
        cost_model: CostModel,
        cost_key = "cost",
    ):
        index_key = fragment_itx.index_key
        pair_graph = DiGraph()
        for node_a1 in component:
            for node_b in fragment_itx[node_a1]:
                # unpack the first edge
                pair1 = (node_a1, node_b)
                alignment_index1, first_index1, second_index1 = fragment_itx[node_a1][node_b][index_key]
                alignment1 = fragment_itx.get_alignment(alignment_index1)
                for node_a2 in fragment_itx[node_b]:
                    if node_a2 != node_a1:
                        # unpack the second edge
                        pair2 = (node_a2, node_b)
                        alignment_index2, first_index2, second_index2 = fragment_itx[node_a2][node_b][index_key]
                        alignment2 = fragment_itx.get_alignment(alignment_index2)
                        # retrieve fragment and interval data
                        fragment_b = None
                        fragment_a1 = None
                        fragment_a2 = None
                        interval_a1 = None
                        interval_a2 = None
                        if node_b == second_index1 == second_index2:
                            assert node_a1 == first_index1 and node_a2 == first_index2
                            fragment_b = fragment_itx.get_second_fragment(node_b)
                            fragment_a1 = fragment_itx.get_first_fragment(node_a1)
                            fragment_a2 = fragment_itx.get_first_fragment(node_a2)
                            interval_a1 = alignment1.first_interval()
                            interval_a2 = alignment2.first_interval()
                        elif node_b == first_index1 == first_index2:
                            assert node_a1 == second_index1 and node_a2 == second_index2
                            fragment_b = fragment_itx.get_first_fragment(node_b)
                            fragment_a1 = fragment_itx.get_second_fragment(node_a1)
                            fragment_a2 = fragment_itx.get_second_fragment(node_a2)
                            interval_a1 = alignment1.second_interval()
                            interval_a2 = alignment2.second_interval()
                        else:
                            raise ValueError(f"indices and edges should be the same")
                        # determine edge direction
                        if interval_a1 < interval_a2:
                            gap_len = max(0, interval_a2[0] - interval_a1[1] - 1)
                            cost = alignment2.score + (cost_model.gap * gap_len)
                            pair_graph.add_edge(alignment_index1, alignment_index2)
                            pair_graph[alignment_index1][alignment_index2][cost_key] = cost
                        elif interval_a2 < interval_a1:
                            gap_len = max(0, interval_a1[0] - interval_a2[1] - 1)
                            cost = alignment1.score + (cost_model.gap * gap_len)
                            pair_graph.add_edge(alignment_index2, alignment_index1)
                            pair_graph[alignment_index2][alignment_index1][cost_key] = cost
        super(FragmentPairGraph, self).__init__(pair_graph, cost_key)

class FragmentChain:

    def __init__(self,
        score: float,
        alignment_chain: list[AlignedPath],
    ):
        # public fields
        self.score = score
        self.chain = alignment_chain
        # private fields
        n_aln = len(alignment_chain)
        first_fragments, second_fragments = zip(*map(lambda aln: aln.fragments(), alignment_chain))
        first_fragment_idx = list(range(n_aln))
        second_fragment_idx = list(range(n_aln))
        print("before filtering")
        print(f"first fragments:\n{first_fragments}\n{first_fragment_idx}\nsecond fragments:\n{second_fragments}\n{second_fragment_idx}\n")
        if alignment_chain[0].first_fragment() == alignment_chain[1].first_fragment():
            first_fragments = first_fragments[1:]
            first_fragment_idx = first_fragment_idx[1:]
        elif alignment_chain[0].second_fragment() == alignment_chain[1].second_fragment():
            second_fragments = second_fragments[1:]
            second_fragments_idx = second_fragments_idx[1:]
        else:
            raise ValueError("broken chain")
        first_fragments = [first_fragments[0]] + [first_fragments[i] for i in range(1, len(first_fragments), 2)]
        first_fragment_idx = [first_fragment_idx[0]] + [first_fragment_idx[i] for i in range(1, len(first_fragment_idx), 2)]
        second_fragments = [second_fragments[0]] + [second_fragments[i] for i in range(1, len(second_fragments), 2)]
        second_fragment_idx = [second_fragment_idx[0]] + [second_fragment_idx[i] for i in range(1, len(second_fragment_idx), 2)]
        print(f"first fragments:\n{first_fragments}\n{first_fragment_idx}\nsecond fragments:\n{second_fragments}\n{second_fragment_idx}\n")