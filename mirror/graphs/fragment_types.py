from typing import Callable, Iterator, Any
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

    @classmethod
    def _sequence_fragments(cls,
        alignment_chain: list[AlignedPath],
    ) -> tuple[list[int],list[int]]:
        left_fragment_sequence = []
        right_fragment_sequence = []
        prev_left = None
        prev_right = None
        for (i, aln) in enumerate(alignment_chain):
            curr_left, curr_right = aln.fragments()
            if curr_left != prev_left:
                left_fragment_sequence.append(i)
            if curr_right != prev_right:
                right_fragment_sequence.append(i)
            prev_left = curr_left
            prev_right = curr_right
        return left_fragment_sequence, right_fragment_sequence
    
    @classmethod
    def _parametize_concatenations(cls,
        alignment_chain: list[AlignedPath],
        fragment_sequence: list[int],
        get_interval: Callable[[AlignedPath], tuple[int, int]],
        score_interval: Callable[[AlignedPath, tuple[int, int]], float],
    ) -> tuple[list[int], list[int]]:
        truncation_sequence = []
        padding_sequence = []
        for (curr_i, next_i) in pairwise(fragment_sequence):
            curr_interval = get_interval(alignment_chain[curr_i])
            next_interval = get_interval(alignment_chain[next_i])
            ### parametize the truncation
            interval_gap = curr_interval[1] - next_interval[0]
            print(f"curr {curr_interval}\nnext {next_interval}\ngap {interval_gap}")
            if interval_gap > 0:
                overlap_interval = (next_interval[0], curr_interval[1])
                print(f"ovlp {overlap_interval}")
                curr_subscore = score_interval(alignment_chain[curr_i], overlap_interval)
                next_subscore = score_interval(alignment_chain[next_i], overlap_interval)
                truncator = interval_gap + 1
                if next_subscore > curr_subscore:
                    # truncate curr_fragment
                    truncation_sequence.append(-truncator)
                    padding_sequence.append(0)
                else:
                    # truncate next_fragment - how can this be implemented?
                    truncation_sequence.append(truncator)
                    padding_sequence.append(0)
            else:
                # no truncation
                truncation_sequence.append(0)
                padding_sequence.append(-(interval_gap + 1))
        return truncation_sequence, padding_sequence
    
    @classmethod
    def _decompose_pad_truncate(cls,
        alignment_chain: list[AlignedPath],
        fragment_sequence: list[int],
        padding: list[int],
        truncations: list[int],
        get_fragment: Callable[[AlignedPath], list[int]],
        get_weights: Callable[[AlignedPath], list[Any]]
    ) -> tuple[list[list[int]], list[list[Any]]]:
        sequence_decomposition = [get_fragment(alignment_chain[fragment_sequence[0]])]
        weight_sequence = [get_weights(alignment_chain[fragment_sequence[0]])]
        for prev_position, (truncator, padding, aln_idx) in enumerate(zip(truncations, padding, fragment_sequence[1:])):
            fragment = get_fragment(alignment_chain[aln_idx])
            weights = get_weights(alignment_chain[aln_idx])
            if truncator > 0:
                sequence_decomposition.append(fragment[truncator:])
                weight_sequence.append(weights[truncator:])
            else:
                if truncator < 0:
                    prev_fragment = sequence_decomposition[prev_position]
                    sequence_decomposition[prev_position] = prev_fragment[:truncator]
                    prev_weight = weight_sequence[prev_position]
                    weight_sequence[prev_position] = prev_weight[:truncator]
                elif padding > 0:
                    prev_fragment = sequence_decomposition[prev_position]
                    sequence_decomposition[prev_position] = prev_fragment + [prev_fragment[-1]] * padding
                    prev_weight = weight_sequence[prev_position]
                    weight_sequence[prev_position] = prev_weight + ['None'] * padding
                sequence_decomposition.append(fragment)
                weight_sequence.append(weights)
        return sequence_decomposition, weight_sequence
    
    @classmethod
    def _concatenate_decomposed_sequences(cls,
        edge_decomposition: list[list[tuple[int, int]]],
        weight_decomposition: list[list[Any]],
    ) -> tuple[list[tuple[int, int]], list[Any]]:
        return (
            list(chain.from_iterable(map(pairwise, edge_decomposition))),
            list(chain.from_iterable(weight_decomposition)))

    def __init__(self,
        score: float,
        alignment_chain: list[AlignedPath],
    ):
        # public fields
        self.score = score
        self.alignment_chain = alignment_chain
        
        # private fields
        ## 1.   determine the correct fragment sequence
        self._left_fragment_sequence, self._right_fragment_sequence = self._sequence_fragments(
            alignment_chain = self.alignment_chain)
        
        ## 2.   parametize concatenations with a truncation value: 
        ##      an integer that is negative if the left term is truncated, 
        ##      positive if the right term is truncated,
        ##      and zero if neither are truncated,
        ##      as well as a padding value:
        ##      if no truncation is occurring, but there is a gap between the intervals,
        ##      the padding value determines how many gap edges are inserted.
        self._left_truncation_sequence, self._left_padding_sequence = self._parametize_concatenations(
            alignment_chain = self.alignment_chain,
            fragment_sequence = self._left_fragment_sequence,
            get_interval = lambda aligned_path: aligned_path.first_interval(),
            score_interval = lambda aligned_path, interval: aligned_path.subscore(left_sub_interval = interval))
        self._right_truncation_sequence, self._right_padding_sequence = self._parametize_concatenations(
            alignment_chain = self.alignment_chain,
            fragment_sequence = self._right_fragment_sequence,
            get_interval = lambda aligned_path: aligned_path.second_interval(),
            score_interval = lambda aligned_path, interval: aligned_path.subscore(right_sub_interval = interval))
        
        ## 3.   construct the truncated decomposition.
        self._left_edge_decomposition, self._left_weight_decomposition = self._decompose_pad_truncate(
            alignment_chain = self.alignment_chain,
            fragment_sequence = self._left_fragment_sequence,
            padding = self._left_padding_sequence,
            truncations = self._left_truncation_sequence,
            get_fragment = lambda aligned_path: aligned_path.first_fragment(),
            get_weights = lambda aligned_path: aligned_path.first_aligned_weights())
        self._right_edge_decomposition, self._right_weight_decomposition = self._decompose_pad_truncate(
            alignment_chain = self.alignment_chain,
            fragment_sequence = self._right_fragment_sequence,
            padding = self._right_padding_sequence,
            truncations = self._right_truncation_sequence,
            get_fragment = lambda aligned_path: aligned_path.second_fragment(),
            get_weights = lambda aligned_path: aligned_path.second_aligned_weights())

        ## 4.   finally, construct the sequence.
        self._left_edge_sequence, self._left_weight_sequence = self._concatenate_decomposed_sequences(
            edge_decomposition = self._left_edge_decomposition,
            weight_decomposition = self._left_weight_decomposition)
        self._right_edge_sequence, self._right_weight_sequence = self._concatenate_decomposed_sequences(
            edge_decomposition = self._right_edge_decomposition,
            weight_decomposition = self._right_weight_decomposition)

    def first_edges(self):
        return self._left_edge_sequence

    def first_aligned_edges(self):
        return list(filter(lambda x: x[0] != x[1], self._left_edge_sequence))

    def first_weights(self):
        return self._left_weight_sequence
    
    def first_aligned_weights(self):
        return list(filter(lambda x: x is not None, self._left_weight_sequence))

    def second_edges(self):
        return self._right_edge_sequence
    
    def second_aligned_edges(self):
        return list(filter(lambda x: x[0] != x[1], self._right_edge_sequence))

    def second_weights(self):
        return self._right_weight_sequence
    
    def second_aligned_weights(self):
        return list(filter(lambda x: x is not None, self._right_weight_sequence))
    
    def __repr__(self):
        return f"""FragmentChain
first edges\t{self.first_edges()}
first weights\t{self.first_weights()}
second edges\t{self.second_weights()}
second weights\t{self.second_edges()}
"""