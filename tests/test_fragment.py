from mirror.graphs.minimal_nodes import *
from mirror.graphs.minimal_paths import *
from mirror.graphs.graph_types import *
from mirror.graphs.align_types import *
from mirror.graphs.align import *
from mirror.graphs.fragment_types import *
from mirror.graphs.fragment import *

from random import shuffle

import unittest

class TestFragment(unittest.TestCase):
    
    def _construct_dag(self, edges, weights, weight_key = "weight"):
        g = DiGraph()
        g.add_edges_from(
            (i,j, {weight_key: w})
            for ((i, j), w) in zip(edges, weights))
        return DAG(g, weight_key)

    def _get_dag_1(self):
        g_edges = [(0, 2), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (5, 7)]
        g_weights = ['a','b','c','d','e','f','g']
        return self._construct_dag(g_edges, g_weights)

    def _get_dag_2(self):
        g_edges = [(0, 2), (1, 2), (2, 3), (4, 5), (5, 6), (5, 7)]
        g_weights = ['a','b','c','e','f','g']
        return self._construct_dag(g_edges, g_weights)

    def _get_dag_3(self):
        g_edges = [(0, 2), (1, 2), (2, 3), (3, 4), (5, 6), (5, 7)]
        g_weights = ['a','b','c','d','f','g']
        return self._construct_dag(g_edges, g_weights)

    def _get_dag_4(self):
        g_edges = [(0, 2), (1, 2), (3, 4), (4, 5), (5, 6), (5, 7)]
        g_weights = ['a','b','d','e','f','g']
        return self._construct_dag(g_edges, g_weights)
    
    def _get_dag_5(self):
        g_edges = [(0, 2), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7), (7, 8), (7,9)]
        g_weights = ['a','b','c','c','d','e','f','g']
        return self._construct_dag(g_edges, g_weights)
    
    def _get_non_ladder_ex(self):
        dag1 = self._construct_dag(
            edges = pairwise(range(6)),
            weights = map(chr, range(ord('a'),ord('a') + 5)))
        dag2 = self._construct_dag(
            edges = [(0,1),(2,3),(4,5)],
            weights = ['a','c','e'])
        return dag1, dag2

    def _get_overlap_ex(self):
        dag1 = self._construct_dag(
            edges = pairwise(range(5)),
            weights = map(chr, range(ord('a'),ord('a') + 4)))
        dag2 = self._construct_dag(
            edges = [
                (0,1),(1,2),(2,3),
                (4,5),(5,6),(6,7)],
            weights = [
                'a','b','c',
                'b','c','d'])
        return dag1, dag2

    def _get_overlap_ex1(self):
        dag1 = self._construct_dag(
            edges = pairwise(range(5)),
            weights = map(chr, range(ord('a'),ord('a') + 4)))
        dag2 = self._construct_dag(
            edges = [
                (0,1),(1,2),(2,3),
                (4,5),(5,6)],
            weights = [
                'a','b','c',
                'c','d'])
        return dag1, dag2
    
    def _all_skips(self, fragment: list[tuple[int, int]]):
        fragment_edges = pairwise(fragment)
        for ((x1, x2), (y1, y2)) in fragment_edges:
            if x1 != y1 and x2 != y2:
                return False
        return True
    
    def _align(self, first: DAG, second: DAG):
        aln = align(
            product_graph = StrongProductDAG(first, second),
            cost_model = LocalCostModel(),
            threshold = 0)
        return [aligned_path for aligned_path in aln if (not self._all_skips(aligned_path.alignment))]
    
    def _test_fragment_itx(self, first_dag, second_dag, tag):
        # alignment
        aln = self._align(first_dag, second_dag)
        # fragment consensus
        frag_itx = FragmentIntersectionGraph(aln)
        fragment_consensii = chain_fragment_pairs(frag_itx, LocalCostModel(), 1000)
        print(f"\nalignment {tag}:\n{aln}")
        print(f"fragment itx:\n{frag_itx}\n{frag_itx.edges(data = True)}\n{frag_itx._fragment_index}")
        print("fragment consensus:")
        for consensus_idx, (score, alignment_sequence) in enumerate(fragment_consensii):
            print(f"consensus[{consensus_idx}]\nscore: {score}\nseq: {alignment_sequence}")
            for alignment_idx in alignment_sequence:
                print(frag_itx.get_alignment(alignment_idx))
            print('-'*20)
        fragment_chains = collate_fragments(aln, LocalCostModel, 1000)
        print(fragment_chains)
    
    def test_non_ladder(self):
        self._test_fragment_itx(*self._get_non_ladder_ex(), "non-ladder")
    
    def test_overlap(self):
        self._test_fragment_itx(*self._get_overlap_ex(), "overlap")

    def test_overlap1(self):
        self._test_fragment_itx(*self._get_overlap_ex1(), "overlap")

    def test_fragment_itx(self):
        dag1 = self._get_dag_1()
        dag2 = self._get_dag_2()
        self._test_fragment_itx(dag1, dag2, "12")

        dag3 = self._get_dag_3()
        self._test_fragment_itx(dag3, dag2, "32")
        
        dag4 = self._get_dag_4()
        self._test_fragment_itx(dag3, dag4, "34")

        dag1 = self._get_dag_1()
        dag5 = self._get_dag_5()
        self._test_fragment_itx(dag1, dag5, "15")
    