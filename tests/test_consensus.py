from mirror.graphs.minimal_nodes import *
from mirror.graphs.minimal_paths import *
from mirror.graphs.graph_types import *
from mirror.graphs.align_types import *
from mirror.graphs.align import *
from mirror.graphs.consensus_types import *
from mirror.graphs.consensus import *

from random import shuffle

import unittest

class TestConsensus(unittest.TestCase):

    def _get_dag_1(self):
        g = DiGraph()
        g_edges = [(0, 2), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (5, 7)]
        g_weights = ['a','b','c','d','e','f','g']
        g_weight_key = "weight"
        g.add_edges_from(
            (i,j, {g_weight_key: w})
            for ((i, j), w) in zip(g_edges, g_weights))
        return DAG(g, g_weight_key)

    def _get_dag_2(self):
        g = DiGraph()
        g_edges = [(0, 2), (1, 2), (2, 3), (4, 5), (5, 6), (5, 7)]
        g_weights = ['a','b','c','e','f','g']
        g_weight_key = "weight"
        g.add_edges_from(
            (i,j, {g_weight_key: w})
            for ((i, j), w) in zip(g_edges, g_weights))
        return DAG(g, g_weight_key)

    def _get_dag_3(self):
        g = DiGraph()
        g_edges = [(0, 2), (1, 2), (2, 3), (3, 4), (5, 6), (5, 7)]
        g_weights = ['a','b','c','d','f','g']
        g_weight_key = "weight"
        g.add_edges_from(
            (i,j, {g_weight_key: w})
            for ((i, j), w) in zip(g_edges, g_weights))
        return DAG(g, g_weight_key)

    def _get_dag_4(self):
        g = DiGraph()
        g_edges = [(0, 2), (1, 2), (3, 4), (4, 5), (5, 6), (5, 7)]
        g_weights = ['a','b','d','e','f','g']
        g_weight_key = "weight"
        g.add_edges_from(
            (i,j, {g_weight_key: w})
            for ((i, j), w) in zip(g_edges, g_weights))
        return DAG(g, g_weight_key)
    
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

    def _print_alignment(self, alignment, first, second):
        for aligned_path in alignment:
            score = aligned_path.score
            fragment = aligned_path.alignment
            fragment_weights = [(first.weight_out(x1, y1), second.weight_out(x2, y2)) for ((x1, x2), (y1, y2)) in pairwise(fragment)]
            print(f"{score}\t{fragment}\n\t{fragment_weights}")
    
    def _test_fragment_itx(self, first_dag, second_dag, tag):
        # alignment
        aln = self._align(first_dag, second_dag)
        # fragment consensus
        frag_itx = FragmentIntersectionGraph(
            alignments = aln,
        )
        assert is_bipartite(frag_itx)
        component = list(connected_components(frag_itx))[0]
        frag_pair = FragmentPairGraph(frag_itx, component, LocalCostModel())
        fragment_consensii = []
        for src in frag_pair.sources():
            nc = propagate(
                topology = frag_pair,
                cost = lambda _, x: x,
                threshold = 1000,
                source = src
            )
            for snk in frag_pair.sinks():
                minimal_paths = backtrace(
                    topology = frag_pair,
                    cost = lambda _, x: x,
                    node_cost = nc,
                    threshold = 1000,
                    source = src,
                    sink = snk,
                )
                fragment_consensii.extend(minimal_paths)
        print(f"\nalignment {tag}:\n{aln}")
        print(f"fragment itx:\n{frag_itx}\n{frag_itx.edges(data = True)}\n{frag_itx._fragment_index}")
        print("fragment consensus:")
        for consensus_idx, (score, alignment_sequence) in enumerate(fragment_consensii):
            print(f"consensus[{consensus_idx}]\nscore: {score}\nseq: {alignment_sequence}")
            for alignment_idx in alignment_sequence:
                print(frag_itx.get_alignment(alignment_idx))
            print('-'*20)
        

    def test_fragment_itx(self):
        dag1 = self._get_dag_1()
        dag2 = self._get_dag_2()
        self._test_fragment_itx(dag1, dag2, "12")

        dag3 = self._get_dag_3()
        self._test_fragment_itx(dag3, dag2, "32")
        
        dag4 = self._get_dag_4()
        self._test_fragment_itx(dag3, dag4, "34")

    def _test_examples(self):
        dag1 = self._get_dag_1()
        dag2 = self._get_dag_2()
        dag3 = self._get_dag_3()
        dag4 = self._get_dag_4()
        print(f"alignment 1 - 2:")
        aln12 = self._align(dag1, dag2)
        self._print_alignment(aln12, dag1, dag2)
        score, aligned_fragments = zip(*aln12)
        realign_fragments(
            aligned_fragments, 
            StrongProductDAG(dag1, dag2),
            LocalCostModel())
        return

        print(f"alignment 3 - 2:")
        aln32 = self._align(dag3, dag2)
        self._print_alignment(aln32, dag3, dag2)
        print(f"alignment 3 - 4:")
        aln34 = self._align(dag3, dag4)
        self._print_alignment(aln34, dag3, dag4)

    def _test_prototype(self):
        from operator import eq, ne
        from functools import reduce
        from itertools import pairwise, chain

        d = DiGraph()
        d.add_edge(0,1,weight='a')
        d.add_edge(1,2,weight='b')
        d.add_edge(2,3,weight='c')
        d.add_edge(3,4,weight='d')
        dag1 = DAG(d, "weight")

        e = DiGraph()
        e.add_edge(0,1,weight='a')
        e.add_edge(1,2,weight='b')
        e.add_edge(3,4,weight='d')
        dag2 = DAG(e, "weight")

        prod = StrongProductDAG(dag1, dag2)
        aln_scores, aln_paths = zip(*align(prod, LocalCostModel()))

        # construct edge -> edge graph
        aln_edges = associate_matched_edges(prod, aln_paths)

        # enumerate path edges
        paths1, paths2 = zip(*map(lambda x: zip(*x), aln_paths))
        path_edges1 = map(lambda x: (x[0], pairwise(x[1])), enumerate(paths1))
        path_edges2 = map(lambda x: (x[0], pairwise(x[1])), enumerate(paths2))

        ## construct consensus graphs
        consensus1 = MultiDAG(edges = filter(lambda x: ne(x[0], x[1]), path_edges1))
        consensus2 = MultiDAG(edges = filter(lambda x: ne(x[0], x[1]), path_edges2))
        
        ## construct edge -> path tables
        ### table 1
        edge_to_paths1 = [dict() for _ in range(dag1.order())]
        for path_idx, path_edges in path_edges1:
            for (src, tgt) in path_edges:
                if ne(src, tgt):
                    if not(tgt in edge_to_paths1[src]):
                        edge_to_paths1[src][tgt] = []
                    edge_to_paths1[src][tgt].append(path_idx)

        ### table 2
        edge_to_paths2 = [dict() for _ in range(dag2.order())]
        for path_idx, path_edges in path_edges2:
            for (src, tgt) in path_edges:
                if ne(src, tgt):
                    if not(tgt in edge_to_paths2[src]):
                        edge_to_paths2[src][tgt] = []
                    edge_to_paths2[src][tgt].append(path_idx)