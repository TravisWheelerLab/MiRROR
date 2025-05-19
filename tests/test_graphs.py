from mirror.graphs.minimal_nodes import *
from mirror.graphs.minimal_paths import *
from mirror.graphs.graph_types import *
from mirror.graphs.align_types import *
from mirror.graphs.align import *
from mirror.graphs.fragment_types import *
from mirror.graphs.fragment import *

import unittest
from random import shuffle

class TestGraphTypes(unittest.TestCase):
    
    @classmethod
    def _construct_dag(cls, edges, weights, weight_key = "weight"):
        g = DiGraph()
        g.add_edges_from(
            (i,j, {weight_key: w})
            for ((i, j), w) in zip(edges, weights))
        return DAG(g, weight_key)

    def test_dag(self):
        dag = self._construct_dag(
            edges = [(0,1),(0,2),(1,3),(2,3)],
            weights = ['a','b','c','d'])

        try:
            dag = self._construct_dag(
                edges = [(0,1),(0,2),(1,3),(3,0)],
                weights = ['a','b','c','d'])
        except ValueError as e:
            self.assertEqual(str(e), "not a directed acyclic graph!")

    def test_products(self):
        # construct DAGs
        dag = self._construct_dag(
            edges = [(0,1),(0,2),(1,3),(2,3)],
            weights = ['a','b','c','d'])

        dag2 = self._construct_dag(
            edges = [(0,1),(1,3),(2,3)],
            weights = ['a','b','d'],
            weight_key = "weight2")
        
        # construct, verify direct product
        d_prod = DirectProductDAG(dag, dag2)
        d_prod_neighbors = [d_prod.unravel(i) for i in d_prod.adj_out(0)]
        expected_neighbors = [(1,1),(2,1)]
        self.assertEqual(d_prod_neighbors, expected_neighbors)

        d_prod_neighbor_weights = [d_prod.weight_out(0, i) for i in d_prod.adj_out(0)]
        expected_neighbor_weights = [('a','a'),('b','a')]
        self.assertEqual(d_prod_neighbor_weights, expected_neighbor_weights)

        d_prod_edges = [(i, j) for i in range(d_prod.order()) for j in d_prod.adj_out(i)]
        expected_num_edges = d_prod.size()
        self.assertEqual(len(d_prod_edges), expected_num_edges)
        
        # construct, verify box product
        b_prod = BoxProductDAG(dag, dag2)
        b_prod_neighbors = [b_prod.unravel(i) for i in b_prod.adj_out(0)]
        expected_neighbors = [(1,0),(2,0),(0,1)]
        self.assertEqual(b_prod_neighbors, expected_neighbors)
        
        b_prod_neighbor_weights = [b_prod.weight_out(0, i) for i in b_prod.adj_out(0)]
        expected_neighbor_weights = [('a',None),('b',None),(None,'a')]
        self.assertEqual(b_prod_neighbor_weights, expected_neighbor_weights)

        b_prod_edges = [(i, j) for i in range(b_prod.order()) for j in b_prod.adj_out(i)]
        expected_num_edges = b_prod.size()
        self.assertEqual(len(b_prod_edges), expected_num_edges)

        # construct, verify strong product as union of direct and box product graphs
        s_prod = StrongProductDAG(dag, dag2)
        s_prod_neighbors = [s_prod.unravel(i) for i in s_prod.adj_out(0)]
        expected_neighbors = list(chain(d_prod_neighbors, b_prod_neighbors))
        self.assertEqual(s_prod_neighbors, expected_neighbors)

class TestMinimalNodes(unittest.TestCase):
    
    def test_minimal_nodes(self):
        dag = TestGraphTypes._construct_dag(
            edges = [(0,1),(0,2),(1,3),(2,3)],
            weights = ['a','b','c','d'])

        dag2 = TestGraphTypes._construct_dag(
            edges = [(0,1),(1,3),(2,3)],
            weights = ['a','c','d'])

        s_prod = StrongProductDAG(dag, dag2)

        cost_table = lambda u, v, x, y: 0. if (x == y) else (1. if ((x != None) and (y != None)) else 2.)
        cost = lambda edge, weight: cost_table(*edge, *weight)
        self.assertEqual(
            0.,
            cost((None, None), ('a', 'a'))
        )
        self.assertEqual(
            1.,
            cost((None, None), ('a', 'b'))
        )
        self.assertEqual(
            2.,
            cost((None, None), ('a', None))
        )

        threshold = 1e4
        source = 0

        node_cost_table = propagate(
            s_prod,
            cost,
            threshold,
            source,
        )

        print("cost table")
        for (k, v) in node_cost_table.items():
            print(f"{v}\t{s_prod.unravel(k)}")

class TestMinimalPaths(unittest.TestCase):
    
    def test1_minimal_paths(self):
        dag = TestGraphTypes._construct_dag(
            edges = [(0,1),(0,2),(1,3),(2,3)],
            weights = ['a','b','c','d'])

        dag2 = TestGraphTypes._construct_dag(
            edges = [(0,1),(1,3),(2,3)],
            weights = ['a','c','d'])

        s_prod = StrongProductDAG(dag, dag2)

        cost_table = lambda u, v, x, y: 0. if (x == y) else (1.5 if ((x != None) and (y != None)) else 2.)
        cost = lambda edge, weight: cost_table(*edge, *weight)
        self.assertEqual(
            0,
            cost((None,None), ('a', 'a'))
        )
        self.assertEqual(
            1.5,
            cost((None,None), ('a', 'b'))
        )
        self.assertEqual(
            2,
            cost((None,None), ('a', None))
        )

        threshold = 4.
        source = 0

        node_cost_table = propagate(
            s_prod,
            cost,
            threshold,
            source,
        )

        sink = 15
        optimal_paths = list(backtrace(
            s_prod,
            cost,
            node_cost_table,
            threshold,
            source,
            sink,
        ))

        print("optimal paths")
        for (path_score, path_nodes) in optimal_paths:
            print(f"{path_score}\t{[s_prod.unravel(x) for x in path_nodes]}")
    
    def test2_substitution(self):
        weight_key = 'weight'

        dag3 = TestGraphTypes._construct_dag(
            edges = [(0,1),(1,2),(2,3),(3,4)],
            weights = ['a','b','c','d'])

        dag4 = TestGraphTypes._construct_dag(
            edges = [(0,1),(1,2),(2,3),(3,4)],
            weights = ['a','b','e','d'])

        s_prod = StrongProductDAG(dag3, dag4)

        cost_table = lambda u, v, x, y: 0. if (x == y) else (1. if ((x != None) and (y != None)) else 2.)
        cost = lambda edge, weight: cost_table(*edge, *weight)

        threshold = 4.
        source = s_prod.ravel(0, 0)

        node_cost_table = propagate(
            s_prod,
            cost,
            threshold,
            source,
        )

        print("cost table")
        for (k, v) in node_cost_table.items():
            print(f"{v}\t{s_prod.unravel(k)}")

        sink = s_prod.ravel(4, 4)
        optimal_paths = list(backtrace(
            s_prod,
            cost,
            node_cost_table,
            threshold,
            source,
            sink,
        ))

        print("optimal paths")
        for (path_score, path_nodes) in sorted(optimal_paths)[::-1]:
            path_len = len(path_nodes)
            path_str = ''.join(str(s_prod.weight_out(path_nodes[i], path_nodes[i + 1])) for i in range(path_len - 1))
            print(f"{path_score}\t{[s_prod.unravel(x) for x in path_nodes]}\n\t{path_str}\n")
    
    def test3_gaps(self):
        weight_key = 'weight'
        dag3 = TestGraphTypes._construct_dag(
            edges = [(0,1),(0,2),(1,3),(2,3)],
            weights = ['a','b','c','d'])

        dag5 = TestGraphTypes._construct_dag(
            edges = [(0,1),(1,3),(2,3)],
            weights = ['a','b','d'])

        s_prod = StrongProductDAG(dag3, dag5)

        # TODO: should be inexpensive to take (x, None) or (None, y) paths if one of the vertices is a source or sink.
        # that is, cost functions need to support local alignment.
        cost_table = lambda u, v, x, y: 0. if (x == y) else (1. if ((x != None) and (y != None)) else 2.)
        cost = lambda edge, weight: cost_table(*edge, *weight)

        threshold = 1e4
        source = s_prod.ravel(0, 0)

        node_cost_table = propagate(
            s_prod,
            cost,
            threshold,
            source,
        )

        print("cost table")
        for (k, v) in node_cost_table.items():
            print(f"{v}\t{s_prod.unravel(k)}")

        sink = s_prod.ravel(4, 4)
        optimal_paths = list(backtrace(
            s_prod,
            cost,
            node_cost_table,
            threshold,
            source,
            sink,
        ))

        print("optimal paths")
        for (path_score, path_nodes) in sorted(optimal_paths)[::-1]:
            path_len = len(path_nodes)
            path_str = ''.join(str(s_prod.weight_out(path_nodes[i], path_nodes[i + 1])) for i in range(path_len - 1))
            print(f"{path_score}\t{[s_prod.unravel(x) for x in path_nodes]}\n\t{path_str}\n")
    
    def test4_filter_parity(self):
        unit_cost_graph = TestGraphTypes._construct_dag(
            edges = [(0,1),(0,2),(1,2),(1,3),(2,4),(3,4),(3,5),(4,6),(5,6)],
            weights = [0.] * 9)
        source = 0
        sink = 6

        weight_id = lambda _, w: w
        node_cost_table = propagate(
            topology = unit_cost_graph,
            cost = weight_id,
            threshold = 0.,
            source = source,
        )

        path_parity_filter = lambda p: (len(p) % 2) == 0
        paths = list(backtrace(
            topology = unit_cost_graph,
            cost = weight_id,
            node_cost = node_cost_table,
            threshold = 0.,
            source = source,
            sink = sink,
            path_filter = path_parity_filter
        ))

        for score, path in paths:
            self.assertEqual(score == len(path))
            self.assertEqual(score % 2, 0)

class TestAlign(unittest.TestCase):

    def test_align(self):
        dag1 = TestGraphTypes._construct_dag(
            edges = [(0,1),(1,2),(2,3),(3,4)],
            weights = ['x','a','b','c'])
        
        dag2 = TestGraphTypes._construct_dag(
            edges = [(0,1),(1,2),(0,3),(3,4),(4,5),(4,6),(6,7)],
            weights = ['a','b','a','b','c','x','c'])

        product_graph = StrongProductDAG(
            first_graph = dag1,
            second_graph = dag2)

        aln_de = align(
            product_graph = product_graph,
            cost_model = LocalCostModel(),
            threshold = 1.,
        )
        for aligned_path in aln_de:
            score = aligned_path.score
            path = aligned_path.alignment
            sequence1 = list(map(
                lambda x: '_' if x == None else x,
                ((dag1.weight_out(path[i][0], path[i + 1][0])) for i in range(len(path) - 1))))
            sequence2 = list(map(
                lambda x: '_' if x == None else x,
                ((dag2.weight_out(path[i][1], path[i + 1][1])) for i in range(len(path) - 1))))
            dual = [w1 if w1 == w2 else f"{w1}/{w2}" for (w1, w2) in zip(sequence1, sequence2)] 
            print(f"{score}\t{path}\n\t{dual}\n")

    def test2_filtered_align(self):
        aaaa_graph = TestGraphTypes._construct_dag(
            edges = [(0,1),(1,2),(2,3),(3,4)],
            weights = list('aaaa'))
        
        bbaa_graph = TestGraphTypes._construct_dag(
            edges = [(0,1),(1,2),(2,3),(3,4)],
            weights = list('bbaa'))
        
        product_graph = StrongProductDAG(
            first_graph = aaaa_graph,
            second_graph = bbaa_graph)
        
        only_a_filter = ProductPathWeightFilter(
            weight_sequence_filter = lambda weights: all(w == 'a' for w in weights),
            graph = product_graph)

        filtered_aln = align(
            product_graph = product_graph,
            cost_model = LocalCostModel(),
            threshold = 10.,
            path_filter = only_a_filter)
        
        for aligned_path in filtered_aln:
            score = aligned_path.score
            path = aligned_path.alignment
            sequence1 = list(map(
                lambda x: '_' if x == None else x,
                ((aaaa_graph.weight_out(path[i][0], path[i + 1][0])) for i in range(len(path) - 1))))
            sequence2 = list(map(
                lambda x: '_' if x == None else x,
                ((bbaa_graph.weight_out(path[i][1], path[i + 1][1])) for i in range(len(path) - 1))))
            dual = [w1 if w1 == w2 else f"{w1}/{w2}" for (w1, w2) in zip(sequence1, sequence2)] 
            print(f"{score}\t{path}\n\t{dual}\n")

class TestFragment(unittest.TestCase):

    def _get_dag_1(self):
        g_edges = [(0, 2), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (5, 7)]
        g_weights = ['a','b','c','d','e','f','g']
        return TestGraphTypes._construct_dag(g_edges, g_weights)

    def _get_dag_2(self):
        g_edges = [(0, 2), (1, 2), (2, 3), (4, 5), (5, 6), (5, 7)]
        g_weights = ['a','b','c','e','f','g']
        return TestGraphTypes._construct_dag(g_edges, g_weights)

    def _get_dag_3(self):
        g_edges = [(0, 2), (1, 2), (2, 3), (3, 4), (5, 6), (5, 7)]
        g_weights = ['a','b','c','d','f','g']
        return TestGraphTypes._construct_dag(g_edges, g_weights)

    def _get_dag_4(self):
        g_edges = [(0, 2), (1, 2), (3, 4), (4, 5), (5, 6), (5, 7)]
        g_weights = ['a','b','d','e','f','g']
        return TestGraphTypes._construct_dag(g_edges, g_weights)
    
    def _get_dag_5(self):
        g_edges = [(0, 2), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7), (7, 8), (7,9)]
        g_weights = ['a','b','c','c','d','e','f','g']
        return TestGraphTypes._construct_dag(g_edges, g_weights)
    
    def _get_non_ladder_ex(self):
        dag1 = TestGraphTypes._construct_dag(
            edges = pairwise(range(6)),
            weights = map(chr, range(ord('a'),ord('a') + 5)))
        dag2 = TestGraphTypes._construct_dag(
            edges = [(0,1),(2,3),(4,5)],
            weights = ['a','c','e'])
        return dag1, dag2

    def _get_overlap_ex(self):
        dag1 = TestGraphTypes._construct_dag(
            edges = pairwise(range(5)),
            weights = map(chr, range(ord('a'),ord('a') + 4)))
        dag2 = TestGraphTypes._construct_dag(
            edges = [
                (0,1),(1,2),(2,3),
                (4,5),(5,6),(6,7)],
            weights = [
                'a','b','c',
                'b','c','d'])
        return dag1, dag2

    def _get_overlap_ex1(self):
        dag1 = TestGraphTypes._construct_dag(
            edges = pairwise(range(5)),
            weights = map(chr, range(ord('a'),ord('a') + 4)))
        dag2 = TestGraphTypes._construct_dag(
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