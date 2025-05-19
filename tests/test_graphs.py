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

    def test_dag(self):
        g = DiGraph()
        g_edges = [(0,1),(0,2),(1,3),(2,3)]
        g_weights = ['a','b','c','d']
        g_weight_key = "weight"
        g.add_edges_from(
            (i,j, {g_weight_key: w})
            for ((i, j), w) in zip(g_edges, g_weights))
        dag = DAG(g, g_weight_key)

        try:
            g = DiGraph()
            g_edges = [(0,1),(0,2),(1,3),(3,0)]
            g_weights = ['a','b','c','d']
            g_weight_key = "weight"
            g.add_edges_from(
                (i,j, {g_weight_key: w})
                for ((i, j), w) in zip(g_edges, g_weights))
            dag = DAG(g, g_weight_key)
        except ValueError as e:
            self.assertEqual(str(e), "not a directed acyclic graph!")

    def test_products(self):
        # construct DAGs
        g = DiGraph()
        g_edges = [(0,1),(0,2),(1,3),(2,3)]
        g_weights = ['a','b','c','d']
        g_weight_key = "weight"
        g.add_edges_from(
            (i,j, {g_weight_key: w})
            for ((i, j), w) in zip(g_edges, g_weights))
        dag = DAG(g, g_weight_key)

        h = DiGraph()
        h_edges = [(0,1),(1,3),(2,3)]
        h_weights = ['a','b','d']
        h_weight_key = "weight2"
        h.add_edges_from(
            (i, j, {h_weight_key: w}) 
            for ((i, j), w) in zip(h_edges, h_weights))
        dag2 = DAG(h, h_weight_key)

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
        g = DiGraph()
        g_edges = [(0,1),(0,2),(1,3),(2,3)]
        g_weights = ['a','b','c','d']
        g_weight_key = "weight"
        g.add_edges_from(
            (i,j, {g_weight_key: w})
            for ((i, j), w) in zip(g_edges, g_weights))
        dag = DAG(g, g_weight_key)

        h = DiGraph()
        h_edges = [(0,1),(1,3),(2,3)]
        h_weights = ['a','c','d']
        h_weight_key = "weight2"
        h.add_edges_from(
            (i, j, {h_weight_key: w}) 
            for ((i, j), w) in zip(h_edges, h_weights))
        dag2 = DAG(h, h_weight_key)

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
        g = DiGraph()
        g_edges = [(0,1),(0,2),(1,3),(2,3)]
        g_weights = ['a','b','c','d']
        g_weight_key = "weight"
        g.add_edges_from(
            (i, j, {g_weight_key: w})
            for ((i, j), w) in zip(g_edges, g_weights))
        dag = DAG(g, g_weight_key)

        h = DiGraph()
        h_edges = [(0,1),(1,3),(2,3)]
        h_weights = ['a','c','d']
        h_weight_key = "weight2"
        h.add_edges_from(
            (i, j, {h_weight_key: w}) 
            for ((i, j), w) in zip(h_edges, h_weights))
        dag2 = DAG(h, h_weight_key)

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

        dag3_edges = [(0,1),(1,2),(2,3),(3,4)]
        dag3_weights = ['a','b','c','d']
        dag3_edge_data = [(i, j, {weight_key: w}) for ((i, j), w) in zip(dag3_edges, dag3_weights)]
        dag3 = DAG(
            graph = DiGraph(incoming_graph_data = dag3_edge_data),
            weight_key = weight_key,
        )

        dag4_edges = [(0,1),(1,2),(2,3),(3,4)]
        dag4_weights = ['a','b','e','d']
        dag4_edge_data = [(i, j, {weight_key: w}) for ((i, j), w) in zip(dag4_edges, dag4_weights)]
        dag4 = DAG(
            graph = DiGraph(incoming_graph_data = dag4_edge_data),
            weight_key = weight_key,
        )

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

        dag3_edges = [(0,1),(1,2),(2,3),(3,4)]
        dag3_weights = ['a','b','c','d']
        dag3_edge_data = [(i, j, {weight_key: w}) for ((i, j), w) in zip(dag3_edges, dag3_weights)]
        dag3 = DAG(
            graph = DiGraph(incoming_graph_data = dag3_edge_data),
            weight_key = weight_key,
        )

        dag5_edges = [(0,1),(1,2),(3,4)]
        dag5_weights = ['a','b','d']
        dag5_edge_data = [(i, j, {weight_key: w}) for ((i, j), w) in zip(dag5_edges, dag5_weights)]
        dag5 = DAG(
            graph = DiGraph(incoming_graph_data = dag5_edge_data),
            weight_key = weight_key,
        )

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

class TestAlign(unittest.TestCase):

    def test_align(self):
        d = DiGraph()
        d.add_edge(0,1,weight='x')
        d.add_edge(1,2,weight='a')
        d.add_edge(2,3,weight='b')
        d.add_edge(3,4,weight='c')
        dag1 = DAG(d, "weight")
        
        e = DiGraph()
        e.add_edge(0,1,weight='a')
        e.add_edge(1,2,weight='b')
        e.add_edge(0,3,weight='a')
        e.add_edge(3,4,weight='b')
        e.add_edge(4,5,weight='c')
        e.add_edge(4,6,weight='x')
        e.add_edge(6,7,weight='c')
        dag2 = DAG(e, "weight")

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
    