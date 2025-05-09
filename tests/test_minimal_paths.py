from mirror.graphs.graph_types import *
from mirror.graphs.minimal_nodes import *
from mirror.graphs.minimal_paths import *

import unittest

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

        # print("optimal paths")
        # for (path_score, path_nodes) in optimal_paths:
        #     print(f"{path_score}\t{[s_prod.unravel(x) for x in path_nodes]}")
    
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

        # print("cost table")
        # for (k, v) in node_cost_table.items():
        #     print(f"{v}\t{s_prod.unravel(k)}")

        sink = s_prod.ravel(4, 4)
        optimal_paths = list(backtrace(
            s_prod,
            cost,
            node_cost_table,
            threshold,
            source,
            sink,
        ))

        # print("optimal paths")
        # for (path_score, path_nodes) in sorted(optimal_paths)[::-1]:
        #     path_len = len(path_nodes)
        #     path_str = ''.join(str(s_prod.weight_in(path_nodes[i], path_nodes[i + 1])) for i in range(path_len - 1))
        #     print(f"{path_score}\t{[s_prod.unravel(x) for x in path_nodes]}\n\t{path_str}\n")
    
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

        # print("cost table")
        # for (k, v) in node_cost_table.items():
        #     print(f"{v}\t{s_prod.unravel(k)}")

        sink = s_prod.ravel(4, 4)
        optimal_paths = list(backtrace(
            s_prod,
            cost,
            node_cost_table,
            threshold,
            source,
            sink,
        ))

        # print("optimal paths")
        # for (path_score, path_nodes) in sorted(optimal_paths)[::-1]:
        #     path_len = len(path_nodes)
        #     path_str = ''.join(str(s_prod.weight_in(path_nodes[i], path_nodes[i + 1])) for i in range(path_len - 1))
        #     print(f"{path_score}\t{[s_prod.unravel(x) for x in path_nodes]}\n\t{path_str}\n")