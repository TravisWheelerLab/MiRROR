from mirror.graphs.graph_types import *
from mirror.graphs.minimal_nodes import *

import unittest

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

        # print("cost table")
        # for (k, v) in node_cost_table.items():
        #     print(f"{v}\t{s_prod.unravel(k)}")