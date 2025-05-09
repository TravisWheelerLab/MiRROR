from mirror.graphs.graph_types import *
from mirror.graphs.consensus_types import *

import unittest

class TestGraphTypes(unittest.TestCase):
    
    def test_bipartite(self):
        b_edges = [(0,1),(0,2),(3,2),(3,4)]
        b = BipartiteGraph(
            edges = b_edges)
        
        try:
            b_edges = [(0,1),(0,2),(1,2)]
            b = BipartiteGraph(
                edges = b_edges)
        except ValueError as e:
            self.assertEqual(str(e), "not a bipartite graph!")

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