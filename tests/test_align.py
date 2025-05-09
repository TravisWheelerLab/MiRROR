from mirror.graphs.align import *
from mirror.graphs.align_types import LocalCostModel

import unittest

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
        for score, path in aln_de:
            sequence1 = list(map(
                lambda x: '_' if x == None else x,
                ((dag1.weight_out(path[i][0], path[i + 1][0])) for i in range(len(path) - 1))))
            sequence2 = list(map(
                lambda x: '_' if x == None else x,
                ((dag2.weight_out(path[i][1], path[i + 1][1])) for i in range(len(path) - 1))))
            dual = [w1 if w1 == w2 else f"{w1}/{w2}" for (w1, w2) in zip(sequence1, sequence2)] 
            #print(f"{score}\t{path}\n\t{dual}\n")