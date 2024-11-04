import unittest
from time import time

import networkx as nx

from mirror.util import collapse_second_order_list
from mirror.paired_paths import *
from mirror.sequence import get_sinks, get_sources, get_weights

WEIGHT_KEY = "weight"

EASY_EXAMPLE = {
    "graph_a" : [((9, 11), 113.08),
                ((11, 13), 57.021),
                ((11, 14), 114.04),
                ((10, 12), 101.04),
                ((12, 15), 113.08),
                ((13, 14), 57.021),
                ((13, 15), 97.041),
                ((13, 16), 128.05),
                ((14, 16), 71.037),
                ((14, 17), 97.041),
                ((15, 17), 57.021),
                ((16, 18), 113.08)],
    
    "graph_b" : [((2, 0), 57.021),
                ((3, 0), 97.041),
                ((3, 1), 71.037),
                ((4, 1), 128.05),
                ((4, 2), 97.041),
                ((4, 3), 57.021),
                ((5, 2), 113.08),
                ((6, 3), 114.04),
                ((6, 4), 57.021),
                ((7, 5), 101.04),
                ((8, 6), 113.08)]
}

EXAMPLE_GRAPHS = [
    EASY_EXAMPLE
]

def _reconstruct_graph(weighted_edgelist, weight_key = WEIGHT_KEY):
    return nx.from_edgelist([(i,j,{WEIGHT_KEY : w}) for ((i,j),w) in weighted_edgelist], create_using=nx.DiGraph)

class Test040_PairedPaths(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(f"\n{cls.__name__}")
    
    def _naiive_paired_paths(self, graph_a: nx.DiGraph, graph_b: nx.DiGraph, weight_key = WEIGHT_KEY):
        sinks_a = get_sinks(graph_a)
        sources_a = get_sources(graph_a)
        paths_a = collapse_second_order_list([nx.all_simple_paths(graph_a, src, sinks_a) for src in sources_a])
        weights_a = [get_weights(graph_a, p, weight_key) for p in paths_a]

        sinks_b = get_sinks(graph_b)
        sources_b = get_sources(graph_b)
        paths_b = collapse_second_order_list([nx.all_simple_paths(graph_b, src, sinks_b) for src in sources_b])
        weights_b = [get_weights(graph_b, p, weight_key) for p in paths_b]

        return [list(zip(p_a, p_b)) 
            for (p_a, w_a) in zip(paths_a, weights_a) 
            for (p_b, w_b) in zip(paths_b, weights_b)
            if w_a == w_b]

    def _dfs_paired_paths(self, graph_a: nx.DiGraph, graph_b: nx.DiGraph, weight_key = WEIGHT_KEY):
        sinks_a = get_sinks(graph_a)
        sources_a = get_sources(graph_a)

        sinks_b = get_sinks(graph_b)
        sources_b = get_sources(graph_b)
    
        weight_transform = lambda x: x
        return collapse_second_order_list([
            weighted_paired_simple_paths(graph_a, src_a, sinks_a, graph_b, src_b, sinks_b, weight_key, weight_transform)
            for src_a in sources_a
            for src_b in sources_b])

    def _test(self, graph_a: nx.DiGraph, graph_b: nx.DiGraph):
        t_init_n = time()
        naiive_paths = self._naiive_paired_paths(graph_a, graph_b)
        t_elap_n = time() - t_init_n

        t_init_d = time()
        dfs_paths = self._dfs_paired_paths(graph_a, graph_b)
        t_elap_d = time() - t_init_d

        return naiive_paths, t_elap_n, dfs_paths, t_elap_d

    def test_examples(self, examples = EXAMPLE_GRAPHS):
        for ex in examples:
            graph_a = _reconstruct_graph(ex["graph_a"])
            graph_b = _reconstruct_graph(ex["graph_b"])
            naiive_result, naiive_time, dfs_result, dfs_time = self._test(graph_a, graph_b)
            # all naiive_results are identified by dfs_result
            self.assertTrue(all((p in dfs_result) for p in naiive_result))