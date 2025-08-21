import unittest
from random import shuffle
import itertools as it

from mirror.graphs.graph_types import *

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

from mirror.graphs.minimal_nodes import *

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

from mirror.graphs.minimal_paths import *

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

from mirror.graphs.align_types import *
from mirror.graphs.align import *

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
            score = aligned_path.score()
            path = aligned_path.alignment_nodes
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
            score = aligned_path.score()
            path = aligned_path.alignment_nodes
            sequence1 = list(map(
                lambda x: '_' if x == None else x,
                ((aaaa_graph.weight_out(path[i][0], path[i + 1][0])) for i in range(len(path) - 1))))
            sequence2 = list(map(
                lambda x: '_' if x == None else x,
                ((bbaa_graph.weight_out(path[i][1], path[i + 1][1])) for i in range(len(path) - 1))))
            dual = [w1 if w1 == w2 else f"{w1}/{w2}" for (w1, w2) in zip(sequence1, sequence2)] 
            print(f"{score}\t{path}\n\t{dual}\n")

from mirror.graphs.ensemble_types import *
from mirror.graphs.ensemble import *
from mirror.graphs.ensemble import _solve_alignment_pair_chains

class TestEnsemble(unittest.TestCase):

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
    
    def _all_skips(self, alignment: list[tuple[int, int]]):
        alignment_edges = pairwise(alignment)
        for ((x1, x2), (y1, y2)) in alignment_edges:
            if x1 != y1 and x2 != y2:
                return False
        return True
    
    def _align(self, first: DAG, second: DAG):
        aln = align(
            product_graph = StrongProductDAG(first, second),
            cost_model = LocalCostModel(),
            threshold = 0)
        return [aligned_path for aligned_path in aln if (not self._all_skips(aligned_path.alignment_nodes))]
    
    def _test_alignment_itx(self, first_dag, second_dag, tag):
        # alignment
        aln = self._align(first_dag, second_dag)
        # alignment consensus
        frag_itx = AlignmentIntersectionGraph(aln)
        alignment_consensii = _solve_alignment_pair_chains(frag_itx, LocalCostModel(), 1000)
        print(f"\nalignment {tag}:\n{aln}")
        print(f"alignment itx:\n{frag_itx}\n{frag_itx.edges(data = True)}\n{frag_itx._alignment_index}")
        print("alignment consensus:")
        for consensus_idx, (score, alignment_sequence) in enumerate(alignment_consensii):
            print(f"consensus[{consensus_idx}]\nscore: {score}\nseq: {alignment_sequence}")
            for alignment_idx in alignment_sequence:
                print(frag_itx.get_alignment(alignment_idx))
            print('-'*20)
        alignment_chains = assemble_alignments(aln, LocalCostModel, 1000)
        print(alignment_chains)
    
    def test_non_ladder(self):
        self._test_alignment_itx(*self._get_non_ladder_ex(), "non-ladder")
    
    def test_overlap(self):
        self._test_alignment_itx(*self._get_overlap_ex(), "overlap")

    def test_overlap1(self):
        self._test_alignment_itx(*self._get_overlap_ex1(), "overlap")

    def test_alignment_itx(self):
        dag1 = self._get_dag_1()
        dag2 = self._get_dag_2()
        self._test_alignment_itx(dag1, dag2, "12")

        dag3 = self._get_dag_3()
        self._test_alignment_itx(dag3, dag2, "32")
        
        dag4 = self._get_dag_4()
        self._test_alignment_itx(dag3, dag4, "34")

        dag1 = self._get_dag_1()
        dag5 = self._get_dag_5()
        self._test_alignment_itx(dag1, dag5, "15")

from mirror.graphs.concatenation_types import *
from mirror.graphs.concatenation import *

class TestConcatenation(unittest.TestCase):

    def _test_concat(self, first_dag, first_node_weights, second_dag, second_node_weights, tag):
        product_graph = StrongProductDAG(first_dag, second_dag)
        local_cost_model = LocalCostModel()
        # local alignments
        local_aln = align(
            product_graph = product_graph,
            cost_model = local_cost_model,
            threshold = 0.)
        # assembled alignments
        ensemble_aln = assemble_alignments(
            alignments = local_aln,
            cost_model = local_cost_model,
            threshold = 1000.)
        # concatenated ensembles
        concat_aln = concatenate_ensembles(
            alignments = ensemble_aln,
            first_node_weights = first_node_weights,
            second_node_weights = second_node_weights,
            cost_model = local_cost_model,
            threshold = 1000.)
        print(f"test-concat[ {tag} ]\n- local alignments:\n{local_aln}\n- ensemble alignments:\n{ensemble_aln}\n- concatenation alignments:\n{concat_aln}")
    
    def test_occluded_edge(self):
        dag1 = TestGraphTypes._construct_dag(
            edges = [(0,1),(1,2),(3,4),(4,5)],
            weights = ['a','b','d','e'])
        node_weights1 = [1., 2., 3., 4., 5., 6.]
        dag2 = TestGraphTypes._construct_dag(
            edges = [(0,1),(1,2),(3,4),(4,5)],
            weights = ['a','b','d','e'])
        node_weights2 = [-1., -2., -3., -4., -5., -6.]
        tag = "Occluded Edge"
        self._test_concat(dag1, node_weights1, dag2, node_weights2, tag)
    
    def test_occluded_node(self):
        dag1 = TestGraphTypes._construct_dag(
            edges = [(0,1),(2,3)],
            weights = ['a','d'])
        node_weights1 = [1., 2., 4., 5.]
        dag2 = TestGraphTypes._construct_dag(
            edges = [(0,1),(2,3)],
            weights = ['a','d'])
        node_weights2 = [-1., -2., -4., -5.]
        tag = "Occluded Node"
        self._test_concat(dag1, node_weights1, dag2, node_weights2, tag)
    
    def test_complex(self):
        pass

from mirror.graphs.spectrum_graphs import SpectrumGraph, construct_spectrum_graphs
from mirror.fragments import FragmentState, ResidueState, PairedFragments, VirtualPivot, BoundaryFragment, ReflectedBoundaryFragment

class TestSpectrumGraphs(unittest.TestCase):

    def _dummy_params(self):
        """a palindromic single series example with a virtual pivot."""
        fragments = [
            FragmentState(0,1.,0,0.,'b',1),
            FragmentState(1,2.,0,0.,'',1),
            FragmentState(2,4.,0,0.,'',1),
            FragmentState(3,5.,0,0.,'',1),
            FragmentState(4,7.,0,0.,'',1),
            FragmentState(5,8.,0,0.,'b',1)]
        residues = [
            ResidueState(1.,0.,0,1.,'a',0,0.,''),
            ResidueState(1.,0.,0,1.,'a',0,0.,''),
            ResidueState(2.,0.,0,2.,'b',0,0.,''),
            ResidueState(1.,0.,0,1.,'a',0,0.,''),
            ResidueState(2.,0.,0,2.,'b',0,0.,''),
            ResidueState(1.,0.,0,1.,'a',0,0.,''),
            ResidueState(1.,0.,0,1.,'a',0,0.,'')]
        pivot_point = 4.5
        # pairs
        pairs = [PairedFragments.from_solution((lf,rf,r))
            for ((lf,rf),r) in zip(pairwise(fragments),residues[1:-1])]
        # virtual pivot
        pivot = VirtualPivot(pivot_point,1.)
        # left boundary
        left_boundaries = [BoundaryFragment.from_solution((None,fragments[0],residues[0]))]
        # right boundary
        right_boundaries = [ReflectedBoundaryFragment.from_solution((None,fragments[-1],residues[-1]),pivot_point)]

        return [(pairs,left_boundaries,right_boundaries,pivot)]

    def _simple_params(self):
        """test cases generated without loss or charge states from the VALIDATION_PEPTIDES set."""

    def _complex_params(self):
        """test cases generated with loss and charge states from the VALIDATION_PEPTIDES set."""

    def _test_construct(self, pairs, left_boundaries, right_boundaries, pivot):
        incr_graph, decr_graph, cut_pairs = construct_spectrum_graphs(pairs, left_boundaries, right_boundaries, pivot)
        print("incr",incr_graph)
        print("decr",decr_graph)
        print("cut",cut_pairs)
        
    def test1_construct(self):
        for params in self._dummy_params():
            self._test_construct(*params)

#    def _test_align(self, graphs, model, threshold):
#        pfx_aln, sfx_aln = align_spectrum_graphs(graphs, model, threshold)
#        print("prefix alignments", pfx_aln)
#        print("suffix alignments", sfx_aln)
#
#    def test2_align(self):
#        for params in self._dummy_params():
#            self._test_align(*self._test_construct(*params))
