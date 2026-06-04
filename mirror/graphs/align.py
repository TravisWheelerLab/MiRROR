import dataclasses
from heapq import heappush, heappop

from .types import Graph, SpectrumGraph, AlignedPaths, AbstractNodeCostModel, AbstractEdgeCostModel, AbstractPathCostModel, AbstractNodeLookup, AugmentedLetter
from ..util import ravel, unravel
from ..sequences.suffix_array import BisectResult

def direct_product_adj(
    first_graph: Graph,
    second_graph: Graph,
    first_pos: int,
    second_pos: int,
):
    print(f"""    direct_product_adj
        first_pos {first_pos}
        second_pos {second_pos}
        first_adj {first_graph.adjacent(first_pos)}
        second_adj {second_graph.adjacent(second_pos)}""")
    return (
        [(i,j) for i in first_graph.adjacent(first_pos) for j in second_graph.adjacent(second_pos)],
        [(i,j) for i in first_graph.edge_index(first_pos) for j in second_graph.edge_index(second_pos)],
    )

def first_box_product_adj(
    first_graph: Graph,
    second_graph: Graph,
    first_pos: int,
    second_pos: int,
):
    return (
        [(i, second_pos) for i in first_graph.adjacent(first_pos)],
        [(i, -1) for i in first_graph.edge_index(first_pos)],
    )

def second_box_product_adj(
    first_graph: Graph,
    second_graph: Graph,
    first_pos: int,
    second_pos: int,
):
    return (
        [(first_pos, j) for j in second_graph.adjacent(second_pos)],
        [(-1, j) for j in second_graph.edge_index(second_pos)],
    )

def align_spectrum_graphs(
    lower: SpectrumGraph,
    upper: SpectrumGraph,
    lower_source: int,
    upper_source: int,
    node_cost_model: AbstractNodeCostModel,
    edge_cost_model: AbstractEdgeCostModel,
    path_cost_model: AbstractPathCostModel,
    lower_node_lookup: AbstractNodeLookup,
    upper_node_lookup: AbstractNodeLookup,
    augmented_alphabet: list[AugmentedLetter],
    threshold: float,
) -> AlignedPaths:
    return AlignedPaths.from_align(
        *_align(
            lower.graph,
            upper.graph,
            lower_source,
            upper_source,
            lower.axial_node,
            upper.axial_node,
            lower.gap_state_node,
            upper.gap_state_node,
            node_cost_model,
            edge_cost_model,
            path_cost_model,
            lower_node_lookup,
            upper_node_lookup,
            augmented_alphabet,
            threshold,
        )
    )

@dataclasses.dataclass(slots=True)
class _HeapItem:
    cost: float
    first_pos: int
    second_pos: int
    anno: list[tuple[AugmentedLetter,float]]
    path: list[int]
    state: tuple[BisectResult,float]

    def __lt__(self, other):
        self.cost < other.cost

    def __iter__(self):
        return iter((self.cost,self.first_pos,self.second_pos,self.anno,self.path,self.state))

def _align(
	first_graph: Graph,
	second_graph: Graph,
	first_source: int,
	second_source: int,
	first_sink: int,
	second_sink: int,
	first_gap_state: int,
	second_gap_state: int,
    node_cost_model: AbstractNodeCostModel,
    edge_cost_model: AbstractEdgeCostModel,
    path_cost_model: AbstractPathCostModel,
    first_node_lookup: AbstractNodeLookup,
    second_node_lookup: AbstractNodeLookup,
	augmented_alphabet: list[AugmentedLetter],
	threshold: float,
) -> tuple[
    list[float],
    list[float],
    list[BisectResult],
    list[list[int]],
    list[list[AugmentedLetter]],
]:
    print("align")
    pq = [
        _HeapItem(
            0.,             # cost
            first_source,   # first pos
            second_source,  # second pos
            [(0,0,0.)],    # anno
            [],             # path
            path_cost_model.initial_state(),
        ),
    ]
    first_graph_order = first_graph.order()
    second_graph_order = second_graph.order()
    tgt_pos = ravel(first_sink, second_sink, second_graph_order)
    costs = [0.,]
    masses = [0,]
    path_states = [None,]
    paths = [[-1,],]
    annotations = [[(-1,-1,0.),],]
    while len(pq) > 0:
        x = heappop(pq)
        # print(x)
        cost, first_pos, second_pos, anno, path, path_state = x
        peptide_mass = anno[-1][-1]
        if len(path) > 0:
            print(unravel(path[-1],second_graph_order), "->", (first_pos,second_pos), anno, cost)
        if cost > threshold:
            print("\tpruned.")
            # prune this path.
            continue

        curr_pos = ravel(first_pos, second_pos, second_graph_order)
        path = path + [curr_pos,]
        if curr_pos == tgt_pos:
            print("\tcomplete!")
            costs.append(cost)
            print("anno",anno)
            masses.append(0.)
            path_states.append(path_state)
            paths.append(path)
            annotations.append(anno)
            continue
        
        direct_tgt, direct_idx = direct_product_adj(
        	first_graph,second_graph,first_pos,second_pos)
        first_box_tgt, first_box_idx = first_box_product_adj(
        	first_graph,second_graph,first_pos,second_pos)
        second_box_tgt, second_box_idx = second_box_product_adj(
        	first_graph,second_graph,first_pos,second_pos)

        nd = len(direct_tgt)
        nb1 = len(first_box_tgt)
        nb2 = len(second_box_tgt)

        if nd == nb1 == nb2 == 0:
            print("\tbranch and bound")
            # branch and bound.
            # the gap state is represented by a special node in each graph that has no neighbors, so it kicks it back into this condition with every step, until either the target mass is reached, or a node is discovered in one or both graphs via lookup.
            for amino_idx, mod_idx, delta_mass in augmented_alphabet:
                putative_mass = peptide_mass + delta_mass
                new_first_pos, aug_first_match = first_node_lookup(putative_mass)
                new_second_pos, aug_second_match = second_node_lookup(putative_mass)
                if new_first_pos is None:
                    new_first_pos = first_gap_state
                if new_second_pos is None:
                    new_second_pos = second_gap_state
                new_node_cost = node_cost_model(new_first_pos, new_second_pos)
                # new_step_anno, new_step_cost = edge_cost_model(first_anno_res,second_anno_res)
                new_step_anno = (amino_idx, mod_idx, delta_mass)
                new_step_cost = 0. # TODO based on aug matches
                new_anno = anno + [new_step_anno,]
                new_path_cost, new_path_state = path_cost_model(path_state, new_step_anno)
                new_cost = cost + new_node_cost + new_step_cost + new_path_cost
                heappush(pq,
                    _HeapItem(          
                        new_cost,       # cost
                        new_first_pos,  # first_pos
                        new_second_pos, # second_pos
                        new_anno,       # anno
                        path,           # path
                        new_path_state, # path state
                    )
                )
        elif nb2 == 0:
            print("\tfirst box")
            # follow first box path and check for reflected nodes in second graph.
            # check for reflected node. this will always fail on the first position of the box path, since the lack of symmetry is why it was entered in the first place.
            # if yes, put that on the queue.
            # if not, just follow the box edges and put those on the queue.
            for i in range(nb1):
                first_edge_idx, second_edge_idx = first_box_idx[i]
                new_edge_cost, new_edge_anno = edge_cost_model(first_edge_idx, second_edge_idx)
                new_anno = anno + [new_edge_anno,]
                # box edge cost, annotation

                putative_mass = peptide_mass + new_edge_anno[-1]
                new_first_pos, _ = first_box_tgt[i]
                new_second_pos, aug_second_match = second_node_lookup(putative_mass)
                if new_second_pos is None:
                    new_second_pos = second_pos
                    new_pos_cost = int(not(aug_second_match))
                else:
                    new_pos_cost = 0.
                # retrieve target position; check if it hits on the other graph, or in the augmented spectrum.

                new_path_cost, new_path_state = path_cost_model(path_state, new_edge_anno)
                new_cost = cost + new_pos_cost + new_edge_cost + new_path_cost
                heappush(pq,
                    _HeapItem(          
                        new_cost,       # cost
                        new_first_pos,  # first_pos
                        new_second_pos, # second_pos
                        new_anno,       # anno
                        path,           # path
                        new_path_state, # path state
                    )
                )
        elif nb1 == 0:
            print("\tsecond box")
            # follow second box path and check for reflected nodes in first graph. same logic as previous case.
            for i in range(nb2):
                first_edge_idx, second_edge_idx = second_box_idx[i]
                new_edge_cost, new_edge_anno = edge_cost_model(first_edge_idx, second_edge_idx)
                new_anno = anno + [new_edge_anno,]
                # box edge cost, annotation

                putative_mass = peptide_mass + new_edge_anno[-1]
                _, new_second_pos = second_box_tgt[i]
                new_first_pos, aug_first_match = first_node_lookup(putative_mass)
                if new_first_pos is None:
                    new_first_pos = first_pos
                    new_pos_cost = int(not(aug_first_match))
                else:
                    new_pos_cost = 0.
                # retrieve target position; check if it hits on the other graph, or in the augmented spectrum.

                new_path_cost, new_path_state = path_cost_model(path_state, new_edge_anno)
                new_cost = cost + new_pos_cost + new_edge_cost + new_path_cost
                heappush(pq,
                    _HeapItem(          
                        new_cost,       # cost
                        new_first_pos,  # first_pos
                        new_second_pos, # second_pos
                        new_anno,       # anno
                        path,           # path
                        new_path_state, # path state
                    )
                )
        else:
            print("\ttraverse")
            # traverse the direct edges of the product graph.
            for i in range(nd):
                new_first_pos, new_second_pos = direct_tgt[i]
                first_edge_idx, second_edge_idx = direct_idx[i]
                new_pos_cost = node_cost_model(new_first_pos, new_second_pos)
                new_edge_cost, new_edge_anno = edge_cost_model(first_edge_idx, second_edge_idx)
                new_anno = anno + [new_edge_anno,]
                new_path_cost, new_path_state = path_cost_model(path_state, new_edge_anno)
                new_cost = cost + new_pos_cost + new_edge_cost + new_path_cost
                heappush(pq,
                    _HeapItem(          
                        new_cost,       # cost
                        new_first_pos,  # first_pos
                        new_second_pos, # second_pos
                        new_anno,       # anno
                        path,           # path
                        new_path_state, # path state
                    )
                )
    return (
        costs,
        masses,
        path_states,
        paths,
        annotations,
    )
