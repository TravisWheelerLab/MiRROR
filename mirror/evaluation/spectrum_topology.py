import itertools as it
from itertools import pairwise

from ..fragments.types import AxesResult, UniqueFragmentIndex
from ..graphs.types import SpectrumGraph, PivotGraph, SymmetricGraph

import numpy as np
import networkx as nx

def construct_spectrum_graphs(
    index: UniqueFragmentIndex,
    axes: AxesResult,
    tolerance: float,
) -> tuple[
    list[SpectrumGraph],
    list[SpectrumGraph],
    list[PivotGraph],
    list[SymmetricGraph],
]:
    n_pairs = len(index.pairs)
    pairs_id = np.arange(n_pairs)
    n_lbound = len(index.lower_boundaries)
    lbound_id = n_pairs + np.arange(n_lbound)
    n_rbounds = [len(x) for x in index.upper_boundaries]
    offset_rbounds = np.cumsum([0,] + n_rbounds)
    rbound_ids = [n_pairs + n_lbound + np.arange(i,j) for i,j in pairwise(offset_rbounds)]
    k = len(axes)
    lower_graphs = [None for _ in range(k)]
    upper_graphs = [None for _ in range(k)]
    pivot_graphs = [None for _ in range(k)]
    symmetric_graphs = [None for _ in range(k)]
    for i in range(k):
        axis = axes.get_axis(i)
        pairs_mass = index.fragment_masses[index.pairs]
        lower_pairs_mask = pairs_mass[:,1] < axis
        expected_mass = 2 * (axis + tolerance)
        upper_pairs_mask = np.logical_and(
            pairs_mass[:,0] > axis,
            pairs_mass[:,1] < expected_mass,
        )
        pivot_pairs_mask = np.logical_not(lower_pairs_mask + upper_pairs_mask)
        pivot_pairs = index.pairs[pivot_pairs_mask]
        lower_sinks = pivot_pairs[:,0]
        lower_sink_mass = index.fragment_masses[lower_sinks]
        lower_sinks = lower_sinks[lower_sink_mass < axis]
        upper_sinks = pivot_pairs[:,1]
        upper_sink_mass = index.fragment_masses[upper_sinks]
        upper_sinks = upper_sinks[upper_sink_mass > axis]
        lower_graphs[i] = SpectrumGraph.from_index(
            index.pairs[lower_pairs_mask],
            pairs_id[lower_pairs_mask],
            index.lower_boundaries,
            lbound_id,
            lower_sinks,
            np.full(len(lower_sinks),-1),
        )
        upper_graphs[i] = SpectrumGraph.from_index(
            index.pairs[upper_pairs_mask],
            pairs_id[upper_pairs_mask],
            index.upper_boundaries[i],
            rbound_ids[i],
            upper_sinks,
            np.full(len(upper_sinks),-1),
        )
        pivot_graphs[i] = PivotGraph.from_index(
            pivot_pairs,
            pairs_id[pivot_pairs_mask],
        )
        symmetric_graphs[i] = SymmetricGraph.from_index(np.vstack([
            np.array([
                [lower_graphs[i].boundary_node, upper_graphs[i].boundary_node],
                [lower_graphs[i].axial_node, upper_graphs[i].axial_node],
            ]),
            index.symmetries[i],
        ]))
    return (
        lower_graphs,
        upper_graphs,
        pivot_graphs,
        symmetric_graphs,
    )
