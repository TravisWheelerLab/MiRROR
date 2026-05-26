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
        expected_mass = 2 * (axis + tolerance)
        pairs_mass = index.fragment_masses[index.pairs]
        lower_pairs_mask = pairs_mass[:,1] < axis
        upper_pairs_mask = np.logical_and(
            pairs_mass[:,0] > axis,
            pairs_mass[:,1] < expected_mass,
        )
        pivot_pairs_mask = np.logical_not(lower_pairs_mask + upper_pairs_mask)
        pivot_pairs = index.pairs[pivot_pairs_mask]
        lower_graphs[i] = SpectrumGraph.from_index(
            index.pairs[lower_pairs_mask],
            pairs_id[lower_pairs_mask],
            pairs.lower_boundaries,
            lbound_id,
            pivot_pairs[:,0],
            np.full(len(pivot_pairs),-1),
        )
        upper_graphs[i] = SpectrumGraph.from_index(
            index.pairs[lower_pairs_mask],
            pairs_id[lower_pairs_mask],
            pairs.upper_boundaries[i],
            rbound_ids[i],
            pivot_pairs[:,1],
            np.full(len(pivot_pairs),-1),
        )
        pivot_graphs[i] = PivotGraph.from_index(
            pivot_pairs,
            pairs_id[pivot_pairs_mask],
        )
        symmetric_graphs[i] = SymmetricGraph.from_index(
            index.symmetries[i],
        )
    return (
        lower_graphs,
        upper_graphs,
        pivot_graphs,
        symmetric_graphs,
    )
