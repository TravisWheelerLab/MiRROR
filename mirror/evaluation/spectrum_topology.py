import itertools as it
from itertools import pairwise

from ..fragments.types import AxesResult, UniqueFragmentIndex, AnnotationIndex
from ..graphs.types import SpectrumGraph, PivotGraph, SymmetricGraph
from .costmodels import SymmetricNodeCostModel, MassConstrainedPathCostModel

import numpy as np
import networkx as nx

def construct_spectrum_topology(
    fragment_index: UniqueFragmentIndex,
    annotation_index: AnnotationIndex,
    axes: AxesResult,
    tolerance: float,
) -> tuple[
    list[SpectrumGraph],
    list[SpectrumGraph],
    list[PivotGraph],
    list[SymmetricGraph],
    list[SymmetricNodeCostModel],
    list[MassConstrainedPathCostModel],
]:
    pairs_id = annotation_index.get_pairs_id()
    lbound_id = annotation_index.get_lower_boundaries_id()
    k = len(axes)
    rbounds_ids = [annotation_index.get_upper_boundaries_id(i) for i in range(k)]
    lower_graphs = [None for _ in range(k)]
    upper_graphs = [None for _ in range(k)]
    pivot_graphs = [None for _ in range(k)]
    symmetric_graphs = [None for _ in range(k)]
    node_cost_models = [None for _ in range(k)]
    path_cost_models = [None for _ in range(k)]
    for i in range(k):
        axis = axes.get_axis(i)
        pairs_mass = fragment_index.fragment_masses[fragment_index.pairs]
        lower_pairs_mask = pairs_mass[:,1] < axis
        expected_mass = 2 * (axis + tolerance)
        upper_pairs_mask = np.logical_and(
            pairs_mass[:,0] > axis,
            pairs_mass[:,1] < expected_mass,
        )
        pivot_pairs_mask = np.logical_not(lower_pairs_mask + upper_pairs_mask)
        pivot_pairs = fragment_index.pairs[pivot_pairs_mask]
        # partition pairs into lower, upper, and pivot by their relation to the axis.

        lower_sinks = pivot_pairs[:,0]
        lower_sink_mass = fragment_index.fragment_masses[lower_sinks]
        lower_sinks = lower_sinks[lower_sink_mass < axis]
        upper_sinks = pivot_pairs[:,1]
        upper_sink_mass = fragment_index.fragment_masses[upper_sinks]
        upper_sinks = upper_sinks[upper_sink_mass > axis]
        # construct sinks as the relative sources of pairs cut in half by the axis.

        node_cost_models[i] = SymmetricNodeCostModel.from_axis(
            fragment_index.fragment_masses,
            axis,
            tolerance,
        )
        path_cost_models[i] = MassConstrainedPathCostModel.from_axis(
            axis,
            tolerance,
        )
        # build costmodels

        lower_graphs[i] = SpectrumGraph.from_index(
            fragment_index.pairs[lower_pairs_mask],
            pairs_id[lower_pairs_mask],
            fragment_index.lower_boundaries,
            lbound_id,
            lower_sinks,
            np.full(len(lower_sinks),-1),
        )
        upper_graphs[i] = SpectrumGraph.from_index(
            fragment_index.pairs[upper_pairs_mask],
            pairs_id[upper_pairs_mask],
            fragment_index.upper_boundaries[i],
            rbounds_ids[i],
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
            fragment_index.symmetries[i],
        ]))
        # build graphs
    return (
        lower_graphs,
        upper_graphs,
        pivot_graphs,
        symmetric_graphs,
        node_cost_models,
        path_cost_models,
    )
