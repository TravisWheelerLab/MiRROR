import itertools as it

from .labeled_peaks import FragmentLabels

from ..fragments.types import PivotResult
from ..graphs.types import SpectrumGraph, PivotGraph

import numpy as np
import networkx as nx

def _construct_spectrum_graphs(
    mz: np.ndarray,                     
    # [float; n]
    pairs: np.ndarray,                  
    # [(int,int); m]
    left_boundaries: np.ndarray,        
    # [int; l]
    right_boundary_arrs: list[np.ndarray], 
    # [[int; _]; p]
    pivots: list[np.ndarray],           
    # [(int,int,int,int); p]
    tolerance: float,
    weight_key: str,
) -> tuple[
        list[SpectrumGraph],
        list[SpectrumGraph],
        list[PivotGraph],
]:
    pair_mz = mz[pairs]
    lb_mz = mz[left_boundaries]
    right_boundary_mz = [mz[right_boundaries] for right_boundaries in right_boundary_arrs]
    # prep mz values for pairs and left boundaries.

    pairs = np.c_[pairs, np.arange(pairs.shape[0])]
    left_boundaries = np.c_[left_boundaries, np.arange(left_boundaries.size)]
    right_boundary_arrs = [np.c_[right_boundaries, np.arange(right_boundaries.size)] for right_boundaries in right_boundary_arrs]
    # add indices to pairs

    k = len(pivots)
    left_adj = [None for _ in range(k)]
    right_adj = [None for _ in range(k)]
    pivot_adj = [None for _ in range(k)]
    for i in range(k):
        pivot = pivots[i]
        rb_mz = right_boundary_mz[i]
        right_boundaries = right_boundary_arrs[i]
        # unpack pivot-specific data
        
        left_pair_mask = pair_mz[:,1] < pivot
        expected_max_mass = 2 * (pivot + tolerance)
        right_pair_mask = np.logical_and(
            pair_mz[:,0] > pivot,
            pair_mz[:,1] < expected_max_mass,
        )
        # bisect pairs into left and right around the pivot.

        pivot_pairs = pairs[np.logical_not(left_pair_mask + right_pair_mask),:]
        left_sinks = pivot_pairs[:,0]
        right_sinks = pivot_pairs[:,1]
        pivot_adj[i] = PivotGraph.from_edges(
            edges = pivot_pairs,
        )
        # construct the pivot graph from undirected pairs intersected by pivot.
        
        left_edges = pairs[left_pair_mask]
        left_boundaries_mask = lb_mz < pivot
        left_sources = left_boundaries[left_boundaries_mask]
        left_adj[i] = SpectrumGraph.from_edges_and_boundaries(
            edges = left_edges,
            boundaries = left_sources,
            pivots = left_sinks,
            weight_key = weight_key,
        )
        # construct the left graph from nodes lower than pivot, ascending w.r.t. mz.

        right_edges = pairs[right_pair_mask]
        right_edges = right_edges[:,[1,0,2]] # transpose edge sources and targets
        right_boundaries_mask = np.logical_and(
            rb_mz > pivot,
            rb_mz < expected_max_mass,
        )
        right_sources = right_boundaries[right_boundaries_mask]
        right_adj[i] = SpectrumGraph.from_edges_and_boundaries(
            edges = right_edges,
            boundaries = right_sources,
            pivots = right_sinks,
            weight_key = weight_key,
        )
        # construct the right graph from nodes higher than pivot, descending w.r.t. mz.

    return (
        left_adj,
        right_adj,
        pivot_adj,
    )

def construct_spectrum_topology(
    fragment_labels: FragmentLabels,
    pivots: PivotResult,
    tolerance: float,
    weight_key: str = float,
) -> tuple[
    list[np.ndarray],
    tuple[
        list[SpectrumGraph],
        list[SpectrumGraph],
        list[PivotGraph],
]]:
    left_adj, right_adj, pivot_adj = _construct_spectrum_graphs(
        fragment_labels.mass,
        fragment_labels.symmetries,
        fragment_labels.lower_boundaries,
        fragment_labels.upper_boundaries,
        pivots.cluster_points,
        tolerance,
        weight_key,
    )
    # for each pivot cluster, build a left, right, and pivot graph.

    sym_nodes = [np.vstack([
            s, 
            [np.array([l.boundary_source, r.boundary_source]),],
            [np.array([l.pivot_sink, r.pivot_sink]),],
        ]) for (s,l,r) in zip(fragment_labels.symmetries,left_adj,right_adj)
    ]
    # amend symmetries with source and sink nodes in each spectrum graph. these nodes do not correspond to peaks, but are defined as symmetric here to simplify the propagation cost model.
    
    return (
        sym_nodes,
        left_adj,
        right_adj,
        pivot_adj,
    )
