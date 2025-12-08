import itertools as it
# standard

from ..spectra.types import Peaks
from ..fragments.types import PairResult, PivotResult, BoundaryResult
from ..graphs.types import SpectrumGraph, PivotGraph
# local

import numpy as np
import networkx as nx

def _reindex_fragment_masses(
    mz: np.ndarray,
    pairs: PairResult,
    left_boundaries: BoundaryResult,
    right_boundaries: list[BoundaryResult],
    pivots: PivotResult,
) -> tuple[
    np.ndarray,         # [float; u]
    np.ndarray,         # [(int,int); m]
    np.ndarray,         # [int; l]
    list[np.ndarray],   # [[int; _]; p]
    list[np.ndarray],   # [(int,int,int,int); p]
]:
    pair_structures = pairs.indices
    pair_indices = pair_structures.flatten()
    pair_charges = pairs.charges.flatten()
    
    lb_indices = left_boundaries.index
    lb_charges = left_boundaries.charge

    right_boundaries_indices = [rb.index for rb in right_boundaries]
    right_boundaries_charges = [rb.charge for rb in right_boundaries]
    rb_indices = np.concat(right_boundaries_indices)
    rb_charges = np.concat(right_boundaries_charges)

    pivot_structures = pivots.pivot_indices
    n_overlap_pivots = next((i for (i,v) in enumerate(pivot_structures) if v is None), len(pivot_structures))
    pivot_indices = np.concat(pivot_structures[:n_overlap_pivots])
    pivot_charges = np.ones_like(pivot_indices)

    symmetric_structures = pivots.symmetries
    symmetric_indices = [sym.flatten() for sym in symmetric_structures]
    sym_indices = np.concat(symmetric_indices)
    sym_charges = np.ones_like(sym_indices)

    concat_indices = np.concat([
        pair_indices,
        lb_indices,
        rb_indices,
        pivot_indices,
        sym_indices,
    ])
    concat_charges = np.concat([
        pair_charges,
        lb_charges,
        rb_charges,
        pivot_charges,
        sym_charges,
    ])
    # concatenate all indices into a single array (and likewise with charges.)

    loc_pairs = pair_indices.size

    loc_lb = loc_pairs + lb_indices.size

    loc_rb = loc_lb + rb_indices.size
    sizes_rb = [x.size for x in right_boundaries_indices]
    offsets_rb = np.cumsum([0] + sizes_rb)

    loc_pivot = loc_rb + pivot_indices.size
    offsets_pivot = np.cumsum([0] + [4] * n_overlap_pivots)

### loc_symmetric = loc_pivot + symmetic_indices.size
    sizes_sym = [x.size for x in symmetric_indices]
    offsets_sym = np.cumsum([0] + sizes_sym)
    # record component locations in the concatenated array.

    all_fragment_masses = mz[concat_indices] * concat_charges
    fragment_masses, reidx_concat_indices = np.unique_inverse(all_fragment_masses)
    # get unique masses and new indices into unique masses.

    reidx_pairs = reidx_concat_indices[:loc_pairs].reshape(pair_structures.shape)

    reidx_left_boundaries = reidx_concat_indices[loc_pairs:loc_lb]

    reidx_right_boundaries = reidx_concat_indices[loc_lb:loc_rb]
    reidx_right_boundaries = [reidx_right_boundaries[i:j] for (i,j) in it.pairwise(offsets_rb)]

    reidx_pivot_structures = reidx_concat_indices[loc_rb:loc_pivot]
    reidx_pivot_structures = [reidx_pivot_structures[i:j] for (i,j) in it.pairwise(offsets_pivot)]

    reidx_symmetric_structures = reidx_concat_indices[loc_pivot:]
    reidx_symmetric_structures = [reidx_symmetric_structures[i:j].reshape(symmetric_structures[k].shape) for (k, (i,j)) in enumerate(it.pairwise(offsets_sym))]
    # decompose and reshape into original arrays.

    return (
        fragment_masses,
        reidx_pairs,
        reidx_left_boundaries,
        reidx_right_boundaries,
        reidx_pivot_structures,
        reidx_symmetric_structures,
    )

def _construct_spectrum_graphs(
    mz: np.ndarray,                     # [float; n]
    pairs: np.ndarray,                  # [(int,int); m]
    left_boundaries: np.ndarray,        # [int; l]
    right_boundary_arrs: list[np.ndarray], # [[int; _]; p]
    pivots: list[np.ndarray],           # [(int,int,int,int); p]
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
    pairs = np.vstack([pairs, np.arange(pairs.shape[1])])
    left_boundaries = np.vstack([left_boundaries, np.arange(left_boundaries.size)])
    right_boundary_arrs = [np.vstack([right_boundaries, np.arange(right_boundaries.size)]) for right_boundaries in right_boundary_arrs]
    # add indices to pairs
    k = len(pivots)
    left_adj = [None for _ in range(k)]
    right_adj = [None for _ in range(k)]
    pivot_adj = [None for _ in range(k)]
    for (i, (right_boundaries, rb_mz, pivot)) in enumerate(zip(right_boundary_arrs, right_boundary_mz, pivots)):
        left_pair_mask = pair_mz[1,:] < pivot
        expected_max_mass = 2 * (pivot + tolerance)
        right_pair_mask = np.logical_and(
            pair_mz[0,:] > pivot,
            pair_mz[1,:] < expected_max_mass,
        )
        # partition the pairs into left and right around the pivot.

        pivot_pairs = pairs[:,np.logical_not(left_pair_mask + right_pair_mask)]
        left_sinks = pivot_pairs[0,:]
        right_sinks = pivot_pairs[1,:]
        pivot_adj[i] = PivotGraph.from_edges(
            edges = pivot_pairs,
        )
        # construct the pivot graph from undirected pairs intersected by pivot.
        
        left_edges = pairs[:,left_pair_mask]
        left_boundaries_mask = lb_mz < pivot
        left_sources = left_boundaries[:,left_boundaries_mask]
        left_adj[i] = SpectrumGraph.from_edges_and_boundaries(
            edges = left_edges,
            boundaries = left_sources,
            pivots = left_sinks,
            weight_key = weight_key,
        )
        # construct the left graph from nodes lower than pivot, ascending w.r.t. mz.

        right_edges = pairs[:,right_pair_mask]
        right_edges = right_edges[[1,0,2],:] # transpose edge sources and targets
        right_boundaries_mask = np.logical_and(
            rb_mz > pivot,
            rb_mz < expected_max_mass,
        )
        right_sources = right_boundaries[:,right_boundaries_mask]
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
    peaks: Peaks,
    pairs: PairResult,
    left_boundaries: BoundaryResult,
    pivots: PivotResult,
    right_boundaries: list[BoundaryResult],
    tolerance: float,
    weight_key: str = float,
) -> tuple[
    np.ndarray,
    list[np.ndarray],
    tuple[
            list[SpectrumGraph],
            list[SpectrumGraph],
            list[PivotGraph],
]]:
    fragment_masses, *graph_data, sym_idx = _reindex_fragment_masses(
        peaks.mz,
        pairs,
        left_boundaries,
        right_boundaries,
        pivots,
    )
    pair_idx, left_boundary_idx, right_boundaries_idx, pivot_idx = graph_data
    left_adj, right_adj, pivot_adj = _construct_spectrum_graphs(
        fragment_masses,
        pair_idx,
        left_boundary_idx,
        right_boundaries_idx,
        pivots.cluster_points,
        tolerance,
        weight_key,
    )
    sym_nodes = [np.vstack([
            s, 
            [np.array([l.boundary_source, r.boundary_source]),],
            [np.array([l.pivot_sink, r.pivot_sink]),],
        ]) for (s,l,r) in zip(sym_idx,left_adj,right_adj)
    ]
    return (
        fragment_masses,
        sym_nodes,
        left_adj,
        right_adj,
        pivot_adj,
    )
