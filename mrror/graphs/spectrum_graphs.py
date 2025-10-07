import itertools as it
# standard

from ..spectra.types import Peaks
from ..fragments.types import PairResult, PivotResult, BoundaryResult
from .types import SparseAdj
# local

import numpy as np

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
    right_boundaries: list[np.ndarray], # [[int; _]; p]
    pivots: list[np.ndarray],           # [(int,int,int,int); p]
) -> tuple[list[SparseAdj],list[SparseAdj],list[SparseAdj]]:
    pair_mz = mz[pairs]
    lb_mz = mz[left_boundaries]
    # prep mz values for pairs and left boundaries.
    k = len(pivots)
    left_adj = [None for _ in range(k)]
    right_adj = [None for _ in range(k)]
    cut_adj = [None for _ in range(k)]
    for (i, (rb, pivot)) in enumerate(zip(right_boundaries, pivots)):
        rb_mz = mz[rb]
        left_pair_mask = pair_mz[1,:] < pivot
        left_pairs = pairs[:,left_pair_mask]
        left_adj[i] = SparseAdj.from_edges(left_pairs)
        # left graph, edges lower than pivot, ascending w.r.t. mz.

        right_pair_mask = pair_mz[0,:] > pivot
        right_pairs = pairs[:,right_pair_mask]
        right_adj[i] = SparseAdj.from_edges(right_pairs, reverse=True)
        # right graph, edges higher than pivot, descending.

        cut_pairs = pairs[:,np.logical_not(left_pair_mask + right_pair_mask)]
        cut_adj[i] = SparseAdj.from_edges(cut_pairs, directed=False)
        # cut graph, undirected edges intersected by pivot, relate sinks between left and right.
    return (
        left_adj,
        right_adj,
        cut_adj,
    )

def construct_spectrum_topology(
    peaks: Peaks,
    pairs: PairResult,
    left_boundaries: BoundaryResult,
    pivots: PivotResult,
    right_boundaries: list[BoundaryResult],
) -> tuple[
    np.ndarray,
    list[np.ndarray],
    tuple[list[SparseAdj],list[SparseAdj],list[SparseAdj]],
]:
    fragment_masses, *graph_data, sym_idx = _reindex_fragment_masses(
        peaks.mz,
        pairs,
        left_boundaries,
        right_boundaries,
        pivots,
    )
    pair_idx, left_boundary_idx, right_boundaries_idx, pivot_idx = graph_data
    spectrum_graphs = _construct_spectrum_graphs(
        fragment_masses,
        pair_idx,
        left_boundary_idx,
        right_boundaries_idx,
        pivots.cluster_points,
    )
    return (
        fragment_masses,
        sym_idx,
        spectrum_graphs
    )
