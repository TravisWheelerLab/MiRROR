import dataclasses
from itertools import pairwise
from typing import Iterator, Iterable
# standard

from ..spectra.types import Peaks, AugmentedPeaks
from ..util import bisect_left, bisect_right, mirror_symmetries, decharge
from .types import PairResult, BoundaryResult, AxesResult, TargetMasses, UniqueFragmentIndex
# local

import numpy as np
import numba

def _find_pairs(
    peaks: np.ndarray, # [float; n]
    target_masses: np.ndarray, # [float; k]
    tolerance: float,
) -> tuple[np.ndarray,np.ndarray]:
    min_target = target_masses[0] - tolerance
    max_target = target_masses[-1] + tolerance
    n = len(peaks)
    results = list()
    queries = list()
    left_ind_arrs = list()
    right_ind_arrs = list()
    query_mass_arrs = list()
    for i in range(n - 1):
        query_peak = peaks[i]
        subpeaks = peaks[i + 1:]
        query_lo = bisect_left(peaks[i + 1:], query_peak + min_target)
        query_hi = bisect_right(peaks[i + 1:], query_peak + max_target)
        query_mass_arrs.append(subpeaks[query_lo:query_hi] - query_peak)
        left_ind_arrs.append(np.full(query_hi - query_lo, i))
        right_ind_arrs.append(np.arange(query_hi - query_lo) + query_lo + i + 1)
    query_masses = np.concat(query_mass_arrs)
    left_indices = np.concat(left_ind_arrs)
    right_indices = np.concat(right_ind_arrs)

    hits_lo = bisect_left(target_masses, query_masses - tolerance)
    hits_hi = bisect_right(target_masses, query_masses + tolerance)
    # find the hit range for each query

    result_mask = hits_hi > hits_lo
    result_data = np.vstack([
        left_indices,
        right_indices,
        hits_lo,
        hits_hi,
    ]).T
    result_data = result_data[result_mask]
    query_masses = query_masses[result_mask]
    # remove results with no hits

    return (
        result_data,
        query_masses,
    )

def _find_boundaries(
    peaks: np.ndarray,
    target_masses: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray,np.ndarray]:
    min_target = target_masses[0] - tolerance
    max_target = target_masses[-1] + tolerance
    n = len(peaks)
    query_lo = bisect_left(peaks, np.array([min_target]))
    query_hi = bisect_right(peaks, np.array([max_target]))
    query_data = np.vstack([
        query_lo,
        query_hi,
    ])
    # find the query range for each left index

    left_mask = (query_hi - query_lo) > 0
    if not(any(left_mask)):
        return (
            np.empty((0,4),dtype=int),
            np.empty((0,),dtype=float),
        )
        # catch empty results before they crash.
    query_data = query_data[:,left_mask]
    query_lo, query_hi = query_data
    # remove indices with empty query ranges

    query_indices = np.hstack([np.arange(lo,hi) for (lo,hi) in zip(query_lo,query_hi)])
    # expand query ranges into indices.

    query_masses = peaks[query_indices]
    # construct queries as the difference between right and left peaks.

    hits_lo = bisect_left(target_masses, query_masses - tolerance)
    hits_hi = bisect_right(target_masses, query_masses + tolerance)
    # find the hit range for each query

    result_mask = (hits_hi - hits_lo) > 0
    result_data = np.vstack([
        query_indices,
        hits_lo,
        hits_hi,
    ]).T
    result_data = result_data[result_mask]
    query_masses = query_masses[result_mask]
    # remove results with no hits

    return (
        result_data,
        query_masses,
    )

def find_pairs(
    peaks: AugmentedPeaks,
    targets: TargetMasses,
    tolerance: float,
) -> PairResult:
    result_data, query_masses = _find_pairs(
        peaks.mz,
        targets.target_masses,
        tolerance,
    )
    if result_data.size == 0:
        return PairResult.empty()
    else:
        augmented_peak_index_pairs = result_data[:,:2]
        original_peak_index_pairs = peaks.get_original_indices(augmented_peak_index_pairs)
        loop_mask = original_peak_index_pairs[:,0] != original_peak_index_pairs[:,1]
        augmented_peak_index_pairs = augmented_peak_index_pairs[loop_mask]
        augmented_peak_charges = peaks.get_augmenting_charges(augmented_peak_index_pairs)
        original_peak_index_pairs = original_peak_index_pairs[loop_mask]
        hit_ranges = result_data[loop_mask,2:]
        query_masses = query_masses[loop_mask]
        return PairResult.from_arrays(
            query_masses = query_masses,
            peak_index_pairs = original_peak_index_pairs,
            augmented_peak_index_pairs = augmented_peak_index_pairs,
            augmented_peak_charges = augmented_peak_charges,
            hit_ranges = hit_ranges,
        )

def find_boundaries(
    peaks: AugmentedPeaks,
    targets: TargetMasses,
    tolerance: float,
) -> BoundaryResult:
    result_data, query_masses = _find_boundaries(
        peaks.mz,
        targets.target_masses,
        tolerance,
    )
    if result_data.size == 0:
        return BoundaryResult.empty()
    else:
        augmented_peak_index = result_data[:,0]
        original_peak_index = peaks.get_original_indices(augmented_peak_index)
        augmented_peak_charges = peaks.get_augmenting_charges(augmented_peak_index)
        hit_ranges = result_data[:,1:]
        return BoundaryResult.from_arrays(
            peak_indices = original_peak_index,
            augmented_peak_indices = augmented_peak_index,
            augmented_peak_charges = augmented_peak_charges,
            query_masses = query_masses,
            hit_ranges = hit_ranges,
        )

def _find_virtual_axes(
    midpoints: Iterable[float],
    bin_width: float,
) -> np.ndarray:
    # restrict the potential axes to the most frequent values
    # the point(s) that are most frequent are likely the ones that induce the greatest mirror symmetry.
    num_bins = int((midpoints.max() - midpoints.min()) / bin_width)
    if num_bins == 0:
        return []
    bin_counts, bin_edges = np.histogram(
        midpoints,
        bins = num_bins)
    bin_values = (bin_edges[:-1] + bin_edges[1:]) / 2
    frequencies = sorted(set(bin_counts))
    mask = bin_counts > frequencies[int(len(frequencies) * 0.75)]
    maximal_bin_values = bin_values[mask]
    maximal_bin_counts = bin_counts[mask]
    # return the bin values and the normalized bin counts
    return maximal_bin_values

@numba.jit(nopython=True)
def _find_overlap_axes(
    spectrum: np.ndarray, # [float; n]
    pairs: np.ndarray, # [[int; m]; 2]
    tolerance: float,
) -> Iterator[tuple[int,int]]:
    n = len(spectrum)
    for (i, i2) in zip(pairs[:,0],pairs[:,1]):
        mass_i = spectrum[i2] - spectrum[i]
        for j in range(i + 1, i2):
            for j2 in range(i2 + 1, n):
                mass_j = spectrum[j2] - spectrum[j]
                dif = abs(mass_i - mass_j)
                # axi = sum(spectrum[[i,j,i2,j2]].sum() / 4
                if dif < tolerance:
                    yield np.array([i,j,i2,j2])
                elif mass_j > mass_i:
                    break

def _find_axes(
    peaks: AugmentedPeaks,
    pairs: PairResult,
    query_tolerance: float,
    find_overlap: bool,
    find_virtual: bool,
) -> Iterator[tuple[float,np.ndarray]]:
    if find_overlap:
        overlap_axes = _find_overlap_axes(
            peaks.mz,
            pairs.get_augmented_peak_pair_indices(), 
            query_tolerance,
        )
        for augmented_indices in overlap_axes:
            mz = peaks.mz[augmented_indices]
            indices = peaks.get_original_indices(augmented_indices)
            yield (
                mz.sum() / 4,
                indices,
            )
    if find_virtual:
        midpoints = _construct_midpoints(pairs.mass[pairs.indices])
        bin_width = query_tolerance / 2
        virtual_axes = _find_virtual_axes(
            midpoints,
            bin_width,
        )
        for axes_point in overlap_axes:
            yield (
                axes_point,
                None,
            )

def _bin_axes(
    axes_points: np.ndarray,
    precision: int,
) -> tuple[np.array,np.array]:
    return np.unique_inverse(np.round(axes_points, precision))

def _screen_axes(
    peaks: AugmentedPeaks, # [float; n]
    cluster_points: np.ndarray, # [float; k]
    axes_cluster_ids: np.ndarray, # [int; p]
    symmetry_tolerance: float,
    score_factor: float,
) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,list[np.ndarray]]:
    tolerance = min(symmetry_tolerance, (peaks.mz[1:] - peaks.mz[:1]).min() / 2)
    # calibrate tolerance against the minimum pairwise distance

    cluster_symmetries, symmetry_deltas = mirror_symmetries(
        peaks.mz,
        cluster_points,
        symmetry_tolerance,
    )

    cluster_scores = np.array([sum(1 - deltas) for deltas in symmetry_deltas])
    # calculate scores from the symmetries of each cluster point's reflection.

    cluster_symmetries = [peaks.get_original_indices(x) for x in cluster_symmetries]
    # fix indices

    score_threshold = cluster_scores.max() * score_factor
    clusters_mask = cluster_scores > score_threshold
    axes_mask = clusters_mask[axes_cluster_ids]
    # mask clusters by their symmetry scores

    old_axes_idx = np.arange(len(axes_cluster_ids))[axes_mask]
    cluster_reindexer = np.arange(len(cluster_points))
    cluster_reindexer[clusters_mask] = np.arange(sum(clusters_mask))
    new_axes_cluster_ids = np.zeros_like(old_axes_idx)
    for (new_pvt_idx, old_pvt_idx) in enumerate(old_axes_idx):
        old_cluster_id = axes_cluster_ids[old_pvt_idx]
        new_cluster_id = cluster_reindexer[old_cluster_id]
        new_axes_cluster_ids[new_pvt_idx] = new_cluster_id.tolist()
    # reindex the axes_cluster_ids

    return (
        axes_mask,
        clusters_mask,
        new_axes_cluster_ids,
        cluster_scores[clusters_mask],
        [sym for (keep,sym) in zip(clusters_mask,cluster_symmetries) if keep],
    )

def deduplicate_by_fragment_mass(
    peaks: Peaks,
    pairs: PairResult,
    lower_boundaries: BoundaryResult,
    axes: AxesResult,
    upper_boundaries: list[BoundaryResult],
) -> UniqueFragmentIndex:
    pairs_mz = peaks.mz[pairs.get_peak_pair_indices().flatten()]
    pairs_charge = pairs.get_augmented_peak_pair_charges().flatten()
    pairs_mass = decharge(pairs_mz, pairs_charge)
    n_pairs = len(pairs_mass)
    # flatten pairs and reconstruct fragment masses

    lbound_mz = peaks.mz[lower_boundaries.get_peak_indices()]
    lbound_charge = lower_boundaries.get_augmented_peak_charges()
    lbound_mass = decharge(lbound_mz, lbound_charge)
    n_lbound = len(lbound_mass)
    # reconstruct lower boundary fragment masses

    axes_mz = peaks.mz[axes.get_axes_peak_indices().flatten()]
    axes_charge = axes.get_axes_charges().flatten()
    axes_mass = decharge(axes_mz, axes_charge)
    n_axes = len(axes_mass)

    ubound_masses = []
    ubound_segments = [0,]
    symmetry_masses = []
    symmetry_segments = [0,]
    sym_peak_idx = axes.get_symmetries_peak_indices()
    sym_charge = axes.get_symmetries_charges()
    k = len(axes)
    for i in range(k):
        ubound = upper_boundaries[i]
        ubound_mz = peaks.mz[ubound.get_peak_indices()]
        ubound_charge = ubound.get_augmented_peak_charges()
        ubound_mass = decharge(ubound_mz, ubound_charge)
        ubound_masses.append(ubound_mass)
        ubound_segments.append(len(ubound_mass))
        # for each upper boundary result, reconstruct fragment masses.
        
        sym_mz = peaks.mz[sym_peak_idx[i]].flatten()
        sym_masses = decharge(sym_mz, sym_charge[i].flatten())
        symmetry_masses.append(sym_masses)
        symmetry_segments.append(len(sym_masses))
        # for each symmetric set, reconstruct fragment masses.
    n_ubound = sum(ubound_segments)
    ubound_offsets = np.cumsum(ubound_segments)
    cat_ubound_mass = np.concat(ubound_masses)

    n_symmetries = sum(symmetry_segments)
    symmetry_offsets = np.cumsum(symmetry_segments)
    cat_symmetry_mass = np.concat(symmetry_masses)

    outer_offsets = np.cumsum([0,n_pairs,n_lbound,n_axes,n_ubound,n_symmetries])
    cat_mass = np.concat([
        pairs_mass,lbound_mass,axes_mass,cat_ubound_mass,cat_symmetry_mass,
    ])
    fragment_masses, fragment_indices = np.unique(cat_mass,return_inverse=True)

    cat_ubound_idx = fragment_indices[outer_offsets[3]:outer_offsets[4]]
    cat_symmetry_idx = fragment_indices[outer_offsets[4]:outer_offsets[5]]
    return UniqueFragmentIndex(
        fragment_masses,
        pairs = fragment_indices[outer_offsets[0]:outer_offsets[1]].reshape((-1,2)),
        lower_boundaries = fragment_indices[outer_offsets[1]:outer_offsets[2]],
        axes = fragment_indices[outer_offsets[2]:outer_offsets[3]].reshape((-1,4)),
        symmetries = [cat_symmetry_idx[i:j].reshape((-1,2))
            for i,j in pairwise(symmetry_offsets)],
        upper_boundaries = [cat_ubound_idx[i:j]
            for i,j in pairwise(ubound_offsets)],
    )
