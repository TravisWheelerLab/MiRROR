import dataclasses
from typing import Iterator, Iterable
# standard

from ..spectra.types import Peaks, AugmentedPeaks
from ..util import bisect_left, bisect_right, mirror_symmetries
from .types import PairResult, BoundaryResult, PivotResult, TargetMassStateSpace
# local

import numpy as np
import numba

def _find_pairs_minimal(
    peaks: np.ndarray, # [float; n]
    tolerance: float,
    target_masses: np.ndarray # [float; k]
) -> tuple[np.ndarray,np.ndarray]:
    min_target = target_masses[0] - tolerance
    max_target = target_masses[-1] + tolerance
    n = len(peaks)
    results = list()
    queries = list()
    for i in range(n - 1):
        query_peak = peaks[i]
        subpeaks = peaks[i + 1:]
        query_lo = bisect_left(peaks[i + 1:], query_peak + min_target)
        query_hi = bisect_right(peaks[i + 1:], query_peak + max_target)
        query_masses = subpeaks[query_lo:query_hi] - query_peak
        hits_lo = bisect_left(target_masses, query_masses - tolerance)
        hits_hi = bisect_right(target_masses, query_masses + tolerance)
        result_mask = hits_hi > hits_lo
        result_data = np.column_stack([
            np.full_like(query_masses, i, dtype=np.int64),
            i + 1 + np.arange(query_lo, query_hi),
            hits_lo,
            hits_hi,
        ])
        results.append(result_data[result_mask])
        queries.append(query_masses[result_mask])
    results = np.vstack(results)
    queries = np.concat(queries)
    return (
        results,
        queries,
    )

def _find_pairs(
    peaks: np.ndarray, # [float; n]
    tolerance: float,
    target_masses: np.ndarray # [float; k]
) -> tuple[np.ndarray,np.ndarray]:
    min_target = target_masses[0] - tolerance
    max_target = target_masses[-1] + tolerance
    n = len(peaks)

    left_indices = np.arange(n - 1)
    query_lo = bisect_left(peaks, peaks[:n - 1] + min_target)
    query_hi = bisect_right(peaks, peaks[:n - 1] + max_target)
    query_data = np.vstack([
        left_indices,
        query_lo,
        query_hi,
    ])
    # find the query range for each left index

    left_mask = query_hi > query_lo
    query_data = query_data[:, left_mask]
    left_indices, query_lo, query_hi = query_data
    # remove indices with empty query ranges

    left_indices = np.hstack([np.full(hi - lo, i) for (i,lo,hi) in zip(left_indices, query_lo, query_hi)])
    right_indices = np.hstack([np.arange(lo,hi) for (lo,hi) in zip(query_lo,query_hi)])
    # expand query ranges into right indices; pair to left indices.

    query_masses = peaks[right_indices] - peaks[left_indices]
    # construct queries as the difference between right and left peaks.

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
        query_masses
    )

def find_pairs(
    peaks: AugmentedPeaks,
    tolerance: float,
    targets: TargetMassStateSpace,
    mode: str = "minimal",
) -> PairResult:
    if mode == "minimal":
        results, queries = _find_pairs_minimal(
            peaks.mz,
            tolerance,
            targets.pair_masses,
        )
    elif mode == "full":
        results, queries = _find_pairs(
            peaks.mz,
            tolerance,
            targets.pair_masses,
        )
    else:
        raise ValueError(f"unrecognized pair search mode {mode}")
    local_indices = results[:,:2]
    global_indices = peaks.get_original_indices(local_indices)
    # reindex augmented peak indices to original peak indices
    loop_mask = global_indices[:,0] != global_indices[:,1]
    # remove loops
    query_masses = queries[loop_mask]
    features, feature_costs, feature_segments = targets.resolve_pairs(results[loop_mask,2:4], query_masses)
    # expand target hit range to features and calculate costs
    return PairResult(
        indices = global_indices[loop_mask],
        charges = peaks.get_augmenting_charges(local_indices[loop_mask]),
        features = features,
        costs = feature_costs,
        segments = feature_segments,
        mass = query_masses,
    )

def _find_boundaries(
    peaks: np.ndarray,
    tolerance: float,
    target_masses: np.ndarray,
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
        query_masses
    )
    
def find_boundaries(
    peaks: AugmentedPeaks,
    tolerance: float,
    targets: TargetMassStateSpace,
    k: int = 1,
) -> BoundaryResult:
    results, queries = _find_boundaries(
        peaks.mz,
        tolerance,
        targets.boundary_masses[k - 1]
    )
    features, feature_costs, feature_segments = targets.resolve_boundaries(results[:,1:], queries, k - 1)
    return BoundaryResult(
        index = peaks.get_original_indices(results[:,0]),
        charge = peaks.get_augmenting_charges(results[:,0]),
        features = features,
        costs = feature_costs,
        segments = feature_segments,
        mass = queries,
    )

def _find_virtual_pivots(
    midpoints: Iterable[float],
    bin_width: float,
) -> np.ndarray:
    # restrict the potential pivots to the most frequent values
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
def _find_overlap_pivots(
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
                if dif < tolerance:
                    yield np.array([i,j,i2,j2])
                elif mass_j > mass_i:
                    break

def _find_pivots(
    peaks: np.ndarray,
    pairs: PairResult,
    query_tolerance: float,
    find_overlap: bool,
    find_virtual: bool,
) -> Iterator[tuple[float,np.ndarray]]:
    if find_overlap:
        overlap_pivots = _find_overlap_pivots(
            peaks,
            pairs.indices, 
            query_tolerance,
        )
        for indices in overlap_pivots:
            yield (
                peaks[indices].sum() / 4,
                indices,
            )
    if find_virtual:
        midpoints = _construct_midpoints(pairs.mass[pairs.indices])
        bin_width = query_tolerance / 2
        virtual_pivots = _find_virtual_pivots(
            midpoints,
            bin_width,
        )
        for pivot_point in overlap_pivots:
            yield (
                pivot_point,
                None,
            )

def _bin_pivots(
    pivot_points: np.ndarray,
    precision: int,
) -> tuple[np.array,np.array]:
    return np.unique_inverse(np.round(pivot_points, precision))

def _screen_pivots(
    peaks: np.ndarray, # [float; n]
    cluster_points: np.ndarray, # [float; k]
    pivot_cluster_ids: np.ndarray, # [int; p]
    symmetry_tolerance: float,
    score_factor: float,
) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,list[np.ndarray]]:
    tolerance = min(symmetry_tolerance, (peaks[1:] - peaks[:1]).min() / 2)
    # calibrate tolerance against the minimum pairwise distance

    cluster_symmetries, symmetry_deltas = mirror_symmetries(
        peaks,
        cluster_points,
        symmetry_tolerance,
    )
    cluster_scores = np.array([sum(1 - deltas) for deltas in symmetry_deltas])
    # calculate scores from the symmetries of each cluster point's reflection.

    clusters_mask = cluster_scores > (cluster_scores.max() * score_factor)
    pivots_mask = clusters_mask[pivot_cluster_ids]
    # mask clusters by their symmetry scores

    # reindex the pivot_cluster_ids
    old_pivot_idx = np.arange(len(pivot_cluster_ids))[pivots_mask]
    cluster_reindexer = np.arange(len(cluster_points))
    cluster_reindexer[clusters_mask] = np.arange(sum(clusters_mask))
    new_pivot_cluster_ids = np.zeros_like(old_pivot_idx)
    for (new_pvt_idx, old_pvt_idx) in enumerate(old_pivot_idx):
        old_cluster_id = pivot_cluster_ids[old_pvt_idx]
        new_cluster_id = cluster_reindexer[old_cluster_id]
        new_pivot_cluster_ids[new_pvt_idx] = new_cluster_id.tolist()

    return (
        pivots_mask,
        clusters_mask,
        new_pivot_cluster_ids,
        cluster_scores[clusters_mask],
        [sym for (keep,sym) in zip(clusters_mask,cluster_symmetries) if keep],
    )
    
def find_pivots(
    peaks: Peaks,
    pairs: PairResult,
    query_tolerance: float,
    symmetry_tolerance: float,
    score_factor: float,
    find_overlap = True,
    find_virtual = False,
    bin_precision = 2,
) -> PivotResult:
    if not(find_overlap or find_virtual):
        raise ValueError("both find_overlap and find_virtual are False. there are no other ways to find pivots.")
    mz = peaks.mz
    pivot_points, pivot_indices = zip(*_find_pivots(
        mz,
        pairs,
        query_tolerance,
        find_overlap,
        find_virtual))
    pivot_points = np.array(pivot_points)
    pivot_indices = np.array(pivot_indices)
    # generate pivots

    cluster_points, pivot_cluster_ids = _bin_pivots(
        pivot_points,
        precision=bin_precision)
    # cluster by pivot point

    scr_result = _screen_pivots(
        mz,
        cluster_points,
        pivot_cluster_ids,
        symmetry_tolerance,
        score_factor,
    )
    pvt_mask, clust_mask, pivot_cluster_ids, clust_scores, clust_sym = scr_result
    pivot_points = pivot_points[pvt_mask]
    pivot_indices = pivot_indices[pvt_mask]
    cluster_points = cluster_points[clust_mask]
    p = len(pivot_points)
    k = len(cluster_points)
    # screen clusters

    clusters = [[] for _ in range(k)]
    for pivot_idx in range(p):
        cluster_id = pivot_cluster_ids[pivot_idx]
        clusters[cluster_id].append(pivot_idx)
    clusters = [np.array(x) for x in clusters]
    # construct cluster membership lists from pivot cluster ids

    return PivotResult.from_data(
        cluster_points,
        clusters,
        clust_scores,
        clust_sym,
        pivot_points,
        pivot_indices,
    )
    
