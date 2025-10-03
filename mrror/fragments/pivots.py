import dataclasses
from typing import Self, Iterator, Iterable
# standard

from ..spectra.types import Peaks
from ..util import mirror_symmetries
from .pairs import PairResult
# local

import numba
import numpy as np

@dataclasses.dataclass(slots=True)
class PivotResult:
    cluster_points: np.ndarray
    # [float; k]
    clusters: list[np.ndarray]
    # [[int; _]; k]
    scores: np.ndarray
    # [float; k]
    symmetries: list[np.ndarray]
    # [[(int,int); _]; k]
    pivot_points: np.ndarray
    # [float; p]
    pivot_indices: np.ndarray
    # [(int,int,int,int); p]

    @classmethod
    def from_data(cls,
        cluster_points: np.ndarray,
        clusters: list[np.ndarray],
        scores: np.ndarray,
        symmetries: list[np.ndarray],
        pivot_points: np.ndarray,
        pivot_indices: np.ndarray,
    ) -> Self:
        assert len(cluster_points) == len(clusters) == len(scores) == len(symmetries)
        assert len(pivot_points) == len(pivot_indices)
        return cls(
            cluster_points,
            clusters,
            scores,
            symmetries,
            pivot_points,
            pivot_indices,
        )

    def __len__(self) -> int:
        return len(self.cluster_points)

def _find_virtual_pivots(
    midpoints: Iterable[float],
    bin_width: float,
) -> np.ndarray:
    # restrict the potential pivots to the most frequent values
    # the point(s) that are most frequent are likely the ones that induce the greatest mirror symmetry.
    num_bins = int((midpoints.max() - midpoints.min()) / bin_width)
    if num_bins == 0:
        return []
    # print(num_bins)
    bin_counts, bin_edges = np.histogram(
        midpoints,
        bins = num_bins)
    # print("bin edges", bin_edges)
    bin_values = (bin_edges[:-1] + bin_edges[1:]) / 2
    # print("bin values", bin_values)
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
    for (i, i2) in zip(pairs[0,:],pairs[1,:]):
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
    # construct cluster membership lists from pivot cluster ids

    return PivotResult.from_data(
        cluster_points,
        clusters,
        clust_scores,
        clust_sym,
        pivot_points,
        pivot_indices,
    )
    
