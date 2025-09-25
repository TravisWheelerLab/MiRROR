import dataclasses
# standard

from ..util import bisect_left, bisect_right
# local

import numpy as np

@dataclasses.dataclass(slots=True)
class PairResult:
    indices: np.ndarray
    # [(int,int); m]
    states: np.ndarray
    # [(int,int); m]
    mass: np.ndarray
    # [float; m]

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
    ]).T
    # find the query range for each left index

    left_mask = (query_hi - query_lo) > 0
    query_data = query_data[left_mask]
    # remove indices with empty query ranges

    left_indices = np.hstack([np.full(hi - lo, i) for (i,lo,hi) in query_data])
    right_indices = np.hstack([np.arange(lo,hi) for (_,lo,hi) in query_data])
    # expand query ranges into right indices; pair to left indices.

    query_masses = peaks[right_indices] - peaks[left_indices]
    # construct queries as the difference between right and left peaks.

    hits_lo = bisect_left(target_masses, query_masses - tolerance)
    hits_hi = bisect_right(target_masses, query_masses + tolerance)
    # find the hit range for each query

    result_mask = (hits_hi - hits_lo) > 0
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
    peaks: np.ndarray, # [float; n]
    tolerance: float,
    target_masses: np.ndarray # [float; k]
) -> PairResult:
    results, queries = _find_pairs(
        peaks,
        tolerance,
        target_masses,
    )
    return PairResult(
        indices = results[:,:2],
        states = results[:,2:4],
        mass = queries,
    )
