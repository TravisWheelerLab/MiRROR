import dataclasses
# standard

from ..util import bisect_left, bisect_right
# local

import numpy as np

@dataclasses.dataclass(slots=True)
class BoundaryResult:
    index: np.ndarray
    # [int; l]
    states: np.ndarray
    # [(int,int); l]
    mass: np.ndarray
    # [float; l]

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
    ]).T
    # find the query range for each left index

    left_mask = (query_hi - query_lo) > 0
    query_data = query_data[left_mask]
    # remove indices with empty query ranges

    query_indices = np.hstack([np.arange(lo,hi) for (lo,hi) in query_data])
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
    peaks: np.ndarray,
    tolerance: float,
    target_masses: np.ndarray,
) -> BoundaryResult:
    results, queries = _find_boundaries(
        peaks,
        tolerance,
        target_masses,
    )
    return BoundaryResult(
        index = results[:,1],
        states = results[:,1:],
        mass = queries,
    )
   
