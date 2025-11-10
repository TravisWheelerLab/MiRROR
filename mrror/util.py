import dataclasses, json, zlib
from typing import Iterator, Iterable, Callable, Any
import itertools as it

import numba
import numpy as np

def ravel(
    i: int,
    j: int,
    n: int,
) -> int:
    return (i * n) + j

def unravel(
    k: int,
    n: int,
) -> tuple[int,int]:
    return (
        k // n,
        k % n,
    )

@numba.jit(nopython=True)
def merge_compare_exact_unique(
    left_arr: Iterable,
    right_arr: Iterable,
) -> Iterator[tuple[int,int]]:
    """Given two unique, ordered Iterables, return the index pairs representing exact matches."""
    n = len(left_arr)
    m = len(right_arr)
    l = 0
    r = 0
    while l < n and r < m:
        x = left_arr[l]
        z = right_arr[r]
        if x == z:
            yield (l,r)
            l += 1
            r += 1
        elif x < z:
            r += 1
        else: # z < x
            l += 1

def dfs(
    adj: list[np.ndarray],
    cost: dict[int,float],
    threshold: float,
    initial_states: list[tuple[float,int,list[int]]],
) -> Iterator[tuple[float,list[int]]]:
    """Depth-first search restricted by path cost.

    Mainly used for debugging. For the functions that enumerate candidate sequences from a weighted product graph, see mrror.graphs.trace.

    arguments:
    - adj: an indexable whose values are collections of nodes. intended to be the `adj` field of a networkx graph.
    - cost: a dictionary from nodes to costs.
    - threshold: max cost of a path may have.
    - initial_states: list of (cost, node, path) states. e.g., to iterate a graph with one source at node zero, pass [(0., 0, [])].

    return:
    - iterator of (cost, [path nodes...]) tuples."""
    q = deque(initial_states)
    while len(q) > 0:
        prev_cost, curr_node, prev_path = q.pop()
        curr_cost = prev_cost + cost[curr_node]
        curr_path = prev_path + [curr_node,]
        if curr_cost <= threshold:
            # terminate paths that exceed the threshold
            degree = len(adj[curr_node])
            if degree == 0:
                yield (curr_cost, curr_path)
                # yield paths that reach sinks
            else:
                for next_node in adj[curr_node]:
                    q.append((
                        curr_cost,
                        next_node.item(),
                        curr_path,
                    ))

def normalize_dict(
    data: dict[Any,float],
) -> dict[Any,float]:
    n = sum(data.values())
    return {k: data[k] / n for k in data}

def bisect_left(
    targets: np.ndarray,
    queries: np.ndarray,
) -> np.ndarray:
    return np.searchsorted(targets, queries, side = 'left')

def bisect_right(
    targets: np.ndarray,
    queries: np.ndarray,
) -> np.ndarray:
    return np.searchsorted(targets, queries, side = 'right')

@numba.jit(nopython=True)
def merge_compare(
    left_arr: np.ndarray,
    right_arr: np.ndarray,
    n: int,
    tolerance: float,
    pivot_point: float,
) -> tuple[np.ndarray,np.ndarray]:
    prev_match = (pivot_point, pivot_point)
    match_left = []
    match_right = []
    l = 0
    r = 0
    while l < n and r < n:
        x = left_arr[l]
        z = right_arr[r]
        if abs(x - z) <= tolerance:
            # hit: count a match, incr indices.
            match_left.append(l)
            match_right.append(r)
            prev_match = (x, z)
            l += 1
            r += 1
        elif abs(z - prev_match[1]) <= tolerance:
            match_left.append(l - 1)
            match_right.append(r)
            r += 1
        elif abs(x - prev_match[0]) <= tolerance:
            match_left.append(l)
            match_right.append(r - 1)
            l += 1
        elif x < z:
            r += 1
        else: # z < x
            l += 1
    #
    return (
        np.array(match_left),
        np.array(match_right),
    )

def _mirror_symmetries(
    sorted_arr: np.ndarray,
    pivot_points: np.ndarray,
    tolerance: float,
) -> Iterator[tuple[np.ndarray,np.ndarray,np.ndarray]]:
    lows = bisect_left(sorted_arr, pivot_points)
    highs = bisect_right(sorted_arr, pivot_points)
    # locate the low and high pivot point indices
    for (lo,hi,pivot_point) in zip(lows,highs,pivot_points):
        left_arr = sorted_arr[:lo][::-1]
        # left_arr is the reverse order of everything below the pivot
        
        right_arr = (2 * pivot_point) - sorted_arr[hi:]
        # right_arr is the reflection of everything above the pivot.
        
        n = min(len(left_arr), len(right_arr))
        match_left, match_right = merge_compare(
            left_arr,
            right_arr,
            n,
            tolerance,
            pivot_point)
        # find the symmetries as matched pairs in the left and right arrays.
        
        deltas = np.abs(right_arr[match_right] - left_arr[match_left])
        # reconstruct deltas

        # convert left_arr, right_arr indices to sorted_arr indices.
        match_left = lo - 1 - match_left
        match_right = lo + match_right
        match_idx = np.vstack([match_left,match_right]).T
        yield (match_idx, deltas)

def mirror_symmetries(
    sorted_arr: np.ndarray,
    pivot_points: np.ndarray,
    tolerance: float,
) -> tuple[Iterable[np.ndarray],Iterable[np.ndarray]]:
    """Finds mirror symmetries in time linear to the length of the array for each pivot point.
    
    Uses bisect to index the pivot point into the sorted array, then applies a merge-like routine to match the set below the pivot point to the reflection of the set above the pivot point. 
    
    Args:
    - sorted_arr: np.ndarray, a list of floats in ascending order.
    - pivot_points: np.ndarray, a list of floats, candidate points of mirror symmetry. sorted_arr[0] < pivot_points < sorted_arr[-1].
    - tolerance: float, the maximum difference between a point and a reflected point such that they can be counted as symmetric.
    
    Returns:
    - symmetries: Iterable[np.ndarray] ~ list[list[tuple[int,int]]], a list of symmetric pairs for each pivot point.
    - deltas: Iterable[np.ndarray] ~ list[list[float]], deltas for each symmetric pair."""
    symmetries, deltas = zip(*_mirror_symmetries(
        sorted_arr,
        pivot_points,
        tolerance,
    ))
    return (
        symmetries,
        deltas,
    )
     
def interleave(arrays: list[np.ndarray]) -> np.ndarray:
    # https://stackoverflow.com/a/5347492
    n = len(arrays)
    c = np.empty(sum(a.size for a in arrays), dtype=arrays[0].dtype)
    for i in range(n):
        c[i::n] = arrays[i]
    return c

def merge_in_order(
    arrays: list[np.ndarray],
    transformation: Callable[np.ndarray,np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    merged_array = np.hstack(arrays)
    if not(transformation is None):
        merged_array = transformation(merged_array)
    order = np.argsort(merged_array)
    indices = np.hstack([np.arange(len(arr)) for arr in arrays])
    source = np.hstack([np.full_like(arrays[i], i, dtype=int) for i in range(len(arrays))])
    return merged_array[order], indices[order], source[order]

def decharge_peaks(
    peaks: np.ndarray, # [float; n]
    charges: np.ndarray, # [int; k]
    transformation: Callable = None,
) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    n = len(peaks)
    k = len(charges)
    decharged_peaks = charges.reshape(k,1) * peaks.reshape(1,n)
    decharged_peaks = np.array([(c * peaks) - (c - 1) for c in charges])
    merged_decharged_peaks, deindexer, charge_table = merge_in_order(decharged_peaks, transformation)
    # construct and re-sort the charge-augmented m/z

    if merged_decharged_peaks[0] <= 0:
        first_pos_idx = np.searchsorted(merged_decharged_peaks, 0., 'right')
        merged_decharged_peaks = merged_decharged_peaks[first_pos_idx:]
        deindexer = deindexer[first_pos_idx:]
        charge_table = charge_table[first_pos_idx:]
    # remove any nonpositive values, which often appear when the transformation is a reflection about a low-mz pivot point.
    
    charge_table = charge_table + 1
    return merged_decharged_peaks, deindexer, charge_table

def prepare_eng_txt(
    txt: str,
    min=7,
    max=50,
    alphabet=['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'L', 'I', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'],
) -> list[str]:
    words = [w.upper().replace('\'','').replace('.','') for line in txt.split('\n') for w in line.split(' ')]
    words = [w for w in words if min <= len(w) <= max]
    words = [w for w in words if all(c in alphabet for c in w)]
    return words

def in_alphabet(
    word: str,
    alphabet: Iterable[str],
) -> bool:
    return all(c in alphabet for c in word)
