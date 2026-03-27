import dataclasses, json, zlib
from typing import Iterator, Iterable, Callable, Any
import itertools as it

import numba
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

HYDROGEN_MASS = 1.007

MESHGRID_INDEXING = 'ij'

def meshgrid(*args, **kwargs) -> tuple:
    """A wrapper around np.meshgrid with the indexing kwarg fixed to the constant MESHGRID_INDEXING. Accordingly, passing 'indexing=...' will throw an exception. This function is provided for the sake of synchronized indexing between meshgrid and it.product."""
    return np.meshgrid(*args, **kwargs, indexing=MESHGRID_INDEXING)

def mesh_ravel(
    arrs: Iterable[np.ndarray],
    dims: tuple,
) -> np.ndarray:
    return np.ravel_multi_index(
        meshgrid(*arrs),
        dims,
    ).flatten()

def mesh_join(
    arrs: Iterable[np.ndarray],
    axis: int = 0,
) -> np.ndarray:
    return np.array(' '.join(x) for x in it.product(*arrs))

def mesh_sum(
    arrs: Iterable[np.ndarray],
    axis: int = 0,
) -> np.ndarray:
    return np.sum(
        meshgrid(*arrs),
        axis=axis,
    ).flatten()

def fuzzy_unique(x: np.ndarray, tolerance: float) -> tuple[np.ndarray,np.ndarray]:
    unique_scaled_x, idx = np.unique_inverse((x / tolerance).astype(int))
    return (
        unique_scaled_x * tolerance,
        idx
    )

def listsum(x):
    """Wrapper function to call sum on lists. Equivalently:
    
            sum(x, [])"""
    return sum(x,[])

def load_config(config_dir, config_name="config"):
    """Loads a hydra config object from a directory of .yaml files.
    
    config_dir: string, an absolute path.
    config_name: string, the name of the top-level config file without its extension."""
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(config_name=config_name)

def pairwise_disjoint(
    a: list,
    b: list,
) -> bool:
    return set(it.pairwise(a)).isdisjoint(it.pairwise(b))

def combine_masses(
    left_masses: np.ndarray,
    right_masses: np.ndarray,
) -> np.ndarray:
    return np.stack([
        np.sum(x, axis=1) 
        for x in (left_masses, right_masses)
    ]).T

def combine_symbols(
    left_symbols: np.ndarray,
    right_symbols: np.ndarray,
    sep: str,
) -> np.ndarray:
    """Combines symbols elementwise to two arrays with the same shape, e.g. with / as separator:
    
        _combine_symbols([[a, b, c], [x, y, z]], [[a, b, d], [u, w, z]]) = [[a, b, c/d], [x/u, y/w, z]]"""
    return np.array([
        x if x == y else x + sep + x
        for (x,y) in zip(
            left_symbols.flatten(),
            right_symbols.flatten(),
        )]).reshape(*left_symbols.shape)

def ravel(
    i: int,
    j: int,
    n: int,
) -> int:
    return int((i * n) + j)

def unravel(
    k: int,
    n: int,
) -> tuple[int,int]:
    return (
        int(k // n),
        int(k % n),
    )

@numba.jit(nopython=True)
def first_match(
    left_arr: Iterable,
    right_arr: Iterable,
) -> tuple[int,int]:
    n = len(left_arr)
    m = len(right_arr)
    for i in range(n):
        for j in range(m):
            if left_arr[i] == right_arr[j]:
                return (i,j)
    return (-1, -1)

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
            l += 1
        else: # z < x
            r += 1

@numba.jit(nopython=True)
def merge_compare_fuzzy_unique(
    left_arr: Iterable,
    right_arr: Iterable,
    tolerance: float,
) -> Iterator[tuple[int,int]]:
    """Given two unique, ordered Iterables, return the index pairs representing matches within a given tolerance."""
    n = len(left_arr)
    m = len(right_arr)
    l = 0
    r = 0
    while l < n and r < m:
        x = left_arr[l]
        z = right_arr[r]
        d = abs(x - z)
        if d < tolerance:
            yield (l,r)
            l += 1
            r += 1
        elif x < z:
            l += 1
        else: # z < x
            r += 1

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
    tolerance: float,
    verbose = False,
) -> tuple[np.ndarray,np.ndarray]:
    prev_match = (np.inf,np.inf)
    match_left = []
    match_right = []
    l = 0
    r = 0
    n = len(left_arr)
    m = len(right_arr)
    while l < n and r < m:
        prev_left_val, prev_right_val = prev_match
        left_val = left_arr[l]
        right_val = right_arr[r]
        if abs(left_val - right_val) <= tolerance:
            # hit: count a match, incr indices.
            match_left.append(l)
            match_right.append(r)
            prev_match = (left_val, right_val)
            l += 1
            r += 1
        elif abs(prev_left_val - right_val) <= tolerance:
            match_left.append(l - 1)
            match_right.append(r)
            prev_match = (prev_left_val, right_val)
            r += 1
        elif abs(left_val - prev_right_val) <= tolerance:
            match_left.append(l)
            match_right.append(r - 1)
            prev_match = (left_val, prev_right_val)
            l += 1
        elif left_val < right_val:
            r += 1
        else:
            l += 1
    #
    return (
        np.array(match_left,dtype=np.int16),
        np.array(match_right,dtype=np.int16),
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
        
        match_left, match_right = merge_compare(
            left_arr,
            right_arr,
            tolerance,
        )
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
    decharged_peaks = np.array([
        (c * peaks) - (HYDROGEN_MASS * (c - 1)) for c in charges])
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
