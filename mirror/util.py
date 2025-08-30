from typing import Iterable, Iterator, Any, Callable
from itertools import chain, combinations
from math import ceil
from bisect import bisect_left, bisect_right

import numba
import numpy as np

# TODO: rewrite
def disjoint_pairs(
    X: list[set],
    Y: list[set],
) -> Iterator[tuple[int,int]]:
    def _construct_membership_table(
        X: list[set],
    ):
        """Creates a dictionary that associates each element in each set in X to the 
        index of the sets in which it is contained."""
        S = set(chain.from_iterable(X))
        table = {
            s: []
            for s in S
        }
        n = len(X)
        for i in range(n):
            for elt in X[i]:
                table[elt].append(i)
        return table
    
    def _find_disjoint(
        x: set,
        n: int,
        membership_table: dict,
    ):
        """Given a set x, the number of sets in the superset, and the membership table representing
        the superset, list the indices of the sets in the superset that do not intersect x."""
        disjoint = [True for _ in range(n)]
        for elt in x:
            for set_idx in membership_table[elt]:
                disjoint[set_idx] = False
        return [i for i in range(n) if disjoint[i]]
        

    Y_membership = _construct_membership_table(Y)
    n = len(X)
    for i in range(n):
        yield from [
            (i,j) for j in _find_disjoint(X[i], n, Y_membership) if j > i]

if __name__ == "__main__":
    def powerset(iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    x = list(powerset([1,2,3]))[1:-1]
    dx = [(x[i],x[j]) for (i,j) in disjoint_pairs(x, x)]
    print(x)
    print(dx)

@numba.njit
def _merge_count(
    left_arr: np.ndarray,
    right_arr: np.ndarray,
    n: int,
    tolerance: float,
    pivot_point: float,
) -> int:
    prev_match = (pivot_point, pivot_point)
    num_symmetries = 0
    l = 0
    r = 0
    while l < n and r < n:
        x = left_arr[l]
        z = right_arr[r]
        if abs(x - z) <= tolerance:
            # hit: count a symmetry, incr indices.
            num_symmetries += 1
            prev_match = (x, z)
            l += 1
            r += 1
        elif abs(z - prev_match[1]) <= tolerance:
            num_symmetries += 1
            r += 1
        elif abs(x - prev_match[0]) <= tolerance:
            num_symmetries += 1    
            l += 1
        elif x < z:
            r += 1
        else: # z < x
            l += 1
    #
    return num_symmetries

def measure_mirror_symmetry(
    sorted_arr: np.ndarray,
    pivot_point: float,
    tolerance: float,
) -> int:
    """Measures mirror symmetry in time linear to the length of the array.
    
    Uses bisect to index the pivot point into the sorted array, then applies a merge-like routine to match the set below the pivot point to the reflection of the set above the pivot point. 
    
    Args:
    - sorted_arr: np.ndarray, a list of floats in ascending order.
    - pivot_point: float, a candidate point of mirror symmetry. sorted_arr[0] < pivot_point < sorted_arr[-1].
    - tolerance: float, the maximum difference between a point and a reflected point such that they can be counted as symmetric.
    
    Returns:
    - num_symmetries: int, the number of points symmetric under the reflection parametized by the pivot point."""
    # left_arr is the reverse order of everything below the pivot
    lo = bisect_left(sorted_arr, pivot_point)
    left_arr = sorted_arr[:lo][::-1]
    # right_arr is the reflection of everything above the pivot.
    hi = bisect_right(sorted_arr, pivot_point)
    right_arr = (2 * pivot_point) - sorted_arr[hi:]
    # merge to count symmetries
    n = min(lo, hi)
    return _merge_count(
        left_arr,
        right_arr,
        n,
        tolerance,
        pivot_point)

def binsort(
    arr: Iterable,
    key: Callable,
) -> tuple[Iterable,list[Iterable]]:
    unique_keys, deindexer = np.unique_inverse([key(x) for x in arr]) 
    bins = [list() for _ in range(len(unique_keys))]
    for (i,x) in zip(deindexer,arr):
        bins[i].append(x)
    return unique_keys, bins

def collapse_second_order_list(llist: list[list]):
    """Associates a list of lists of elements to a flat list of elements.
    
        list(itertools.chain.from_iterable(llist))

    :llist: a second-order iterable."""
    return list(chain.from_iterable(llist))

def _recursive_collapse(items, index, count):
    # because strings are iterables, including characters, they will lead to infinite recursion if not treated separately.
    if not(isinstance(items, str)) and isinstance(items, Iterable):
        new_items = []
        new_index = []
        for x in items:
            new_x, new_i, count = _recursive_collapse(x, index, count)
            new_items.append(new_x)
            new_index.append(new_i)
        return collapse_second_order_list(new_items), new_index, count
    else:
        count += 1
        return [items], count, count

def recursive_collapse(items):
    return _recursive_collapse(items, [], -1)[:2]

def recursive_uncollapse(flat_items, index):
    if isinstance(index, Iterable):
        items = []
        for subindex in index:
            new_item = recursive_uncollapse(flat_items, subindex)
            items.append(new_item)
        return items
    else:
        return flat_items[index]

def mask_ambiguous_residues(res: chr):
    "Maps residues \'L\' and \'I\' to \"I/L\"."
    if res == "L" or res == "I":
        return "L"
    else:
        return res

def interleave(arrays: list[np.ndarray]) -> np.ndarray:
    # https://stackoverflow.com/a/5347492
    n = len(arrays)
    c = np.empty(sum(a.size for a in arrays), dtype=arrays[0].dtype)
    for i in range(n):
        c[i::n] = arrays[i]
    return c

def merge_in_order(
    arrays: list[np.ndarray],
    transformation: Callable = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    merged_array = np.hstack(arrays)
    if not(transformation is None):
        merged_array = transformation(merged_array)
    order = np.argsort(merged_array)
    indices = np.hstack([np.arange(len(arr)) for arr in arrays])
    source = np.hstack([np.full_like(arrays[i], i, dtype=int) for i in range(len(arrays))])
    return merged_array[order], indices[order], source[order]

def print_alignment(
    score: float,
    alignment_edges: Iterator[tuple[Any, Any]], 
    alignment_weights: Iterator[tuple[Any, Any]],
    name: str == None,
):
    alignment_edges = list(alignment_edges)
    alignment_weights = list(alignment_weights)
    if name == None:
        header = f"score: {score}\n"
    else:
        header = f"{name}\nscore: {score}\n"
    # construct the alignment representation
    first_nodes = []
    first_weights = []
    horizontal_separator = []
    second_weights = []
    second_nodes = []
    for (((first_node_left, first_node_right), (second_node_left, second_node_right)), (first_weight, second_weight)) in zip(alignment_edges, alignment_weights):
        if first_weight is None:
            first_nodes.append('...')
            first_weights.append('')
            horizontal_separator.append('')
            second_weights.append(str(second_weight))
            second_nodes.append(f"{second_node_left}-{second_node_right}")
        elif second_weight is None:
            first_nodes.append(f"{first_node_left}-{first_node_right}")
            first_weights.append(str(first_weight))
            horizontal_separator.append('')
            second_weights.append('')
            second_nodes.append('...')
        else:
            first_nodes.append(f"{first_node_left}-{first_node_right}")
            first_weights.append(str(first_weight))
            horizontal_separator.append('|')
            second_weights.append(str(second_weight))
            second_nodes.append(f"{second_node_left}-{second_node_right}")
    # left-justify the alignment representation 
    pad_len = max(map(
        len, 
        chain(first_nodes, first_weights, horizontal_separator, second_weights, second_nodes)))
    left_pad_len = ceil(pad_len / 2)
    fill_char = ' '
    first_nodes, second_nodes = map(
            lambda symbols: map(
                lambda s: s.ljust(pad_len, fill_char),
                symbols),
            [first_nodes, second_nodes])
    first_weights, horizontal_separator, second_weights = map(
            lambda symbols: map(
                lambda s: s.rjust(left_pad_len, fill_char).ljust(pad_len, fill_char),
                symbols),
            [first_weights, horizontal_separator, second_weights])
    # set the footer
    footer = '\n' + ((len(alignment_edges) + 1) * pad_len * '-')
    # done 
    return header + '\n'.join(map(
        lambda symbols: ' '.join(symbols),
        [first_nodes, first_weights, horizontal_separator, second_weights, second_nodes]
    )) + footer
