import numpy as np
from typing import Iterable, Iterator, Any, Callable
from itertools import chain
from math import ceil
from bisect import bisect_left, bisect_right

def measure_mirror_symmetry(
    sorted_arr: np.ndarray,
    pivot_point: float,
    tolerance: float):
    """Reflects sorted_arr about pivot_point to create the query set; uses bisect to find the minimum distance of each query to any point in the original sorted_arr; returns the sum these minimum distances."""
    reflected_arr = 2 * pivot_point - sorted_arr
    n = len(sorted_arr)
    left_bound = sorted_arr[0]
    right_bound = sorted_arr[-1]
    symmetries = 0
    for j,q in enumerate(reflected_arr):
        I = range(
            bisect_left(sorted_arr, q - tolerance),
            bisect_right(sorted_arr, q + tolerance))
        for i in I:
            p = sorted_arr[i]
            dif = abs(p - q)
            if dif <= tolerance:
                symmetries += 1
                break
    return symmetries

def binsort(
    arr: Iterable,
    bins: int,
    key: Callable = None,
) -> list[list]:
    B = [list() for _ in range(bins)]
    for (i, k) in enumerate(map(key, arr)):
        B[k].append(arr[i])
    return B

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

def merge_in_order(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    merged_array = np.hstack(arrays)
    order = np.argsort(merged_array)
    indices = np.hstack([np.arange(len(arr)) for arr in arrays])
    source = np.hstack([np.full_like(arrays[i], i, dtype=int) for i in range(len(arrays))])
    return merged_array[order], indices[order], source[order]
    #N = [len(A) for A in arrays]
    #I = [0 for _ in arrays]
    #m = len(arrays)
    #minimum = min(min(A) for A in arrays)
    #for _ in range(sum(N)):
    #    for k in range(m):
    #        i = I[k]
    #        n = N[k]
    #        arr = arrays[k]

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
