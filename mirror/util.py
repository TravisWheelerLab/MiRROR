import numpy as np
from typing import Iterator, Any
from itertools import chain
from math import ceil

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