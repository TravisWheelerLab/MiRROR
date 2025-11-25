import dataclasses
import itertools as it

from .graphs.types import PivotGraph, WeightedProductGraph
from .pathspaces import AnnotatedResiduePathSpace
from .costmodels import MISMATCH_SEPARATOR

import numpy as np

def orient_affixes(
    affixes: AnnotatedResiduePathSpace,
    b_anno: list[str],
    y_anno: list[str],
    sep: str = MISMATCH_SEPARATOR
) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Given a bag of undifferentiated affixes, and the b and y annotation symbols, orient as prefix, suffix, or infix, and identify the best series annotation responsible for the categorization."""
    n = len(affixes)
    pfx = []
    sfx = []
    ifx = []
    for i in range(n):
        terminal_annotation = affixes[i][2][-1]
        terminal_loss = terminal_annotation[:, 2]
        found_b = False
        found_y = False
        for (j, loss_pair) in enumerate(terminal_loss):
            split_loss_pair = loss_pair.split(sep)
            has_b = any(np.isin(split_loss_pair, b_anno))
            has_y = any(np.isin(split_loss_pair, y_anno))
            if has_b and has_y:
                continue 
                # if all boundaries look like this it's an infix (non-orientable)
            if has_b and not(found_b):
                pfx.append((i, j))
                found_b = True
            if has_y and not(found_y):
                sfx.append((i, j))
                found_y = True
        if not(found_b or found_y):
            ifx.append((i, -1))
    # construct orientations: the first column denotes the index of the affix, the second column denotes the index of the annotation. for infixes, the second column is -1 because infixes are identified by their lack of orientable annotation.
    # NOTE: right terminal edges are annotated from the reflected spectrum. a reflected b ion is a y ion, and likewise a reflected y ion is a b ion, so prefixes are recognized by b-b pairings and suffixes by y-y pairings.
    return (
        np.array(pfx,dtype=int).reshape((len(pfx),2)),
        np.array(sfx,dtype=int).reshape((len(sfx),2)),
        np.array(ifx,dtype=int).reshape((len(ifx),2)),
    )
    # reshape to stop empty arrs of shape (0,) gumming up concatenators 
    # likewise, force the dtype to be int even if it's empty.

def _refine_affixes(
    affixes: AnnotatedResiduePathSpace,
    consistent_anno: list[str],
    conflicting_anno: list[str],
    sep: str,
) -> tuple[np.ndarray,np.ndarray]:
    """Given affixes, consistent annotation symbols, and conflicting annotation symbols, refine the affixes into consistent and inconsistent categories."""
    n = len(affixes)
    cst = []
    cfl = []
    for i in range(n):
        terminal_annotation = affixes[i][2][-1]
        terminal_loss = terminal_annotation[:, 2]
        found_cst = False
        for (j, loss_pair) in enumerate(terminal_loss):
            split_loss_pair = loss_pair.split(sep)
            has_cst = any(np.isin(split_loss_pair, consistent_anno))
            has_cfl = any(np.isin(split_loss_pair, conflicting_anno))
            if has_cst and not(found_cst) and not(has_cfl):
                cst.append((i, j))
                found_cst = True
        if not(found_cst):
            cfl.append((i, -1))
    return (
        np.array(cst,dtype=int).reshape((len(cst),2)),
        np.array(cfl,dtype=int).reshape((len(cfl),2)),
    )

def refine_affixes(
    reverse_affixes: AnnotatedResiduePathSpace,
    forward_affixes: AnnotatedResiduePathSpace,
    b_anno: list[str],
    y_anno: list[str],
    sep: str = MISMATCH_SEPARATOR
) -> tuple[AnnotatedResiduePathSpace,np.ndarray,np.ndarray,np.ndarray]:
    """Given candidate prefixes as reverse affixes, candidate suffixes as forward affixes, and the b and y annotation symbols, orient as prefix, suffix, or infix, and identify the best series annotation responsible for the categorization. Returns the concatenated path space of reverse + forward affixes as well as the three categorizations."""
    pfx, inconsistent_pfx = _refine_affixes(reverse_affixes, b_anno, y_anno, sep)
    sfx, inconsistent_sfx = _refine_affixes(forward_affixes, y_anno, b_anno, sep)
    print(inconsistent_pfx, inconsistent_sfx)
    # construct component categories
    num_rev = len(reverse_affixes)
    affixes = reverse_affixes + forward_affixes
    # concatenate affixes
    sfx[:,0] += num_rev
    inconsistent_sfx[:,0] += num_rev
    # shift suffix indices to match concatenated affixes
    # unfortunately the cleanest way to express this operation is by modifying the arrays in-place. :(
    ifx = np.concat([inconsistent_pfx, inconsistent_sfx], dtype=int)
    # construct infixes
    return (
        affixes,
        pfx,
        sfx,
        ifx,
    )

def pair_affixes(
    pfx_idx: np.ndarray,
    sfx_idx: np.ndarray,
    aligned_affixes: AnnotatedResiduePathSpace,
    product_topology: WeightedProductGraph,
    pivot_topology: PivotGraph,
) -> np.ndarray:
    unraveled_pfx_nodes = np.array([
        product_topology.unravel(aligned_affixes.get_path(i)[0]) 
        for (i,_) in pfx_idx
    ])
    sfx_nodes = np.array([aligned_affixes.get_path(i)[0] for (i,_) in sfx_idx])
    # unpack and unravel initial nodes
    unraveled_pfx_clusters, pfx_clust_id = np.unique(unraveled_pfx_nodes, axis=0, return_inverse=True)
    sfx_clusters, sfx_clust_id = np.unique_inverse(sfx_nodes)
    # cluster nodes
    pivot_adj = pivot_topology.graph.adj
    pfx_adj_nodes = [[product_topology.ravel(v,x) for (v,x) in it.product(pivot_adj[u],pivot_adj[w])] for (u,w) in unraveled_pfx_clusters]
    pfx_adj_offset = np.cumsum([0,] + [len(x) for x in pfx_adj_nodes])
    pfx_adj_nodes = np.concat(pfx_adj_nodes)
    pfx_adj_sfx = np.searchsorted(pfx_adj_nodes, sfx_nodes)
    pfx_sfx_clust_pairs = [(i,j) for (i,(l,r)) in enumerate(it.pairwise(pfx_adj_offset)) for j in pfx_adj_sfx[l:r]]
    pfx_sfx_clust_pairs = np.unique(pfx_sfx_clust_pairs, axis=0)
    n_clust_pair = len(pfx_sfx_clust_pairs)
    print(n_clust_pair)
    print(pfx_sfx_clust_pairs)
    # produce cluster-level pairs via the actions of pivot_adj on the nodes
    n_pfx_clust = len(unraveled_pfx_clusters)
    n_sfx_clust = len(sfx_clusters)
    clustered_pfx_idx = [[] for _ in range(n_pfx_clust)]
    for (i,(p_idx,_)) in enumerate(pfx_idx):
        clustered_pfx_idx[pfx_clust_id[i]].append(p_idx)
    clustered_sfx_idx = [[] for _ in range(n_sfx_clust)]
    for (i,(s_idx,_)) in enumerate(sfx_idx):
        clustered_sfx_idx[sfx_clust_id[i]].append(s_idx)
    unraveled_sfx_clusters = [product_topology.unravel(i) for i in sfx_clusters]
    # group the affix indices by their node cluster membership
    affix_pairs = []
    for (i,(pfx_clust_id,sfx_clust_id)) in enumerate(pfx_sfx_clust_pairs):
        pfx_left_node, pfx_right_node = unraveled_pfx_clusters[pfx_clust_id]
        sfx_left_node, sfx_right_node = unraveled_sfx_clusters[sfx_clust_id]
        left_anno = pivot_topology.get_weight(pfx_left_node,sfx_right_node)
        right_anno = pivot_topology.get_weight(pfx_right_node,sfx_left_node)
        # associate pivot edge annotations to cluster pairs
        for p_idx in clustered_pfx_idx[pfx_clust_id]:
            for s_idx in clustered_sfx_idx[sfx_clust_id]:
                print(p_idx)
                print(s_idx)
                print(left_anno,right_anno)
                affix_pairs.append((p_idx,s_idx,left_anno,right_anno))
    # iterate the index-level pairs generated by each cluster-level pair
    return np.array(affix_pairs)
