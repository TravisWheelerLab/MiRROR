import dataclasses

from .graphs.types import PivotGraph
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
    affixes: AnnotatedResiduePathSpace,
    pivot_topology: PivotGraph,
) -> tuple[None]:
    pass
