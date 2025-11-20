import dataclasses
from .graphs.types import PivotGraph, PathSpace
from .costmodels import MISMATCH_SEPARATOR

def orient_affixes(
    affixes: PathSpace, # TODO, specifically a path space weighted by the OrderedResidueCostModel
    b_anno: list[str],
    y_anno: list[str],
    sep: str = MISMATCH_SEPARATOR
) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    if not(isinstance(affixes.state[0][0],list)):
        raise ValueError("PathSpace states must be OrderedResiduePathState.")
        # obviously not a complete check but will make noise if a SuffixArrayPathState is passed instead.
    n = len(affixes)
    pfx = []
    sfx = []
    ifx = []
    for i in range(n):
        terminal_annotation = affixes.state[i][-1][1]
        terminal_loss = terminal_annotation[:,2]
        found_b = False
        found_y = False
        for (j, loss_pair) in terminal_loss:
            split_loss_pair = loss_pair.split(sep)
            if any(np.isin(split_loss_pair,b_anno)) and not(found_b):
                pfx.append((i,j))
                found_b = True
            if any(np.isin(split_loss_pair,y_anno)) and not(found_y):
                sfx.append((i,j))
                found_y = True
        if not(found_b or found_y):
            ifx.append((i,-1))
    # construct orientations: the first column denotes the index of the affix, the second column denotes the index of the annotation. for infixes, the second column is -1 because infixes are identified by their lack of orientable annotation.
    # NOTE: right terminal edges are annotated from the reflected spectrum. a reflected b ion is a y ion, and likewise a reflected y ion is a b ion, so prefixes are recognized by b-b pairings and suffixes by y-y pairings.
    return (
        np.array(pfx),
        np.array(sfx),
        np.array(ifx),
    )

def pair_affixes(
    affixes: PathSpace,
    pivot_topology: PivotGraph,
) -> tuple[]:
    pass
