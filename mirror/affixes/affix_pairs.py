from .affix_types import Affix
from ..graph_utils import find_edge_disjoint_dual_path_pairs

import numpy as np

#=============================================================================#

def find_affix_pairs(
    affixes: np.ndarray
) -> list[tuple[int,int]]:
    """Lists the indices of pairs of affixes whose dual paths do not share any edges.

        np.array(find_edge_disjoint_dual_path_pairs(afx.path() for afx in affixes))
        
    :affixes: a numpy array of Affix objects."""
    return np.array(find_edge_disjoint_dual_path_pairs(afx.path() for afx in affixes))

AffixPair = tuple[Affix, Affix]