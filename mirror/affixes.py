#

import numpy as np

from .util import residue_lookup
from .graph_utils import nx, SingularPath, DualPath, GraphPair, unzip_dual_path, path_to_edges, find_edge_disjoint_dual_path_pairs
from .spectrum_graphs import GAP_KEY

#=============================================================================#

class Affix:
    """Interface to a dual path and its residue sequences, which correspond to 
    a potential affix (indeterminately either prefix or suffix) of a candidate sequence."""

    def __init__(self, dual_path, translations):
        self._dual_path = dual_path
        self._translations = translations
    
    def path(self) -> DualPath:
        "The dual path underlying this Affix in the spectrum graph pair."
        return self._dual_path

    def translate(self) -> tuple[str, str]:
        "The pair of string types created by sequencing the edge weights in the spectrum graph pair along this Affix's dual path."
        return self._translations
    
    def __repr__(self):
        return f"""Affix(
    paths = {self._dual_path}
    translations = {self.translate()}
)"""

AffixPair = tuple[Affix, Affix]

#=============================================================================#

def create_affix(
    dual_path: DualPath,
    spectrum_graph_pair: GraphPair,
) -> Affix:
    "Create an affix object from a dual_path and the spectrum_graph_pair which supports the paths."
    translations = _translate_dual_path(dual_path, spectrum_graph_pair)
    return Affix(dual_path, translations)

# the following two functions could probably be a lot faster written as a comprehension of a 2-array
def _translate_dual_path(
    dual_path: DualPath,
    spectrum_graph_pair: GraphPair,
) -> tuple[str, str]:
    # translates a dual path to a sequence-like object
    asc_graph, desc_graph = spectrum_graph_pair
    asc_path, desc_path = unzip_dual_path(dual_path)
    return (
        _translate_singular_path(asc_path, asc_graph), 
        _translate_singular_path(desc_path, desc_graph))

def _translate_singular_path(
    singular_path: SingularPath,
    spectrum_graph: nx.DiGraph,
    weight_key = GAP_KEY,
) -> str:
    path_edges = path_to_edges(singular_path)
    return ' '.join([residue_lookup(spectrum_graph[i][j][weight_key]) for (i,j) in path_edges])

#=============================================================================#

def filter_affixes(
    affixes: np.ndarray,
    path_to_suffix_array: str,
    occurrence_threshold: int = 0,
) -> np.ndarray:
    """Removes affixes which do not have either translation appearing in a suffix array.
        
        NOT YET IMPLEMENTED.
    
    :affixes: a numpy array of Affix objects.
    :path_to_suffix_array: str, path to a suffix array created by sufr.
    :occurrence_threshold: int, any Affix with less than or equal to this value is filtered out. defaults to 0: any Affix that occurs in the suffix array is kept."""
    if len(affixes) == 0:
        return np.array([])
    # admits an affix as long as one of its translations occurs in the suffix array 
    translations = np.array([afx.translate() for afx in affixes])
    asc_occurrences = _count_occurrences(translations[:, 0], path_to_suffix_array)
    desc_occurrences = _count_occurrences(translations[:, 1], path_to_suffix_array)
    occurrence_mask = asc_occurrences + desc_occurrences > 0
    return affixes[occurrence_mask]

def _count_occurrences(
    biosequences: np.ndarray,
    path_to_suffix_array: str
) -> list[int]:
    # associates strings, assumed to be biosequence, to their occurrence in a suffix array.
    # TODO: these will be calls to pylibsufr 
    return np.ones_like(biosequences)

#=============================================================================#

def find_affix_pairs(
    affixes: np.ndarray
) -> list[tuple[int,int]]:
    """Lists the indices of pairs of affixes whose dual paths do not share any edges.

        find_edge_disjoint_dual_path_pairs(afx.path() for afx in affixes)
        
    :affixes: a numpy array of Affix objects."""
    dual_paths = [afx.path() for afx in affixes]
    return find_edge_disjoint_dual_path_pairs(dual_paths)