import numpy as np
from .graph_utils import nx, SingularPath, DualPath, GraphPair, unzip_dual_path, path_to_edges, find_edge_disjoint_dual_path_pairs
from .spectrum_graphs import GAP_KEY

#=============================================================================#

class Affix:

    def __init__(self, dual_path, translations):
        self._dual_path = dual_path
        self._translations = translations
    
    def path(self) -> DualPath:
        return self._dual_path

    def translate(self) -> tuple[str, str]:
        return self._translations

AffixPair = tuple[Affix, Affix]

#=============================================================================#

def create_affix(
    dual_path: DualPath,
    spectrum_graph_pair: GraphPair,
) -> Affix:
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
        _translate_dual_path(desc_path, desc_graph))

def _translate_singular_path(
    singular_path: SingularPath,
    spectrum_graph: nx.Digraph,
    weight_key = GAP_KEY,
) -> str:
    path_edges = path_to_edges(singular_path)
    return (spectrum_graph[i][j][weight_key] for (i,j) in path_edges)

#=============================================================================#

def filter_affixes(
    affixes: np.ndarray,
    path_to_suffix_array: str,
    occurrence_threshold: int = 0,
) -> np.ndarray:
    # admits an affix as long as one of its translations occurs in the suffix array 
    translations = np.array([afx.translate() for afx in affixes])
    asc_occurrences = _count_occurrences(translations[:, 0], path_to_suffix_array)
    desc_occurrences = _count_occurrences(translations[:, 1], path_to_suffix_array)
    return affixes[asc_occurrences + desc_occurrences > 0]

def _count_occurrences(
    biosequences: np.ndarray,
    path_to_suffix_array: str
) -> list[int]:
    # associates strings, assumed to be biosequence, to their occurrence in a suffix array.
    # TODO: these will be calls to pylibsufr 
    pass

#=============================================================================#

def find_affix_pairs(
    affixes: np.ndarray
) -> list[tuple[int,int]]:
    dual_paths = [afx.path() for afx in affixes]
    return find_edge_disjoint_dual_path_pairs(dual_paths)