from ..util import residue_lookup
from ..graph_utils import nx, SingularPath, DualPath, GraphPair, unzip_dual_path, path_to_edges
from ..spectrum_graphs import GAP_KEY

#=============================================================================#

class Affix:
    """Interface to a dual path and its residue sequences, which correspond to 
    a potential affix (indeterminately either prefix or suffix) of a candidate sequence."""

    def __init__(self, dual_path, translations, called_sequence):
        self._dual_path = dual_path
        self._translations = translations
        self._called_sequence = called_sequence
    
    def path(self) -> DualPath:
        "The dual path underlying this Affix in the spectrum graph pair."
        return self._dual_path

    def translate(self) -> tuple[str, str]:
        "The pair of string types created by sequencing the edge weights in the spectrum graph pair along this Affix's dual path."
        return (''.join(self._translations[0]), ''.join(self._translations[1]))

    def reverse_translate(self) -> tuple[str, str]:
        "The pair of string types created by sequencing the edge weights in the spectrum graph pair along this Affix's dual path."
        return (''.join(self._translations[0][::-1]), ''.join(self._translations[1][::-1]))
    
    def call(self) -> str:
        return ''.join(self._called_sequence)
    
    def reverse_call(self) -> str:
        return ''.join(self._called_sequence[::-1])
    
    def __repr__(self):
        return f"""Affix(
    paths = {self._dual_path}
    translations = {self.translate()}
)"""

#=============================================================================#

def create_affix(
    dual_path: DualPath,
    spectrum_graph_pair: GraphPair,
) -> Affix:
    "Create an affix object from a dual_path and the spectrum_graph_pair which supports the paths."
    translations = _translate_dual_path(dual_path, spectrum_graph_pair)
    called_sequence = _call_sequence_from_translations(*translations)
    return Affix(dual_path, translations, called_sequence)

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
    return [residue_lookup(spectrum_graph[i][j][weight_key]) for (i,j) in path_edges]

def _call_sequence_from_translations(tr1: str, tr2: str):
    sequence = []
    for (res1, res2) in zip(tr1, tr2):
        if res1 == 'X' and res2 != 'X':
            sequence.append(res2)
        elif res1 != 'X' and res2 == 'X':
            sequence.append(res1) 
        elif res1 == res2:
            sequence.append(res1)
        else:
            sequence.append(f"{res1}/{res2}")
    return sequence