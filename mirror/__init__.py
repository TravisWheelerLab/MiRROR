__version__ = "0.0.1"

"""FUNCTION NAMING CONVENTION
prefix      | meaning  
------------+-------------------------------------------------------
find_<X>    | returns a list or numpy array of objects of type X that may be empty.
create_<X>  | returns a single object of type X.
filter_<X>  | returns a subset from a list or numpy array of type X."""

from .gaps import find_gaps
from .pivots import find_pivots
from .boundaries import find_boundary_peaks, create_augmented_spectrum
from .spectrum_graphs import create_spectrum_graph_pair
from .graph_utils import find_dual_paths, find_extended_paths, find_edge_disjoint_dual_path_pairs

from .affixes import create_affix, filter_affixes, find_affix_pairs
from .candidates import create_candidates, filter_candidate_sequences
