__version__ = "0.0.1"

"""FUNCTION NAMING CONVENTION
prefix      | meaning                                                               |
------------+-----------------------------------------------------------------------+
load_<X>    | open, read, and format data from storage into memory.                 |
read_<X>    | read and format data from an open file into memory.                   |
            | (generally wrapped y a load_<X> function.)                            |
save_<X>    | open, format, and write data from memory into storage.                |
write_<X>   | format and write data from memory into an opened file.                |
            | (generally wrapped by a save_<X> function.)                           |
find_<X>    | returns a list or numpy array of objects of type X that may be empty. |
create_<X>  | returns a single object of type X.                                    |
filter_<X>  | returns a subset from a list or numpy array of type X.                |"""             

from .io import load_spectrum_from_mzML, read_fasta_records, load_fasta_records, load_fasta_as_strings, write_strings_to_fasta, save_strings_as_fasta, write_peaks_to_csv, save_peaks_as_csv
from .preprocessing import create_spectrum_bins, filter_spectrum_bins

from .gaps import find_all_gaps
from .pivots import find_pivots
from .boundaries import find_boundary_peaks, create_augmented_spectrum

from .spectrum_graphs import create_spectrum_graph_pair
from .graph_utils import find_dual_paths, find_extended_paths, find_edge_disjoint_dual_path_pairs

from .affixes import create_affix, filter_affixes, find_affix_pairs
from .candidates import create_candidates, filter_candidate_sequences
