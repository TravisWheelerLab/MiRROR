import uuid
import pathlib

from .affix_types import Affix
from ..types import Iterator, Iterable
from ..util import mask_ambiguous_residues

import numpy as np
import numpy.typing as npt
from pylibsufr import read_sequence_file, SufrBuilderArgs, SuffixArray as SufrSuffixArray, SufrMetadata, CountOptions, CountResult

#=============================================================================#

class SuffixArray:
    """Wraps a `pylibsufr.SuffixArray` object, itself a binding to the `SuffixArray` struct in the `libsufr` Rust crate.
    For now, only the create, read, and count operations are implemented here."""
    def __init__(self,
        suffix_array: SufrSuffixArray,
        path_to_suffix_array: str
    ):
        self._suffix_array = suffix_array
        self._path = path_to_suffix_array

    @classmethod
    def _builder_args(cls, path_to_fasta: str, path_to_suffix_array: str = None, low_memory = False):
        if path_to_suffix_array == None:
            # if an output path wasn't passed, name the suffix array file after the fasta file.
            fasta_stem = pathlib.Path(path_to_fasta).stem
            path_to_suffix_array = f"./{fasta_stem}.sufr"
        # read the fasta into libsufr's internal representation.
        sequence_file_data = read_sequence_file(path_to_fasta)
        seq_str = ''.join(mask_ambiguous_residues(c) for c in sequence_file_data.seq().decode())    
        #print(f"SFD:\n\t{seq_str}")
        # construct the arguments
        sufr_builder_args = SufrBuilderArgs(
            seq_str.encode(),
            path_to_suffix_array,
            sequence_file_data.start_positions(),
            sequence_file_data.sequence_names(),
            is_dna = False,
            low_memory = low_memory,
        )
        return sufr_builder_args, path_to_suffix_array

    @classmethod
    def write(cls, path_to_fasta: str, path_to_suffix_array: str = None, low_memory = False):
        sufr_builder_args, path_to_suffix_array = cls._builder_args(path_to_fasta, path_to_suffix_array, low_memory = low_memory)
        return SufrSuffixArray.write(sufr_builder_args)

    @classmethod
    def read(cls, path_to_suffix_array: str, low_memory = False):
        """Read a suffix array from its file."""
        return cls(
            SufrSuffixArray.read(path_to_suffix_array, low_memory),
            path_to_suffix_array)

    @classmethod
    def create(cls, path_to_fasta: str, path_to_suffix_array: str = None, low_memory = False):
        """Create a suffix array from fasta records. 
        If no output path is given, the suffix array file will be named after the input fasta file."""
        sufr_builder_args, path_to_suffix_array = cls._builder_args(path_to_fasta, path_to_suffix_array, low_memory = low_memory)
        return cls(
            SufrSuffixArray(sufr_builder_args),
            path_to_suffix_array)
    
    def count(self, queries: Iterator[str]) -> list[int]:
        """Given an iterator of strings, return a list of the same size counting the occurrences of each query in the suffix array."""
        count_options = CountOptions(list(queries))
        count_results = self._suffix_array.count(count_options)
        occurrences = [res.count for res in count_results]
        #print(f"COUNT:\n\t{list(zip(queries,occurrences))}")
        return occurrences

#=============================================================================#

def filter_affixes(
    affixes: np.ndarray,
    suffix_array: SuffixArray,
    occurrence_threshold: int = 0,
) -> np.ndarray:
    """Removes affixes which do not have either translation appearing in a suffix array.
    
    :affixes: a numpy array of Affix objects.
    :path_to_suffix_array: str, path to a suffix array created by sufr.
    :occurrence_threshold: int, any Affix with less than or equal to this value is filtered out. defaults to 0; any Affix that occurs in the suffix array is kept."""
    if len(affixes) == 0:
        return np.array([])
    occurrence_mask = mask_nonoccurring_affixes(affixes, suffix_array, occurrence_threshold = occurrence_threshold)
    return affixes[occurrence_mask]

def mask_nonoccurring_affixes(
    affixes: Iterable[Affix],
    suffix_array: SuffixArray,
    occurrence_threshold: int = 0,
) -> np.ndarray:
    n = len(affixes)
    if n == 0:
        return np.array([])
    # admits an affix as long as one of its translations occurs in the suffix array 
    calls = np.array([afx.call() for afx in affixes])
    rev_calls = np.array([afx.reverse_call() for afx in affixes])
    
    queries = np.hstack([calls, rev_calls])
    query_occ = np.array(suffix_array.count(queries))

    call_occ = query_occ[:n]
    rev_call_occ = query_occ[n:]
    
    return (call_occ + rev_call_occ) > 0
