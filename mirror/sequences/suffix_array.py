import uuid
import pathlib
from dataclasses import dataclass

from typing import Iterator, Iterable

import numpy as np
import numpy.typing as npt
from pylibsufr import read_sequence_file, SufrBuilderArgs, SuffixArray as SufrSuffixArray, SufrMetadata, CountOptions, CountResult, BisectOptions, BisectResult

#=============================================================================#

class SuffixArray:
    """Wraps a `pylibsufr.SuffixArray` object, itself a binding to the `SuffixArray` 
    struct in the `libsufr` Rust crate. Only the `write`, `read`, `create`, `bisect`, 
    and `count` methods are implemented."""
    def __init__(self,
        suffix_array: SufrSuffixArray,
        path_to_suffix_array: str,
    ):
        self._suffix_array = suffix_array
        self._path = path_to_suffix_array
        self.query_low_memory = False
        self.max_query_len = None

    @classmethod
    def _builder_args(cls,
        path_to_fasta: str,
        path_to_suffix_array: str = None,
        low_memory = False
    ):
        if path_to_suffix_array == None:
            # if an output path wasn't passed, name the suffix array file after the fasta file.
            fasta_stem = pathlib.Path(path_to_fasta).stem
            path_to_suffix_array = f"./{fasta_stem}.sufr"
        # read the fasta into libsufr's internal representation.
        sequence_file_data = read_sequence_file(path_to_fasta)   
        seq_str = ''.join(c for c in sequence_file_data.seq().decode())
        #print(seq_str)
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
    def write(cls,
        path_to_fasta: str,
        path_to_suffix_array: str = None,
        low_memory = False,
    ):
        sufr_builder_args, path_to_suffix_array = cls._builder_args(path_to_fasta, path_to_suffix_array, low_memory = low_memory)
        return SufrSuffixArray.write(sufr_builder_args)

    @classmethod
    def read(cls,
        path_to_suffix_array: str,
        low_memory = False,
    ):
        """Read a suffix array from its file."""
        return cls(
            SufrSuffixArray.read(path_to_suffix_array, low_memory),
            path_to_suffix_array)

    @classmethod
    def create(cls, 
        path_to_fasta: str, 
        path_to_suffix_array: str = None, 
        low_memory = False,
    ):
        """Create a suffix array from fasta records. 
        If no output path is given, the suffix array 
        file will be named after the input fasta file."""
        sufr_builder_args, path_to_suffix_array = cls._builder_args(path_to_fasta, path_to_suffix_array, low_memory = low_memory)
        return cls(
            SufrSuffixArray(sufr_builder_args),
            path_to_suffix_array)
    
    def count(self, 
        queries: Iterator[str],
    ) -> list[int]:
        """Given an iterator of strings, return a list of the same size 
        counting the occurrences of each query in the suffix array."""
        count_options = CountOptions(
            queries = list(queries),
            max_query_len = self.max_query_len,
            low_memory = self.query_low_memory)
        count_results = self._suffix_array.count(count_options)
        return [res.count for res in count_results]
    
    def bisect(self, 
        queries: Iterator[str], 
        prefix_result: BisectResult = None,
    ) -> list[BisectResult]:
        """Given an iterator of strings and an optional prefix result,
        count the occurrences of the query strings. If a prefix result
        is passed, search is restricted to the range of the prefix.
        Unlike count, this function returns its result type, BisectResult.
        The occurrence quantities can be accessed via the `count` field of
        the BisectResult object."""
        return self._suffix_array.bisect(BisectOptions(
            queries = list(queries),
            max_query_len = self.max_query_len,
            low_memory = self.query_low_memory,
            prefix_result = prefix_result))