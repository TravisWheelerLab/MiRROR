from dataclasses import dataclass 
from multiprocessing import Pool
import functools as ft
import itertools as it

from .gaps import find_kmers
from .spectra import SpectrumAlignmentScoreModel, simulate_spectrum, align_spectra
from .sequences import SuffixArray

from .alignment import AlignmentResult

@dataclass
class FiltrationParams:
    suffix_array: SuffixArray                                      # 
    max_solvable_gap: int                                          # max gap length solvable via kmer mass table.
    spectrum_alignment_score_threshold: float                      # min spectrum alignment score
    spectrum_alignment_score_model: SpectrumAlignmentScoreModel    # spectrum alignment strategy
    candidate_score_threshold: float                               # min candidate score aggregate

def _expand(
    alignment_result: AlignmentResult,
    params: FiltrationParams,
) -> Iterator[Candidate]:
     pass   

def _filtrate(
    candidates: Iterator[Candidate],
    params: FiltrationParams,
) -> list[Candidate]:
    pass

def filtrate(
    alignments: Iterator[AlignmentResult],
    params: FiltrationParams,
) -> Iterator[list[Candidate]]:
    with Pool() as pool:
        candidates = pool.map(
            ft.partial(_expand, params = params),
            alignments)
        return pool.map(
            ft.partial(_filtrate, params = params),
            candidates)
