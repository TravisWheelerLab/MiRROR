from typing import Iterable
from dataclasses import dataclass 
from multiprocessing import Pool
import functools as ft
import itertools as it

from .spectra import PeakList, SpectrumRegistrationScoreModel, simulate_spectrum, register_spectra
from .sequences import SuffixArray, Candidate

from .alignment import AlignmentResult

@dataclass
class FiltrationParams:
    max_solvable_gap: int                                       # max gap length solvable via kmer mass table.
    suffix_array: SuffixArray                                   # kmer solution space
    fragment_state_space: FragmentStateSpace                    # ...
    simulation_model: SpectrumSimulationModel                   # spectrum simulator configuration
    registration_score_threshold: float                         # min spectrum registration score
    registration_score_model: SpectrumRegistrationScoreModel    # spectrum registration strategy
    candidate_score_threshold: float                            # min candidate score aggregate

class FiltrationResult:
    def __init__(self,
        index: Iterable[int],
        candidates: Iterable[Candidate],
    ):
        num_bins = index[-1] # index is sorted, so the last element is its max.
        self._candidates = [[] for _ in range(num_bins)]
        for (i, c) in zip(index, candidates):
            self._candidates[i].append(c)
        for i in range(num_bins):
            self._candidates[i].sort()

    def flattened(self) -> Iterator[Candidate]:
        return sorted(it.chain(self._candidates))

    def granular(self) -> Iterator[list[Candidate]]:
        return sorted(
            self._candidates,
            key = lambda x: max(c.score() for c in x))

def expand(
    alignment_result: AlignmentResult,
    params: FiltrationParams,
) -> Iterator[list[Candidate]]:
    for affix_pair in alignment_result:
        yield Candidate.construct_equivalence_class(
            affix_pair = affix_pair,
            suffix_array = params.suffix_array)
    
def filtrate(
    true_spectrum: PeakList,
    candidates: Iterator[list[Candidate]],
    params: FiltrationParams,
) -> FiltrationResult:
    # flatten the candidates into a first-order collection, but retain each candidate's position in the second-order collection.
    index, candidates_flat = map(
        np.array,
        zip(*it.chain.from_iterable(zip(
            it.repeat(i),
            c) for (i, c) in enumerate(candidates))))
    # apply the spectrum simulation model to generate a spectrum for each candidate.
    candidate_spectra = list(map(
        ft.partial(
            simulate_spectra,
            model = params.spectrum_simulation_model),
        map(
            lambda x: x.get_sequence(),
            candidates_flat)))
    # perform one-to-many registration from the true spectrum to each candidate spectrum. retain only the candidates w/ a registered spectrum.
    hit_ids, hit_scores = register_spectra(
        query = true_spectrum,
        targets = candidate_spectra,
        threshold = params.registration_score_threshold,
        model = params.registration_score_model)
    registered_candidates = candidates[hit_ids]
    registered_index = index[hit_ids]
    # rescore and filter the registered candidates
    final_candidates, final_index = rescore_candidates(
        candidates = registered_candidates,
        index = registered_index,
        registration_scores = hit_scores,
        score_threshold = params.candidate_score_threshold)
    # the FiltrationResult object will reorder and restructure the remaining candidates.
    return FiltrationResult(
        index = final_index,
        candidates = final_candidates)
