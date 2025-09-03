from typing import Self
import dataclasses

from .types import PeakList, AnnotatedPeakList

import numpy as np
from fastdtw import fastdtw

@dataclasses.dataclass(slots=True)
class SpectrumRegistration:
    """The result of registering a query spectrum against a reference spectrum using fast dynamic time warping. Computes a score based on the properties of the aligned peaks."""
    query_idx: int
    query: PeakList
    query_registration: list[int]
    ref: PeakList
    ref_registration: list[int]
    distance: float

    @classmethod
    def from_fastdtw(cls,
        query_idx: int,
        query: PeakList,
        ref: PeakList,
        result: tuple[float,list[int,int]],
    ) -> Self:
        distance, registration = result
        query_registration, ref_registration = zip(*registration)
        return cls(
            query_idx = query_idx,
            query = query,
            query_registration = list(query_registration),
            ref = ref,
            ref_registration = list(ref_registration),
            distance = distance,
        )

    def score(self) -> float:
        """Aggregates the raw distance score of the registration with peak data from both spectra."""
        # TODO: need to rework .types, merge what is now AnnotatedPeakList into PeakList, change the BenchmarkPeakList inheritance accordingly, and rewrite AnnotatedPeakList as an actual annotation, as in, the result of the applying the annotation step to the peaks. it should associate scores to peaks by the intersecting fragment states. for now, this is just a weighted sum of intensities.
        return (self.distance 
            + (np.sqrt(
                self.query.intensity[self.query_registration] 
                + self.ref.intensity[self.ref_registration])
            ).sum())

def _register(
    ref: np.array,
    query: np.array,
    radius: int,
) -> tuple[float,list[int]]:
    return fastdtw(
        x = ref,
        y = query,
        radius = radius,
        dist = lambda x,y: 2 * abs(x - y) / (x + y),
    )

def register_spectra(
    ref: PeakList,
    queries: list[PeakList],
    radius: int = 1,
) -> list[SpectrumRegistration]:
    """Register many query peak lists to a reference peak list using fastdtw, parametized by a radius window.
    
    Args:
    - ref: AnnotatedPeakList, a reference spectrum.
    - queries: list[PeakList], a collection of query spectra.
    - radius: int = 1, optional parameter to increase the search window used by fastdtw.
    Returns:
    - list[SpectrumRegistration], the result of registering each query to the reference."""
    n = len(queries)
    return [SpectrumRegistration.from_fastdtw(
            query_idx = i,
            query = queries[i],
            ref = ref,
            result = _register(queries[i].mz, ref.mz, radius),
        ) for i in range(n)]
