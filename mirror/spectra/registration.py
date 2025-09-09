from typing import Self
import dataclasses

from .types import PeakList, AnnotatedPeakList

import numpy as np
from fastdtw import fastdtw

@dataclasses.dataclass(slots=True)
class Registration:
    """The result of registering a query spectrum against a reference spectrum using fast dynamic time warping. Computes a score based on the properties of the aligned peaks."""
    query: PeakList
    query_registration: list[int]
    query_dels: list[int]
    ref: PeakList
    ref_registration: list[int]
    ref_dels: list[int]
    distance: float

    @classmethod
    def from_fastdtw(cls,
        query: PeakList,
        ref: PeakList,
        result: tuple[float,list[int,int]],
    ) -> Self:
        distance, registration = result
        query_registration = []
        query_dels = []
        ref_registration = []
        ref_dels = []
        prev_i = -1
        prev_j = -1
        for (i, j) in registration:
            query_registration.append(i)
            ref_registration.append(j)
            if i - prev_i > 1:
                query_dels.append(i)
            if j - prev_j > 1:
                ref_dels.append(j)
        return cls(
            query = query,
            query_registration = query_registration,
            query_dels = query_dels,
            ref = ref,
            ref_registration = ref_registration,
            ref_dels = ref_dels,
            distance = distance,
        )

    def decomposition(self) -> tuple[int,int,int,float]:
        pass
    
    def score(self, model=(1,1,1)) -> float:
        pass        

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
) -> list[Registration]:
    """Register many query peak lists to a reference peak list using fastdtw, parametized by a radius window.
    
    Args:
    - ref: AnnotatedPeakList, a reference spectrum.
    - queries: list[PeakList], a collection of query spectra.
    - radius: int = 1, optional parameter to increase the search window used by fastdtw.
    Returns:
    - list[Registration], the result of registering each query to the reference."""
    n = len(queries)
    return [Registration.from_fastdtw(
            query = queries[i],
            ref = ref,
            result = _register(queries[i].mz, ref.mz, radius),
        ) for i in range(n)]
