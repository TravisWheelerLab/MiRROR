import dataclasses
from typing import Self, Iterable, Iterator

from ..util import decharge_peaks

from .simulation import generate_fragment_spectrum, list_mz, list_intensity, list_ion_data, simulate_pivot, DEFAULT_PARAM, COMPLEX_PARAM

import numpy as np

@dataclasses.dataclass(slots=True)
class SpectraParams:
    initial_b: bool

    @classmethod
    def from_dict(cls, cfg: dict):
        return cls(
            initial_b = cfg['initial_b']
        )

@dataclasses.dataclass(slots=True)
class Peaks:
    mz: np.ndarray
    # [float; n]
    intensity: np.ndarray
    # [float; n]

    def __len__(self) -> int:
        return len(self.mz)

    @classmethod
    def from_data(cls,
        mz: Iterable,
        intensity: Iterable = None,
    ) -> Self:
        return cls(
            mz = np.array(mz),
            intensity = np.array(intensity) if intensity else np.ones_like(mz),
        )

@dataclasses.dataclass(slots=True)
class SimulatedPeaks(Peaks):
    mz: np.ndarray          # [float; n]
    intensity: np.ndarray   # [float; n]
    ion_data: np.ndarray    # [str; n]
    pivot: float
    peptide: str

    # TODO 2: should probably use an enum for this
    @classmethod
    def from_peptide(cls,
        peptide: str,
        mode: str = 'simple',
        initial_b: bool = False,
    ) -> Self:
        param = COMPLEX_PARAM if mode == "complex" else DEFAULT_PARAM
        if initial_b:
            param.setValue("add_first_prefix_ion", "true")
        spectrum = generate_fragment_spectrum(
            peptide,
            param,
            max_charge= 1 if mode == "simple" else 3,
        ) 
        return cls(
            mz = list_mz(spectrum),
            intensity = list_intensity(spectrum),
            ion_data = list_ion_data(spectrum),
            pivot = simulate_pivot(peptide),
            peptide = peptide,
        )

@dataclasses.dataclass(slots=True)
class PeaksDataset:
    _data: list[Peaks]
    _input: str

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, i) -> Peaks:
        return self._data[i]

    def __iter__(self) -> Iterator[Peaks]:
        return self._data.__iter__()
    
    @classmethod
    def from_peptide(cls,
        input: str,
        initial_b: bool = False,
    ) -> Self:
        return cls(
            _data = [SimulatedPeaks.from_peptide(*input.split('::'), initial_b=initial_b)],
            _input = input,
        )

    @classmethod
    def from_fasta(cls,
        input: str, 
        **kwargs,
    ) -> Self:
        pass
        
    @classmethod
    def from_mgf(cls,
        input: str, 
        **kwargs,
    ) -> Self:
        pass
        
    @classmethod
    def from_mzlib(cls,
        input: str, 
        **kwargs,
    ) -> Self:
        pass
        
@dataclasses.dataclass(slots=True)
class AugmentedPeaks(Peaks):
    mz: np.ndarray
    # [float; n]
    intensity: np.ndarray
    # [float; n]
    indices: np.ndarray
    # [int; n]
    charges: np.ndarray
    # [int; n]

    def get_original_indices(self,
        augmented_indices: np.ndarray,
    ) -> np.ndarray:
        return self.indices[augmented_indices]

    def get_augmenting_charges(self,
        augmented_indices: np.ndarray,
    ) -> np.ndarray:
        return self.charges[augmented_indices]

    @classmethod
    def from_data(cls,
        mz: Iterable[float],
        intensity: Iterable[float],
        indices: Iterable[int],
        charges: Iterable[int],
    ) -> Self:
        return cls(
            mz = np.array(mz),
            intensity = np.array(mz),
            indices = np.array(indices),
            charges = np.array(charges),
        )

    @classmethod
    def from_peaks(cls,
        peaks: Peaks,
        charges: np.ndarray = None,
        pivot_point: float = None,
    ) -> Self:
        reflection = None
        if pivot_point:
            reflector = 2 * pivot_point
            reflection = lambda x: reflector - x
        # construct the optional reflection

        mz, deindexer, charge_table = decharge_peaks(
            peaks.mz,
            charges,
            transformation=reflection,
        )
        # decharge peaks

        intensity = peaks.intensity[deindexer]
        # remap intensity to the new peaks

        return cls(
            mz = mz,
            intensity = intensity,
            indices = deindexer,
            charges = charge_table
        )
