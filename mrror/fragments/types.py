import dataclasses
from typing import Self
# standard

import numpy as np

@dataclasses.dataclass
class FragmentStateSpace:
    loss_masses: np.ndarray
    # [float; m]
    loss_symbols: np.ndarray
    # [str; m]
    applicable_losses: np.ndarray
    # [[int; _]; m] 
    # amino idx -> {applicable loss indices}.
    charges: np.ndarray

    @classmethod
    def trivial(cls) -> Self:
        return cls(
            loss_masses = np.array([0.]),
            loss_symbols = np.array(['']),
            applicable_losses = [np.array([0]) for _ in range(20)],
            charges = np.array([1]))

    def n_losses(self) -> int:
        return len(self.loss_masses)

@dataclasses.dataclass
class ResidueStateSpace:
    amino_masses: np.ndarray
    # [float; k]
    amino_symbols: np.ndarray
    # [str; k]
    modification_masses: np.ndarray
    # [float; k]
    modification_symbols: np.ndarray
    # [str; k]
    applicable_modifications: list[np.ndarray] 
    # [[int; _]; k]
    # amino_idx -> {modification_idx applicable to amino}.
    max_num_modifications: int
    # restricts the number of modifications that can be applied to a residue.

    def n_aminos(self) -> int:
        return len(self.amino_masses)

    def n_modifications(self, amino_id: int) -> int:
        return len(self.applicable_modifications[amino_id])

    def get_modifications(self, amino_id: int) -> list[float]:
        return self.modification_masses[self.applicable_modifications[amino_id]]
