from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class FragmentState:
    peak_idx: float
    peak_mz: float
    loss_id: int
    loss_mass: float
    loss_symbol: str
    charge: int

@dataclass
class ResidueState:
    residue_mass: float
    amino_id: int
    amino_mass: float
    amino_symbol: str
    modification_id: int
    modification_mass: float
    modification_symbol: str

@dataclass
class FragmentStateSpace:
    residue_masses: list[float]
    residue_symbols: list[str]
    modification_masses: list[float]
    modification_symbols: list[str]
    applicable_modifications: list[list[int]] # residue_idx -> {modification_idx applicable to residue}
    loss_masses: list[float]
    loss_symbols: list[str]
    applicable_losses: list[list[int]] # residue_idx -> {loss_idx applicable to residue}
    charges: list[int]

class AbstractFragmentSolver(ABC)
    @abstractmethod
    def set_reference(self, mz: float = 0., annotate_losses = False, annotate_charge = False):
        """Set the reference m/z and state space. If nothing is passed, defaults to mz=0, annotate_losses = False, annotate_charge = False, so any queries will be solved as singletons without loss or charge annotations on the reference, such as with *BoundaryFragment types. If values are passed, loss and charge will be solved for the reference as well as the query."""
    @abstractmethod
    def set_query(self, mz: float):
        """Set the query m/z. Loss and charge will always be annotated for this value. If previously set, the reference m/z is subtracted from the query m/z to construct the mass from which a ResidueState is resolved."""
    @abstract
    def get_solutions(self) -> Iterator[tuple[ResidueState, FragmentState, FragmentState]]:
        """Iterate residue and fragment states that solve the query - reference value within a margin of error. Uses query and reference data to solve for a ResidueState and one or two FragmentStates, depending on the reference state space. If no reference was passed, the second FragmentState is trivial and can be discarded."""
