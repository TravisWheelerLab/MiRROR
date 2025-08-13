from typing import Self, Iterator, Iterable, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
from bisect import bisect_left, bisect_right
import functools as ft
import itertools as it
from .. import util

import numpy as np

@dataclass
class FragmentStateSpace:
    loss_masses: list[float]
    loss_symbols: list[str]
#    applicable_losses: list[list[int]] # amino_idx -> {loss_idx applicable to amino}.
#    max_num_losses: int                # restricts the number of losses that can be applied to a fragment.
    charges: list[int]

    @classmethod
    def trivial(cls) -> Self:
        return cls(
            loss_masses = [0.],
            loss_symbols = [''],
            charges = [1])

    def n_losses(self) -> int:
        return len(self.loss_masses)

@dataclass
class FragmentState:
    peak_idx: int
    fragment_mass: float
    loss_id: int
    loss_mass: float
    loss_symbol: str
    charge: int

    def __eq__(self, other: Self) -> bool:
        return (self.fragment_mass - other.fragment_mass < 1e10) and (self.loss_id == other.loss_id) and (self.charge == other.charge)

    @classmethod
    def from_index(cls,
        peak_idx: int,
        fragment_mass: float,
        loss_id: int,
        charge: int,
        state_space: FragmentStateSpace,
    ) -> Self:
        return cls(
            peak_idx = peak_idx,
            fragment_mass = fragment_mass,
            loss_id = loss_id,
            loss_mass = state_space.loss_masses[loss_id],
            loss_symbol = state_space.loss_symbols[loss_id],
            charge = charge)
        
@dataclass
class ResidueStateSpace:
    amino_masses: list[float]
    amino_symbols: list[str]
    modification_masses: list[float]
    modification_symbols: list[str]
    applicable_modifications: list[list[int]] # amino_idx -> {modification_idx applicable to amino}.
    max_num_modifications: int                # restricts the number of modifications that can be applied to a residue.

    def n_aminos(self) -> int:
        return len(self.amino_masses)

    def n_modifications(self, amino_id: int) -> int:
        return len(self.applicable_modifications[amino_id])

    def get_modifications(self, amino_id: int) -> list[float]:
        return [self.modification_masses[i] for i in self.applicable_modifications[amino_id]]

@dataclass
class ResidueState:
    residue_mass: float
    amino_id: int
    amino_mass: float
    amino_symbol: str
    modification_id: int
    modification_mass: float
    modification_symbol: str

    def __eq__(self, other: Self) -> bool:
        return (self.amino_id == other.amino_id) and (self.modification_id == other.modification_id)

    @classmethod
    def from_index(cls,
        residue_mass: float,
        amino_id: int,
        modification_id: int,
        state_space: ResidueStateSpace,
    ) -> Self:
        global_modification_id = state_space.applicable_modifications[amino_id][modification_id]
        return cls(
            residue_mass = residue_mass,
            amino_id = amino_id,
            amino_mass = state_space.amino_masses[amino_id],
            amino_symbol = state_space.amino_symbols[amino_id],
            modification_id = modification_id,
            modification_mass = state_space.modification_masses[global_modification_id],
            modification_symbol = state_space.modification_masses[global_modification_id])

class AbstractFragmentSolver(ABC):

    # abstract methods -- these must be implemented in child classes.

    @classmethod
    @abstractmethod
    def from_state_space(cls,
        mz: list[float],
        tolerance: float,
        state_space: tuple[FragmentStateSpace,FragmentStateSpace,ResidueStateSpace],
        target_mass_data: tuple[list[float],list[tuple[int,int,int,int]]] = None,
    ) -> Self:
        """Construct a fragment solver object from an m/z array, a match threshold float value, and a state space triple (FragmentStateSpace,FragmentStateSpace,ResidueStateSpace). If target_mass_data is passed, that value will be used instead of constructing a new target mass list and deindexer from the state space."""

    @classmethod
    def from_half_space(cls,
        mz: Iterable[float],
        tolerance: float,
        state_space: tuple[FragmentStateSpace,ResidueStateSpace],
        transformation: Callable = None,
        target_mass_data: tuple[list[float],list[tuple[int,int,int,int]]] = None,
    ) -> Self:
        """Construct a fragment solver from a right fragment space and a residue space. The left mz and left state space are assumed to be trivial, meaning every reference fragment will be set to mz 0, charge 1, with no losses."""

    @abstractmethod
    def n_reference(self) -> int:
        """Return the number of reference indices."""

    @abstractmethod
    def n_query(self) -> int:
        """Return the number of query indices."""

    @abstractmethod
    def set_reference(self, idx: int) -> tuple[int,int]:
        """Set the reference mass by index. Returns the unraveled (peak index, charge) pair."""

    @abstractmethod
    def set_query(self, idx: int) -> tuple[int,int,float]:
        """Set the query mass by index. Returns the unraveled peak index and charge, as well as the mass delta from the reference m/z."""

    @abstractmethod
    def get_solutions(self) -> Iterator[tuple[FragmentState,FragmentState,ResidueState]]:
        """Iterate residue and fragment states that solve the query - reference value within a margin of error. The solution triple (FragmentState,FragmentState,ResidueState) is drawn from the state space triple. The first fragment state is a member of the first fragment state space, the second of the second, and the residue state is a member of the residue state space. Any constraints placed on those spaces will restrict the corresponding output."""

    # utility methods -- these do not need to be overridden.

    @staticmethod
    def _decharge_peaks(
        peaks: list[float],
        state_space: FragmentStateSpace,
        transformation: Callable = None,
    ) -> tuple[list[float],list[tuple[int, int]]]:
        # construct and reorder the charge-augmented m/z
        mz_arr = np.array(peaks)
        decharged_arrs = [mz_arr if (c == 1) else (c * mz_arr) - (c - 1) for c in state_space.charges]
        merged_decharged_arr, deindexer, charge_table = util.merge_in_order(decharged_arrs, transformation)
        charge_table = charge_table + 1
        # collect the deindexer,charge_table pairs as the index unraveler.
        return merged_decharged_arr, list(zip(deindexer, charge_table))

    @staticmethod
    def augment_masses(
        state_space: tuple[FragmentStateSpace,FragmentStateSpace,ResidueStateSpace],
    ) -> tuple[list[float],list[tuple[int,int,int,int]]]:
        left_fragment_space, right_fragment_space, residue_space = state_space
        # generate the unconditional augments as the 3-tensor sum of orthogonal amino, left loss, and right loss vectors.
        # it is shaped/typed as a 4-tensor because we'll need the extra dimension later when adding in the conditional
        # modification augments.
        amino_masses = np.array(residue_space.amino_masses).reshape(residue_space.n_aminos(), 1, 1, 1)
        left_loss_masses = np.array(left_fragment_space.loss_masses).reshape(1, left_fragment_space.n_losses(), 1, 1)
        right_loss_masses = np.array(right_fragment_space.loss_masses).reshape(1, 1, right_fragment_space.n_losses(), 1)
        ## recall: the residue mass = right fragment mass - left fragment mass. that means:
        ## left loss decreases the left fragment mass, which increases the observed residue mass.
        ## right loss decreases the right fragment mass, which decreases the observed residue mass.
        unconditional_augment_tensor = amino_masses + left_loss_masses - right_loss_masses
        # iterate along the amino (first) axis, construct the conditional augment vectors and apply them to the unconditional augment matrix associated to each amino id.
        augmented_masses = []
        augmented_indices = []
        for amino_id in range(residue_space.n_aminos()):
            unc_aug_matrix = unconditional_augment_tensor[amino_id, :, :, :]
            con_aug_vector = np.array(residue_space.get_modifications(amino_id)).reshape(1, 1, 1, residue_space.n_modifications(amino_id))
            amino_aug_tensor = unc_aug_matrix + con_aug_vector
            amino_aug_masses = amino_aug_tensor.flatten()
            augmented_masses.extend(amino_aug_masses)
            _, left_loss_ind, right_loss_ind, modification_ind = np.unravel_index(range(len(amino_aug_masses)), amino_aug_tensor.shape)
            amino_aug_indices = zip(it.repeat(amino_id), left_loss_ind, right_loss_ind, modification_ind)
            augmented_indices.extend(amino_aug_indices)
        # argsort by augmented mass and reorder the indices accordingly to construct the index unaveler.
        augmented_masses = np.array(augmented_masses)
        augmented_indices = np.array(augmented_indices)
        index_key = np.argsort(augmented_masses)
        return augmented_masses[index_key], augmented_indices[index_key]
   
    @staticmethod
    def _generate_fragment_states(
            loss_indices: Iterator[int],
            peak_idx: int,
            fragment_mass: float,
            charge: int,
            state_space: FragmentStateSpace,
    ) -> Iterator[FragmentState]:
        return map(
            lambda loss_id: FragmentState.from_index(
                peak_idx = int(peak_idx),
                fragment_mass = float(fragment_mass),
                loss_id = int(loss_id),
                charge = int(charge),
                state_space = state_space),
            loss_indices)

    @staticmethod
    def _generate_residue_states(
        amino_indices: Iterator[int],
        modification_indices: Iterator[int],
        residue_mass: float,
        state_space: ResidueStateSpace,
    ) -> Iterator[FragmentState]:
        return map(
            lambda x: ResidueState.from_index(
                residue_mass = float(residue_mass),
                amino_id = int(x[0]),
                modification_id = int(x[1]),
                state_space = state_space),
            zip(amino_indices, modification_indices))

    @staticmethod
    def _filter_solutions(
        solutions: Iterator[tuple[FragmentState,FragmentState,ResidueState]]
    ) -> Iterator[tuple[FragmentState,FragmentState,ResidueState]]:
        for soln in solutions:
            not_loop = (soln[0].fragment_mass != soln[1].fragment_mass)
            nontrivial_loss = (soln[0].loss_id != soln[1].loss_id)
            trivial_left_loss = (soln[0].loss_id == 0)
            if not_loop and (nontrivial_loss or trivial_left_loss):
                yield soln

class BisectFragmentSolver(AbstractFragmentSolver):
    def __init__(self,
        state_space: tuple[FragmentStateSpace,FragmentStateSpace,ResidueStateSpace],
        tolerance: float,
        left_mass: list[float],
        left_mass_index_unraveler: list[tuple[int,int]], # -> (original mz idx, charge augment)
        right_mass: list[float],
        right_mass_index_unraveler: list[tuple[int,int]], # -> (original mz idx, charge augment)
        target_masses: list[float],
        target_mass_index_unraveler: list[tuple[int,int,int,int]], # -> (left loss, right loss, amino, modification)
    ):
        self._left_fragment_space, self._right_fragment_space, self._residue_space = state_space
        self._tolerance = tolerance
        self._left_n = len(left_mass)
        self._left_mass = left_mass
        self._left_mass_index_unraveler = left_mass_index_unraveler
        self._right_n = len(right_mass)
        self._right_mass = right_mass
        self._right_mass_index_unraveler = right_mass_index_unraveler
        self._m = len(target_masses)
        self._target_masses = target_masses
        self._target_mass_index_unraveler = target_mass_index_unraveler
    
    @classmethod
    def from_state_space(cls,
        mz: Iterable[float],
        tolerance: float,
        state_space: tuple[FragmentStateSpace,FragmentStateSpace,ResidueStateSpace],
        target_mass_data: tuple[list[float],list[tuple[int,int,int,int]]] = None,
    ) -> Self:
        """Construct a fragment solver object from an m/z array, a match threshold float value, and a state space triple (FragmentStateSpace,FragmentStateSpace,ResidueStateSpace)."""
        left_fragment_space, right_fragment_space, residue_space = state_space
        left_mass, left_unraveler = cls._decharge_peaks(
            peaks = mz,
            state_space = left_fragment_space)
        right_mass, right_unraveler = cls._decharge_peaks(
            peaks = mz,
            state_space = right_fragment_space)
        target_masses, target_unraveler = cls.augment_masses(state_space) if target_mass_data is None else target_mass_data
        return cls(
            state_space = state_space,
            tolerance = tolerance,
            left_mass = left_mass,
            left_mass_index_unraveler = left_unraveler,
            right_mass = right_mass,
            right_mass_index_unraveler = right_unraveler,
            target_masses = target_masses,
            target_mass_index_unraveler = target_unraveler)

    @classmethod
    def from_half_space(cls,
        mz: Iterable[float],
        tolerance: float,
        state_space: tuple[FragmentStateSpace,ResidueStateSpace],
        transformation: Callable = None,
        target_mass_data: tuple[list[float],list[tuple[int,int,int,int]]] = None,
    ) -> Self:
        state_space = (FragmentStateSpace.trivial(), *state_space)
        left_fragment_space, right_fragment_space, residue_space = state_space
        left_mass, left_unraveler = cls._decharge_peaks(
            peaks = [0.],
            state_space = left_fragment_space)
        right_mass, right_unraveler = cls._decharge_peaks(
            peaks = mz,
            state_space = right_fragment_space,
            transformation = transformation)
        target_masses, target_unraveler = cls.augment_masses(state_space) if target_mass_data is None else target_mass_data
        return cls(
            state_space = state_space,
            tolerance = tolerance,
            left_mass = left_mass,
            left_mass_index_unraveler = left_unraveler,
            right_mass = right_mass,
            right_mass_index_unraveler = right_unraveler,
            target_masses = target_masses,
            target_mass_index_unraveler = target_unraveler)

    def get_extremal_masses(self) -> tuple[float,float]:
        return (
            self._target_masses[0] - self._tolerance,
            self._target_masses[-1] + self._tolerance)

    def n_reference(self) -> int:
        return len(self._left_mass)

    def n_query(self) -> int:
        return len(self._right_mass)

    def _bound_bisection(self, i) -> int:
        return max(0, min(self._m - 1, i))
    
    def _bisect_range(self, query: float) -> tuple[int, int]:
        return (self._bound_bisection(bisect_left(self._target_masses, query - self._tolerance)),
            self._bound_bisection(bisect_right(self._target_masses, query + self._tolerance)))

    def set_reference(self, idx: int) -> tuple[int,int]:
        self._ref_mass = self._left_mass[idx]
        self._ref_peak_idx, self._ref_charge = self._left_mass_index_unraveler[idx]
        return self._ref_peak_idx, self._ref_charge

    def set_query(self, idx: int) -> tuple[int,int,float]:
        self._query_mass = self._right_mass[idx] 
        self._query_peak_idx, self._query_charge = self._right_mass_index_unraveler[idx]
        self._query_mass_delta = self._query_mass - self._ref_mass
        return self._query_peak_idx, self._query_charge, self._query_mass_delta

    def get_solutions(self) -> Iterator[tuple[FragmentState,FragmentState,ResidueState]]:
        match_lo, match_hi = self._bisect_range(self._query_mass_delta)
        if match_hi <= match_lo:
            return []
        else:
            amino_indices, left_loss_indices, right_loss_indices, modification_indices = zip(*map(
                lambda i: self._target_mass_index_unraveler[i],
                range(match_lo, match_hi)))
            left_fragment_states = self._generate_fragment_states(
                loss_indices = left_loss_indices,
                peak_idx = self._ref_peak_idx,
                fragment_mass = self._ref_mass,
                charge = self._ref_charge,
                state_space = self._left_fragment_space)
            right_fragment_states = self._generate_fragment_states(
                loss_indices = right_loss_indices,
                peak_idx = self._query_peak_idx,
                fragment_mass = self._query_mass,
                charge = self._query_charge,
                state_space = self._right_fragment_space)
            residue_states = self._generate_residue_states(
                amino_indices = amino_indices,
                modification_indices = modification_indices,
                residue_mass = self._query_mass_delta,
                state_space = self._residue_space)
            return self._filter_solutions(zip(left_fragment_states, right_fragment_states, residue_states))
