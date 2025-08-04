from typing import Self
from dataclasses import dataclass
from abc import ABC, abstractmethod
import functools as ft
import itertools as it

@dataclass
class FragmentStateSpace:
    loss_masses: list[float]
    loss_symbols: list[str]
#    applicable_losses: list[list[int]] # amino_idx -> {loss_idx applicable to amino}.
    max_num_losses: int                # restricts the number of losses that can be applied to a fragment.
    charges: list[int]

    def n_losses(self) -> int:
        return len(self.loss_masses)

@dataclass
class FragmentState:
    peak_idx: int
    peak_mz: float
    loss_id: int
    loss_mass: float
    loss_symbol: str
    charge: int

    @classmethod
    def from_index(cls,
        peak_idx: int,
        peak_mz: float,
        loss_id: int,
        charge: int,
        state_space: FragmentStateSpace,
    ) -> Self:
        return cls(
            peak_idx = peak_idx,
            peak_mz = peak_mz,
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
        return [self.modification_masses[i] for i in self.applicable_modofications[i]]

@dataclass
class ResidueState:
    residue_mass: float
    amino_id: int
    amino_mass: float
    amino_symbol: str
    modification_id: int
    modification_mass: float
    modification_symbol: str

    @classmethod
    def from_index(cls,
        residue_mass: float,
        amino_id: int,
        modification_id: int,
        state_space: ResidueStateSpace,
    ) -> Self:
        return cls(
            residue_mass = residue_mass,
            amino_id = amino_id,
            amino_mass = state_space.amino_masses[amino_id],
            amino_symbol = state_space.amino_symbols[amino_id],
            modification_id = modification_id,
            modification_mass = state_space.modification_masses[modification_id],
            modification_symbols = state_space.modification_masses[modification_id])

class AbstractFragmentSolver(ABC)
    # abstract methods -- these must be implemented in child classes.
    @classmethod
    @abstractmethod
    def from_state_space(cls,
        mz: list[float],
        tolerance: float,
        state_space: tuple[FragmentStateSpace,FragmentStateSpace,ResidueStateSpace],
    ) -> Self:
        """Construct a fragment solver object from an m/z array, a match threshold float value, and a state space triple (FragmentStateSpace,FragmentStateSpace,ResidueStateSpace). If the first m/z is not zero, a new list mz_new will be created such that mz_new[0] = 0 and otherwise mz_new[i + 1] = mz_old[i]."""

    @abstractmethod
    def set_reference(self, idx: int) -> None:
        """Set the reference m/z by index."""

    @abstractmethod
    def set_query(self, idx: int) -> None:
        """Set the query m/z by index."""

    @abstractmethod
    def get_solutions(self) -> Iterator[tuple[FragmentState,FragmentState,ResidueState]]:
        """Iterate residue and fragment states that solve the query - reference value within a margin of error. The solution triple (FragmentState,FragmentState,ResidueState) is drawn from the state space triple. The first fragment state is a member of the first fragment state space, the second of the second, and the residue state is a member of the residue state space. Any constraints placed on those spaces will restrict the corresponding output."""

    # utility methods -- these do not need to be overridden.
    @staticmethod
    def _augment_peaks(
        peaks: list[float],
        state_space: FragmentStateSpace,
    ) -> tuple[list[float],list[tuple[int, int]]]:
        # if the peaks list does not include a dummy value 0. at index 0, create it.
        if peaks[0] != 0.:
            peaks = [0.] + peaks
        # construct and reorder the charge-augmented m/z
        mz_arr = np.array(peaks)
        augmented_arrs = [mz if (c == 1) else c * mz for c in state_space.charges]
        merged_mz_arr, deindexer, charge_table = util.merge_in_order(augmented_mz_arrays)
        # collect the deindexer,charge_table pairs as the index unraveler.
        return merged_mz_arr, list(zip(deindexer, charge_table))

    @staticmethod
    def _augment_masses(
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
            left_loss_ind, right_loss_ind, modification_ind = np.unravel_index(range(len(amino_aug_masses), amino_aug_tensor.shape))
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
            peak_mz: float,
            charge: int,
            state_space: FragmentStateSpace,
    ) -> Iterator[FragmentState]:
        return map(
            ft.partial(
                FragmentState.from_index,
                peak_idx = peak_idx,
                peak_mz = peak_mz,
                charge = charge
                state_space = state_space),
            loss_indices)

    @staticmethod
    def _generate_residue_states(
        amino_indices: Iterator[int],
        modification_indices: Iterator[int],
        residue_mass: float,
        state_space: ResidueStateSpace:
    ) -> Iterator[FragmentState]:
        residue_state_factory = ft.partial(
            ResidueState.from_index,
            residue_mass = residue_mass,
            state_space = state_space)
        return map(
            lambda x: residue_state_factory(*x),
            zip(amino_indices, modification_indices))

    @staticmethod
    def _filter_solutions(
        solutions: Iterator[tuple[FragmentState,FragmentState,ResidueState]]
    ) -> Iterator[tuple[FragmentState,FragmentState,ResidueState]]:
        return filter(
            lambda soln: soln[0].peak_idx != soln[1].peak_idx,
            solutions)

class BisectFragmentSolver(AbstractFragmentSolver):
    def __init__(self,
        state_space: tuple[FragmentStateSpace,FragmentStateSpace,ResidueStateSpace],
        tolerance: float,
        left_mz: list[float],
        left_mz_index_unraveler: list[tuple[int,int]] # -> (original mz idx, charge augment)
        right_mz: list[float],
        right_mz_index_unraveler: list[tuple[int,int]] # -> (original mz idx, charge augment)
        target_masses: list[float],
        target_mass_index_unraveler: list[tuple[int, int, int, int]], # -> (left loss, right loss, amino, modification)
    ):
        self._left_fragment_space, self._right_fragment_space, self._residue_space = state_space
        self._tolerance = tolerance
        self._left_n = len(left_mz)
        self._left_mz = left_mz
        self._left_mz_index_unraveler = left_mz_index_unraveler
        self._right_n = len(right_mz)
        self._right_mz = right_mz
        self._right_mz_index_unraveler = right_mz_index_unraveler
        self._m = len(target_masses)
        self._target_masses = target_masses
        self._target_mass_index_unraveler = target_mass_index_unraveler
    
    @classmethod
    def from_state_space(cls,
        mz: list[float],
        tolerance: float,
        state_space: tuple[FragmentStateSpace,FragmentStateSpace,ResidueStateSpace],
    ) -> Self:
        """Construct a fragment solver object from an m/z array, a match threshold float value, and a state space triple (FragmentStateSpace,FragmentStateSpace,ResidueStateSpace)."""
        left_fragment_space, right_fragment_space, residue_space = state_space
        left_mz, left_unraveler = augment_peaks(
            peaks = mz,
            state_space = left_fragment_space)
        right_mz, right_unraveler = augment_peaks(
            peaks = mz,
            state_space = right_fragment_space)
        target_masses, target_unraveler = augment_masses(
            state_space = state_space)
        return self(
            state_space = state_space,
            tolerance = tolerance,
            left_mz = left_mz,
            left_mz_index_unravler = left_unraveler,
            right_mz = right_mz,
            right_mz_index_unraveler = right_unraveler,
            target_masses = target_masses,
            target_mass_index_unraveler = target_unravler)

    def _bound_bisection(self, i) -> int:
        return max(0, min(self._m - 1, i))
    
    def _bisect_range(self, query: float) -> tuple[int, int]:
        return (self._bound_bisection(bisect_left(self._target_masses, query - self._tolerance)),
            self._bound_bisection(bisect_right(self._target_masses, query + self._tolerance)))

    def set_reference(self, idx: int) -> None:
        self._ref_mz = self._left_mz[idx]
        self._ref_peak_idx, self._ref_charge = self._left_mz_index_unraveler[idx]

    def set_query(self, idx: int) -> None:
        self._query_mz = self._right_mz[idx] 
        self._query_peak_idx, self._query_charge = self._right_mz_index_unraveler[idx]
        self._match_lo, self._match_hi = self._bisect_range(self._query_mz - self._ref_mz)

    def get_solutions(self) -> Iterator[tuple[FragmentState,FragmentState,ResidueState]]:
        left_loss_indices, right_loss_indices, amino_indices, modification_indices = zip(*map(
            lambda i: self._target_mass_index_unraveler[i],
            range(self._match_lo, self._match_hi + 1)))
        left_fragment_states = self._generate_fragment_states(
            loss_indices = left_loss_indices,
            peak_idx = self._ref_peak_idx,
            peak_mz = self._ref_mz,
            charge = self._ref_charge,
            state_space = self._left_fragment_space))
        right_fragment_states = self._generate_fragment_states(
            loss_indices = right_loss_indices,
            peak_idx = self._query_peak_idx,
            peak_mz = self._query_mz,
            charge = self._query_charge,
            state_space = self._right_fragment_space))
        residue_states = self._generate_residue_states(
            amino_indices = amino_indices,
            modification_indices = modification_indices,
            residue_mass = self._query_mz - self._ref_mz,
            state_space = self._residue_space)
        return self._filter_solutions(zip(left_fragment_states, right_fragment_states, residue_states))
