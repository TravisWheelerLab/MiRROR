from typing import Type, Iterable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from bisect import bisect_left, bisect_right

import numpy as np

@dataclass
class MassTransformation:
    """A collection of parameters describing the loss, charge, and modification
    states of a pair of peaks whose difference corresponds to a transformation
    of a residue mass. Other than the residue index, values of 0 imply the null
    state, e.g., charge +1, no neutral losses, or no modification."""
    # index within the charge-augmented peak list
    inner_index: tuple[int, int]
    # index into the original peak list
    peaks: tuple[int, int]
    peaks_mass: tuple[float, float]
    # residue mass without transformation
    residue: int
    residue_mass: float
    # post-translational modification
    modification: int
    modification_mass: float
    # neutral losses
    losses: tuple[int, int]
    losses_mass: tuple[float, float]
    # charge states
    charge_states: tuple[int, int]
    # difference between the transformed left peak and the observed right peak
    mass_error: float

    @classmethod
    def dummy(cls, index_pair: tuple[int, int]):
        return cls(
            inner_index = index_pair,
            peaks = index_pair,
            peaks_mass = (0., 0.),
            residue = 0,
            residue_mass = 0.,
            modification = 0,
            modification_mass = 0.,
            losses = (0, 0),
            losses_mass = (0., 0.),
            charge_states = (1, 1),
            mass_error = 0.)
    
    @classmethod
    def without_index(cls,
        query_mass: float,
        residue: int,
        residue_mass: float,
        modification: int,
        modification_mass: float,
        losses: tuple[int, int],
        losses_mass: tuple[float, float],
        charge_states: tuple[int, int],
        mass_error: float,
    ):
        return cls(
            inner_index = (-1, 0),
            peaks = (-1, 0),
            peaks_mass = (0., query_mass),
            residue = residue,
            residue_mass = residue_mass,
            modification = modification,
            modification_mass = modification_mass,
            losses = losses,
            losses_mass = losses_mass,
            charge_states = charge_states,
            mass_error = mass_error)

@dataclass
class MassTransformationSpace:
    """A model of every potential transformation under a given parametization.
    Transformations are represented as shifts, so the space is a glorified list of floats."""
    residue_symbols: list[str]
    residue_masses: list[float]
    loss_masses: list[float]
    residue_losses: dict[str, list[int]]
    modification_masses: list[float]
    residue_modifications: dict[str, list[int]]

    def get_extremal_delta(self):
        max_mass = max(self.residue_masses)
        min_loss = min(self.loss_masses)
        max_loss = max(self.loss_masses)
        max_modification = max(self.modification_masses)
        return max_mass + max_loss + max_modification - min_loss

    def get_transformation_tensors(self):
        masses = np.array(self.residue_masses)
        losses = self.loss_masses
        modifications = self.modification_masses
        # mass tensor - one dimensional along axis 0.
        n_masses = len(masses)
        mass_tensor = masses.reshape(n_masses, 1, 1, 1)
        # inverse loss tensors - one dimensional, respectively along axes 1 and 2.
        n_losses = len(losses) + 1
        left_loss_tensor = -1 * np.concatenate(([-0.0], losses)).reshape(1, n_losses, 1, 1)
        right_loss_tensor = -1 * np.concatenate(([-0.0], losses)).reshape(1, 1, n_losses, 1)
        # inverse modification tensor - one dimensional along axis 3.
        n_modifications = len(modifications) + 1
        modification_tensor = -1 * np.concatenate(([-0.0], modifications)).reshape(1, 1, 1, n_modifications)
        # compute the extremal delta
        return mass_tensor, left_loss_tensor, right_loss_tensor, modification_tensor

    def get_key(self, i: int):
        return self.residue_symbols[i]

    def get_residue_mass(self, i: int):
        return self.residue_masses[i]
    
    def get_loss_mass(self, key: str, loss_id: int):
        i = loss_id - 1
        if i == -1:
            return 0.
        elif i in self.residue_losses[key]: 
            return self.loss_masses[i]
        else:
            return np.inf
    
    def get_modification_mass(self, key: str, mod_id: int):
        i = mod_id - 1
        if i == -1:
            return 0.
        elif i in self.residue_modifications[key]: 
            return self.modification_masses[i]
        else:
            return np.inf

@dataclass
class AbstractMassTransformationSolver(ABC):
    transformation_space: MassTransformationSpace
    mz: np.ndarray
    tolerance: float

    @abstractmethod
    def __post_init__(self) -> None:
        """Restructures the transformation space."""

    @abstractmethod
    def set_left_peak(self, left_mz: float) -> None:
        """Sets the left index for gap search, and performs any work needed for subsequent operations."""
    
    @abstractmethod
    def set_right_peak(self, right_mz: float) -> None:
        """Sets the right index for gap search, and performs any work needed to retrieve the optimal result for (left_mz, right_mz)."""

    @abstractmethod
    def get_solutions(self) -> tuple[int, int, int, int]:
        """Having set left and right peaks as `l`, and `r`, respectively, 
        determine the 4-tuple `(mass_idx, left_loss_idx, right_loss_idx, modification_idx)`
        which optimally decomposes the difference `l - r` as
        `self.mass_tensor[mass_idx] - self.left_loss_tensor[left_loss_idx] + self.right_loss_tensor[right_loss_indx] + self.modification_tensor[modification_idx]`."""

    def _filter_solution(self, optimizers: Iterable[tuple[int, int, int, int]]) -> Iterable[tuple[int, int, int, int]]:
        return filter(lambda x: ((x[1] == 0) or (x[1] != x[2])), optimizers)

class TensorMassTransformationSolver(AbstractMassTransformationSolver):
    def __post_init__(self):
        """self.modification_tensor can be very large, so there's no reason to compute self.right_loss_tensor + self.modification_tensor more than once."""
        mass_tensor, left_loss_tensor, right_loss_tensor, modification_tensor = self.transformation_space.get_transformation_tensors()
        self.extremal_delta = self.transformation_space.get_extremal_delta()
        self._left_tensor = left_loss_tensor.reshape(1, *left_loss_tensor.shape)
        self._right_tensor = right_loss_tensor + modification_tensor
        self._right_tensor = self._right_tensor.reshape(1, *self._right_tensor.shape)
        self._mass_tensor = mass_tensor.reshape(1, *mass_tensor.shape)
        # initialize dynamic fields
        self._outer_index = -1
        self._inner_index = -1
        self._lower_inner_index = -1
        self._upper_inner_index = -1
        self._results = None
        self._local_results = None
    
    def _inner_upper_bound(self) -> int:
        idx = bisect_right(self.mz, self.mz[self._outer_index] + self.extremal_delta)
        return max(0, min(self.mz.size - 1, idx + 1))
    
    def _construct_results(self) -> None:
        left_peaks = self.mz[self._outer_index] + self._left_tensor
        self._lower_inner_index = self._outer_index + 1
        self._upper_inner_index = self._inner_upper_bound()
        right_queries = self.mz[self._lower_inner_index: self._upper_inner_index + 1]
        right_queries = right_queries.reshape(right_queries.size, 1, 1, 1, 1)
        right_peaks = right_queries + self._right_tensor
        differences = right_peaks - left_peaks
        differences[differences < 0] = np.inf
        self._results = np.abs(differences - self._mass_tensor)
        
    def set_left_peak(self, left_index: int) -> None:
        self._outer_index = left_index
        self._construct_results()

    def _construct_local_results(self) -> None:
        self._local_results = self._results[self._local_inner_index, :, :, :, :]
        
    def set_right_peak(self, right_index: int) -> None:
        self._local_inner_index = right_index - self._lower_inner_index
        self._construct_local_results()
        
    def get_solutions(self) -> tuple[int, int, int, int]:
        soln_tensor = self._local_results <= self._local_results.min() + self.tolerance
        return self._filter_solution(zip(*(soln_tensor).nonzero()))

class BisectMassTransformationSolver(AbstractMassTransformationSolver):
    def __post_init__(self):
        """construct the target mass values from the transformation tensors"""
        mass_tensor, left_loss_tensor, right_loss_tensor, modification_tensor = self.transformation_space.get_transformation_tensors()
        # transform target masses
        subtractive_transformations = modification_tensor + right_loss_tensor
        transformed_masses = mass_tensor + left_loss_tensor - subtractive_transformations
        self._n = transformed_masses.size
        self._shape = transformed_masses.shape
        self._transformed_target_masses = transformed_masses.flatten()
        indices = np.arange(self._n)
        self._unraveled_indices = np.array([np.unravel_index(i, self._shape) for i in indices])
        # reorder
        order = np.argsort(self._transformed_target_masses)
        self._transformed_target_masses = self._transformed_target_masses[order]
        self._unraveled_indices = [tuple(i) for i in self._unraveled_indices[order]]
        # initialize dynamic fields
        self._left_peak = np.nan
        self._match_lo = -1
        self._match_hi = -1
    
    def _unravel(self, i):
        return self._unraveled_indices[i]

    def _bound_bisection(self, i):
        return max(0, min(self._n - 1, i))
    
    def _bisect_range(self, query: float):
        return (self._bound_bisection(bisect_left(self._transformed_target_masses, query - self.tolerance)),
            self._bound_bisection(bisect_right(self._transformed_target_masses, query + self.tolerance)))

    def set_left_peak(self, left_index: int) -> None:
        self._left_peak = self.mz[left_index]

    def set_right_peak(self, right_index: int) -> None:
        query = self.mz[right_index] - self._left_peak
        self._match_lo, self._match_hi = self._bisect_range(query)
    
    def get_solutions(self) -> tuple[int, int, int, int]:
        return self._filter_solution(map(self._unravel, range(self._match_lo, self._match_hi + 1)))

@dataclass
class ResidueParams:
    # search parameters
    tolerance: float
    comparative_tolerance: float
    strategy: Type[AbstractMassTransformationSolver]
    # transformation parameters
    ## charge states
    charge_symbols: list[str]
    charge_states: list[int]
    ## neutral losses
    loss_symbols: list[str]
    loss_masses: list[float]
    ## modifications
    modification_symbols: list[str]
    modification_masses: list[float]
    ## residues and residue -> loss,modification tables
    residue_symbols: list[str]
    residue_masses: list[float]
    residue_losses: dict[str, list[int]]
    residue_modifications: dict[str, list[int]]