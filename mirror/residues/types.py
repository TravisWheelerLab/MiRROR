from typing import Type, Iterable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from bisect import bisect_left, bisect_right

import numpy as np

@dataclass
class MassTransformationSpace:
    """A model of every potential transformation under a given parametization.
    Transformations are represented as shifts, so the space is a glorified list of floats."""
    residue_symbols: list[str]
    residue_masses: list[float]
    loss_symbols: list[str]
    loss_masses: list[float]
    residue_losses: dict[str, list[int]]
    modification_symbols: list[str]
    modification_masses: list[float]
    residue_modifications: dict[str, list[int]]

    def _construct_transformation_tensors(self):
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

    def __post_init__(self):
        self.mass_tensor, self.left_loss_tensor, self.right_loss_tensor, self.modification_tensor = self._construct_transformation_tensors()

    def get_extremal_delta(self):
        max_mass = max(self.residue_masses)
        min_loss = min(self.loss_masses)
        max_loss = max(self.loss_masses)
        max_modification = max(self.modification_masses)
        return max_mass + max_loss + max_modification - min_loss
    
    def get_transformation_tensors(self):
        return self.mass_tensor, self.left_loss_tensor, self.right_loss_tensor, self.modification_tensor

    def get_residue_symbol(self, i: int):
        return self.residue_symbols[i]
    
    def get_loss_symbol(self, i: int):
        if i == 0:
            return ''
        else:
            return self.loss_symbols[i - 1]
    
    def get_modification_symbol(self, i: int):
        if i == 0:
            return ''
        else:
            return self.modification_symbols[i - 1]

    def get_residue_mass(self, i: int):
        return self.residue_masses[i]
    
    def get_loss_mass(self, key: str, i: int):
        loss_id = i - 1
        if i == 0 or loss_id in self.residue_losses[key]:
            return self.left_loss_tensor[0, i, 0, 0]
        else:
            return np.inf
    
    def get_modification_mass(self, key: str, i: int):
        modification_id = i - 1
        if i == 0 or modification_id in self.residue_losses[key]:
            return self.modification_tensor[0, 0, 0, i]
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

@dataclass
class MassTransformation:
    """A collection of parameters describing the loss, charge, and modification
    states of a pair of peaks whose difference corresponds to a transformation
    of a residue mass. Other than the residue index, values of 0 imply the null
    state, e.g., charge +1, no neutral losses, or no modification."""
    # index within the charge-augmented peak list
    inner_index: tuple[int, int]
    # index into the original peak list
    peaks_index: tuple[int, int]
    peaks_mass: tuple[float, float]
    # residue mass without transformation
    residue_index: int
    residue_mass: float
    residue_symbol: str
    # post-translational modification
    modification_index: int
    modification_mass: float
    modification_symbol: str
    # neutral losses
    losses_index: tuple[int, int]
    losses_mass: tuple[float, float]
    losses_symbol: str
    # charge states
    charges_index: tuple[int, int]
    charges_symbol: str
    # difference between the transformed left peak and the observed right peak
    mass_error: float

    @classmethod
    def dummy(cls, index_pair: tuple[int, int]):
        return cls(
            inner_index = index_pair,
            peaks_index = index_pair,
            peaks_mass = (0.,0.,),
            residue_index = -1,
            residue_mass = 0.,
            residue_symbol = '',
            modification_index = 0,
            modification_mass = 0.,
            modification_symbol = '',
            losses_index = (0, 0),
            losses_mass = (0., 0.),
            losses_symbol = ('', ''),
            charges_index = (0, 0),
            charges_symbol = ('', ''),
            mass_error = 0.)
    
    @classmethod
    def without_index(cls,
        query_mass: float,
        residue_index: int,
        modification_index: int,
        losses_index: tuple[int, int],
        charges_index: tuple[int, int],
        mass_error: float,
        params: ResidueParams,
    ):
        return cls(
            inner_index = (-1, 0),
            peaks = (-1, 0),
            peaks_mass = (0., query_mass),
            residue_index = residue_index,
            residue_mass = params.residue_masses[residue_index],
            residue_symbol = params.residue_symbols[residue_index],
            modification_index = modification_index,
            modification_mass = modification_mass,
            modification_symbol = params.modification_symbols[modification_index],
            losses_index = losses_index,
            losses_mass = (params.loss_masses[losses_index[0]], params.loss_masses[losses_index[1]]),
            losses_symbol = (params.loss_symbols[losses_index[0]], params.loss_symbols[losses_index[1]]),
            charges_index = charges_index,
            charges_symbol = (params.charge_symbols[charges_index[0]], params.charge_symbols[charges_index[1]]),
            mass_error = mass_error)

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
