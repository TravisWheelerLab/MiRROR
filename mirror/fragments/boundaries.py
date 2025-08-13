from typing import Self, Iterator, Iterable, Callable
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

from ..spectra.types import PeakList
from .solvers import FragmentState, FragmentStateSpace, ResidueState, ResidueStateSpace, BisectFragmentSolver
from .pivots import AbstractPivot

class AbstractBoundaryFragment(ABC):
    """A (fragment,residue) solution that was discovered using a solver with trivial left fragment space."""

    @classmethod
    @abstractmethod
    def from_solution(cls,
        soln: tuple[FragmentState,FragmentState,ResidueState],
        series: str,
    ) -> Self:
        """Construct a boundary from its solution, as returned by a solver, and the series symbol."""

    @abstractmethod
    def get_fragment(self) -> float:
        """The fragment state associated to this boundary."""

    @abstractmethod
    def get_residue(self) -> ResidueState:
        """The residue state associated to this boundary."""

    @abstractmethod
    def get_series(self) -> str:
        """The symbol for the series whose offset was used to discover this boundary."""

@dataclass 
class LeftBoundaryFragment(AbstractBoundaryFragment):
    """A boundary fragment occurring on the low end of the spectrum. Typically corresponds to the first b-series ion, or a prefix of the b series."""
    fragment: FragmentState
    residue: ResidueState
    series: str

    @classmethod
    def from_solution(cls,
        soln: tuple[FragmentState,FragmentState,ResidueState],
    ) -> Self:
        """Construct a boundary from its solution, as returned by a solver, and the series symbol."""
        fragment = soln[1]
        series = ''
        if 'y' in fragment.loss_symbol:
            series = 'y'
        elif 'b' in fragment.loss_symbol:
            series = 'b'
        return cls(
            fragment = fragment,
            residue = soln[2],
            series = series)

    def get_fragment(self) -> float:
        """The fragment state associated to this boundary."""
        return self.fragment

    def get_residue(self) -> ResidueState:
        """The residue state associated to this boundary."""
        return self.residue

    def get_series(self) -> str:
        """The symbol for the series whose offset was used to discover this boundary."""
        return self.series

    def __repr__(self) -> str:
        dict_repr = str(asdict(self))
        return f"LeftBoundaryFragment{dict_repr}"

@dataclass
class RightBoundaryFragment(AbstractBoundaryFragment):
    """A boundary fragment occurring on the high end of the spectrum. Typically corresponds to the last y-series ion, or a prefix of the y series. Implements the additional get_pivot_point method which returns the float value about which the fragment m/z of this boundary was reflected."""
    fragment: FragmentState
    residue: ResidueState
    series: str
    pivot_point: float

    @classmethod
    def from_solution(cls,
        soln: tuple[FragmentState,FragmentState,ResidueState],
        pivot_point: float,
    ) -> Self:
        """Construct a boundary from its solution, as returned by a solver, and the series symbol."""
        fragment = soln[1]
        series = 'unk'
        if 'y' in fragment.loss_symbol:
            series = 'y'
        elif 'b' in fragment.loss_symbol:
            series = 'b'
        return cls(
            fragment = fragment,
            residue = soln[2],
            series = series,
            pivot_point = pivot_point)

    def get_fragment(self) -> float:
        """The fragment state associated to this boundary."""
        return self.fragment

    def get_residue(self) -> ResidueState:
        """The residue state associated to this boundary."""
        return self.residue

    def get_series(self) -> str:
        """The symbol for the series whose offset was used to discover this boundary."""
        return self.series

    def get_pivot_point(self) -> float:
        return self.pivot_point

def _find_boundaries(
    peaks: PeakList,    
    tolerance: float,
    residue_state_space: ResidueStateSpace,
    fragment_state_space: FragmentStateSpace,
    transformation: Callable = None,
    target_data: tuple[list[float],list[tuple[int,int,int,int]]] = None,
) -> Iterator[tuple[FragmentState,FragmentState,ResidueState]]:
    solver = BisectFragmentSolver.from_half_space(
        mz = peaks,
        tolerance = tolerance,
        state_space = (fragment_state_space, residue_state_space),
        transformation = transformation,
        target_mass_data = target_data)
    print("input mz\n", peaks.mz)
    print("solver mz/idx\n", list(zip(solver._right_mass.tolist(), [int(x[0]) for x in solver._right_mass_index_unraveler])))
    print("solver targets/idx\n", list(zip(solver._target_masses.tolist(),solver._target_mass_index_unraveler.tolist())))
    min_residue_mass, max_residue_mass = solver.get_extremal_masses()
    ref_peak, ref_charge = solver.set_reference(0)
    assert ref_peak == 0. and ref_charge == 1
    for i in range(solver.n_query()):
        peak, charge, mass = solver.set_query(i)
        if mass > max_residue_mass:
            break
        elif mass > min_residue_mass:
            yield from solver.get_solutions()

def find_left_boundaries(
    peaks: PeakList,
    tolerance: float,
    residue_state_space: ResidueStateSpace,
    fragment_state_space: FragmentStateSpace,
    target_data: tuple[list[float],list[tuple[int,int,int,int]]] = None,
) -> Iterator[LeftBoundaryFragment]:
    boundaries = _find_boundaries(
        peaks = peaks,
        tolerance = tolerance,
        residue_state_space = residue_state_space,
        fragment_state_space = fragment_state_space,
        transformation = None,
        target_data = target_data)
    yield from [
        LeftBoundaryFragment.from_solution(soln)
        for soln in boundaries]

def find_right_boundaries(
    pivots: list[AbstractPivot],
    peaks: PeakList,
    tolerance: float,
    residue_state_space: ResidueStateSpace,
    fragment_state_space: FragmentStateSpace,
    target_data: tuple[list[float],list[tuple[int,int,int,int]]] = None,
) -> list[list[RightBoundaryFragment]]:
    n = len(pivots)
    results = [[] for _ in range(n)]
    for i in range(n):
        pivot_point = pivots[i].get_pivot_point()
        reflector = 2 * pivot_point
        boundaries = _find_boundaries(
            peaks = peaks,
            tolerance = tolerance,
            residue_state_space = residue_state_space,
            fragment_state_space = fragment_state_space,
            transformation = lambda x: reflector - x,
            target_data = target_data)
        results[i].extend([
            RightBoundaryFragment.from_solution(soln, pivot_point)
            for soln in boundaries])
    return results

def rescore_pivots(
    pivots: Iterator[AbstractPivot],
    left_boundaries: Iterator[LeftBoundaryFragment],
    right_boundaries: Iterable[Iterator[RightBoundaryFragment]],
    peaks: PeakList,
    symmetry_tolerance: float,
    score_threshold: float,
) -> Iterator[AbstractPivot]:
    left_bound = min(lb.fragment.peak_idx for lb in left_boundaries)
    for (i, pivot) in enumerate(pivots):
        right_bound = max(rb.fragment.peak_idx for rb in right_boundaries)
        assymmetry_score = util.measure_mirror_symmetry(
            sorted_arr = pivots.mz[left_bound: right_bound + 1],
            pivot_point = pivot.pivot_point())
        if assymmetry_score < score_threshold:
            yield pivot.rescore(assymmetry_score)
