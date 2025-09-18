from typing import Self, Iterator, Iterable, Callable, Any
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

from ..util import measure_mirror_symmetry
from ..spectra.types import PeakList
from .solvers import FragmentState, FragmentStateSpace, ResidueState, ResidueStateSpace, BisectFragmentSolver
from .pivots import Pivot

@dataclass(slots=True)
class BoundaryFragment:
    """A boundary fragment occurring on the low end of the spectrum."""
    fragment: FragmentState
    residue: ResidueState
    series: str

    @classmethod
    def from_solution(cls,
        soln: tuple[FragmentState,FragmentState,ResidueState],
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
            series = series)

    @classmethod
    def from_dict(cls,
        data: dict[str,Any],
    ) -> Self:
        return cls(
            fragment = FragmentState(**data["fragment"]),
            residue = ResidueState(**data["residue"]),
            series = data["series"])

    def cost(self) -> float:
        """Aggregate cost from fragment and residue components."""
        return self.fragment.cost() + self.residue.cost()

    def __repr__(self) -> str:
        dict_repr = str(asdict(self))
        return f"BoundaryFragment{dict_repr}"

@dataclass(slots=True)
class ReflectedBoundaryFragment(BoundaryFragment):
    """A boundary fragment occurring on the high end of the spectrum. Implements the additional get_pivot_point method which returns the float value about which the fragment m/z of this boundary was reflected."""
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
        if 'y' in fragment.loss_symbol:
            series = 'y'
        elif 'b' in fragment.loss_symbol:
            series = 'b'
        else:
            series = 'unk'
        return cls(
            fragment = fragment,
            residue = soln[2],
            series = series,
            pivot_point = pivot_point)

    @classmethod
    def from_dict(cls,
        data: dict[str,Any],
    ) -> Self:
        return cls(
            fragment = FragmentState(**data["fragment"]),
            residue = ResidueState(**data["residue"]),
            series = data["series"],
            pivot_point = data["pivot_point"])

def _find_boundaries(
    peaks: PeakList,
    min_mz: float,
    max_mz: float,
    tolerance: float,
    residue_space: ResidueStateSpace,
    fragment_space: FragmentStateSpace,
    transformation: Callable = None,
    target_data: tuple[list[float],list[tuple[int,int,int,int]]] = None,
) -> Iterator[tuple[FragmentState,FragmentState,ResidueState]]:
    solver = BisectFragmentSolver.from_half_space(
        mz = peaks,
        tolerance = tolerance,
        state_space = (fragment_space, residue_space),
        transformation = transformation,
        target_mass_data = target_data)
    min_residue_mass, max_residue_mass = solver.get_extremal_masses()
    ref_peak, ref_charge = solver.set_reference(0)
    assert ref_peak == 0. and ref_charge == 1
    for i in range(solver.n_query()):
        peak_idx, charge, mass = solver.set_query(i)
        peak_mz = peaks[peak_idx] * charge
        if mass > max_residue_mass or not(min_mz < peak_mz < max_mz):
            break
        elif mass > min_residue_mass:
            yield from solver.get_solutions()

def find_left_boundaries(
    peaks: PeakList,
    tolerance: float,
    residue_space: ResidueStateSpace,
    fragment_space: FragmentStateSpace,
    target_data: tuple[list[float],list[tuple[int,int,int,int]]] = None,
) -> Iterator[BoundaryFragment]:
    boundaries = _find_boundaries(
        peaks = peaks,
        min_mz = 0,
        max_mz = peaks[-1] + 1,
        tolerance = tolerance,
        residue_space = residue_space,
        fragment_space = fragment_space,
        transformation = None,
        target_data = target_data)
    yield from [
        BoundaryFragment.from_solution(soln)
        for soln in boundaries]

def find_right_boundaries(
    pivot_points: list[float],
    peaks: PeakList,
    tolerance: float,
    residue_space: ResidueStateSpace,
    fragment_space: FragmentStateSpace,
    target_data: tuple[list[float],list[tuple[int,int,int,int]]] = None,
) -> list[list[ReflectedBoundaryFragment]]:
    n = len(pivot_points)
    results = [[] for _ in range(n)]
    for i in range(n):
        pt = pivot_points[i]
        reflector = 2 * pt
        boundaries = _find_boundaries(
            peaks = peaks,
            min_mz = pt,
            max_mz = fragment_space.charges[-1] * (peaks[-1] + 1),
            tolerance = tolerance,
            residue_space = residue_space,
            fragment_space = fragment_space,
            transformation = lambda x: reflector - x,
            target_data = target_data)
        results[i].extend([
            ReflectedBoundaryFragment.from_solution(soln, pt)
            for soln in boundaries])
    return results
