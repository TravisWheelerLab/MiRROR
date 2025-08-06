from typing import Self, Iterator
from dataclasses import dataclass

from ..spectra.types import PeakList
from .solvers import ResidueState, ResidueStateSpace, FragmentState, FragmentStateSpace, BisectFragmentSolver

def calculate_offset(
    soln: tuple[FragmentState,FragmentState,ResidueState],
) -> float:
    pass

@dataclass
class PairedFragments:
    left_fragment: FragmentState
    right_fragment: FragmentState
    residue: ResidueState
    offset: float

    def from_solution(cls,
        soln: tuple[FragmentState,FragmentState,ResidueState],
    ) -> Self:
        return cls(
            *soln,
            offset = calculate_offset(soln))

def find_pairs(
    peaks: PeakList,
    tolerance: float,
    residue_space: ResidueStateSpace,
    fragment_space: FragmentStateSpace,
    target_data: tuple[list[float],list[tuple[int,int,int,int]]] = None,
) -> Iterator[PairedFragments]:
    solver = BisectFragmentSolver.from_state_space(
        mz = peaks,
        tolerance = tolerance,
        state_space = (fragment_space, fragment_space, residue_space),
        target_mass_data = target_data)
    min_residue_mass, max_residue_mass = solver.get_extremal_masses()
    n = solver.n_query()
    for i in range(n):
        peak_i, charge_i = solver.set_reference(i)
        for j in range(i + 1, n):
            peak_j, charge_j, mass_delta = solver.set_query(j)
            if mass_delta < min_residue_mass:
                continue
            elif mass_delta > max_residue_mass:
                break
            elif peak_j > peak_i:
                yield from map(
                    PairedFragments.from_solution,
                    solver.get_solutions())
