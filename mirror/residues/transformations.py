from typing import Iterable

from .types import ResidueParams, MassTransformation, MassTransformationSpace, AbstractMassTransformationSolver
from ..spectra.types import PeakList
from ..util import merge_in_order

import numpy as np

def _solve(
    mass: float,
) -> None:
    pass

def solve_single(
    mz: float
) -> MassTransformation:
    """Determine the transformation responsible for a single residue mass."""

def solve_pair(
    left_mz: float,
    right_mz: float
) -> MassTransformation:
    delta_mz = right_mz - left_mz

def _augment_charge(
    peaks: PeakList,
    charge_states: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mz = np.array(peaks.mz)
    augmented_mz_arrays = [mz if (c == 1) else c * mz for c in charge_states]
    merged_augmented_mz, deindexer, charge_table = merge_in_order(augmented_mz_arrays)
    return (merged_augmented_mz, deindexer, charge_table)

def solve_peak_list(
    peaks: PeakList,
    params: ResidueParams,
) -> Iterable[MassTransformation]:
    """Find all pairs of peaks related by a MassTransformation."""
    # 1. init search strategy (tensor or bisect)
    transformation_space = MassTransformationSpace(
        residue_symbols = params.residue_symbols,
        residue_masses = params.residue_masses,
        loss_masses = params.loss_masses,
        residue_losses = params.residue_losses,
        modification_masses = params.modification_masses,
        residue_modifications = params.residue_modifications)
    # 2. construct charge-augmented list of m/z values
    augmented_peaks, deindexer, charge_table = _augment_charge(peaks, params.charge_states)
    # 3. enumerate and solve candidate pairs
    tolerance = params.tolerance
    extremal_delta = transformation_space.get_extremal_delta()
    transformation_solver = params.strategy(
        transformation_space = transformation_space,
        mz = augmented_peaks,
        tolerance = tolerance)
    #print(f"tolerance {tolerance}\nextremal delta {extremal_delta}\nsolver {transformation_solver}")
    for (i, left_peak) in enumerate(augmented_peaks[:-1]):
        # set up the left peak
        transformation_solver.set_left_peak(i)
        left_index = deindexer[i]
        left_charge = charge_table[i]
        slice_offset = i + 1
        for (local_j, right_peak) in enumerate(augmented_peaks[slice_offset:]):
            j = slice_offset + local_j
            mass_query = right_peak - left_peak
            # bounded from above
            if mass_query - extremal_delta > tolerance:
                break
            elif False: #TODO: bound from below
                continue
            else: 
                # set up the right peak
                transformation_solver.set_right_peak(j)
                right_index = deindexer[j]
                right_charge = charge_table[j]
                # enumerate solutions
                min_delta = np.inf
                #print(f"peak[{i}]ₗ = {left_peak}\tpeak[{j}]ᵣ = {right_peak}\tΔ = {mass_query}")
                for solution_parameters in transformation_solver.get_solutions():
                    # unwrap solution
                    residue_idx, left_loss_idx, right_loss_idx, modification_idx = solution_parameters
                    #print(f"- candidate:\n\t{solution_parameters}")
                    # measure the error
                    ## retrieve components.
                    key = transformation_space.get_key(residue_idx)
                    residue_mass = transformation_space.get_residue_mass(residue_idx)
                    left_loss_mass = transformation_space.get_loss_mass(key, left_loss_idx)
                    right_loss_mass = transformation_space.get_loss_mass(key, right_loss_idx)
                    modification_mass = transformation_space.get_modification_mass(key, modification_idx)
                    ## calculate the difference between observation and explanation
                    ## each subtraction introduces numeric instability; do as few as possible.
                    left_inverse_transformed_peak = left_peak + left_loss_mass
                    right_inverse_transformed_peak = right_peak + right_loss_mass + modification_mass
                    inverse_transformed_mass = right_inverse_transformed_peak - left_inverse_transformed_peak
                    delta_mass = abs(inverse_transformed_mass - residue_mass)
                    # if a solution exists, encode it as a MassTransformation
                    #print(f"\tres = {key}\tmass = {residue_mass}\tleft = {left_loss_mass}\tright = {right_loss_mass}\tmod = {modification_mass}\tΔ = {delta_mass}")
                    if delta_mass <= tolerance:
                        yield MassTransformation(
                            inner_index = (i, j),
                            peaks = (left_index, right_index),
                            peaks_mass = (left_peak, right_peak),
                            residue = residue_idx,
                            residue_mass = residue_mass,
                            modification = modification_idx,
                            modification_mass = modification_mass,
                            losses = (left_loss_idx, right_loss_idx),
                            losses_mass = (left_loss_mass, right_loss_mass),
                            charge_states = (left_charge, right_charge),
                            mass_error = delta_mass,
                        )