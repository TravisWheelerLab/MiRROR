from typing import Iterator
import itertools as it

from .types import ResidueParams, MassTransformation, MassTransformationSpace, AbstractMassTransformationSolver
from ..spectra.types import PeakList, BenchmarkPeakList, NineSpeciesBenchmarkPeakList
from ..util import merge_in_order

import numpy as np

def _augment_charge(
    peaks: PeakList,
    charge_states: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mz = np.array(peaks.mz)
    augmented_mz_arrays = [mz if (c == 1) else c * mz for c in charge_states]
    merged_augmented_mz, deindexer, charge_table = merge_in_order(augmented_mz_arrays)
    return (merged_augmented_mz, deindexer, charge_table)

def solve_singletons(
    mz: Iterator[float],
    params: ResidueParams,
) -> list[Iterator[MassTransformation]]:
    pass

def solve_pairs(
    pairs: Iterator[tuple[float, float]],
    params: ResidueParams,
) -> list[Iterator[MassTransformation]]:
    pass

def solve_peak_list(
    peaks: PeakList,
    params: ResidueParams,
) -> Iterator[MassTransformation]:
    """Find all pairs of peaks related by a MassTransformation."""
    # 1. init search strategy (tensor or bisect)
    transformation_space = MassTransformationSpace(
        residue_symbols = params.residue_symbols,
        residue_masses = params.residue_masses,
        loss_symbols = params.loss_symbols,
        loss_masses = params.loss_masses,
        residue_losses = params.residue_losses,
        modification_symbols = params.modification_symbols,
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
    for (i, left_peak) in enumerate(augmented_peaks[:-1]):
        # set up the left peak
        transformation_solver.set_left_peak(i)
        left_index = deindexer[i]
        left_charge_index = charge_table[i]
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
                right_charge_index = charge_table[j]
                # enumerate solutions
                min_delta = np.inf
                for solution_parameters in transformation_solver.get_solutions():
                    # unwrap solution
                    residue_idx, left_loss_idx, right_loss_idx, modification_idx = solution_parameters
                    # measure the error
                    ## retrieve components.
                    residue_symbol = transformation_space.get_residue_symbol(residue_idx)
                    residue_mass = transformation_space.get_residue_mass(residue_idx)
                    left_loss_mass = transformation_space.get_loss_mass(residue_symbol, left_loss_idx)
                    right_loss_mass = transformation_space.get_loss_mass(residue_symbol, right_loss_idx)
                    modification_mass = transformation_space.get_modification_mass(residue_symbol, modification_idx)
                    if np.inf in (left_loss_mass, right_loss_mass, modification_mass):
                        continue
                    left_charge_symbol = params.charge_symbols[left_charge_index]
                    left_loss_symbol = transformation_space.get_loss_symbol(left_loss_idx)
                    right_charge_symbol = params.charge_symbols[right_charge_index]
                    right_loss_symbol = transformation_space.get_loss_symbol(right_loss_idx)
                    modification_symbol = transformation_space.get_modification_symbol(modification_idx)
                    ## calculate the difference between observation and explanation
                    left_inverse_transformed_peak = left_peak + left_loss_mass
                    right_inverse_transformed_peak = right_peak + right_loss_mass + modification_mass
                    inverse_transformed_mass = right_inverse_transformed_peak - left_inverse_transformed_peak
                    delta_mass = abs(inverse_transformed_mass - residue_mass)
                    # if a solution exists, encode it as a MassTransformation
                    if delta_mass <= tolerance:
                        yield MassTransformation(
                            inner_index = (i, j),
                            peaks_index = (left_index, right_index),
                            peaks_mass = (left_peak, right_peak),
                            residue_index = residue_idx,
                            residue_mass = residue_mass,
                            residue_symbol = residue_symbol,
                            modification_index = modification_idx,
                            modification_mass = modification_mass,
                            modification_symbol = modification_symbol,
                            losses_index = (left_loss_idx, right_loss_idx),
                            losses_mass = (left_loss_mass, right_loss_mass),
                            losses_symbol = (left_loss_symbol, right_loss_symbol),
                            charges_index = (left_charge_index, right_charge_index),
                            charges_symbol = (left_charge_symbol, right_charge_symbol),
                            mass_error = delta_mass,
                        )
                    #else:
                    #    print(f"candidate: {(left_index, right_index)} {residue_idx} {left_loss_idx} {right_loss_idx} {modification_idx} {delta_mass} | query: {mass_query} = {right_peak} - {left_peak} | decomposition: {residue_mass}  {left_loss_mass} {right_loss_mass} {modification_mass} ~ {inverse_transformed_mass} = {right_inverse_transformed_peak} - {left_inverse_transformed_peak} = ({right_peak} + {right_loss_mass} + {modification_mass}) - ({left_peak} + {left_loss_mass})")

def _transformation_index_pairs_from_series_data(
    series_idx: list[int],
    series_pos: list[int],
) -> Iterator[tuple[int, int]]:
    for i, pos in zip(series_idx, series_pos):
        for j, pos2 in zip(series_idx[i + 1:], series_pos[i + 1:]):
            if pos2 == pos1 + 1:
                yield (i, j)

def _benchmark_transformations_from_index_pairs(
    bpl: NineSpeciesBenchmarkPeakList,
    pairs: Iterator[tuple[int, int]],
    mass_transformation_space: MassTransformationSpace,
) -> Iterator[MassTransformation]:
    peptide = bpl.get_peptide()
    for (i, j) in pairs:
        left_mz = bpl[i]
        right_mz = bpl[j]
        left_charge, left_losses, left_series, left_position = bpl.get_state(i)
        right_charge, right_losses, right_series, right_position = bpl.get_state(j)

        residue_index = mass_transformation_space.get_residue_index(residue)
        residue_mass = mass_transformation_space.get_residue_mass(residue_index)
        yield MassTransformation(
            inner_index=(i, j),
            peaks_index=(i, j),
            peaks_mass=(left_mz,right_mz),
            residue_index=residue_index,
            residue_mass=residue_mass,
            residue_symbol=residue,
            modification_index=modification_index,
            modification_mass=modification,
            modification_symbol=modification,
            losses_index=(left_losses_index,right_losses_index),
            losses_mass=(left_losses_mass,right_losses_mass),
            losses_symbol=(left_losses_symbol,right_losses_symbol),
            charges_index=(left_charge_index,right_charge_index),
            charges_symbol=(left_charge_symbol,right_charge_symbol),
            mass_error=abs(left_mass_error) + abs(right_mass_error))

def benchmark_transformations(
    bpl: NineSpeciesBenchmarkPeakList,
    params: ResidueParams,
) -> Iterator[MassTransformation]:
    #
    y_idx = bpl.get_y_series_peaks()
    y_pos = [bpl.get_position[i] for i in y_idx]
    y_index_pairs = _transformation_index_pairs_from_series_data(y_idx, y_pos)
    y_transformations = _benchmark_transformations_from_index_pairs(bpl, y_index_pairs)
    #
    b_idx = bpl.get_b_series_peaks()
    b_pos = [bpl.get_position[i] for i in b_idx]
    b_index_pairs = _transformation_index_pairs_from_series_data(b_idx, b_pos)
    b_transformations = _benchmark_transformations_from_index_pairs(bpl, b_index_pairs)
    #
    return it.chain(y_transformations, b_transformations)
    