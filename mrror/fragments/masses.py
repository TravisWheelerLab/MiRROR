# =============================================================================
# this module describes methods for the construction and transformation 
# of target masses. for class descriptions, refer to mrror.fragments.types.
# =============================================================================
import dataclasses
import itertools as it
from typing import Self, Iterable

from ..util import HYDROGEN_MASS, mesh_sum, mesh_join, mesh_ravel
from ..spectra.types import Peaks
from .types import TargetMasses, MultiResidueTargetMasses, ResidueStateSpace, FragmentStateSpace, PairResult, PivotResult, BoundaryResult

import numpy as np

def augment_pair_masses(
    left_fragment_space: FragmentStateSpace,
    right_fragment_space: FragmentStateSpace,
    residue_space: ResidueStateSpace,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """Calculate augmented masses for boundaries, applying modifications conditionally and losses unconditionally."""
    n_aminos = residue_space.n_aminos()
    n_left_losses = len(left_fragment_space.loss_masses)
    n_right_losses = len(right_fragment_space.loss_masses)
    n_aminos_with_mods = sum(residue_space.n_modifications(i) for i in range(n_aminos))
    n_augmented_masses = n_left_losses * n_right_losses * n_aminos_with_mods
    # count augments.
    
    augmented_masses = np.empty((n_augmented_masses,), dtype=float)
    augmented_indices = np.empty((n_augmented_masses,4), dtype=int)
    pos = 0
    for amino_id in range(n_aminos):
        amino_mass = residue_space.amino_masses[amino_id]
        for mod_id in residue_space.get_modifications(amino_id):
            mod_mass = residue_space.modification_masses[mod_id]
            for left_loss_id in range(n_left_losses):
                left_loss_mass = left_fragment_space.loss_masses[left_loss_id]
                for right_loss_id in range(n_right_losses):
                    right_loss_mass = right_fragment_space.loss_masses[right_loss_id]
                    augmented_masses[pos] = amino_mass + left_loss_mass - right_loss_mass + mod_mass
                    augmented_indices[pos] = [amino_id, left_loss_id, right_loss_id, mod_id]
                    pos += 1
    # iterate aminos, applying modifications conditionally and losses unconditionally.

    return (augmented_masses, augmented_indices)

def augment_boundary_masses(
    left_fragment_space: FragmentStateSpace,
    right_fragment_space: FragmentStateSpace,
    residue_space: ResidueStateSpace,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """Calculate augmented masses for boundaries, applying modifications and right losses conditionally."""
    n_aminos = residue_space.n_aminos()
    n_left_losses = len(left_fragment_space.loss_masses)
    n_augmented_masses = n_left_losses * sum(
        residue_space.n_modifications(i) * right_fragment_space.n_losses(i) for i in range(n_aminos))
    # count augments and set up arrays.

    augmented_masses = np.empty((n_augmented_masses,), dtype=float)
    augmented_indices = np.empty((n_augmented_masses,4), dtype=int)
    pos = 0
    for amino_id in range(n_aminos):
        amino_mass = residue_space.amino_masses[amino_id]
        mod_ids = residue_space.get_modifications(amino_id)
        right_loss_ids = right_fragment_space.get_losses(amino_id)
        for mod_id in mod_ids:
            mod_mass = residue_space.modification_masses[mod_id]
            for left_loss_id in range(n_left_losses):
                left_loss_mass = left_fragment_space.loss_masses[left_loss_id]
                for right_loss_id in right_loss_ids:
                    right_loss_mass = right_fragment_space.loss_masses[right_loss_id]
                    augmented_masses[pos] = amino_mass + left_loss_mass - right_loss_mass + mod_mass
                    augmented_indices[pos] = [amino_id, left_loss_id, right_loss_id, mod_id]
                    pos += 1
    # iterate aminos, applying modifications and right losses conditionally, and left losses unconditionally.
    
    return (augmented_masses, augmented_indices)

def _construct_target_masses(
    target_masses,
    target_states,
    residue_space,
    left_fragment_space,
    right_fragment_space,
) -> TargetMasses:
    """Given augmented target masses and the state spaces used to create them, prepare the data and construct the final TargetMass object."""
    key = np.argsort(target_masses)
    return TargetMasses(
        target_masses = target_masses[key],
        target_states = target_states[key],
        null_states = [
            left_fragment_space.loss_null_indices,
            right_fragment_space.loss_null_indices,
            residue_space.modification_null_indices,
        ],
        residue_space = residue_space,
        left_fragment_space = left_fragment_space,
        right_fragment_space = right_fragment_space,
    )

def construct_pair_target_masses(
    residue_space: ResidueStateSpace,
    fragment_space: FragmentStateSpace,
) -> TargetMasses:
    """Construct a TargetMasses object for detecting pair fragments."""
    target_masses, target_states = augment_pair_masses(
        fragment_space,
        fragment_space,
        residue_space,
    )
    return _construct_target_masses(
        target_masses,
        target_states,
        residue_space,
        fragment_space,
        fragment_space,
    )

def construct_boundary_target_masses(
    residue_space: ResidueStateSpace,
    fragment_space: FragmentStateSpace,
):
    """Construct a TargetMasses object for detecting boundary fragment."""
    trivial_fragment_space = FragmentStateSpace.trivial()
    # target_masses, target_states = augment_boundary_masses(
    target_masses, target_states = augment_pair_masses(
        trivial_fragment_space,
        fragment_space,
        residue_space,
    )
    return _construct_target_masses(
        target_masses,
        target_states,
        residue_space,
        trivial_fragment_space,
        fragment_space,
    )

def cluster_combinations_by_mass(
    combinations: list[np.ndarray], # [[int; k]; _]
    masses: np.ndarray,             # [float; n]
) -> list[list[np.ndarray]]:        # [[[int; k]; _]; __]
    combined_masses = [sum(masses[x]) for x in combinations]
    unique_masses, reindex = np.unique_inverse(combined_masses)
    n = len(unique_masses)
    clustered_combinations = [[] for _ in range(n)]
    for (old_idx, new_idx) in enumerate(reindex):
        clustered_combinations[new_idx].append(combinations[old_idx])
    return clustered_combinations

def _combine_applicability(
    application_arrs: Iterable[list[np.ndarray]],
    dims: tuple,
) -> list[np.ndarray]:
    return [
        mesh_ravel(
            x,
            dims,
        )
        for x in it.product(*application_arrs)
    ]

def _combine_fragment_spaces(
    spaces: list[FragmentStateSpace],
) -> FragmentStateSpace:
    combined_n_losses = tuple([x.n_total_losses() for x in spaces])
    return FragmentStateSpace(
        loss_masses = mesh_sum([x.loss_masses for x in spaces]),
        loss_symbols = mesh_join([x.loss_symbols for x in spaces]),
        loss_null_indices = mesh_ravel(
            [x.loss_null_indices for x in spaces],
            combined_n_losses,
        ),
        applicable_losses = _combine_applicability(
            [x.applicable_losses for x in spaces],
            combined_n_losses,
        ),
    )

def _combine_residue_spaces(
    spaces: list[ResidueStateSpace],
) -> ResidueStateSpace:
    combined_n_modifications = tuple([x.n_total_modifications() for x in spaces])
    return ResidueStateSpace(
        amino_masses = mesh_sum([x.amino_masses for x in spaces]),
        amino_symbols = mesh_join([x.amino_symbols for x in spaces]),
        modification_masses = mesh_sum([x.modification_masses for x in spaces]),
        modification_symbols = mesh_join([x.modification_symbols for x in spaces]),
        modification_null_indices = mesh_ravel(
            [x.modification_null_indices for x in spaces],
            combined_n_modifications,
        ),
        applicable_modifications = _combine_applicability(
            [x.applicable_modifications for x in spaces],
            combined_n_modifications,
        ),
        max_num_modifications = sum([x.max_num_modifications for x in spaces]),
    )

def combine_target_masses(
    targets: list[TargetMasses],
) -> MultiResidueTargetMasses:
    """Takes the product of a list of target masses. Every mass in the product is the sum of one mass from each operand. For example, the combination of target masses [1.0, 2.0] and [0.3, 0.7] is [1.3, 1.7, 2.3, 2.7]. Duplicate masses are clustered, but their unique annotations are retained and re-constructed after search.
    
    The underlying spaces are combined in a similar way to facilitate state and symbol retrieval using the same TargetMasses interface."""
    combined_residue_space = _combine_residue_spaces([
        x.residue_space for x in targets])
    combined_left_fragment_space = _combine_fragment_spaces([
        x.left_fragment_space for x in targets])
    combined_right_fragment_space = _combine_fragment_spaces([
        x.right_fragment_space for x in targets])
    target_masses = mesh_sum([x.target_masses for x in targets])
    key = np.argsort(target_masses)
    return MultiResidueTargetMasses(
        target_masses = target_masses[key],
        target_states = np.c_[ # form a 2d array by stacking columns
            mesh_ravel( # combined amino states
                [x.target_states[:,0] for x in targets],
                tuple([x.residue_space.n_aminos() for x in targets]),
            ),
            mesh_ravel( # combined left loss states
                [x.target_states[:,1] for x in targets],
                tuple([x.left_fragment_space.n_total_losses() for x in targets]),
            ),
            mesh_ravel( # combined right loss states
                [x.target_states[:,2] for x in targets],
                tuple([x.right_fragment_space.n_total_losses() for x in targets]),
            ),
            mesh_ravel( # combined modification states
                [x.target_states[:,3] for x in targets],
                tuple([x.residue_space.n_total_modifications() for x in targets]),
            ),
        ][key],
        null_states = [
            combined_left_fragment_space.loss_null_indices,
            combined_right_fragment_space.loss_null_indices,
            combined_residue_space.modification_null_indices,
        ],
        residue_space = combined_residue_space,
        left_fragment_space = combined_left_fragment_space,
        right_fragment_space = combined_right_fragment_space,
        num_residues = len(targets),
    )
