import dataclasses
import itertools as it
from typing import Self

from ..util import HYDROGEN_MASS
from ..spectra.types import Peaks
from .types import TargetMasses, ResidueStateSpace, FragmentStateSpace, PairResult, PivotResult, BoundaryResult

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
    left_null_losses = left_fragment_space.loss_null_indices
    right_null_losses = right_fragment_space.loss_null_indices
    null_mods = residue_space.modification_null_indices
    null_indices = [left_null_losses, right_null_losses, null_mods]
    # collect loss and modification states corresponding to null / no augmentation.

    sort_key = np.argsort(target_masses)
    # sort by mass.

    return TargetMasses(
        target_masses = target_masses[sort_key],
        target_states = target_states[sort_key],
        null_states = null_indices,
        residue_space = residue_space,
        left_fragment_space = left_fragment_space,
        right_fragment_space = right_fragment_space,
    )

def construct_pair_target_masses(
    residue_space: ResidueStateSpace,
    fragment_space: FragmentStateSpace,
) -> TargetMasses:
    """Construct a TargetMasses object for detecting pair fragment."""
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
