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

def _reindex_fragment_masses(
    mz: np.ndarray,
    intensity: np.ndarray,
    pairs: PairResult,
    pivots: PivotResult,
    lower_boundaries: BoundaryResult,
    upper_boundaries: list[BoundaryResult],
):# -> tuple[
#    np.ndarray,         # [float; u]
#    np.ndarray,         # [(int,int); m]
#    np.ndarray,         # [int; l]
#    list[np.ndarray],   # [[int; _]; p]
#    list[np.ndarray],   # [(int,int,int,int); p]
#]:
    pair_structures = pairs.indices
    pair_indices = pair_structures.flatten()
    pair_charges = pairs.charges.flatten()
    pair_losses = [
        pairs.features[i:j,k] 
        for (i,j) in it.pairwise(pairs.segments)
        for k in range(1,3)
    ]
    pair_mods = [
        np.empty((0,),dtype=int) if k == 0 else pairs.features[i:j,3]
        for (i,j) in it.pairwise(pairs.segments)
        for k in range(2)
    ]
    pair_target_indices = [
        pairs.features[i:j,4] 
        for (i,j) in it.pairwise(pairs.segments)
        for k in range(2)
    ]
    pair_costs = [
        pairs.costs[i:j]
        for (i,j) in it.pairwise(pairs.segments)
        for k in range(2)
    ]

    print("pair charges",pair_charges)

    # pair_features = [pairs.features[i:j] for (i,j) in it.pairwise(pairs.segments)]
    
    lb_indices = lower_boundaries.index
    lb_charges = lower_boundaries.charge
    lb_losses = [lower_boundaries.features[i:j,2] for (i,j) in it.pairwise(lower_boundaries.segments)]
    lb_mods = [lower_boundaries.features[i:j,3] for (i,j) in it.pairwise(lower_boundaries.segments)]
    lb_target_indices = [lower_boundaries.features[i:j,4] for (i,j) in it.pairwise(lower_boundaries.segments)]
    lb_costs = [lower_boundaries.costs[i:j] for (i,j) in it.pairwise(lower_boundaries.segments)]

    print("lb charges", lb_charges)

    upper_boundaries_indices = [ub.index for ub in upper_boundaries]
    upper_boundaries_charges = [ub.charge for ub in upper_boundaries]
    upper_boundaries_losses = [
        [ub.features[i:j,2] for (i,j) in it.pairwise(ub.segments)]
        for ub in upper_boundaries
    ]
    upper_boundaries_mods = [
        [ub.features[i:j,3] for (i,j) in it.pairwise(ub.segments)]
        for ub in upper_boundaries
    ]
    upper_boundaries_target_indices = [
        [ub.features[i:j,4] for (i,j) in it.pairwise(ub.segments)]
        for ub in upper_boundaries
    ]
    upper_boundaries_costs = [[ub.costs[i:j] for (i,j) in it.pairwise(ub.segments)] for ub in upper_boundaries]
    ub_indices = np.concat(upper_boundaries_indices)
    ub_charges = np.concat(upper_boundaries_charges)
    ub_losses = sum(upper_boundaries_losses,[])
    ub_mods = sum(upper_boundaries_mods,[])
    ub_target_indices = sum(upper_boundaries_target_indices,[])
    ub_costs = sum(upper_boundaries_costs,[])

    print("ub charges", ub_charges)

    pivot_structures = pivots.pivot_indices
    n_overlap_pivots = next((i for (i,v) in enumerate(pivot_structures) if v is None), len(pivot_structures))
    pivot_indices = np.concat(pivot_structures[:n_overlap_pivots])
    pivot_charges = np.ones_like(pivot_indices)
    pivot_losses = [[-1,] for _ in pivot_indices]
    pivot_mods = [[-1,] for _ in pivot_indices]
    pivot_target_indices = [[-1,] for _ in pivot_indices]
    pivot_costs = [[np.inf,] for _ in pivot_indices]

    symmetric_structures = pivots.symmetries
    symmetric_indices = [sym.flatten() for sym in symmetric_structures]
    sym_indices = np.concat(symmetric_indices)
    sym_charges = np.ones_like(sym_indices)
    sym_losses = [[-1] for _ in sym_indices]
    sym_mods = [[-1] for _ in sym_indices]
    sym_target_indices = [[-1] for _ in sym_indices]
    sym_costs = [[np.inf,] for _ in sym_indices]

    concat_indices = np.concat([
        pair_indices,
        lb_indices,
        ub_indices,
        pivot_indices,
        sym_indices,
    ])
    concat_charges = np.concat([
        pair_charges,
        lb_charges,
        ub_charges,
        pivot_charges,
        sym_charges,
    ])
    concat_losses = pair_losses + lb_losses + ub_losses + pivot_losses + sym_losses
    concat_mods = pair_mods + lb_mods + ub_mods + pivot_mods + sym_mods
    concat_target_indices = pair_target_indices + lb_target_indices + ub_target_indices + pivot_target_indices + sym_target_indices
    concat_costs = pair_costs + lb_costs + ub_costs + pivot_costs + sym_costs
    # concatenate all indices into a single array (and likewise with charges and features)

    loc_pairs = pair_indices.size

    loc_lb = loc_pairs + lb_indices.size

    loc_ub = loc_lb + ub_indices.size
    sizes_ub = [x.size for x in upper_boundaries_indices]
    offsets_ub = np.cumsum([0] + sizes_ub)

    loc_pivot = loc_ub + pivot_indices.size
    offsets_pivot = np.cumsum([0] + [4] * n_overlap_pivots)

### loc_symmetric = loc_pivot + symmetic_indices.size
    sizes_sym = [x.size for x in symmetric_indices]
    offsets_sym = np.cumsum([0] + sizes_sym)
    # record component locations in the concatenated array.

    annotated_masses = (mz[concat_indices] * concat_charges) - (HYDROGEN_MASS * (concat_charges - 1))
    fragment_masses, reidx_concat_indices = np.unique_inverse(annotated_masses)
    print("reidx", reidx_concat_indices)
    assert len(reidx_concat_indices) == len(concat_costs)
    # get unique masses and new indices into unique masses.
    
    annotated_intensities = intensity[concat_indices]
    fragment_intensities = np.empty_like(fragment_masses)
    fragment_charges = [[] for _ in fragment_masses]
    fragment_losses = [[] for _ in fragment_masses]
    fragment_mods = [[] for _ in fragment_masses]
    fragment_target_indices = [[] for _ in fragment_masses]
    fragment_costs = [[] for _ in fragment_masses]
    for (old_idx, new_idx) in enumerate(reidx_concat_indices):
        fragment_intensities[new_idx] += annotated_intensities[old_idx]
        fragment_charges[new_idx].append(concat_charges[old_idx])
        fragment_losses[new_idx].append(concat_losses[old_idx])
        fragment_mods[new_idx].append(concat_mods[old_idx])
        fragment_target_indices[new_idx].append(concat_target_indices[old_idx])
        fragment_costs[new_idx].append(concat_costs[old_idx])
        # TODO, how to do this for losses and mods? there's one mod for every pair, so it has to be assigned to the right fragment. the left fragment is not annotated by that pair. boundary annotations fit the shape fine though.
    # aggregate intensity, charge, loss, and mod annotations for each fragment mass.

    reidx_pairs = reidx_concat_indices[:loc_pairs].reshape(pair_structures.shape)

    reidx_lower_boundaries = reidx_concat_indices[loc_pairs:loc_lb]

    reidx_upper_boundaries = reidx_concat_indices[loc_lb:loc_ub]
    reidx_upper_boundaries = [reidx_upper_boundaries[i:j] for (i,j) in it.pairwise(offsets_ub)]

    reidx_pivot_structures = reidx_concat_indices[loc_ub:loc_pivot]
    reidx_pivot_structures = [reidx_pivot_structures[i:j] for (i,j) in it.pairwise(offsets_pivot)]

    reidx_symmetric_structures = reidx_concat_indices[loc_pivot:]
    reidx_symmetric_structures = [reidx_symmetric_structures[i:j].reshape(symmetric_structures[k].shape) for (k, (i,j)) in enumerate(it.pairwise(offsets_sym))]
    # decompose and reshape into original arrays.

    return (
        fragment_masses,
        fragment_intensities,
        fragment_charges,
        [[np.array(x, dtype=int) for x in a] for a in fragment_losses],
        [[np.array(x, dtype=int) for x in a] for a in fragment_mods],
        [[np.array(x, dtype=int) for x in a] for a in fragment_target_indices],
        [[np.array(x, dtype=float) for x in a] for a in fragment_costs],
        reidx_pairs,
        reidx_pivot_structures,
        reidx_symmetric_structures,
        reidx_lower_boundaries,
        reidx_upper_boundaries,
    )

def construct_unique_fragment_masses(
    peaks: Peaks,
    pairs: PairResult,
    pivots: PivotResult,
    lower_boundaries: BoundaryResult,
    upper_boundaries: list[BoundaryResult],
# ) -> FragmentMasses:
):
    mass, intensity, charges, losses, mods, target_indices, costs, pair_idx, pivot_idx, sym_idx, lower_bound_idx, upper_bound_idx = _reindex_fragment_masses(
        peaks.mz,
        peaks.intensity,
        pairs,
        pivots,
        lower_boundaries,
        upper_boundaries
    )
    return ()
    # return FragmentMasses(
        # mass = mass,
        # intensity = intensity,
        # charges = charges,
        # losses = losses,
        # modifications = mods,
        # target_indices = target_indices,
        # costs = costs,
        # pairs = pair_idx,
        # pivots = pivot_idx,
        # symmetries = sym_idx,
        # lower_boundaries = lower_bound_idx,
        # upper_boundaries = upper_bound_idx,
    # )
