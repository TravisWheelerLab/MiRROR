import dataclasses
from typing import Self, Iterator
import itertools as it
# standard

from .pairs import PairResult
from .pivots import PivotResult
from .boundaries import BoundaryResult
#local

import numpy as np

@dataclasses.dataclass(slots=True)
class FragmentStateSpace:
    loss_masses: np.ndarray
    # [float; m]
    loss_symbols: np.ndarray
    # [str; m]
    applicable_losses: np.ndarray
    # [[int; _]; m] 
    # amino idx -> {applicable loss indices}.
    charges: np.ndarray

    @classmethod
    def trivial(cls) -> Self:
        return cls(
            loss_masses = np.array([0.]),
            loss_symbols = np.array(['']),
            applicable_losses = [np.array([0]) for _ in range(20)],
            charges = np.array([1]))

    def n_losses(self) -> int:
        return len(self.loss_masses)

@dataclasses.dataclass(slots=True)
class ResidueStateSpace:
    amino_masses: np.ndarray
    # [float; k]
    amino_symbols: np.ndarray
    # [str; k]
    modification_masses: np.ndarray
    # [float; k]
    modification_symbols: np.ndarray
    # [str; k]
    applicable_modifications: list[np.ndarray] 
    # [[int; _]; k]
    # amino_idx -> {modification_idx applicable to amino}.
    max_num_modifications: int
    # restricts the number of modifications that can be applied to a residue.

    def n_aminos(self) -> int:
        return len(self.amino_masses)

    def n_modifications(self, amino_id: int) -> int:
        return len(self.applicable_modifications[amino_id])

    def get_modifications(self, amino_id: int) -> list[float]:
        return self.modification_masses[self.applicable_modifications[amino_id]]

@dataclasses.dataclass(slots=True)
class MultiResidueStateSpace(ResidueStateSpace):
    amino_masses: np.ndarray
    # [float; k]
    amino_symbols: list[np.ndarray]
    # [[str; _]; k]
    modification_masses: np.ndarray
    # [float; k]
    modification_symbols: np.ndarray
    # [str; k]
    applicable_modifications: list[np.ndarray] 
    # [[int; _]; k]
    # amino_idx -> {modification_idx applicable to amino}.
    max_num_modifications: int
    # restricts the number of modifications that can be applied to a residue.

    @classmethod
    def from_nonunique_pairs(
        cls,
        masses: np.ndarray,
        words: np.ndarray,
        #mod_masses: np.ndarray,
        #mod_symbols: np.ndarray,
        #applicable_mods: list[np.ndarray],
        #max_num_mods: int,
    ) -> Self:
        unique_masses, reindexer = np.unique_inverse(masses)
        clustered_words = [[] for _ in range(len(unique_masses))]
        clustered_mods = [[] for _ in range(len(unique_masses))]
        for (i, w) in enumerate(words):
            clustered_words[reindexer[i]].append(w)
        return cls(
            unique_masses,
            [np.array(x) for x in clustered_words],
            np.array([0]),#mod_masses,
            np.array(['']),#mod_symbols,
            [np.array([0]) for _ in unique_masses],#applicable_mods,
            0,#max_num_mods,
        )

    def n_aminos(self) -> int:
        return len(self.amino_masses)

    def n_modifications(self, amino_id: int) -> int:
        return len(self.applicable_modifications[amino_id])

    def get_modifications(self, amino_id: int) -> list[float]:
        selected_mod = self.applicable_modifications[amino_id]
        return self.modification_masses[self.applicable_modifications[amino_id]]

@dataclasses.dataclass(slots=True)
class TargetMassStateSpace:
    residue_spaces: list[ResidueStateSpace]
    fragment_space: FragmentStateSpace
    boundary_space: FragmentStateSpace
    pair_masses: np.ndarray
    pair_indices: np.ndarray
    boundary_masses: list[np.ndarray]
    boundary_indices: list[np.ndarray]

    @staticmethod
    def _query_index(
        states: np.ndarray,     # [(int,int); n] < m
        indices: np.ndarray,    # [[int; k]; m]
        feature_axis: int,      # int < k
        features: np.ndarray,   # [_; l]
    ) -> Iterator[np.ndarray]:
        """For each range in states yield an array of the queried feature."""
        # for (l, r) in states:
        for l, r in zip(*states):
            yield features[indices[l:r,feature_axis]]

    def get_pair_residues(
        self,
        states: np.ndarray,
    ) -> list[np.ndarray]:
        return list(self._query_index(
            states,
            self.pair_indices,
            0,
            self.residue_spaces[0].amino_symbols,
        ))

    def get_boundary_residues(
        self,
        states: np.ndarray,
    ) -> list[np.ndarray]:
        return list(self._query_index(
            states,
            self.boundary_indices[0],
            0,
            self.residue_spaces[0].amino_symbols,
        ))

    def get_boundary_kmers(
        self,
        states: np.ndarray,
        k: int = 2,
    ) -> list[np.ndarray]:
        max_k = len(self.residue_spaces)
        if k >= max_k:
            raise ValueError(f"k too large! {k} >= {max_k}")
        return list(self._query_index(
            states,
            self.boundary_indices[k - 1],
            0,
            self.residue_spaces[k - 1].amino_symbols,
        ))
    
    # this method could be faster, but it only gets called once per AnnotationParams, so it's probably ok.
    @staticmethod
    def _augment_masses(
        left_fragment_space: FragmentStateSpace,
        right_fragment_space: FragmentStateSpace,
        residue_space: ResidueStateSpace,
    ) -> tuple[list[float],list[tuple[int,int,int,int]]]:
        # generate the unconditional augments as the 3-tensor sum of orthogonal amino, left loss, and right loss vectors.
        # it is shaped/typed as a 4-tensor because we'll need the extra dimension later when adding in the conditional
        # modification augments.
        amino_masses = residue_space.amino_masses.reshape(residue_space.n_aminos(), 1, 1, 1)
        left_loss_masses = left_fragment_space.loss_masses.reshape(1, left_fragment_space.n_losses(), 1, 1)
        right_loss_masses = right_fragment_space.loss_masses.reshape(1, 1, right_fragment_space.n_losses(), 1)
        ## recall: the residue mass = right fragment mass - left fragment mass. that means:
        ## left loss decreases the left fragment mass, which increases the observed residue mass.
        ## right loss decreases the right fragment mass, which decreases the observed residue mass.
        unconditional_augment_tensor = amino_masses + left_loss_masses - right_loss_masses
        # iterate along the amino (first) axis, construct the conditional augment vectors and apply them to the unconditional augment matrix associated to each amino id.
        augmented_masses = []
        augmented_indices = []
        for amino_id in range(residue_space.n_aminos()):
            unc_aug_matrix = unconditional_augment_tensor[amino_id, :, :, :]
            con_aug_vector = residue_space.get_modifications(amino_id).reshape(1, 1, 1, residue_space.n_modifications(amino_id))
            amino_aug_tensor = unc_aug_matrix + con_aug_vector
            amino_aug_masses = amino_aug_tensor.flatten()
            augmented_masses.extend(amino_aug_masses)
            _, left_loss_ind, right_loss_ind, modification_ind = np.unravel_index(np.arange(len(amino_aug_masses)), amino_aug_tensor.shape)
            amino_aug_indices = zip(it.repeat(amino_id), left_loss_ind, right_loss_ind, modification_ind)
            augmented_indices.extend(amino_aug_indices)
        # argsort by augmented mass and reorder the indices accordingly to construct the index unaveler.
        augmented_masses = np.array(augmented_masses)
        augmented_indices = np.array(augmented_indices)
        index_key = np.argsort(augmented_masses)
        return augmented_masses[index_key], augmented_indices[index_key]

    @classmethod
    def from_state_spaces(cls,
        residue_spaces: list[ResidueStateSpace],
        fragment_space: FragmentStateSpace,
        boundary_space: FragmentStateSpace,
    ):
        pair_masses, pair_indices = cls._augment_masses(
            fragment_space,
            fragment_space,
            residue_spaces[0],
        )
        boundary_masses, boundary_indices = zip(*[cls._augment_masses(
            FragmentStateSpace.trivial(),
            boundary_space,
            residue_space,
        ) for residue_space in residue_spaces])
        return cls(
            residue_spaces,
            fragment_space,
            boundary_space,
            pair_masses,
            pair_indices,
            boundary_masses,
            boundary_indices,
        )
