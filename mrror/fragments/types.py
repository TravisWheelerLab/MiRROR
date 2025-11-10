import dataclasses, abc
from typing import Self, Iterator
import itertools as it

import numpy as np

@dataclasses.dataclass(slots=True)
class FragmentStateSpace:
    loss_masses: np.ndarray
    # [float; m]
    loss_symbols: np.ndarray
    # [str; m]
    loss_null_indices: np.ndarray
    # [int; m]
    applicable_losses: list[np.ndarray]
    # [[int; _]; m] 
    # amino idx -> {applicable loss indices}.
    charges: np.ndarray

    @classmethod
    def trivial(cls) -> Self:
        return cls(
            loss_masses = np.array([0.]),
            loss_symbols = np.array(['']),
            loss_null_indices = np.ndarray([0]),
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
    modification_null_indices: np.ndarray
    # [int; k]
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
    modification_null_indices: np.ndarray
    # [int; k]
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
            np.array([0.]),#mod_masses,
            np.array(['']),#mod_symbols,
            np.array([0]),#mod_nulls,
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
    null_indices: list[list[np.ndarray]]
    pair_masses: np.ndarray
    pair_indices: np.ndarray
    boundary_masses: list[np.ndarray]
    boundary_indices: list[np.ndarray]

    @staticmethod
    def _count_nonnull(
        features: np.ndarray,
        null_indices: list[np.ndarray],
    ) -> np.ndarray:
        return np.sum(
            [x != nidx for (x,nidx) in zip(features.T,null_indices)],
            axis=0,
        )

    @classmethod
    def _resolve_hits(
        cls,
        hit_ranges: np.ndarray,
        query_masses: np.ndarray,
        target_masses: np.ndarray,
        target_features: np.ndarray,
        null_indices: list[np.ndarray],
    ) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
        features = [target_features[l:r,:] for (l,r) in hit_ranges.T]
        offsets = [qm - target_masses[l:r] for (qm,(l,r)) in zip(query_masses,hit_ranges.T)]
        complexities = [cls._count_nonnull(x[:,1:],null_indices) for x in features]
        costs = [o * (1 + c) for (o,c) in zip(offsets,complexities)]
        segments = np.cumsum(hit_ranges[1,:] - hit_ranges[0,:]) + 1
        return (
            np.concat(features),
            np.concat(costs),
            np.concat([[0], segments]),
        )
    
    def resolve_pairs(
        self,
        hit_ranges: np.ndarray,
        query_masses: np.ndarray,
    ) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
        return self._resolve_hits(
            hit_ranges,
            query_masses,
            self.pair_masses,
            self.pair_indices,
            self.null_indices[0],
        )

    def resolve_boundaries(
        self,
        hit_ranges: np.ndarray,
        query_masses: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
        return self._resolve_hits(
            hit_ranges,
            query_masses,
            self.boundary_masses[k],
            self.boundary_indices[k],
            self.null_indices[k],
        )
    
    # this method could be faster, but it only gets called once per AnnotationParams, so it's probably ok.
    @staticmethod
    def _augment_masses(
        left_fragment_space: FragmentStateSpace,
        right_fragment_space: FragmentStateSpace,
        residue_space: ResidueStateSpace,
    ) -> tuple[list[float],list[tuple[int,int,int,int]]]:
        amino_masses = residue_space.amino_masses.reshape(residue_space.n_aminos(), 1, 1, 1)
        left_loss_masses = left_fragment_space.loss_masses.reshape(1, left_fragment_space.n_losses(), 1, 1)
        right_loss_masses = right_fragment_space.loss_masses.reshape(1, 1, right_fragment_space.n_losses(), 1)
        # generate the unconditional augments as the 3-tensor sum of orthogonal amino, left loss, and right loss vectors. it is shaped/typed as a 4-tensor because we'll need the extra dimension later when adding in the conditional
        unconditional_augment_tensor = amino_masses + left_loss_masses - right_loss_masses
        ## recall: the residue mass = right fragment mass - left fragment mass. that means:
        ## left loss decreases the left fragment mass, which increases the observed residue mass.
        ## right loss decreases the right fragment mass, which decreases the observed residue mass.
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
        # iterate along the amino (first) axis, construct the conditional augment vectors and apply them to the unconditional augment matrix associated to each amino id.
        augmented_masses = np.array(augmented_masses)
        augmented_indices = np.array(augmented_indices)
        index_key = np.argsort(augmented_masses)
        # argsort by augmented mass and reorder the indices accordingly to construct the index unaveler.
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
        # construct masses and feature indices
        null_losses = fragment_space.loss_null_indices
        null_indices = [[null_losses, null_losses, x.modification_null_indices] for x in residue_spaces]
        # retrieve null indices for feature states that should not be penalized
        return cls(
            residue_spaces,
            fragment_space,
            boundary_space,
            null_indices,
            pair_masses,
            pair_indices,
            boundary_masses,
            boundary_indices,
        )

@dataclasses.dataclass(slots=True)
class AbstractAnnotationResult(abc.ABC):

    def __post_init__(self):
        print("AnnotationResult:")
        print(self.features)
        print(self.costs)
        print(self.segments)
        print(self.features.shape, self.costs.shape, self.segments.shape)

    def get_annotation(self, i: int) -> tuple[float,np.ndarray]:
        # print(self.features, self.costs, self.segments)
        l, r = self.segments[i:i+2]
        print("get_annotation", i, self.segments.shape, self.costs.shape, (l, r))
        return (
            self.costs[l:r],
            self.features[l:r,:],
        )

@dataclasses.dataclass(slots=True)
class PairResult(AbstractAnnotationResult):
    indices: np.ndarray
    # [(int,int); m]

    charges: np.ndarray
    # [(int,int); m]

    features: np.ndarray
    # [(int,int,int,int); n]

    costs: np.ndarray
    # [float; n]

    segments: np.ndarray
    # [int; m + 1]

    mass: np.ndarray
    # [float; m]

@dataclasses.dataclass(slots=True)
class BoundaryResult(AbstractAnnotationResult):
    index: np.ndarray
    # [int; l]

    charge: np.ndarray
    # [int; l]

    features: np.ndarray
    # [(int,int,int,int); n]

    costs: np.ndarray
    # [float; n]

    segments: np.ndarray
    # [int; l + 1]

    mass: np.ndarray
    # [float; l]

@dataclasses.dataclass(slots=True)
class PivotResult:
    cluster_points: np.ndarray
    # [float; k]
    
    clusters: list[np.ndarray]
    # [[int; _]; k]
    
    scores: np.ndarray
    # [float; k]
    
    symmetries: list[np.ndarray]
    # [[(int,int); _]; k]
    
    pivot_points: np.ndarray
    # [float; p]
    
    pivot_indices: np.ndarray
    # [(int,int,int,int); p]
    
    @classmethod
    def from_data(cls,
        cluster_points: np.ndarray,
        clusters: list[np.ndarray],
        scores: np.ndarray,
        symmetries: list[np.ndarray],
        pivot_points: np.ndarray,
        pivot_indices: np.ndarray,
    ) -> Self:
        assert len(cluster_points) == len(clusters) == len(scores) == len(symmetries)
        assert len(pivot_points) == len(pivot_indices)
        return cls(
            cluster_points,
            clusters,
            scores,
            symmetries,
            pivot_points,
            pivot_indices,
        )

    def __len__(self) -> int:
        return len(self.cluster_points)
