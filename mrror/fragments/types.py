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

    def n_losses(self, amino_id: int) -> int:
        return len(self.applicable_losses[amino_id])

    def get_losses(self, amino_id: int) -> list[float]:
        return self.applicable_losses[amino_id]

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
        return self.applicable_modifications[amino_id]

@dataclasses.dataclass(slots=True)
class MultiResidueStateSpace(ResidueStateSpace):
    amino_masses: np.ndarray
    # [float; k]
    amino_symbols: list[np.ndarray]
    # [[str; _]; k]
    amino_clusters: list[list[np.ndarray]]
    # [[int; _]; k]
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
        mod_masses: np.ndarray,
        mod_symbols: np.ndarray,
        mod_nulls: np.ndarray,
        applicable_mods: list[np.ndarray],
        max_num_mods: int,
        alphabet: np.ndarray,
    ) -> Self:
        unique_masses, reindexer = np.unique_inverse(masses)
        n_masses = len(unique_masses)
        # construct mass clusters

        clustered_words = [[] for _ in range(n_masses)]
        for (idx, word) in enumerate(words):
            clustered_words[reindexer[idx]].append(word)
        clustered_words = [np.array(x) for x in clustered_words]
        # group words by mass clusters

        amino_ids = np.arange(len(alphabet))
        alphabet = np.array(alphabet)
        alphabet_order = np.argsort(alphabet)
        sorted_alphabet = np.array(alphabet)[alphabet_order]
        sorted_amino_ids = amino_ids[alphabet_order]
        clustered_ids = [[] for _ in range(n_masses)]
        for i in range(n_masses):
            id_cluster = set()
            for word in clustered_words[i]:
                word_amino_ids = sorted(np.searchsorted(sorted_alphabet, np.array(list(word))))
                word_amino_ids = sorted_amino_ids[word_amino_ids]
                id_cluster.add(tuple(word_amino_ids))
            clustered_ids[i] = [np.array(list(x)) for x in id_cluster]
        # group amino ids by mass clusters

        return cls(
            amino_masses = unique_masses,
            amino_symbols = clustered_words,
            amino_clusters = clustered_ids,
            modification_masses = mod_masses,
            modification_symbols = mod_symbols,
            modification_null_indices = mod_nulls,
            applicable_modifications = applicable_mods,
            max_num_modifications = max_num_mods,
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
    pair_null_indices: list[np.ndarray]
    boundary_masses: list[np.ndarray]
    boundary_indices: list[np.ndarray]
    boundary_null_indices: list[list[np.ndarray]]

    @staticmethod
    def _count_nonnull(
        features: np.ndarray,
        null_indices: list[np.ndarray],
    ) -> np.ndarray:
        return np.sum(
            [np.all([x != i for i in nidx],axis=0) for (x,nidx) in zip(features.T,null_indices)],
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
        features = [target_features[l:r,:] for (l,r) in hit_ranges]
        offsets = [np.abs(qm - target_masses[l:r]) for (qm,(l,r)) in zip(query_masses,hit_ranges)]
        complexities = [cls._count_nonnull(x[:,1:],null_indices) for x in features]
        costs = [(o * (1 + c)) for (o,c) in zip(offsets,complexities)]
        segments = np.cumsum(hit_ranges[:,1] - hit_ranges[:,0]) # + 1
        return (
            np.concat(features),
            np.concat(costs),
            np.concat([[0], segments]),
            np.concat(offsets),
            np.concat([target_masses[l:r] for (l,r) in hit_ranges])
        )
    
    def resolve_pairs(
        self,
        hit_ranges: np.ndarray,
        query_masses: np.ndarray,
    ) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
        features, costs, segments, *_ =  self._resolve_hits(
            hit_ranges,
            query_masses,
            self.pair_masses,
            self.pair_indices,
            self.pair_null_indices,
        )
        return (features, costs, segments)

    def resolve_boundaries(
        self,
        hit_ranges: np.ndarray,
        query_masses: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
        features, costs, segments, offsets, hit_masses = self._resolve_hits(
            hit_ranges,
            query_masses,
            self.boundary_masses[k],
            self.boundary_indices[k],
            self.boundary_null_indices[k],
        )
        return (features, costs, segments)

    @staticmethod
    def _weigh_features(
        features: np.ndarray,
        amino_masses: np.ndarray,
        left_loss_masses: np.ndarray,
        right_loss_masses: np.ndarray,
        modification_masses: np.ndarray,
    ) -> np.ndarray:
        if (features == -1).all():
            return np.zeros(features.shape[0], dtype=float)
        else:
            return np.array([
                [
                    amino_masses[i],
                    left_loss_masses[l],
                    -1 * right_loss_masses[r],
                    modification_masses[j],
                ]  for (i,l,r,j) in features
            ], dtype=float)

    def weigh_pairs(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        return self._weigh_features(
            features,
            self.residue_spaces[0].amino_masses,
            self.fragment_space.loss_masses,
            self.fragment_space.loss_masses,
            self.residue_spaces[0].modification_masses,
        )

    def weigh_boundaries(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        return self._weigh_features(
            features,
            self.residue_spaces[0].amino_masses,
            [0.,],
            self.boundary_space.loss_masses,
            self.residue_spaces[0].modification_masses,
        )
    
    @staticmethod
    def _symbolize_features(
        features: np.ndarray,
        amino_symbols: np.ndarray,
        left_loss_symbols: np.ndarray,
        right_loss_symbols: np.ndarray,
        modification_symbols: np.ndarray,
    ) -> np.ndarray:
        if (features == -1).all():
            return np.full_like(features, '')
        else:
            return np.vstack([
                [amino_symbols[i],left_loss_symbols[l],right_loss_symbols[r],modification_symbols[j]] for (i,l,r,j) in features
            ])

    def symbolize_pairs(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        return self._symbolize_features(
            features,
            self.residue_spaces[0].amino_symbols,
            self.fragment_space.loss_symbols,
            self.fragment_space.loss_symbols,
            self.residue_spaces[0].modification_symbols,
        )

    def symbolize_boundaries(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        return self._symbolize_features(
            features,
            self.residue_spaces[0].amino_symbols,
            ['',],
            self.boundary_space.loss_symbols,
            self.residue_spaces[0].modification_symbols,
        )

    def get_series_loss_symbols(self, series=['b','y']) -> tuple[list[str],list[str]]:
        series_loss_symbols = self.boundary_space.loss_symbols
        return ([x for x in series_loss_symbols if x.startswith(series_sym)] for series_sym in series)

    @staticmethod
    def _augment_single_residue_masses(
        left_fragment_space: FragmentStateSpace,
        right_fragment_space: FragmentStateSpace,
        residue_space: ResidueStateSpace,
    ) -> tuple[list[float],list[tuple[int,int,int,int]]]:
        n_aminos = residue_space.n_aminos()
        n_augments_per_amino = [
            left_fragment_space.n_losses(i) * right_fragment_space.n_losses(i) * residue_space.n_modifications(i)
            for i in range(n_aminos)
        ]
        n_augmented_masses = sum(n_augments_per_amino)
        augmented_masses = np.empty((n_augmented_masses,), dtype=float)
        augmented_indices = np.empty((n_augmented_masses,4), dtype=int)
        # count augments and set up augment arrays

        pos = 0
        for amino_id in range(n_aminos):
            amino_mass = residue_space.amino_masses[amino_id]
            mods = residue_space.get_modifications(amino_id)
            left_losses = left_fragment_space.get_losses(amino_id)
            right_losses = right_fragment_space.get_losses(amino_id)
            for mod_id in mods:
                mod_mass = residue_space.modification_masses[mod_id]
                for left_loss_id in left_losses:
                    left_loss_mass = left_fragment_space.loss_masses[left_loss_id]
                    for right_loss_id in right_losses:
                        right_loss_mass = right_fragment_space.loss_masses[right_loss_id]
                        augmented_masses[pos] = amino_mass + left_loss_mass - right_loss_mass + mod_mass
                        augmented_indices[pos] = [amino_id, left_loss_id, right_loss_id, mod_id]
                        pos += 1
        # iterate aminos, conditionally iterate modifications and losses.

        augmented_masses = np.array(augmented_masses)
        augmented_indices = np.array(augmented_indices)
        index_key = np.argsort(augmented_masses)
        # argsort by augmented mass and reorder the indices accordingly to construct the index unaveler.

        return augmented_masses[index_key], augmented_indices[index_key]

    @staticmethod
    def _augment_multi_residue_masses(
        left_fragment_space: FragmentStateSpace,
        right_fragment_space: FragmentStateSpace,
        residue_space: MultiResidueStateSpace,
    ) -> tuple[list[float],list[tuple[int,int,int,int]]]:
        raise NotImplementedError()
        n_words = residue_space.n_aminos()
        n_augments_per_word = [
            np.prod([
                left_fragment_space.n_losses(i) * right_fragment_space.n_losses(i) * residue_space.n_modifications(i)
                for amino_ids in residue_space.amino_clusters[x]
                for i in amino_ids
            ]) for x in range(n_words)
        ]
        n_augmented_masses = sum(n_augments_per_word)
        augmented_masses = np.empty((n_augmented_masses,), dtype=float)
        augmented_indices = np.empty((n_augmented_masses,4), dtype=int)
        # count augments and set up augment arrays

        pos = 0
        for word_id in range(n_words):
            word_amino_mass = residue_space.amino_masses[word_id]
            amino_id_cluster = residue_space.amino_clusters[word_id]
            mods = residue_space.get_modifications(amino_id_cluster)
            left_losses = left_fragment_space.get_losses(amino_id_cluster)
            right_losses = right_fragment_space.get_losses(amino_id_cluster)
            for mod_id in mods:
                mod_mass = residue_space.modification_masses[mod_id]
                for left_loss_id in left_losses:
                    left_loss_mass = left_fragment_space.loss_masses[left_loss_id]
                    for right_loss_id in right_losses:
                        right_loss_mass = right_fragment_space.loss_masses[right_loss_id]
                        augmented_masses[pos] = amino_mass + left_loss_mass - right_loss_mass + mod_mass
                        augmented_indices[pos] = [word_id, left_loss_id, right_loss_id, mod_id]
                        pos += 1
        # iterate aminos, conditionally iterate modifications and losses.

        augmented_masses = np.array(augmented_masses)
        augmented_indices = np.array(augmented_indices)
        index_key = np.argsort(augmented_masses)
        # argsort by augmented mass and reorder the indices accordingly to construct the index unaveler.

        return augmented_masses[index_key], augmented_indices[index_key]

    @classmethod
    def _augment_masses(
        cls,
        left_fragment_space: FragmentStateSpace,
        right_fragment_space: FragmentStateSpace,
        residue_space: ResidueStateSpace,
    ) -> tuple[list[float],list[tuple[int,int,int,int]]]:
        if type(residue_space) is MultiResidueStateSpace:
            return cls._augment_multi_residue_masses(
                left_fragment_space,
                right_fragment_space,
                residue_space,
            )
        else:
            return cls._augment_single_residue_masses(
                left_fragment_space,
                right_fragment_space,
                residue_space,
            )

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
        ) for residue_space in residue_spaces[:1]])
        # construct masses and feature indices
        pair_null_losses = fragment_space.loss_null_indices
        pair_null_indices = [pair_null_losses, pair_null_losses, residue_spaces[0].modification_null_indices]
        boundary_null_losses = boundary_space.loss_null_indices
        boundary_null_indices = [[np.array([0,]), boundary_null_losses, kmer_space.modification_null_indices] for kmer_space in residue_spaces]
        # retrieve null indices for feature states that should not be penalized
        return cls(
            residue_spaces,
            fragment_space,
            boundary_space,
            pair_masses,
            pair_indices,
            pair_null_indices,
            boundary_masses,
            boundary_indices,
            boundary_null_indices,
        )

@dataclasses.dataclass(slots=True)
class AbstractAnnotation(abc.ABC):

    def len(self):
        return len(self.segments) - 1

    def get_annotation(self, i: int) -> tuple[float,np.ndarray]:
        if i == -1:
            return (
                np.array([0.,]),
                np.full((1,self.features.shape[1]),-1),
            )
            # null annotation
        else:
            l, r = self.segments[i:i+2]
            data = (
                self.costs[l:r],
                self.features[l:r,:],
            )
            return data

    def _symbolize(self, features: np.ndarray, sym: list[str], segments: np.ndarray, idx: np.ndarray, costs: np.ndarray) -> str:
        for (i, (l, r)) in enumerate(it.pairwise(segments)):
            cost = costs[l:r]
            ord = np.argsort(cost)
            feat = features[l:r]
            sym.append(f"{i} {idx[i]} {[' '.join(x).strip() for x in feat[ord]]} {cost[ord]}")
        return '\n'.join(sym)
        
    @abc.abstractmethod
    def symbolize(self, targets: TargetMassStateSpace) -> str:
        """Generate symbolic representations of annotated features."""

    @classmethod
    @abc.abstractmethod
    def empty(cls) -> Self:
        """Create an empty annotation."""

@dataclasses.dataclass(slots=True)
class PairResult(AbstractAnnotation):
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

    def symbolize(self, targets: TargetMassStateSpace) -> str:
        features = targets.symbolize_pairs(self.features)
        sym = ["PairResult"]
        return self._symbolize(features, sym, self.segments, self.indices, self.costs)

    @classmethod
    def empty(cls) -> Self:
        return cls(
            indices = np.empty((0,2),dtype=int),
            charges = np.empty((0,2),dtype=int),
            features = np.empty((0,4),dtype=int),
            costs = np.empty((0,),dtype=float),
            segments = np.empty((0,),dtype=int),
            mass= np.empty((0,),dtype=float),
        )

@dataclasses.dataclass(slots=True)
class BoundaryResult(AbstractAnnotation):
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

    def symbolize(self, targets: TargetMassStateSpace) -> str:
        features = targets.symbolize_boundaries(self.features)
        sym = ["BoundaryResult"]
        return self._symbolize(features, sym, self.segments, self.index, self.costs)

    @classmethod
    def empty(cls) -> Self:
        return cls(
            index = np.empty((0,),dtype=int),
            charge = np.empty((0,),dtype=int),
            features = np.empty((0,4),dtype=int),
            costs = np.empty((0,),dtype=float),
            segments = np.empty((0,),dtype=int),
            mass= np.empty((0,),dtype=float),
        )

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
