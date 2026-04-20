import dataclasses, abc
from typing import Self, Iterator
import itertools as it

from ..util import HYDROGEN_MASS, mesh_ravel, fuzzy_unique

import numpy as np
from omegaconf.dictconfig import DictConfig

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
    # amino index -> {applicable loss indices}.

    def n_losses(self, amino_id: int) -> int:
        return len(self.applicable_losses[amino_id])

    def n_total_losses(self) -> int:
        return len(self.loss_masses)

    def get_losses(self, amino_id: int) -> list[int]:
        return self.applicable_losses[amino_id]

    def get_all_losses(self) -> list[int]:
        return list(range(len(self.loss_masses)))

    def tabulate(self) -> str:
        ""

    @classmethod
    def trivial(cls) -> Self:
        return cls(
            loss_masses = np.array([0.],dtype=float),
            loss_symbols = np.array([''],dtype=str),
            loss_null_indices = np.array([0],dtype=int),
            applicable_losses = [np.array([0],dtype=int) for _ in range(20)],
        )

    @classmethod
    def from_config_to_pairs(
        cls,
        cfg: DictConfig,
    ) -> Self:
        return cls(
            loss_masses = np.array(cfg.loss.masses),
            loss_symbols = np.array(cfg.loss.symbols),
            loss_null_indices = np.array(cfg.loss.nulls),
            applicable_losses = [np.array(x) for x in cfg.loss.application],
        )

    @classmethod
    def _from_config_to_boundaries(
        cls,
        series_masses,
        series_symbols,
        loss_masses,
        loss_symbols,
        loss_nulls,
        applicable_losses,
        sep = ' ',
    ) -> Self:
        n_ser = series_masses.size
        n_loss = loss_masses.size
        boundary_nulls = np.array([i + (j * n_loss) for i in loss_nulls for j in range(n_ser)])
        boundary_appl = [
            np.concat([boundary_nulls,] + 
                      [x[1:] + (i * n_loss) for i in range(n_ser)]) 
            for x in applicable_losses]
        return cls(
            loss_masses = (series_masses.reshape(n_ser,1) + loss_masses.reshape(1,n_loss)).flatten(),
            loss_symbols = np.strings.strip(series_symbols.reshape(n_ser,1) + sep + loss_symbols.reshape(1,n_loss)).flatten(),
            loss_null_indices = boundary_nulls,
            applicable_losses = boundary_appl,
        )

    @classmethod
    def from_config_to_boundaries(
        cls,
        cfg: DictConfig,
        reflect = False,
    ) -> Self:
        key = -1 if reflect else 1
        return cls._from_config_to_boundaries(
            series_masses = np.concat([cfg.series.prefix_masses, cfg.series.suffix_masses][::key]),
            series_symbols = np.concat([cfg.series.prefix_symbols, cfg.series.suffix_symbols][::key]),
            loss_masses = key * np.array(cfg.loss.masses),
            # reflection transposes prefix and suffix ion series (b and y, a and x, etc.)
            # reflection flips the sign of losses; r - (x - l) = r - x + l.
            # so b_n - H2O1 reflects to y_1 + H2O1.
            loss_symbols = np.array(cfg.loss.symbols),
            loss_nulls = np.array(cfg.loss.nulls),
            applicable_losses = [np.array(x) for x in cfg.loss.application],
        )

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

    def n_total_modifications(self) -> int:
        return len(self.modification_masses)

    def n_modifications(self, amino_id: int) -> int:
        return len(self.applicable_modifications[amino_id])

    def get_modifications(self, amino_id: int) -> list[int]:
        return self.applicable_modifications[amino_id]

    @classmethod
    def from_config(
        cls,
        cfg: DictConfig,
        reflect = False,
    ) -> Self:
        key = -1 if reflect else 1
        return cls(
            amino_masses = np.array(cfg.res.masses),
            amino_symbols = np.array(cfg.res.symbols),
            modification_masses = key * np.array(cfg.mod.masses),
            # reflection flips the sign of losses and modifications; r - (x - l) = r - x + l.
            # so b_n - H2O1 reflects to y_1 + H2O1.
            modification_symbols = np.array(cfg.mod.symbols),
            modification_null_indices = np.array(cfg.mod.nulls),
            applicable_modifications = [np.array(x) for x in cfg.mod.application],
            max_num_modifications = cfg.mod.max_num,
        )

@dataclasses.dataclass(slots=True)
class TargetMasses(abc.ABC):
    """A collection of masses (and underlying annotations) spanned by a given triplet of a residue space, a left fragment space, and a right fragment space. The classmethod constructor from_unclustered is implemented but it is strongly recommended to use the fragments.masses.construct_*_target_masses functions instead."""
    target_masses: np.ndarray                   # [float; n]
    # truncated_masses: np.ndarray                # [float; m]
    # target_clusters: np.ndarray                 # [[int; 2]; m]
    target_states: np.ndarray                   # [[int; 4]; n]
    null_states: list[np.ndarray]               # [[int; _]; 3]
    residue_space: ResidueStateSpace
    left_fragment_space: FragmentStateSpace
    right_fragment_space: FragmentStateSpace

    @staticmethod
    def _sort_and_cluster(
        target_masses,
        target_states,
        tolerance
    ) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        n_targets = len(target_masses)
        sort_key = np.argsort(target_masses)
        target_masses = target_masses[sort_key]
        target_states = target_states[sort_key]
        truncated_masses, cluster_idx = fuzzy_unique(target_masses, tolerance)
        print("  ", n_targets, "->", len(truncated_masses))
        target_clusters = [0,] + sum([[i - 1, i] for i in range(1, n_targets) if cluster_idx[i - 1] != cluster_idx[i]],[]) + [n_targets - 1,]
        target_clusters = np.array(target_clusters).reshape((-1,2))
        assert len(truncated_masses) == len(target_clusters)
        return (
            target_masses,
            truncated_masses,
            target_clusters,
            target_states,
        )

    @classmethod
    def from_unclustered(
        cls,
        target_masses,
        target_states,
        null_states,
        residue_space,
        left_fragment_space,
        right_fragment_space,
        tolerance: float,
    ) -> Self:
        print("TargetMasses")
        target_masses, truncated_masses, target_clusters, target_states = cls._sort_and_cluster(target_masses,target_states,tolerance)
        return cls(
            target_masses,
            truncated_masses,
            target_clusters,
            target_states,
            null_states,
            residue_space,
            left_fragment_space,
            right_fragment_space,
            tolerance,
        )

    @staticmethod
    def _count_nonnull(
        hit_states: np.ndarray,
        null_states: list[np.ndarray],
    ) -> np.ndarray:
        state_nulls = [
            np.all([x != i for i in nullstate], axis=0) 
            for (x,nullstate) in zip(hit_states.T,null_states)
        ]
        return np.sum(state_nulls, axis=0)

    def get_series_loss_symbols(self, series=['b','y']) -> tuple[list[str],list[str]]:
        series_loss_symbols = self.boundary_space.loss_symbols
        return ([x for x in series_loss_symbols if x.startswith(series_sym)] for series_sym in series)

    def get_hit_states(
        self,
        hit_ranges: np.ndarray,
        query_masses: np.ndarray,
    ) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
        # hit_ranges = np.clip(hit_ranges,0,len(self.target_clusters) - 1)
        # expanded_hit_ranges = np.array([(self.target_clusters[l,0],self.target_clusters[r,1]) for (l,r) in hit_ranges])
        # hit_states = [self.target_states[l:r,:] for (l,r) in expanded_hit_ranges]
        # offsets = [np.abs(qm - self.target_masses[l:r]) for (qm,(l,r)) in zip(query_masses,expanded_hit_ranges)]
        hit_states = [self.target_states[l:r,:] for (l,r) in hit_ranges]
        offsets = [np.abs(qm - self.target_masses[l:r]) for (qm,(l,r)) in zip(query_masses,hit_ranges)]
        complexities = [self._count_nonnull(x[:,1:4],self.null_states) for x in hit_states]
        costs = [(o * (1 + c)) for (o,c) in zip(offsets,complexities)]
        # segments = np.cumsum(expanded_hit_ranges[:,1] - expanded_hit_ranges[:,0]) # + 1
        segments = np.cumsum(hit_ranges[:,1] - hit_ranges[:,0]) # + 1
        return (
            np.concat(hit_states),
            np.concat(costs),
            np.concat([[0], segments]),
            np.concat(offsets),
            np.concat([self.target_masses[l:r] for (l,r) in hit_ranges])
        )

    def get_state_weights(
        states: np.ndarray,
    ) -> np.ndarray:
        if (states == -1).all():
            return np.zeros(states.shape[0], dtype=float)
            # empty states, usually from the spectrum graph edge connecting pivot peaks to the virtual pivot sink.
        else:
            return np.array([
                [
                    self.residue_space.amino_masses[i],
                    self.left_fragment_space.loss_masses[l],
                    -1 * self.right_fragment_space.loss_masses[r],
                    self.residue_space.modification_masses[j],
                ]  for (i,l,r,j) in states
            ], dtype=float)

    def get_state_symbols(
        states: np.ndarray,
    ) -> np.ndarray:
        if (states == -1).all():
            return np.full_like(states, '')
            # empty states, usually from the spectrum graph edge connecting pivot peaks to the virtual pivot sink.
        else:
            return np.array([
                [
                    self.residue_space.amino_symbols[i],
                    self.left_fragment_space.loss_symbols[l],
                    self.right_fragment_space.loss_symbols[r],
                    self.residue_space.modification_symbols[j],
                ] for (i,l,r,j) in states
            ])
        # NOTE, in case this breaks - the return was a 'np.vstack' call but that seemed unnecessary.

@dataclasses.dataclass(slots=True)
class MultiResidueTargetMasses(TargetMasses):
    """A TargetMasses object with the additional num_residues field.
    Like TargetMasses, a classmethod constructor is deliberately not provided in lieu of fragments.masses.combine_target_masses."""
    num_residues: int

    @classmethod
    def from_unclustered(
        cls,
        target_masses,
        target_states,
        null_states,
        residue_space,
        left_fragment_space,
        right_fragment_space,
        tolerance: float,
        num_residues: int,
    ) -> Self:
        print("MultiResidueTargetMasses")
        target_masses, truncated_masses, target_clusters, target_states = cls._sort_and_cluster(target_masses,target_states,tolerance)
        return cls(
            target_masses,
            truncated_masses,
            target_clusters,
            target_states,
            null_states,
            residue_space,
            left_fragment_space,
            right_fragment_space,
            tolerance,
            num_residues,
        )
    
@dataclasses.dataclass(slots=True)
class AbstractAnnotation(abc.ABC):

    def __len__(self):
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
    def symbolize(self, targets: TargetMasses) -> str:
        """Generate symbolic representations of annotated features."""
        # TODO- replace this with a 'tabulate' method.

    @classmethod
    @abc.abstractmethod
    def empty(cls) -> Self:
        """Create an empty annotation."""

@dataclasses.dataclass(slots=True)
class PairResult(AbstractAnnotation):
    indices: np.ndarray
    # [(int,int); m]

    inner_indices: np.ndarray
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

    def symbolize(self, targets: TargetMasses) -> str: # TODO convert to tabulate, use TargetMasses
        features = targets.symbolize_pairs(self.features)
        sym = ["PairResult"]
        return self._symbolize(features, sym, self.segments, self.indices, self.costs)

    @classmethod
    def empty(cls) -> Self:
        return cls(
            indices = np.empty((0,2),dtype=int),
            inner_indices = np.empty((0,2),dtype=int),
            charges = np.empty((0,2),dtype=int),
            features = np.empty((0,5),dtype=int),
            costs = np.empty((0,),dtype=float),
            segments = np.empty((0,),dtype=int),
            mass= np.empty((0,),dtype=float),
        )

@dataclasses.dataclass(slots=True)
class BoundaryResult(AbstractAnnotation):
    index: np.ndarray
    # [int; l]

    inner_index: np.ndarray
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

    def symbolize(self, targets: TargetMasses) -> str: # TODO convert to tabulate, use TargetMasses
        features = targets.symbolize_boundaries(self.features)
        sym = ["BoundaryResult"]
        return self._symbolize(features, sym, self.segments, self.index, self.costs)

    @classmethod
    def empty(cls) -> Self:
        return cls(
            index = np.empty((0,),dtype=int),
            inner_index = np.empty((0,),dtype=int),
            charge = np.empty((0,),dtype=int),
            features = np.empty((0,5),dtype=int),
            costs = np.empty((0,),dtype=float),
            segments = np.empty((0,),dtype=int),
            mass = np.empty((0,),dtype=float),
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
    def empty(cls):
        return cls(
            cluster_points = np.empty(shape=(0,), dtype=float),
            clusters = [],
            scores = np.empty(shape=(0,), dtype=float),
            symmetries = [],
            pivot_points = np.empty(shape=(0,), dtype=float),
            pivot_indices = np.empty(shape=(0,4), dtype=float),
        )

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
