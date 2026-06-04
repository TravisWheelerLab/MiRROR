import dataclasses
from typing import Self

from ..util import bisect_left, bisect_right
from ..spectra.types import AugmentedPeaks
from ..fragments.types import FragmentStateSpace, LossDistribution, UniqueFragmentIndex, AnnotationIndex
from ..graphs.align import AbstractNodeLookup

import numpy as np

@dataclasses.dataclass(slots=True)
class PeptideMassLookup(AbstractNodeLookup):
    topological_masses: np.ndarray
    topological_indices: np.ndarray
    loss_augmented_peaks: np.ndarray
    tolerance: float

    def __call__(self, peptide_mass: float) -> tuple[int,bool]:
        query_lo = peptide_mass - self.tolerance
        query_hi = peptide_mass + self.tolerance
        topo_lo = bisect_left(
            self.topological_masses,
            query_lo,
        )
        topo_hi = bisect_right(
            self.topological_masses,
            query_hi,
        )
        if topo_lo < topo_hi:
            topo_hit_masses = self.topological_masses[topo_lo:topo_hi]
            err = np.abs(topo_hit_masses - peptide_mass)
            topo_hit_indices = self.topological_indices[topo_lo:topo_hi]
            return (
                topo_hit_indices[np.argmin(err)],
                None,
            )
        else:
            peak_lo = bisect_left(
                self.loss_augmented_peaks,
                query_lo,
            )
            peak_hi = bisect_right(
                self.loss_augmented_peaks,
                query_hi,
            )
            return (
                None,
                peak_lo < peak_hi,
            )

    @classmethod
    def from_masses(
        cls,
        topological_masses: np.ndarray,
        topological_indices: np.ndarray,
        loss_augmented_peaks: np.ndarray,
        tolerance: float,
    ) -> Self:
        topo_order = np.argsort(topological_masses)
        return cls(
            topological_masses = topological_masses[topo_order],
            topological_indices = topological_indices[topo_order],
            loss_augmented_peaks = np.sort(loss_augmented_peaks),
            tolerance = tolerance,
        )

def construct_peptide_mass_lookup(
    decharged_peaks: AugmentedPeaks,
    loss_distribution: LossDistribution,
    fragment_index: UniqueFragmentIndex,
    annotation_index: AnnotationIndex,
    left_pair_fragment_space: FragmentStateSpace,
    right_pair_fragment_space: FragmentStateSpace,
    lower_boundary_fragment_space: FragmentStateSpace,
    upper_boundary_fragment_space: FragmentStateSpace,
    tolerance: float,
) -> PeptideMassLookup:
    fragment_masses = fragment_index.fragment_masses
    misc_frag_masses = list(set(decharged_peaks.mz).difference(fragment_masses))
    if len(misc_frag_masses) == 0:
        misc_peptide_masses = np.empty(0,dtype=float)
    else:
        misc_loss_masses = loss_distribution.query_loss_by_mass(misc_frag_masses)[1]
        misc_peptide_masses = np.concat([
            frag_mass + loss_masses
            for (frag_mass,loss_masses) in zip(misc_frag_masses,misc_loss_masses)
        ])
    # transform decharged peaks that were not annotated into peptide masses by applying the inverse (positive) mass of every possible loss state to every such peak.

    fragment_loss_masses = [
        [np.empty(0,dtype=float),]
        for _ in range(len(fragment_index))
    ]
    def collect_fragment_losses(fragment_ids,loss_states,fragment_space):
        for (frag_id,losses) in zip(fragment_ids,loss_states):
            loss_masses = fragment_space.get_loss_mass(losses)
            fragment_loss_masses[frag_id].append(loss_masses)
    collect_fragment_losses(
        fragment_index.lower_boundaries,
        annotation_index.get_right_loss_state(
            annotation_index.get_lower_boundaries_id()
        ),
        lower_boundary_fragment_space,
    )
    collect_fragment_losses(
        fragment_index.pairs[:,0],
        annotation_index.get_left_loss_state(
            annotation_index.get_pairs_id()
        ),
        left_pair_fragment_space,
    )
    collect_fragment_losses(
        fragment_index.pairs[:,1],
        annotation_index.get_right_loss_state(
            annotation_index.get_pairs_id()
        ),
        right_pair_fragment_space,
    )
    ubounds_frag_ids = fragment_index.upper_boundaries
    for (i, ubound_frag_ids) in enumerate(ubounds_frag_ids):
        collect_fragment_losses(
            ubound_frag_ids,
            annotation_index.get_right_loss_state(
                annotation_index.get_upper_boundaries_id(i)
            ),
            upper_boundary_fragment_space,
        )
    fragment_loss_masses = [np.unique(np.concat(x)) for x in fragment_loss_masses]    
    n = len(fragment_masses)
    peptide_masses = np.concat([
        fragment_masses[i] + fragment_loss_masses[i]
        for i in range(n)
    ])
    frag_ids = np.concat([
        np.repeat(i,len(fragment_loss_masses[i]))
        for i in range(n)
    ])
    # derive peptide masses from annotations by applying inverse losses to fragment masses.

    return PeptideMassLookup.from_masses(
        topological_masses = peptide_masses,
        topological_indices = frag_ids,
        loss_augmented_peaks = misc_peptide_masses,
        tolerance = tolerance,
    )
