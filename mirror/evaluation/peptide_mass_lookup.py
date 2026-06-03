import dataclasses
from typing import Self

from ..util import bisect_left, bisect_left
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
        query_lo = peptide_mass - tolerance
        query_hi = peptide_mass + tolerance
        topo_lo = bisect_left(
            self.topological_masses,
            query_lo,
        )
        topo_hi = bisect_right(
            self.topological_masses,
            query_hi,
        )
        if topo_lo < topo_hi:
            topo_hit_masses = topological_masses[topo_lo:topo_hi]
            err = np.abs(topo_hits - peptide_mass)
            topo_hit_indices = topological_indices[topo_lo:topo_hi]
            return (
                topo_indices[np.argmin(err)],
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
    misc_frag_mass = list(set(decharged_peaks.mz).difference(fragment_index.fragment_masses))
    if len(misc_frag_mass) == 0:
        misc_peptide_mass = np.empty(0,dtype=float)
    else:
        misc_loss_mass = loss_distribution.query_loss_by_mass(misc_frag_mass)[1]
        misc_peptide_mass = np.concat([
            frag_mass + loss_mass
            for (frag_mass,loss_mass) in zip(misc_frag_mass,misc_loss_mass)
        ])
    # transform decharged peaks that were not annotated into peptide masses by applying the inverse (positive) mass of every possible loss state to every such peak.

    lbound_anno_ids = annotation_index.get_lower_boundaries_id()
    lbound_loss = annotation_index.get_right_loss_state(lbound_anno_ids)
    lbound_loss_mass = lower_boundary_fragment_space.get_loss_mass(np.concat(lbound_loss))
    pair_anno_ids = annotation_index.get_pairs_id()
    left_pair_loss = annotation_index.get_left_loss_state(pair_anno_ids)
    left_pair_loss_mass = left_pair_fragment_space.get_loss_mass(np.concat(left_pair_loss))
    right_pair_loss = annotation_index.get_right_loss_state(pair_anno_ids)
    right_pair_loss_mass = right_pair_fragment_space.get_loss_mass(np.concat(right_pair_loss))
    k = len(fragment_index.upper_boundaries)
    ubounds_anno_ids = [
        annotation_index.get_upper_boundaries_id(i)
        for i in range(k)
    ]
    ubounds_loss = [
        annotation_index.get_right_loss_state(ids)
        for ids in ubounds_anno_ids
    ]
    ubounds_loss_mass = [
        upper_boundary_fragment_space.get_loss_mass(np.concat(loss))
        for loss in ubounds_loss
    ]
    loss_mass = np.concat([lbound_loss_mass,left_pair_loss_mass,right_pair_loss_mass] + ubounds_loss_mass)
    lbound_frag_ids = fragment_index.lower_boundaries
    repeat_lbound_frag_ids = [np.repeat(id,len(loss)) for (id,loss) in zip(lbound_frag_ids,lbound_loss)]
    left_pair_frag_ids = fragment_index.pairs[:,0]
    repeat_left_pair_frag_ids = [np.repeat(id,len(loss)) for (id,loss) in zip(left_pair_frag_ids,left_pair_loss)]
    right_pair_frag_ids = fragment_index.pairs[:,1]
    repeat_right_pair_frag_ids = [np.repeat(id,len(loss)) for (id,loss) in zip(right_pair_frag_ids,right_pair_loss)]
    repeat_ubounds_frag_ids = [
        [np.repeat(id,len(loss)) for (id,loss) in zip(ubound_frag_ids,ubound_loss)]
        for (ubound_frag_ids,ubound_loss) in zip(fragment_index.upper_boundaries,ubounds_loss)
    ]
    frag_ids = np.concat(sum([repeat_lbound_frag_ids, repeat_left_pair_frag_ids, repeat_right_pair_frag_ids, *repeat_ubounds_frag_ids],[]))
    frag_mass = fragment_index.fragment_masses[frag_ids]
    peptide_mass = frag_mass + loss_mass
    # derive peptide masses from annotations by applying the inverse of every annotated loss state; retain fragment indices for lookup into spectrum graphs.

    return PeptideMassLookup.from_masses(
        topological_masses = peptide_mass,
        topological_indices = frag_ids,
        loss_augmented_peaks = misc_peptide_mass,
        tolerance = tolerance,
    )
