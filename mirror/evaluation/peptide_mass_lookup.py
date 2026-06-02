import dataclasses

from ..util import enumerate_samples, bisect_left
from ..fragments.types import FragmentStateSpace, ResidueStateSpace
from ..spectra.types import AugmentedPeaks

import numpy as np

@dataclasses.dataclass(slots=True)
class LossDistribution:
    loss_state_distributions: list[np.ndarray]
    loss_mass_distributions: list[np.ndarray]
    min_mass_per_length: np.ndarray

    @classmethod
    def from_state_spaces(
        cls,
        fragment_space: FragmentStateSpace,
        residue_space: ResidueStateSpace,
        k = 50,
    ) -> tuple[np.ndarray,np.ndarray]:
        n_losses = fragment_space.n_total_losses()
        n_aminos = residue_space.n_aminos()
        loss_tally = np.zeros((n_aminos,n_losses),dtype=int)
        for i in range(n_aminos):
            for j in fragment_space.get_losses(i)[1:]: # discard null loss 0 at idx 0.
                loss_tally[i,j] += 1
        loss_applicator = np.max(loss_tally,axis=0)
        # construct loss applicator and minimum loss-augmented amino mass.
        
        max_num_losses = fragment_space.max_num_losses
        min_k = min(k, max_num_losses)
        loss_occurrences_per_length = [
            np.clip(
                loss_applicator * peptide_length,
                0,
                max_num_losses,
            )
            for peptide_length in range(1, min_k + 1)
        ]
        loss_distr = [
            list(enumerate_samples(
                n_losses,
                max_num_losses,
                loss_occurrences,
            ))
            for loss_occurrences in loss_occurrences_per_length
        ]
        loss_state_distr = [
            np.zeros((len(distr),max_num_losses),dtype=int)
            for distr in loss_distr
        ]
        loss_mass_distr = [
            np.zeros(len(distr),dtype=float)
            for distr in loss_distr
        ]
        for (peptide_length,loss_dist) in enumerate(loss_distr):
            for (i,loss_state) in enumerate(loss_dist):
                for (j,loss_id) in enumerate(loss_state):
                    loss_state_distr[peptide_length][i,j] = loss_id
                state = loss_state_distr[peptide_length][i,:]
                mass = np.sum(fragment_space.loss_masses[state])
                loss_mass_distr[peptide_length][i] = mass
        # construct the loss distribution for each peptide length.

        loss_augmented_residue_masses = []
        for i in range(n_aminos):
            amino_mass = residue_space.amino_masses[i]
            for loss_state in enumerate_samples(n_losses,max_num_losses,loss_tally[i]):
                loss_mass = np.sum(fragment_space.loss_masses[list(loss_state)])
                loss_augmented_residue_masses.append(amino_mass - loss_mass)
        min_mass = min(loss_augmented_residue_masses)
        min_mass_per_length = [min_mass * i for i in range(1, min_k + 1)]
        # enumerate the minimum mass peptide for each length between 1 and min_k.
    
        return cls(
            loss_state_distributions = loss_state_distr,
            loss_mass_distributions = loss_mass_distr,
            min_mass_per_length = min_mass_per_length,
        )

    def query_peptide_mass(
        self,
        query_masses: np.ndarray,
    ) -> tuple[list[np.ndarray],list[np.ndarray]]:
        min_mass_peptide_lengths = np.clip(
            bisect_left(
                self.min_mass_per_length,
                query_masses,
            ),
            min = 1,
        ) - 1
        print(min_mass_peptide_lengths)
        return (
            [self.loss_state_distributions[i] for i in min_mass_peptide_lengths],
            [self.loss_mass_distributions[i] for i in min_mass_peptide_lengths],
        )
