import dataclasses
import itertools as it
from itertools import chain, product, combinations
from typing import Iterable, Iterator, Self
from bisect import bisect_left, bisect_right

from .suffix_array import SuffixArray, BisectResult
from ..fragments.types import FragmentStateSpace, ResidueStateSpace, SingletonResult
from ..spectra.types import AugmentedPeaks

import numpy as np

def generate_unordered_combinations(
    residues: np.ndarray,       # [str; n]
    length: int,            
    suffix_array: SuffixArray,
) -> list[np.ndarray]:          # [[int; k]; _]
    indices = np.arange(residues.size)
    ordered_tuples = list(it.product(indices, repeat=length))
    # setup.

    ordered_residues = [''.join(residues[list(x)]) for x in ordered_tuples]
    occurrences = suffix_array.count(ordered_tuples)
    # filter by suffix array.

    comb = set()
    for (tup, count) in zip(ordered_tuples, occurrences):
        if count > 0:
            key = tuple(sorted(tup))
            comb.add(key)
    # project ordered tuples onto unordered combinations.

    return [np.array(x) for x in comb]

def _all_kmers(
    k: int,
    masses: np.ndarray,
    symbols: np.ndarray,
    applicable_modifications: list[list[int]],
) -> tuple[np.ndarray,np.ndarray,list[np.ndarray]]:
    """arg 'k' is the int kmer length, arg 'masses' is a 1d np array of n floats. arg 'symbols' is a 1d np array of n strings. returns a list of n^k (float, str) tuples, where each float is a sum of 'k' elements in 'masses', and each str is a concatenation of 'k' elements in 'symbols'."""
    n = len(masses)
    ind = list(range(n))
    kmer_ind = [list(i) for i in it.product(*[ind] * k)]
    kmer_masses = np.array([sum(masses[i]) for i in kmer_ind])
    kmer_symbols = np.array([''.join(symbols[i].tolist()) for i in kmer_ind])
    kmer_mods = [np.array(list(set(x for j in i for x in applicable_modifications[j]))) for i in kmer_ind]
    return (
        kmer_masses,
        kmer_symbols,
        kmer_mods,
    )

def all_kmers(
    residue_space: ResidueStateSpace,
    k: int,
) -> tuple[np.ndarray,np.ndarray,list[np.ndarray]]:
    return _all_kmers(
        k,
        residue_space.amino_masses,
        residue_space.amino_symbols,
        residue_space.applicable_modifications,
    )

@dataclasses.dataclass(slots=True)
class PeptideResult:
    query_mass: float
    mass_errors: np.ndarray     # [float; N]
    sequences: str              # [char; S]
    seq_offsets: np.ndarray     # [int; N + 1]
    modifications: np.ndarray   # [[int; M]; S]
    losses: np.ndarray          # [[int; L]; N]
    peak_indices: np.ndarray    # [int; P]
    peak_offsets: np.ndarray    # [int; N + 1]

    def __len__(self) -> int:
        return len(self.mass_errors)

    def get_mass_error(self, i: int) -> float:
        return self.mass_errors[i]

    def get_peptide(self, i: int) -> str:
        return self.sequences[self.seq_offsets[i]:self.seq_offsets[i+1]]

    def get_modifications(self, i: int) -> np.ndarray:
        return self.modifications[self.seq_offsets[i]:self.seq_offsets[i+1]]

    def get_losses(self, i: int) -> np.ndarray:
        return self.losses[i,:]

    def get_singleton_peak_indices(self, i: int) -> np.ndarray:
        return self.peak_indices[self.peak_offsets[i]:self.peak_offsets[i+1]]

    def __iter__(self) -> Iterator[tuple[float, str, np.ndarray]]:
        for i in range(len(self)):
            yield (
                self.mass_errors[i],
                self.get_peptide(i),
                self.get_modifications(i),
                self.get_losses(i),
                self.get_singleton_peak_indices(i),
            )

    @classmethod
    def from_hits(
        cls,
        query_mass: float,
        hits: list[tuple[str,BisectResult,float,list,tuple]],
        num_mods: int,
        num_losses: int,
    ) -> Self:
    #     print("PeptideResult.from_hits")
    #     print(hits)
        seqs, res, mass_err, mods, losses = zip(*hits)
        seq_str = ''.join(seqs)
        seq_offsets = np.cumsum([0,] + [len(x) for x in seqs])
        s = len(seq_str)
        modifications = np.zeros((s, num_mods),dtype=np.uint8)
        for (seq_num,mod_by_pos) in enumerate(mods):
            off = seq_offsets[seq_num]
    #         print(mod_by_pos)
            for x in mod_by_pos:
                if x == ():
                    continue
                pos, mod = x
                modifications[off + pos, mod] += 1
        n = len(mass_err)
        losses = np.zeros((n, num_losses),dtype=np.uint8)
        for (seq_num,loss_ids) in enumerate(losses):
            for i in loss_ids:
                losses[seq_num, i] += 1
        return cls(
            query_mass = query_mass,
            mass_errors = np.array(mass_err),
            sequences = seq_str,
            seq_offsets = seq_offsets,
            modifications = modifications,
            losses = losses,
            peak_indices = np.array([],dtype=int), # TODO
            peak_offsets = np.zeros_like(seq_offsets),
        )

def ord_to_idx(
    symbols: list[str],
) -> np.ndarray:
    lookup = np.full(128, -1, dtype=np.int8)
    for i, ch in enumerate(symbols):
        lookup[ord(ch)] = i
    return lookup

def _query_peaks(
    query: float,
    tolerance: float,
    mz: np.ndarray,
):
    """Returns an integer index if a hit is found, otherwise None."""
    hits = np.arange(
        bisect_left(mz, query - tolerance),
        bisect_right(mz, query + tolerance))
    if len(hits) == 0 or len(hits) == len(mz): # no match -> None
        return None
    if len(hits) == 1: # one match
        return hits.item()
    else: # multiple matches -> best fit
        return hits[np.argmin(np.abs(query - mz[hits]))]

def multi_combos(
    items: list,
    size: int,
):
    # print("multi_combos", items, size)
    yield ()
    # print('\t', ())
    for n in range(1, min(len(items), size) + 1):
    #     print('\t', list(combinations(items, n)))
        yield from combinations(items, n)

def mod_combinations(
    amino_idx: int,
    appl_mods: list[np.ndarray],
    max_num_mods: int,
):
    mods = appl_mods[amino_idx][1:].tolist() # mod 0 is the null mod, ignore it.
    return multi_combos(mods, max_num_mods)

def loss_combinations(
    loss_tally: np.ndarray,
    max_num_losses: int,
):
    losses = [i for (i, t) in enumerate(loss_tally) for _ in range(t)]
    return multi_combos(losses, max_num_losses)

def query_by_mass(
    query_mass: float,
    tolerance: float,
    suffix_array: SuffixArray,
    residue_space: ResidueStateSpace,
    fragment_space: FragmentStateSpace,
    peaks: AugmentedPeaks = AugmentedPeaks.empty(),
    max_peptide_length: int = 30,
) -> tuple[
    PeptideResult,
    SingletonResult,
]:
    amino_masses = residue_space.amino_masses
    n_aminos = len(amino_masses)
    amino_symbols = residue_space.amino_symbols
    amino_indexer = ord_to_idx(amino_symbols)
    seed_results = suffix_array.bisect(amino_symbols, prefix=None)
    mod_masses = residue_space.modification_masses
    appl_mods = residue_space.applicable_modifications
    max_num_mods = residue_space.max_num_modifications
    mods_per_residue = [
        list(mod_combinations(idx, appl_mods, max_num_mods))
        for idx in range(len(amino_masses))
    ]
    mods_max_mass = max(
        mod_masses[np.array(x)].sum()
        for x in chain.from_iterable(mods_per_residue)
        if len(x) > 0
    )
    # print("mods max mass", mods_max_mass)
    loss_masses = fragment_space.loss_masses
    appl_loss = fragment_space.applicable_losses
    max_num_loss = 3
    loss_max_mass = max_num_loss * np.max(fragment_space.loss_masses)
    # max_num_loss = fragment_space.max_num_losses TODO

    stack = []
    for (amino, result, mass) in zip(amino_symbols, seed_results, amino_masses):
        idx = amino_indexer[ord(amino)]
        # amino character -> ordinal value -> index from 0 to 19

        pos = 0
        initial_mods = [
            tuple([(pos,mod) for mod in combo]) 
            for combo in mods_per_residue[idx]
        ]
        mods_per_pos = [initial_mods,]
        # all possible combinations of mods at the first position.

        loss_tally = np.zeros(fragment_space.n_total_losses(),dtype=np.uint8)
        for i in appl_loss[idx][1:]:
            loss_tally[i] += 1
        # loss_tally is an encoding of a multiset of losses.
        
        stack.append((amino,idx,result,mass,loss_tally,mods_per_pos))

    hits = []
    counter = 0
    while stack:
        counter += 1
        seq, idx, result, mass, loss_tally, mods_per_pos = stack.pop()
    #     print(seq, mass, loss_tally, mods_per_pos)

        minimal_mass = mass - loss_max_mass
        if len(seq) > max_peptide_length or result.count == 0 or minimal_mass > query_mass + tolerance:
    #         print("prune", len(seq), result.count, minimal_mass)
            continue

        mod_space = [
            list(chain.from_iterable(state))
            for state in product(*mods_per_pos)
        ]
        # [[(i,j); _]; __] :: i is a position in seq and j is a mod id.

        loss_space = list(loss_combinations(loss_tally, max_num_loss))
        # [[i; _]; __] :: i is a loss id.

    #     print("mods",len(mod_space),"losses",len(loss_space))
        for (mod_state, loss_ids) in product(mod_space, loss_space):
    #         print("mod state =",mod_state)
            if mod_state == [()] or len(mod_state) == 0:
                mod_mass = 0
            else:
                mod_positions, mod_ids = zip(*mod_state)
                mod_mass = mod_masses[list(mod_ids)].sum()
    #         print("loss state=",loss_ids)
            loss_mass = loss_masses[list(loss_ids)].sum()
            augmented_mass = mass + mod_mass - loss_mass
    #         print(mod_state, mod_mass, loss_ids, loss_mass, augmented_mass)
            mass_error = augmented_mass - query_mass
            if abs(mass_error) < tolerance:
                hits.append((seq, result, mass_error, mod_state, loss_ids))

            # TODO - find singleton peaks

        new_res = suffix_array.bisect(amino_symbols,prefix = result)
        for new_idx in range(n_aminos):
            pos = 0
            new_mods = [
                tuple([(pos,mod) for mod in combo]) 
                for combo in mods_per_residue[new_idx]
            ]
            # all possible combinations of mods at the first position.

            new_loss_tally = np.copy(loss_tally)
            for i in appl_loss[new_idx][1:]:
                new_loss_tally[i] += 1
            # loss_tally is an encoding of a multiset of losses.
            
            amino_sym = amino_symbols[new_idx]
            amino_mass = amino_masses[new_idx]
            stack.append((
                seq + amino_sym,
                np.append(idx,new_idx),
                new_res[new_idx],
                mass + amino_mass,
                new_loss_tally,
                [x for x in mods_per_pos] + [new_mods,],
            ))
    # print("counter",counter)
    return (
        PeptideResult.from_hits(
            query_mass,
            hits,
            len(mod_masses),
            len(loss_masses),
        ),
        SingletonResult.empty(),
    )
