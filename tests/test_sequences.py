import unittest
import itertools as it

import numpy as np

from mirror.presets import MONO_RESIDUE_SPACE, FRAGMENT_SPACE

from mirror.sequences import SuffixArray
class TestSuffixArray(unittest.TestCase):

    @staticmethod    
    def get_suffix_array():
        return SuffixArray.create(
            path_to_fasta = "./data/sequences/test.fasta",
            path_to_suffix_array = "./data/output/test.sufr")
    
    def test_create(self):
        self.get_suffix_array()
    
    def test_count(self):
        queries = ["HVV", "ILE", "TPP"]
        counts = [2,2,9]
        self.assertEqual(
            counts,
            self.get_suffix_array().count(queries))

    @staticmethod
    def get_kmers(
        k: int,
        alphabet = MONO_RESIDUE_SPACE.amino_symbols,
        join = lambda x: ''.join(x),
    ) -> list[str]:
        return [join(x) for x in it.product(*it.repeat(alphabet, k))]

    def test_bisect(self):
        # set up
        K = 5
        alphabet = MONO_RESIDUE_SPACE.amino_symbols
        suffix_array = self.get_suffix_array()
        # iter seqs
        seqs = ['']
        states = [None]
        for k in range(K):
            # print(f"k{k}")
            new_seqs = []
            new_states = []
            for (seq, state) in zip(seqs, states):
                # print(f"seq {seq}")
                for c in alphabet:
                    new_seq = seq + c
                    new_state = suffix_array.bisect([c],state)[0]
                    bisect_count = new_state.count
                    count = suffix_array.count([new_seq])[0]
                    # print(f"\tnew seq {new_seq} {bisect_count} {count}")
                    self.assertEqual(bisect_count, count)
                    if bisect_count > 0:
                        new_seqs.append(new_seq)
                        new_states.append(new_state)
            seqs = new_seqs
            states = new_states

    def test_repr(self):
        self.get_suffix_array().__repr__()

from mirror.sequences import PeptideMassQueryEngine, query_mass, query_kmers
from random import choice
class TestQueries(unittest.TestCase):

    @staticmethod
    def get_engine():
        return PeptideMassQueryEngine(
            fragment_space = FRAGMENT_SPACE,
            residue_space = MONO_RESIDUE_SPACE,
            suffix_array = TestSuffixArray.get_suffix_array(),
            max_num_losses = 0, # unused
            max_num_modifications = 0, # unused
            max_iter = 10,
        )

    def test_PeptideMassQueryEngine(self):
        """Test the PeptideMassQueryEngine constructor."""
        # create the engine
        engine = self.get_engine()
        # initialize its state
        seqs = [""]
        masses = [0.]
        states = [None]
        symbols = engine.residue_space.amino_symbols
        # iterate once: seqs should be all single-residue symbols
        seqs, masses, states = engine.step(seqs,masses,states)
        self.assertTrue(
            all(seqs == symbols)
        )
        # iterate again: seqs should be a subset of double-residue symbols
        seqs, masses, states = engine.step(seqs,masses,states)
        self.assertTrue(
            set(seqs).issubset(
                [r1 + r2 for (r1, r2) in it.product(symbols, symbols)])
        )

    def test_query_kmers(self):
        """Test the kmer query function."""
        engine = self.get_engine()
        for k in range(5):
            print(f"k{k+1}",end=',',flush=True)
            hits = [seq.tolist() for (_,seq) in query_kmers(engine, k + 1)]
            gt = [seq for seq in TestSuffixArray.get_kmers(k + 1) if engine.suffix_array.count([seq])[0] > 0]
            self.assertEqual(
                hits,
                gt,
            )

    def test_query_mass(self):
        """Test the mass query function."""
        engine = self.get_engine()
        engine.max_iter = 5
        # set up the solution space
        sym_to_mass = {s: m for (s,m) in zip(MONO_RESIDUE_SPACE.amino_symbols, MONO_RESIDUE_SPACE.amino_masses)}
        all_outputs = [
            x
            for k in range(5)
            for x in query_kmers(engine, k + 1)]
        # run the queries
        T = 200
        t = 0.01
        for i in range(T):
            print(f"i{i}",end=',',flush=True)
            mass, seq = choice(all_outputs)
            gt = [(m,s) for (m,s) in all_outputs if abs(mass - m) < t]
            hits = query_mass(engine, mass, t)
            # print("\ngt\n\t",gt)
            # print("\nhits\n\t",hits)
            self.assertEqual(
                gt,
                hits,
            )

