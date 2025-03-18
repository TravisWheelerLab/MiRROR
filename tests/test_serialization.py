import unittest

import mirror

import numpy as np

class TestSerialization(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.peptide = "AALLLLLLLLLLR"
        cls.residue_seq = list(cls.peptide)
        cls.mass_seq = [mirror.util.RESIDUE_MONO_MASSES[res] for res in cls.residue_seq]
        cls.peaks = mirror.util.simulate_peaks(cls.peptide)
        cls.symmetry_factor = np.pi
        cls.boundary_padding = 3
        cls.gap_key = "gap"
        cls.sufr_file = "_temp.sufr"
        cls.occurrence_threshold = 1
        cls.test_spectrum_file = "_temp.ts"


    def test_serialization_TestSpectrum(self):
        mirror.io.save_strings_as_fasta("_temp.fa", [self.peptide])
        mirror.affixes.SuffixArray.write("_temp.fa", self.sufr_file)
        test_spectrum = mirror.TestSpectrum(
            self.residue_seq,
            self.mass_seq,
            np.zeros_like(self.mass_seq),
            np.zeros_like(self.mass_seq),
            np.zeros_like(self.mass_seq),
            np.array([]),
            self.peaks,
            target_gaps = [],
            target_pivot = None,
            gap_search_parameters = mirror.DEFAULT_GAP_SEARCH_PARAMETERS,
            intergap_tolerance = mirror.util.INTERGAP_TOLERANCE,
            symmetry_factor = self.symmetry_factor,
            terminal_residues = mirror.util.TERMINAL_RESIDUES,
            boundary_padding = self.boundary_padding,
            gap_key = self.gap_key,
            suffix_array_file = self.sufr_file,
            occurrence_threshold = self.occurrence_threshold,
        )
        test_spectrum.write(self.test_spectrum_file)

        test_spectrum2 = mirror.TestSpectrum.read(self.test_spectrum_file)

        """
        print(len(test_spectrum._boundaries), len(test_spectrum2._boundaries))
        for (b1, b2) in zip(test_spectrum._boundaries, test_spectrum2._boundaries):
            print(b1)
            print(b2)
            print(b1 == b2, str(b1) == str(b2))
            print()
        """
        self.maxDiff = None
        self.assertEqual(
            list(test_spectrum._annotated_peaks), 
            list(test_spectrum2._annotated_peaks))
        self.assertEqual(
            list(map(lambda x: x.get_index_pairs(), test_spectrum._gaps)), 
            list(map(lambda x: x.get_index_pairs(), test_spectrum2._gaps)))
        self.assertEqual(test_spectrum._y_terminii, test_spectrum2._y_terminii)
        self.assertEqual(test_spectrum._pivots, test_spectrum2._pivots)
        self.assertEqual(test_spectrum._boundaries, test_spectrum2._boundaries)
        self.assertEqual(test_spectrum._graph_pairs, test_spectrum2._graph_pairs)
        self.assertEqual(
            list(map(list, mirror.util.collapse_second_order_list(test_spectrum._affixes))), 
            list(map(list, mirror.util.collapse_second_order_list(test_spectrum2._affixes))))
        self.assertEqual(test_spectrum._candidates, test_spectrum2._candidates)
        self.assertEqual(test_spectrum._indices, test_spectrum2._indices)