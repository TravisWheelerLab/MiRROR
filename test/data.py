import unittest
from pivotpoint import *
import random
from os import path

class DataGen(unittest.TestCase):
    def setUp(self):
        self.dir = "./test/input/"
        self.gapset = AMINO_ACID_MASS_AVERAGE
        sprot = read_fasta_to_list("/home/georgeglidden/datasets/sprot.fasta")
        random.shuffle(sprot)
        small_sprot = sprot[0:100]
        small_sequences = [x[2:27] for x in small_sprot]
        write_list_to_fasta("/home/georgeglidden/projects/pivotpoint/test/input/test_sequences_small.fasta",small_sequences)
        
    def test_create_pivots_from_random_params(self):
        # dataset from random parametization
        nsamples = 100
        value_range = [100,2000]
        radius_range = [1 + min(value_range) + max(self.gapset),5]
        path_rand_pivots = path.join(self.dir, "test_pivots_random.txt")
        create_random_pivots_data(path_rand_pivots,self.gapset,nsamples,value_range,radius_range)

    def test_create_pivots_from_sequence_data(self):
        # datasets from sequence
        path_sequences = path.join(self.dir, "test_sequences_small.fasta")
        sequences = read_fasta_to_list(path_sequences)
        path_format_sequence_pivots = path.join(self.dir, "test_pivots_sequence")
        create_sequence_yb_pivots_data(path_format_sequence_pivots,self.gapset,sequences)

    def test_create_pivots_from_noKR_sequence_data(self):
        # datasets from sequence
        path_sequences = path.join(self.dir, "noKR_sequences.fasta")
        sequences = read_fasta_to_list(path_sequences)
        path_format_sequence_pivots = path.join(self.dir, "noKR_pivots_sequence")
        create_sequence_yb_pivots_data(path_format_sequence_pivots,self.gapset,sequences)

def create_random_pivots_data(outpath: str, gapset: list[float], nsamples: int, value_range: tuple[float,float], radius_range: tuple[float,float]):
    data = [pivot_from_params(
                random.uniform(value_range[0],value_range[1]),
                gapset[random.randint(0,len(gapset) - 1)],
                random.uniform(radius_range[0],radius_range[1]))
            for _ in range(nsamples)]
    io.write_pivots_to_csv(outpath,data)

def create_sequence_yb_pivots_data(outpath: str, gapset: list[float], sequences: list[str]):
    n = len(sequences)
    for i in range(n):
        sequence = sequences[i]
        outpath_sequence_pivots = f"{outpath}{i}.txt"
        mz_values = generate_spectrum_from_sequence(sequence,parametize_yb_spectrum())
        data = find_pivoting_interval_pairs(mz_values,gapset,0.1)
        io.write_pivots_to_csv(outpath_sequence_pivots,data)

def create_clusters_data():
    pass

def create_samples_data():
    pass

