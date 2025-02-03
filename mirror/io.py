import pyopenms as oms
from Bio import Seq, SeqRecord, SeqIO

from .types import *
from .util import comma_separated

#=============================================================================#
# spectrum input

def load_spectrum_from_mzML(path_to_mzML: str):
    exp = oms.MSExperiment()
    oms.MzMLFile().load(path_to_mzML, exp)
    return exp

#=============================================================================#
# FASTA input

def read_fasta_records(handle):
    return list(SeqIO.parse(handle, "fasta"))

def load_fasta_records(path_to_fasta: str):
    records = []
    with open(path_to_fasta) as handle:
        return read_fasta_records(handle)

def load_fasta_as_strings(path_to_fasta: str):
    return list(map(lambda x: str(x.seq), load_fasta_records(path_to_fasta)))

# FASTA output

def write_records_to_fasta(handle, records):
    return SeqIO.write(records, handle, "fasta")

def save_strings_as_fasta(path_to_fasta: str, seqs: list[str], get_id: lambda i: str(i)):
    n = len(seqs)
    records = [SeqRecord.SeqRecord(Seq.Seq(seqs[i]), id=get_id(i), name="", description="") for i in range(n)]
    with open(path_to_fasta, "w") as handle:
        return write_records_to_fasta(handle, records)

#=============================================================================#
# peaks input

def read_peaks_from_csv(handle):
    intensities, peaks = handle.readlines()
    return split_commas(intensities, float), split_commas(peaks, float)

def load_peaks_from_csv(path_to_csv):
    with open(path_to_csv, 'r') as handle:
        return read_peaks_from_csv(handle)

# peaks output

def write_peaks_to_csv(handle, intensities: list[float], peaks: list[float]):
    return handle.write('\n'.join([
        comma_separated(intensities),
        comma_separated(peaks)
    ]))

def save_peaks_as_csv(path_to_csv: str, intensities: list[float], peaks: list[float]):
    with open(path_to_csv, 'w') as handle:
        return write_peaks_to_csv(handle, intensities, peaks)

#=============================================================================#
# target groups input

def read_target_groups(handle) -> list[TargetGroup]:
    pass

def write_target_groups(handle, target_groups: list[TargetGroup]):
    pass

#=============================================================================#
# gaps input

def read_gaps(handle) -> list[list[Gap]]:
    pass

def load_gaps(path_to_csv: str):
    with open(path_to_csv, 'r') as handle:
        return read_gaps_from_csv(handle)

# gaps output

def write_gaps(handle, gap_structure: list[list[Gap]]):
    pass

def save_gaps(path_to_csv: str, gap_structure: list[list[Gap]]):
    with open(path_to_csv, 'w') as handle:
        write_gaps_to_csv(handle, gap_structure)