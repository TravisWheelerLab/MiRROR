import pyopenms as oms
import mzspeclib as mzlib
from mzspeclib.validate import ValidationWarning
from Bio import Seq, SeqRecord, SeqIO

import warnings
warnings.filterwarnings(action="ignore", category = ValidationWarning)

# the io module is a terminal object; it cannot be imported by any other local module.
from .types import *
from .gaps import read_gap_result, write_gap_result, read_gap_params, write_gap_params
from .util import collapse_second_order_list, comma_separated, split_commas, RESIDUES, MASSES, MONO_MASSES
from .preprocessing import MzSpecLib

#=============================================================================#
# spectrum input

def load_spectrum_from_mzML(path_to_mzML: str):
    "From a path string, load a .mzML file as a pyopenms.MSExperiment object."
    exp = oms.MSExperiment()
    oms.MzMLFile().load(path_to_mzML, exp)
    return exp

def load_spectra_from_mzSpecLib(path_to_mzlib: str):
    """From a path string, load a .mzlib file as an MzSpecLib;
    wraps the mzspeclib.SpectrumLibrary object."""
    with warnings.catch_warnings(action="ignore"):
        return MzSpecLib(mzlib.SpectrumLibrary(filename = path_to_mzlib))

#=============================================================================#
# FASTA input

def read_fasta_records(handle):
    """From an IO stream of a FASTA file, list fasta records:

        list(SeqIO.parse(handle, "fasta"))"""
    return list(SeqIO.parse(handle, "fasta"))

def load_fasta_records(path_to_fasta: str):
    """Given a path string to a FASTA file, open the file and list fasta records."""
    with open(path_to_fasta) as handle:
        return read_fasta_records(handle)

def load_fasta_as_strings(path_to_fasta: str):
    """Given a path string to a FASTA file, open the file and list the sequences of fasta records as strings."""
    return list(map(lambda x: str(x.seq), load_fasta_records(path_to_fasta)))

# FASTA output

def write_records_to_fasta(handle, records):
    """To an IO stream, write a list of records.
    
        SeqIO.write(records, handle, \"fasta\")"""
    return SeqIO.write(records, handle, "fasta")

def save_strings_as_fasta(path_to_fasta: str, seqs: list[str], get_id = lambda i: str(i)):
    """Given a path, open the file and write a list of sequences using the BioPython SeqRecord interface."""
    n = len(seqs)
    records = [SeqRecord.SeqRecord(Seq.Seq(seqs[i]), id=get_id(i), name="", description="") for i in range(n)]
    with open(path_to_fasta, "w") as handle:
        return write_records_to_fasta(handle, records)

#=============================================================================#
# peaks input

def read_peaks_from_csv(handle):
    """From an IO stream to a CSV file of peaks, read the intensities and peaks as lists of floats."""
    intensities, peaks = handle.readlines()
    return split_commas(intensities, float), split_commas(peaks, float)

def load_peaks_from_csv(path_to_csv):
    """Given a path string to a CSV file of peaks, open the file and read the intensities and peaks as lists of floats."""
    with open(path_to_csv, 'r') as handle:
        return read_peaks_from_csv(handle)

# peaks output

def write_peaks_to_csv(handle, intensities: list[float], peaks: list[float]):
    """To an IO stream, write intensity and peak lists in CSV format."""
    return handle.write('\n'.join([
        comma_separated(intensities),
        comma_separated(peaks)
    ]))

def save_peaks_as_csv(path_to_csv: str, intensities: list[float], peaks: list[float]):
    """Given a path string, open the file and write intensity and peak lists in CSV format."""
    with open(path_to_csv, 'w') as handle:
        return write_peaks_to_csv(handle, intensities, peaks)
