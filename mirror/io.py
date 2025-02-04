import pyopenms as oms
import mzspeclib as mzlib
from mzspeclib.validate import ValidationWarning
from Bio import Seq, SeqRecord, SeqIO

import warnings
warnings.filterwarnings(action="ignore", category = ValidationWarning)

from .types import *
from .gaps import GapResult
from .util import collapse_second_order_list, comma_separated, split_commas, AMINOS, AMINO_MASS, AMINO_MASS_MONO
from .preprocessing import MzSpecLib

#=============================================================================#
# spectrum input

def load_spectrum_from_mzML(path_to_mzML: str):
    exp = oms.MSExperiment()
    oms.MzMLFile().load(path_to_mzML, exp)
    return exp

def load_spectra_from_mzSpecLib(path_to_mzlib: str):
    with warnings.catch_warnings(action="ignore"):
        return MzSpecLib(mzlib.SpectrumLibrary(filename = path_to_mzlib))

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

def save_strings_as_fasta(path_to_fasta: str, seqs: list[str], get_id = lambda i: str(i)):
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

def read_target_groups(handle) -> tuple[list[TargetGroup], list[str]]:
    residues = []
    target_groups = []
    for line in handle.readlines():
        residue, groupstr = line.split(':')
        group_values = split_commas(groupstr, float)
        residues.append(residue)
        target_groups.append(group_values)
    return (target_groups, residues)

def load_target_groups(path_to_dlm):
    with open(path_to_dlm, 'r') as handle:
        return read_target_groups(handle)

# target groups output

def write_target_groups(handle, target_groups: list[TargetGroup], residues: list[str]):
    for (residue, group_values) in zip(residues, target_groups):
        groupstr = comma_separated(group_values)
        handle.write(f"{residue}:{groupstr}\n")

def save_target_groups(path_to_dlm, target_groups: list[TargetGroup], residues: list[str]):
    with open(path_to_dlm, 'w') as handle:
        write_target_groups(handle, target_groups, residues)

def save_default_target_groups(path_to_dlm):
    target_groups = [[x] for x in AMINO_MASS]
    residues = AMINOS
    save_target_groups(
        path_to_dlm, 
        target_groups,
        residues)

def save_mono_target_groups(path_to_dlm):
    target_groups = [[x] for x in AMINO_MASS_MONO]
    residues = AMINOS
    save_target_groups(
        path_to_dlm, 
        target_groups,
        residues)

def save_combined_target_groups(path_to_dlm):
    target_groups = [[m, m_mono] for (m, m_mono) in zip(AMINO_MASS, AMINO_MASS_MONO)]
    residues = AMINOS
    save_target_groups(
        path_to_dlm, 
        target_groups,
        residues)

def save_complete_target_groups(path_to_dlm):
    target_groups = [[m, m_mono] for (m, m_mono) in zip(AMINO_MASS, AMINO_MASS_MONO)]
    target_groups = [collapse_second_order_list([v, v - 17, v - 18, v + 17, v + 18] for v in x) for x in target_groups]
    residues = AMINOS
    save_target_groups(
        path_to_dlm, 
        target_groups,
        residues)


#=============================================================================#
# gaps input

def read_gap_results(handle) -> list[GapResult]:
    results = []
    counter = -1
    group_id = -1
    group_res = ""
    group_vals = []
    gap_vals = []
    gap_left_indices = []
    gap_right_indices = []
    local_ids = []
    for line in handle.readlines():
        counter += 1
        if counter == 0:
            group_id += 1
            group_res = line
        elif counter == 1:
            group_vals = split_commas(line, float)
        elif counter == 2:
            gap_vals = split_commas(line, float)
        elif counter == 3:
            gap_left_indices = split_commas(line, int)
        elif counter == 4:
            gap_right_indices = split_commas(line, int)
        elif counter == 5:
            local_ids = split_commas(line, int)
        elif line == "":
            gap_list = list(zip(gap_vals, zip(gap_left_indices, gap_right_indices), local_ids))
            results.append(GapResult(group_id, group_res, target_group, gap_list))
            counter = -1
            group_vals = []
            gap_vals = []
            gap_left_indices = []
            gap_right_indices = []
            local_ids = []
    if counter != -1:
        raise Exception("malformed gap result file, at least one record could not be read.")
    return results

def load_gap_results(path_to_csv: str):
    with open(path_to_csv, 'r') as handle:
        return read_gaps_from_csv(handle)

# gaps output

def _write_gap_result(handle, result: GapResult):
    group_res = result.group_residue
    group_vals = comma_separated(result.group_values)
    gap_vals = comma_separated(result.values())
    index_arr = result.indices()
    gap_left_indices = comma_separated(index_arr[:, 0])
    gap_right_indices = comma_separated(index_arr[:, 1])
    local_ids = comma_separated(result.local_ids())
    for x in [group_res, group_vals, gap_vals, gap_left_indices, gap_right_indices, local_ids, "end"]:
        if x == "end":
            handle.write('\n')
        elif len(x) > 0:
            handle.write(x+'\n')
        else:
            handle.write(',\n')
def write_gap_results(handle, gap_structure: list[GapResult]):
    gap_structure.sort(key = lambda result: result.group_id)
    for result in gap_structure:
        _write_gap_result(handle, result)

def save_gap_results(path_to_csv: str, gap_structure: list[GapResult]):
    with open(path_to_csv, 'w') as handle:
        write_gaps_to_csv(handle, gap_structure)