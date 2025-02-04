import pyopenms as oms
from Bio import Seq, SeqRecord, SeqIO

from .types import *
from .gaps import GapResult
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

def read_gap_results(handle) -> list[GapResult]:
    results = []
    counter = -1
    group_id = -1
    group_vals = []
    gap_vals = []
    gap_left_indices = []
    gap_right_indices = []
    local_ids = []
    for line in handle.readlines():
        counter += 1
        if counter == 0:
            group_id = int(line)
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
            results.append(GapResult(group_id, target_group, gap_list))
            counter = -1
            group_id = -1
            group_vals = []
            gap_vals = []
            gap_left_indices = []
            gap_right_indices = []
            local_ids = []
    if counter != -1:
        raise Exception("malformed gap result file, at least one record could not be read.")
    result.sort(key = lambda x: x.group_id)
    return results

def load_gap_results(path_to_csv: str):
    with open(path_to_csv, 'r') as handle:
        return read_gaps_from_csv(handle)

# gaps output

def _write_gap_result(handle, result: GapResult):
    group_id = str(result.group_id)
    group_vals = comma_separated(result.group_values)
    gap_vals = comma_separated(result.values())
    index_arr = result.indices()
    gap_left_indices = comma_separated(index_arr[:, 1])
    gap_right_indices = comma_separated(index_arr[:, 2])
    local_ids = comma_separated(result.local_ids())
    for x in [group_id, group_vals, gap_vals, gap_left_indices, gap_right_indices, local_ids, ""]:
        handle.writeline(x)

def write_gap_results(handle, gap_structure: list[GapResult]):
    for result in gap_structure:
        _write_gap_result(handle, result)

def save_gap_results(path_to_csv: str, gap_structure: list[GapResult]):
    with open(path_to_csv, 'w') as handle:
        write_gaps_to_csv(handle, gap_structure)