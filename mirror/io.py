import pyopenms as oms
from Bio import Seq, SeqRecord, SeqIO

def load_spectrum_from_mzML(path_to_mzML: str):
    exp = oms.MSExperiment()
    oms.MzMLFile().load(path_to_mzML, exp)
    return exp

def load_fasta_records(path_to_fasta: str):
    records = []
    with open(path_to_fasta) as handle:
        records = list(SeqIO.parse(handle, "fasta"))
    return records

def save_strings_to_fasta(path_to_fasta: str, seqs: list[str], get_id: lambda i: str(i)):
    n = len(seqs)
    records = [SeqRecord.SeqRecord(Seq.Seq(seqs[i]), id=get_id(i), name="", description="") for i in range(n)]
    with open(path_to_fasta, "w") as handle:
        return SeqIO.write(records, handle, "fasta")