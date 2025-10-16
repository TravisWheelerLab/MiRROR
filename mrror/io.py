from typing import Iterator

import mzspeclib as mzlib
import pyopenms as oms
from mzspeclib.validate import ValidationWarning
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import warnings
warnings.filterwarnings(action="ignore", category = ValidationWarning)

def write_str_to_fa(
    txt: Iterator[str],
    path_to_fa: str,
):
    return SeqIO.write(
        [SeqRecord(Seq(x), id=f"{i}") for (i, x) in enumerate(txt)],
        path_to_fa,
        "fasta",
    )

def read_str_from_fa(
    path_to_fa: str,
) -> Iterator[str]:
    with open(path_to_fa, 'r') as f:
        for record in SeqIO.parse(f, "fasta"):
            yield str(record.seq)

def read_mzml(
    filepath: str,
    **kwargs,
) -> oms.MSExperiment:
    "From a path string, load a .mzML file as a pyopenms.MSExperiment object."
    exp = oms.MSExperiment()
    oms.MzMLFile().load(path_to_mzML, exp, **kwargs)
    return exp

def read_mzlib(
    filepath: str,
    **kwargs,
) -> mzlib.SpectrumLibrary:
    """From a path string, load a .mzlib file as an MzSpecLib;
    wraps the mzspeclib.SpectrumLibrary object."""
    with warnings.catch_warnings(action="ignore"):
        return mzlib.SpectrumLibrary(filename = path_to_mzlib)

def read_mgf():
    pass

def write_mzlib():
    pass

def write_mgf():
    pass
