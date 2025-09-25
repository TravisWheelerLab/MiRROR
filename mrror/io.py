import pyopenms as oms
import mzspeclib as mzlib
from mzspeclib.validate import ValidationWarning
from Bio import Seq, SeqRecord, SeqIO

import warnings
warnings.filterwarnings(action="ignore", category = ValidationWarning)

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
