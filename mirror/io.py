from pyteomics import mgf
import mzspeclib as mzlib
from mzspeclib.validate import ValidationWarning as mzlib_ValidationWarning
import warnings
warnings.filterwarnings(action="ignore", category = mzlib_ValidationWarning)


def read_mzlib(path_to_mzlib: str) -> mzlib.SpectrumLibrary:
    """From a path string, load the mzlib file as an mzspeclib.SpectrumLibrary."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mzlib.SpectrumLibrary(filename = path_to_mzlib)

def read_mgf(path_to_mgf: str) -> mgf.IndexedMGF:
    """From a path string, load the MGF file as a pyteomics.mgf.IndexedMGF."""
    return mgf.read(path_to_mgf)