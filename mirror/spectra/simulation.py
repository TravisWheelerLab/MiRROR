import pyopenms as oms

def default_param():
    """Creates a pyopenms.Param() object with add_b_ions, add_y_ions, and add_metainfo set to true."""
    param = oms.Param()
    param.setValue("add_b_ions", "true")
    param.setValue("add_y_ions", "true")
    param.setValue("add_metainfo", "true")
    return param

DEFAULT_PARAM = default_param()

def complex_param():
    """Creates a pyopenms.Param() object with add_b_ions, add_y_ions, add_metadata,
    add_all_precursor_charges, and add_losses set to true."""
    param = oms.Param()
    param.setValue("add_b_ions", "true")
    param.setValue("add_y_ions", "true")
    param.setValue("add_all_precursor_charges", "true")
    param.setValue("add_losses", "true")
    param.setValue("add_metainfo", "true")
    return param

COMPLEX_PARAM = complex_param()

def generate_fragment_spectrum(seq: str, param: oms.Param, min_charge = 1, max_charge = 1):
    """From a string and a pyopenms.Param() object, uses a pyopenms.TheoreticalSpectrumGenerator
    object to create a simulated fragment spectrum as a pyopenms.MSSpectrum object."""
    tsg = oms.TheoreticalSpectrumGenerator()
    spec = oms.MSSpectrum()
    peptide = oms.AASequence.fromString(seq)
    tsg.setParameters(param)
    tsg.getSpectrum(spec, peptide, min_charge, max_charge)
    return spec

def list_mz(spec: oms.MSSpectrum):
    """Creates a numpy array of the peaks of a pyopenms.MSSpectrum object.
    
        np.array([peak.getMZ() for peak in spec])
    
    :spec: a pyopenms.MSSpectrum object."""
    return [peak.getMZ() for peak in spec]

def list_intensity(spec: oms.MSSpectrum):
    """Creates a numpy array of the intensities of a pyopenms.MSSpectrum object.
    
        np.array([peak.getIntensity() for peak in spec])
    
    :spec: a pyopenms.MSSpectrum object."""
    return [peak.getIntensity() for peak in spec]

def simulate_simple_peaks(peptide: str):
    return list_mz(generate_fragment_spectrum(peptide, DEFAULT_PARAM))

def simulate_complex_peaks(peptide: str):
    return list_mz(generate_fragment_spectrum(peptide, COMPLEX_PARAM))
    
