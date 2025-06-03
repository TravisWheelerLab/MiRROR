import pyopenms as oms

def default_param():
    """Creates a pyopenms.Param() object with add_b_ions, add_y_ions, and add_metainfo set to true."""
    param = oms.Param()
    param.setValue("add_b_ions", "true")
    param.setValue("add_y_ions", "true")
    param.setValue("add_metainfo", "true")
    return param

DEFAULT_PARAM = default_param()

def generate_fragment_spectrum(seq: str, param: oms.Param):
    """From a string and a pyopenms.Param() object, uses a pyopenms.TheoreticalSpectrumGenerator
    object to create a simulated fragment spectrum as a pyopenms.MSSpectrum object."""
    tsg = oms.TheoreticalSpectrumGenerator()
    spec = oms.MSSpectrum()
    peptide = oms.AASequence.fromString(seq)
    tsg.setParameters(param)
    tsg.getSpectrum(spec, peptide, 1, 1)
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
    spectrum = generate_fragment_spectrum(peptide, DEFAULT_PARAM)
    return list_mz(spectrum)