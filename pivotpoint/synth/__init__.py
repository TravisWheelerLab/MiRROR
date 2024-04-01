from pyopenms import TheoreticalSpectrumGenerator, MSSpectrum, AASequence, Param
from random import uniform

def parametize_yb_spectrum():
    p = Param()
    
    p.setValue("add_b_ions", "true")
    p.setValue("add_a_ions", "false")
    
    return p

def parametize_yba_spectrum():
    p = Param()
    
    p.setValue("add_b_ions", "true")
    p.setValue("add_a_ions", "true")
    
    return p

def parametize_full_spectrum():
    p = Param()
    
    p.setValue("add_b_ions", "true")
    p.setValue("add_a_ions", "true")
    
    p.setValue("add_first_prefix_ion", "true")
    p.setValue("add_precursor_peaks", "true")
    p.setValue("add_all_precursor_charges", "true")
    p.setValue("add_losses", "true")
    
    return p

def generate_spectrum_from_sequence(seq: str, p: Param):
    generator = TheoreticalSpectrumGenerator()
    generator.setParameters(p)
    
    spectrum = MSSpectrum()
    peptide = AASequence.fromString(seq)
    
    # populate the spectrum
    generator.getSpectrum(spectrum, peptide, 1, 1)
    
    return list(map(lambda peak: peak.getMZ(), spectrum))

def generate_random_mz(lo: float, hi: float, nsample: int):
    return [uniform(lo,hi) for _ in range(nsample)]

def generate_gapset_from_sequence(seq: str, p: Param,max_gap=150):
    mz = generate_spectrum_from_sequence(seq,p) 
    n = len(mz)
    gapset = []
    for i in range(n):
        for j in range(i + 1,n):
            d = mz[j] - mz[i]
            if d < max_gap:
                gapset.append(d)
            else:
                break
    return gapset 