import os, dataclasses, shutil, glob, enum
from typing import Self
# standard

import hydra
from omegaconf import DictConfig, OmegaConf
# hydra

from mrror.io import read_mzlib, read_mgf
from mrror.util import in_alphabet
from mrror.fragments.types import TargetMassStateSpace
from mrror.spectra.types import PeaksDataset, Peaks, SimulatedPeaks

from mrror.annotation import AnnotationParams, AnnotationResult, annotate
from mrror.alignment import AlignmentParams, AlignmentResult, align
from mrror.enumeration import EnumerationParams, EnumerationResult, enumerate_candidates
# library

def _sum_profiles(results, precision=4):
    return round(sum(t for res in results for t in res._profile.values()), precision)

@dataclasses.dataclass(slots=True)
class AppParams:
    name: str
    version: str
    descr: str
    verbosity: int
    time_precision: int
    header_symbol: str

    @classmethod
    def from_dict(
        cls,
        cfg: DictConfig,
        header_symbol = "::",
    ) -> Self:
        return cls(
            **cfg,
            header_symbol = header_symbol,
        )

    def __repr__(self) -> str:
        symbol = ' ' + self.header_symbol + ' '
        repeats = shutil.get_terminal_size().columns // len(symbol)
        header = symbol * repeats
        if self.verbosity == 0:
            return f"{self.name} v{self.version}"
        else:
            return f"{self.name} v{self.version}\n{header}\n{self.descr}"

class InputType(enum.Enum):
    MZLIB = "mzSpecLib"
    MGF = "MGF"
    FASTA = "Fasta"
    PEPTIDE = "Peptide"
    UNKNOWN = "???"

@hydra.main(version_base=None, config_path="params", config_name="config")
def main(cfg: DictConfig) -> None:
    app_cfg = AppParams.from_dict(cfg['app'])
    print(app_cfg)
    # configures the application.
    
    anno_params = AnnotationParams.from_config(cfg['annotation'])
    alphabet = anno_params.residue_space.amino_symbols
    if app_cfg.verbosity > 1:
        print(anno_params)
    # configures spectrum annotation, describes the state spaces of fragments and residues, as well as search and filtration parameters.
    
    algn_params = AlignmentParams.from_config(cfg['alignment'])
    if app_cfg.verbosity > 1:
        print(algn_params)
    # configures graph alignment. describes a cost model, alignment mode, and cost threshold.
    
    enmr_params = EnumerationParams.from_config(cfg['enumeration'])
    if app_cfg.verbosity > 1:
        print(enmr_params)
    # configures candidate enumeration.
    
    input = cfg['input']
    input_type = (
        InputType.MZLIB if input.endswith(".mzlib.txt") else 
        InputType.MGF if input.endswith(".mgf") else 
        InputType.FASTA if (input.endswith(".fasta") or input.endswith(".fa")) else
        InputType.PEPTIDE if (len(input) > 0 and in_alphabet(input.split('::')[0],alphabet)) else 
        InputType.UNKNOWN
    )
    peaks_data = None
    if input_type == InputType.MZLIB:
        peaks_data = PeaksDataset.from_mzml(input)
    elif input_type == InputType.MGF:
        peaks_data = PeaksDataset.from_mgf(input)
    elif input_type == InputType.FASTA:
        peaks_data = PeaksDataset.from_fasta(input)
    elif input_type == InputType.PEPTIDE:
        peaks_data = PeaksDataset.from_peptide(input)
    else:
        raise ValueError(f"Unrecognized input type. The input string {input} cannot be parsed!")
    print(f"* {input} [{input_type}]")
    if app_cfg.verbosity > 2:
        print(peaks_data)
    # parses the input field and from it constructs a spectrum.

    targets = TargetMassStateSpace.from_state_spaces(
        anno_params.residue_space,
        anno_params.fragment_space,
        anno_params.boundary_fragment_space
    )
    # admissible target masse for pair deltas and boundaries.

    enumerate_candidates(align(annotate(SimulatedPeaks.from_peptide("PEPTIDE"), targets, anno_params, verbose=True), targets, algn_params, verbose=True), targets, enmr_params, verbose=True)
    # precompile numba.jit functions
    
    anno_results = [
        annotate(peaks, targets, anno_params, verbose=app_cfg.verbosity > 1)
        for peaks in peaks_data
    ]
    anno_runtime = _sum_profiles(anno_results, precision=app_cfg.time_precision)
    print(f"| annotated in {anno_runtime}s")
    # runs spectrum annotation. identifies initial and terminal fragments, and pairs of fragments whose difference encodes the mass of a residue.
    
    algn_results = [
    #    align(anno_res, targets, algn_params, verbose=app_cfg.verbosity > 1)
    #    for anno_res in anno_results
    ]
    algn_runtime = _sum_profiles(algn_results, precision=app_cfg.time_precision)
    print(f"| aligned in {algn_runtime}s")
    # runs graph alignment. constructs spectrum graphs and their (sparse) product via cost propagation.
    
    enmr_results = [
    #    enumerate_candidates(algn_res, targets, enmr_params, verbose=app_cfg.verbosity > 1)
    #    for algn_res in algn_results
    ]
    enmr_runtime = _sum_profiles(enmr_results, precision=app_cfg.time_precision)
    print(f"| enumerated in {enmr_runtime}s")
    # runs candidate enumeration. candidate substrings are generated by depth-first search on the product graph, then stitched together using annotated data and an optional suffix array.

if __name__ == "__main__":
    main()
