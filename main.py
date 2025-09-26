import os, dataclasses, shutil, glob, enum
from typing import Self
# standard

import hydra
from omegaconf import DictConfig, OmegaConf
# hydra

from mrror.io import read_mzlib, read_mgf
from mrror.util import in_alphabet
from mrror.spectra.types import PeaksDataset, Peaks, SimulatedPeaks
from mrror.annotation import AnnotationParams, AnnotationResult, annotate
# from mrror.alignment import AlignmentParams, AlignmentResult
# from mrror.enumeration import EnumerationParams, EnumerationResult
# library

@dataclasses.dataclass(slots=True)
class AppParams:
    name: str
    version: str
    descr: str
    verbosity: int
    _header: str

    @classmethod
    def from_dict(
        cls,
        cfg: DictConfig,
        symbol = "::",
    ) -> Self:
        symbol = ' ' + symbol + ' '
        repeats = shutil.get_terminal_size().columns // len(symbol)
        return cls(
            **cfg,
            _header = symbol * repeats,
        )

    def __repr__(self) -> str:
        if self.verbosity == 0:
            return f"{self.name} v{self.version}"
        else:
            return f"{self.name} v{self.version}\n{self._header}\n{self.descr}"

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
    # prints the name + version unless app.minimal_input=false, in which case the MRROR title card is printed.

    anno_params = AnnotationParams.from_config(cfg['annotation'])
    alphabet = anno_params.residue_space.amino_symbols
    annotate(SimulatedPeaks.from_peptide("TEST"), anno_params)
    if app_cfg.verbosity > 1:
        print(anno_params)
    # configures spectrum annotation, describing the state spaces of fragments and residues, as well as search and filtration parameters.

    # configures graph alignment, setting alignment model, strategy, and threshold.

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
    print(f"> {input} [{input_type}]")
    if app_cfg.verbosity > 2:
        print(peaks_data)
    # parses the input field and from it constructs a spectrum.

    anno_results = [annotate(peaks, anno_params, verbose=cfg['app'].verbosity > 1) for peaks in peaks_data]
    anno_runtime = sum(t for res in anno_results for t in res._profile.values())
    print(f"| annotated in {anno_runtime}s")
    algn_results = []
    algn_runtime = sum(t for res in algn_results for t in res._profile.values())
    print(f"| aligned in {algn_runtime}s")
    enmr_results = []
    enmr_runtime = sum(t for res in enmr_results for t in res._profile.values())
    print(f"| enumerated in {enmr_runtime}s")
    # run annotation, alignment, and enumeration phases.

if __name__ == "__main__":
    main()
