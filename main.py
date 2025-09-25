import os, dataclasses, shutil, glob, enum
from typing import Self
# standard

import hydra
from omegaconf import DictConfig, OmegaConf
# hydra

from mrror.io import read_mzlib, read_mgf
from mrror.util import in_alphabet
from mrror.spectra.types import PeaksDataset, Peaks
from mrror.annotation import AnnotationParams, AnnotationResult, annotate
# from mrror.alignment import AlignmentParams, AlignmentResult
# from mrror.enumeration import EnumerationParams, EnumerationResult
# library

@dataclasses.dataclass(slots=True)
class AppParams:
    name: str
    version: str
    descr: str
    minimal_output: bool
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
        if self.minimal_output:
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
    print(f"> {input} [{input_type}]\n\n{peaks_data}")
    # parses the input field and from it constructs a spectrum.

    # run annotation, alignment, and enumeration phases.
    anno_results = [annotate(peaks, anno_params) for peaks in peaks_data]
    algn_results = None
    enmr_results = None

if __name__ == "__main__":
    main()
