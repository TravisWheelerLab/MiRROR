import os, dataclasses, shutil, glob, enum, pathlib
from typing import Self
from time import time
import itertools as it
# standard

import hydra
from omegaconf import DictConfig, OmegaConf
# hydra

from mrror.io import read_mzlib, read_mgf, reverse_fasta
from mrror.util import in_alphabet
from mrror.fragments.types import TargetMassStateSpace
from mrror.spectra.types import SpectraParams, PeaksDataset, Peaks, SimulatedPeaks
from mrror.sequences.suffix_array import SuffixArray
from mrror.evaluation.costmodels import AnnotatedResiduePathCostModel
from mrror.evaluation.annotated_peaks import AnnotatedPeaks
from mrror.annotation import AnnotationParams, AnnotationResult, annotate
from mrror.alignment import AlignmentParams, AlignmentResult, align
from mrror.enumeration import EnumerationParams, EnumerationResult, enumerate_candidates
# library

import numpy as np
import editdistance as ed

def _sum_profiles(results, precision=4):
    return round(sum(t for res in results for t in res._profile.values()), precision)

@dataclasses.dataclass(slots=True)
class AppParams:
    name: str
    version: str
    descr: str
    verbosity: int
    time_precision: int
    mode: str
    working_dir: str
    output_dir: str
    output_shape: str
    session_name: str

    @classmethod
    def from_dict(
        cls,
        cfg: DictConfig,
    ) -> Self:
        return cls(
            **cfg,
        )

    def __repr__(self) -> str:
        symbol = '-'
        repeats = shutil.get_terminal_size().columns // len(symbol)
        header = symbol * repeats
        if self.verbosity == -1:
            return f"{self.name} v{self.version}\n{header}\n{self.descr}"
        else:
            return f"{self.name} v{self.version}"

class InputType(enum.Enum):
    MZLIB = "mzSpecLib"
    MGF = "MGF"
    FASTA = "Fasta"
    PEPTIDE = "Peptide"
    UNKNOWN = "???"

def setup(cfg):
    app_cfg = AppParams.from_dict(cfg.app)
    # configures the application.

    spec_cfg = SpectraParams.from_dict(cfg.spectra)
    # configures spectra simulation.

    output_dir = pathlib.Path(app_cfg.output_dir)
    if not(output_dir.exists()):
        output_dir.mkdir(parents=True, exist_ok=True)
    # constructs the output dir

    working_dir = pathlib.Path(app_cfg.working_dir)
    if not(working_dir.exists()):
        working_dir.mkdir(parents=True, exist_ok=True)
    # constructs the working dir
    
    suffix_array, reversed_suffix_array = (None, None)
    transcriptome = pathlib.Path(cfg.transcriptome)
    if transcriptome.suffix in ('.fa','.fasta'):
        suffix_array = SuffixArray.create(
            path_to_fasta = str(transcriptome),
            path_to_suffix_array = str(working_dir / f"{transcriptome.stem}.sufr"),
        )
        reversed_suffix_array = SuffixArray.create(
            path_to_fasta = reverse_fasta(str(transcriptome)),
            path_to_suffix_array = str(working_dir / f"{transcriptome.stem}_reversed.sufr"),
        )
    elif transcriptome.suffix == '.sufr':
        suffix_array = SuffixArray.read(str(transcriptome))
        reversed_transcriptome = transcriptome.parent / (transcriptome.stem + '_reversed' + transcriptome.suffix)
        if reversed_transcriptome.exists():
            reversed_suffix_array = SuffixArray.read(str(reversed_transcriptome))
        else:
            raise ValueError(f"A sufr file {transcriptome} was passed but there is no corresponding reversed suffix array, which was expected at {reversed_transcriptome}.")
    else:
        print(f"The transcriptome argument was \"{transcriptome}\" which cannot be parsed as either a .fasta or .sufr file. Evaluation will proceed without a suffix array.")
    suffix_arrays = (suffix_array, reversed_suffix_array)
    # constructs the suffix array and its reversal, which restrict the space of peptides that can be called from spectra.
    
    anno_params = AnnotationParams.from_config(cfg.annotation)
    if app_cfg.verbosity > 1:
        print(anno_params)
    print("residues\n",anno_params.residue_space)
    print("pair fragments\n",anno_params.fragment_space)
    print("boundary fragments\n",anno_params.boundary_fragment_space)
    input()
    # parametizes spectrum annotation, describes the state spaces of fragments and residues, as well as search and filtration parameters.
    
    algn_params = AlignmentParams.from_config(cfg.alignment)
    if app_cfg.verbosity > 1:
        print(algn_params)
    # parametizes graph alignment. describes a cost model, alignment mode, and cost threshold.
    
    enmr_params = EnumerationParams.from_config(cfg.enumeration)
    if app_cfg.verbosity > 1:
        print(enmr_params)
    # parametizes candidate enumeration.
    
    targets = TargetMassStateSpace.from_state_spaces(
        [anno_params.residue_space, anno_params.dimer_space, anno_params.trimer_space],
        anno_params.fragment_space,
        anno_params.boundary_fragment_space,
    )
    # admissible target masses for pair deltas and boundaries.

    return (cfg, app_cfg, spec_cfg, output_dir, working_dir, suffix_arrays, anno_params, algn_params, enmr_params, targets)

def evaluate(cfg, app_cfg, spec_cfg, output_dir, working_dir, suffix_arrays, anno_params, algn_params, enmr_params, targets) -> None:
    t = time()

    _anno = annotate(SimulatedPeaks.from_peptide("PEPTIDE"), targets, anno_params, verbose=False); enumerate_candidates(_anno, align(_anno, targets, algn_params, verbose=False), targets, suffix_arrays, enmr_params, verbose=False)
    precompile_runtime = time() - t
    # precompile numba.jit functions

    input_str = cfg.input
    alphabet = anno_params.residue_space.amino_symbols
    input_type = (
        InputType.MZLIB if input_str.endswith(".mzlib.txt") else 
        InputType.MGF if input_str.endswith(".mgf") else 
        InputType.FASTA if (input_str.endswith(".fasta") or input_str.endswith(".fa")) else
        InputType.PEPTIDE if (len(input_str) > 0 and in_alphabet(input_str.split('::')[0],alphabet)) else 
        InputType.UNKNOWN
    )
    peaks_data = None
    if input_type == InputType.MZLIB:
        peaks_data = PeaksDataset.from_mzml(input_str)
    elif input_type == InputType.MGF:
        peaks_data = PeaksDataset.from_mgf(input_str)
    elif input_type == InputType.FASTA:
        peaks_data = PeaksDataset.from_fasta(input_str)
    elif input_type == InputType.PEPTIDE:
        peaks_data = PeaksDataset.from_peptide(input_str, initial_b=spec_cfg.initial_b)
    else:
        raise ValueError(f"Unrecognized input type. The input string {input_str} cannot be parsed!")
    print(f"\n* {input_str} [{input_type}]")
    if app_cfg.verbosity > 2:
        print(peaks_data)
    # parses the input field and from it constructs a spectrum.
    
    anno_results = [
        annotate(peaks, targets, anno_params, verbose=app_cfg.verbosity > 1)
        for peaks in peaks_data
    ]
    anno_runtime = _sum_profiles(anno_results, precision=app_cfg.time_precision)
    for (i,res) in enumerate(anno_results):
        output_path = output_dir / f"{app_cfg.session_name}_{i}.anno"
        res.save(output_path)
    if app_cfg.verbosity > 0:
        print(f"| annotation\t{anno_runtime}s")
    # runs and records spectrum annotation. identifies initial and terminal fragments, and pairs of fragments whose difference encodes the mass of a residue. exports peak annotations. constructs spectrum graphs.
    
    algn_results = [
       align(anno_res, targets, algn_params, verbose=app_cfg.verbosity > 1)
       for anno_res in anno_results
    ]
    algn_runtime = _sum_profiles(algn_results, precision=app_cfg.time_precision)
    for (i,res) in enumerate(algn_results):
        output_path = output_dir / f"{app_cfg.session_name}_{i}.algn"
        res.save(output_path)
    if app_cfg.verbosity > 0:
        print(f"| alignment\t{algn_runtime}s")
    # runs and records graph alignment. constructs a sparse product between spectrum graphs using propagation.
    
    enmr_results = [
       enumerate_candidates(anno_res, algn_res, targets, suffix_arrays, enmr_params, verbose=app_cfg.verbosity > 1)
       for (anno_res, algn_res) in zip(anno_results, algn_results)
    ]
    enmr_runtime = _sum_profiles(enmr_results, precision=app_cfg.time_precision)
    for (i,res) in enumerate(enmr_results):
        output_path = output_dir / f"{app_cfg.session_name}_{i}.enmr"
        res.save(output_path)
    if app_cfg.verbosity > 0:
        print(f"| enumeration\t{enmr_runtime}s")
    # runs and records candidate enumeration. candidate substrings are generated by depth-first search on the product graph, then stitched together using annotated data and an optional suffix array.

    eval_total_runtime = round(anno_runtime + algn_runtime + enmr_runtime, app_cfg.time_precision)
    total_runtime = round(time() - t, app_cfg.time_precision)
    if app_cfg.verbosity > 0:
        print(f"\nevaluate: {eval_total_runtime}s, runtime: {total_runtime}s")

def test(cfg, app_cfg, spec_cfg, output_dir, working_dir, suffix_arrays, anno_params, algn_params, enmr_params, targets, *args, **kwargs):
    print("MRROR test!")
    app_cfg = AppParams.from_dict(cfg['app'])

    spec_cfg = SpectraParams.from_dict(cfg['spectra'])
    
    anno_params = AnnotationParams.from_config(cfg['annotation'])
    alphabet = anno_params.residue_space.amino_symbols
    
    algn_params = AlignmentParams.from_config(cfg['alignment'])
    
    enmr_params = EnumerationParams.from_config(cfg['enumeration'])
    
    targets = TargetMassStateSpace.from_state_spaces(
        [anno_params.residue_space, anno_params.dimer_space, anno_params.trimer_space],
        anno_params.fragment_space,
        anno_params.boundary_fragment_space,
    )

    peaks = SimulatedPeaks.from_peptide(cfg['input'],mode='complex',initial_b=spec_cfg.initial_b)


    print("peptide\n- ", peaks.peptide)
    print("peaks\n- ", peaks.mz)
    print("pivot\n- ", peaks.pivot)
    print("left boundaries\n- ", peaks.lower_boundaries())
    print("right boundaries\n- ", peaks.upper_boundaries())
    print("pairs\n- ", peaks.pairs())
    print("paths\n- ", peaks.paths())
    try:
        true_alignments = peaks.alignments()
        print("alignments\n- ", true_alignments)
    except ValueError:
        print("alignments could not be produced for this spectrum.")

    anno_res = annotate(peaks, targets, anno_params, verbose=app_cfg.verbosity > 3)
    anno_res.save(output_dir / f"{app_cfg.session_name}.anno")

    print("annotated pivots\n", anno_res.pivots.cluster_points)
    print("annotated pairs\n", anno_res.pairs.symbolize(targets))
    print("annotated boundaries\n- left\n", anno_res.lower_boundaries.symbolize(targets))
    print("- right")
    for rb in anno_res.upper_boundaries:
        print(rb.symbolize(targets))
    print("anno symmetries: ", [sym.tolist() for sym in anno_res.pivots.symmetries], '\n', [np.round(peaks.mz[sym], 4).tolist() for sym in anno_res.pivots.symmetries])

    algn_res = align(anno_res, targets, algn_params, verbose=app_cfg.verbosity > 3)

    print("algn symmetries: ", [sym[:-2].tolist() for sym in algn_res.symmetries], '\n', [np.round(algn_res.fragment_masses[sym[:-2,:]], 4).tolist() for sym in algn_res.symmetries])

    sym_lookup = [
        {
            tuple(y.tolist()): tuple(x.tolist())
            for (x,y) in zip(
                anno_res.pivots.symmetries[i],
                algn_res.symmetries[i],
            )
        } 
        for i in range(len(anno_res))
    ]

    print("sym lookup:",sym_lookup)
    
    masses = algn_res.fragment_masses
    print("fragment masses:", masses)

    if app_cfg.verbosity > 0:
        print("left topologies", '\n'.join([x.display() for x in algn_res.lower_topology]))
        print("right topologies", '\n'.join([x.display() for x in algn_res.upper_topology]))

    enmr_res = enumerate_candidates(anno_res, algn_res, targets, suffix_arrays, enmr_params, verbose=app_cfg.verbosity > 3)

    print("aligned affixes:", sum([len(x) for x in enmr_res.aligned_affixes]))
    print("prefixes + suffixes:", sum([len(x) for x in enmr_res.prefixes] + [len(x) for x in enmr_res.suffixes]))
    n = len(enmr_res)
    for i in range(n):
        for (tag,affix_category) in (("prefix",enmr_res.prefixes[i]),("suffix",enmr_res.suffixes[i]),("infix",enmr_res.infixes[i])):
            for (afx_idx,bnd_idx) in affix_category:
                cost, path, anno = enmr_res.aligned_affixes[i][afx_idx][:3]
                anno_residues = [u[:,0] for u in anno]
                anno_loss = [u[:,2] for u in anno]
                term = anno_loss[-1][bnd_idx]
                unraveled_path = [[int(y) for y in algn_res.prod_topology[i].unravel(x)] for x in path]
                translated_path = []
                for x in unraveled_path[:-1]:
                    z = (-1,-1)
                    try:
                        z = sym_lookup[i][tuple(x)]
                    except KeyError:
                        z = ('?','?')
                    translated_path.append(z)
                if tag == "infix":
                    print(f"{tag} {afx_idx} {bnd_idx} {cost} {[str(v[0]) for v in anno_residues]}")
                else:
                    print(cost, path, anno)
                    l_path, r_path = zip(*translated_path)
                    print(f"{tag} {afx_idx} {bnd_idx} {cost} {[str(v[0]) for v in anno_residues]} l={l_path} r={r_path} {term}")

    print("called candidates:", sum([len(x) for x in enmr_res.candidates]))
    c = 0
    print("peaks:", peaks, len(peaks))
    for (i, cand_res) in enumerate(enmr_res.candidates):
        n_cand = len(cand_res)
        print(f"cluster {i} [{2 * anno_res.pivots.cluster_points[i]}] produced {n_cand} candidate{'s' if n_cand != 1 else ''}.")
        if n_cand > 0:
            abs_offset = np.abs(cand_res.offset)
            min_offset = abs_offset.min()
            max_offset = abs_offset.max()
            for (j, cand) in enumerate(cand_res):
                if c < cfg.num_out:
                    c += 1
                    cost, seq, path, anno, offset = cand
                    normalized_offset = (np.abs(offset) - min_offset) / (max_offset - min_offset) if n_cand > 1 else cand_res.offset[0]
                    print(c, (i, j), float(cost.round(4)), seq.replace(' ',''), float(offset.round(4)))
                    annotated_peaks = AnnotatedPeaks.from_evaluation(
                        peaks = peaks,
                        annotation = (anno_res, anno_params),
                        cluster = i,
                        alignment = (algn_res, algn_params),
                        enumeration = (enmr_res, enmr_params),
                        candidate = j,
                    )
                else:
                    break

@hydra.main(version_base=None, config_path="params", config_name="config")
def main(cfg: DictConfig) -> None:
    mode = cfg.app.mode
    if mode == 'evaluate':
        return evaluate(*setup(cfg))
    elif mode == 'test':
        return test(*setup(cfg))
    else:
        print(f"Unrecognized mode {mode}")
        
if __name__ == "__main__":
    main()
