import pathlib, dataclasses

import hydra
from omegaconf import DictConfig, OmegaConf

from .io import reverse_fasta, SerializableDataclass, serialize_dataclass, deserialize_dataclass
from .fragments.types import FragmentStateSpace, ResidueStateSpace, TargetMasses, MultiResidueTargetMasses, PairResult, PivotResult, BoundaryResult
from .fragments.masses import construct_pair_target_masses, construct_boundary_target_masses
from .sequences.suffix_array import SuffixArray
from .annotation import AnnotationParams
from .alignment import AlignmentParams
from .enumeration import EnumerationParams

def make_session_dir(
    config: DictConfig,
) -> pathlib.Path:
    """The first step in setup: create an appropriately-named session directory. Both the parent directory and session name are specified in the config. If the session name has already been used, it will be made unique using an incremental numerical suffix."""
    session_parent_dir = pathlib.Path(config.session.dir)
    if not(session_parent_dir.exists()):
        session_parent_dir.mkdir()
    session_dir = session_parent_dir / config.session.name
    if session_dir.exists():
        if '_' in config.session.name and config.session.name.split('_')[-1].isnumeric():
            *session_name, suffix = config.session.name.split('_')
            session_name = '_'.join(session_name)
            i = int(suffix) + 1
            # this is a bit dense. it handles cases where an incremented name is passed as a session name.
            # e.g., there is a session named 'sesh' and one named 'sesh_1'. if the config session name is
            # 'sesh_1' but that already exists, this logic ensures that the new name will be 'sesh_2'
            # rather than 'sesh_1_1'. it is a little more complicated because it has to handle names that
            # also have underscores elsewhere, like 'sesh_with_underscores_1' needs to become 
            # 'sesh_with_underscores_2' without trying to cast any of the preceding string parts to int.
        else:
            session_name = config.session.name
            i = 1
            # otherwise, just try to increment the session name.
        session_dir = session_parent_dir / (session_name + f"_{i}")
        while session_dir.exists():
            i += 1
            session_dir = session_parent_dir / (session_name + f"_{i}")
        print(f"The session name [{config.session.name}] has already been used. This session has been renamed to [{session_dir.stem}]")
    session_dir.mkdir()
    config.session.name = session_dir.stem
    return session_dir

def construct_params(
    config: DictConfig,
) -> tuple[
    AnnotationParams,
    AlignmentParams,
    EnumerationParams,
]:
    """The second step in setup: create parameter objects for the three proceeding phases of the algorithm. The *Params are dataclasses that hold flattened subsets of the nested data in config."""
    return (
        AnnotationParams.from_config(config.annotation),
        AlignmentParams.from_config(config.alignment),
        EnumerationParams.from_config(config.enumeration),
    )

def load_suffix_arrays(
    config: DictConfig,
    session_dir: pathlib.Path,
) -> tuple[
    SuffixArray,
    SuffixArray,
]:
    """Third step in setup: decide whether forward and reverse suffix arrays can be read directly from storage or must be created from a provided transcriptome fasta. In absence of either prebuilt suffix arrays or a transcriptome, the tuple (None, None) will be returned."""
    suf_path = config.suffix_array
    rev_suf_path = config.reverse_suffix_array
    tr_path = config.transcriptome
    if not((suf_path is None) or (rev_suf_path is None)):
        print("Reading suffix arrays from storage.")
        return (
            SuffixArray.read(suf_path),
            SuffixArray.read(rev_suf_path),
        )
    elif not(tr_path is None):
        print("Suffix array arguments were not provided. MiRROR will construct a reversed transcriptome, then create suffix arrays.")
        rev_tr_path = reverse_fasta(pathlib.Path(tr_path), pathlib.Path(session_dir))
        suf_path = str(session_dir / "forward.sufr")
        config.suffix_array = suf_path
        rev_suf_path = str(session_dir / "reverse.sufr")
        config.reverse_suffix_array = rev_suf_path
        return (
            SuffixArray.create(tr_path, suf_path),
            SuffixArray.create(rev_tr_path, rev_suf_path),
        )
    else:
        print("Neither suffix array nor transcriptome arguments were provided. MiRROR will proceed without suffix arrays.")
        return (
            None,
            None,
        )

def construct_targets(
    config: DictConfig,
    forward_suffix_array: SuffixArray,
    reverse_suffix_array: SuffixArray,
) -> tuple[
    list[TargetMasses],
    list[TargetMasses],
    list[TargetMasses],
]:
    """Fourth step in setup: enumerate all viable combinations of amino acids, losses, and modifications given in the config. If configured for multi-residue boundaries, the suffix arrays constrain the space of residue sequences."""
    config = config.annotation
    pair_fragment_space = FragmentStateSpace.from_config_to_pairs(config)
    residue_space = ResidueStateSpace.from_config(config)
    pair_targets = construct_pair_target_masses(
        residue_space,
        pair_fragment_space,
    )
    # target masses for fragments observed as the difference of two consecutive peaks.

    lower_boundary_fragment_space = FragmentStateSpace.from_config_to_boundaries(config)
    lower_boundary_targets = construct_boundary_target_masses(
        residue_space,
        lower_boundary_fragment_space,
    )
    # target masses for low-mz boundaries observed as single peaks.

    reflected_residue_space = ResidueStateSpace.from_config(config, reflect=True)
    reflected_upper_boundary_fragment_space = FragmentStateSpace.from_config_to_boundaries(config, reflect=True)
    reflected_upper_boundary_targets = construct_boundary_target_masses(
        reflected_residue_space,
        reflected_upper_boundary_fragment_space,
    )
    # target masses for high-mz boundaries observed as the reflections of single peaks.

    max_k = config.max_k
    multi_lower_boundary_targets = [None for _ in range(max_k - 1)]
    multi_reflected_upper_boundary_targets = [None for _ in range(max_k - 1)]
    if max_k > 1:
        # TODO: constrain by suffix arrays.
        for k in range(2, max_k + 1):
            operand = [pair_targets,] * (k - 1)
            multi_lower_boundary_targets[k - 2] = combine_target_masses(
                [boundary_targets,] + operand)
            multi_reflected_upper_boundary_targets[k - 2] = combine_target_masses(
                [reflected_upper_boundary_targets,] + operand)
    return (
        [pair_targets,],
        [lower_boundary_targets, *multi_lower_boundary_targets],
        [reflected_upper_boundary_targets, *multi_reflected_upper_boundary_targets],
    )

@dataclasses.dataclass(slots=True)
class Session:
    session_dir: pathlib.Path
    anno_params: AnnotationParams
    algn_params: AlignmentParams
    enmr_params: EnumerationParams
    forward_suffix_array: SuffixArray
    reverse_suffix_array: SuffixArray
    pair_targets: list[TargetMasses]
    boundary_targets: list[TargetMasses]
    reverse_boundary_targets: list[TargetMasses]

def setup(
    config: DictConfig,
) -> Session:
    """Expand the config into a Session object, containing parameters for each phase, suffix arrays, and target masses."""
    session_dir = make_session_dir(config)
    anno_params, algn_params, enmr_params = construct_params(config)
    suffix_array, rev_suffix_array = load_suffix_arrays(config, session_dir)
    pair_tgt, boundary_tgt, rev_boundary_tgt = construct_targets(config, suffix_array, rev_suffix_array)
    if config.session.serialize:
        OmegaConf.save(config, session_dir / "config.yaml")
    return Session(
        session_dir,
        anno_params,
        algn_params,
        enmr_params,
        suffix_array,
        rev_suffix_array,
        pair_tgt,
        boundary_tgt,
        rev_boundary_tgt,
    )
