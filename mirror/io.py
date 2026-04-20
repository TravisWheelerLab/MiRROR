import dataclasses, json, zlib, base64
from typing import Iterator, Iterable

import numpy as np
import networkx as nx
import mzspeclib as mzlib
import pyopenms as oms
from pyteomics import mgf
from mzspeclib.validate import ValidationWarning
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import warnings
warnings.filterwarnings(action="ignore", category = ValidationWarning)

def load_amino_table(
    fpath: str,
    row_dlm: str = '\n',
    col_dlm: str = ' \t',
) -> np.ndarray:
    with open(fpath, 'r') as f:
        txt = f.read()
    rows = txt.split(row_dlm)[:-1]
    data = [x.split(col_dlm) for x in rows]
    return np.array(data)
    
def encode_graph(
    d: nx.DiGraph,
) -> str:
    return base64.b64encode(zlib.compress(bytes(json.dumps(nx.adjacency_data(d),cls=DataclassEncoder), 'utf8'))).decode('ascii')

def decode_graph(
    b: bytes,
) -> nx.DiGraph:
    return nx.adjacency_graph(decode_structure(json.loads(zlib.decompress(base64.b64decode(b)))))

def encode_arr(
    a: np.ndarray,
) -> tuple[
    str,
    str,
    tuple,
]:
    return (
        base64.b64encode(zlib.compress(a.tobytes())).decode('ascii'),
        str(a.dtype),
        list(a.shape),
    )

def decode_arr(
    b: str,
    ty: str,
    sh: list[int],
) -> np.ndarray:
    return np.frombuffer(
        zlib.decompress(base64.b64decode(b)), 
        dtype=np.dtype(ty),
    ).reshape(sh)

ARRAY_SYMBOL = '#'
GRAPH_SYMBOL = '~'

class DataclassEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return (ARRAY_SYMBOL, encode_arr(obj))
        elif isinstance(obj, nx.DiGraph):
            return (GRAPH_SYMBOL, encode_graph(obj))
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return super(DataclassEncoder, self).default(obj)

def encode_structure(
    s,
    array_symbol = ARRAY_SYMBOL,
    graph_symbol = GRAPH_SYMBOL
):
    if isinstance(s, np.ndarray):
        return (array_symbol, encode_arr(s))
    elif isinstance(s, nx.DiGraph):
        return (graph_symbol, encode_graph(s))
    elif isinstance(s, tuple):
        return tuple([encode_structure(v) for v in s])
    elif isinstance(s, dict):
        for (k, v) in s.items():
            s[k] = encode_structure(v)
        return s
    elif isinstance(s, list):
        for (i, v) in enumerate(s):
            s[i] = encode_structure(v)
        return s
    else:
        return s

def decode_structure(
    s,
    array_symbol = ARRAY_SYMBOL,
    graph_symbol = GRAPH_SYMBOL,
):
    if isinstance(s, tuple) or isinstance(s, list):
        if len(s) == 2:
            sym = s[0]
            data = s[1]
            if sym == array_symbol:
                return decode_arr(*data)
            elif sym == graph_symbol:
                return decode_graph(data)
    
    if isinstance(s, tuple):
        return tuple([decode_structure(v) for v in s])
    elif isinstance(s, dict):
        for (k, v) in s.items():
            s[k] = decode_structure(v)
        return s
    elif isinstance(s, list):
         for (i, v) in enumerate(s):
             s[i] = decode_structure(v)
         return s
    else:
        return s

def serialize_dataclass(x, readable=False):
    return json.dumps(dataclasses.asdict(x), cls=DataclassEncoder)

def dataclass_from_dict(ty, d):
    try:
        fieldtypes = ty.__annotations__
        return ty(**{f: dataclass_from_dict(fieldtypes[f], d[f]) for f in d})
    except KeyError as e:
        if ty is nx.DiGraph:
            return d
        else:
            raise e
    except AttributeError:
        if isinstance(d, (tuple, list)):
            return [dataclass_from_dict(ty.__args__[0], f) for f in d]
        return d

def deserialize_dataclass(ty, xstr):
    return dataclass_from_dict(ty, decode_structure(json.loads(xstr)))

@dataclasses.dataclass(slots=True)
class SerializableDataclass:

    def save(self, fpath):
        with open(fpath, 'w') as f:
            f.write(serialize_dataclass(self))
        
    @classmethod
    def load(cls, fpath):
        with open(fpath, 'r') as f:
            return deserialize_dataclass(cls, f.read())

def write_str_to_fa(
    txt: Iterator[str],
    path_to_fa: str,
):
    return SeqIO.write(
        [SeqRecord(Seq(x), id=f"{i}") for (i, x) in enumerate(txt)],
        path_to_fa,
        "fasta",
    )

def read_str_from_fa(
    path_to_fa: str,
) -> Iterator[str]:
    with open(path_to_fa, 'r') as f:
        for record in SeqIO.parse(f, "fasta"):
            yield str(record.seq)

def reverse_fasta(
    path_to_fa: str,
) -> str:
    filename, ext = path_to_fa.split('.')
    try:
        new_path = f"{filename}_REVERSED.{ext}"
        write_str_to_fa(
            (x[::-1] for x in read_str_from_fa(path_to_fa)),
            new_path,
        )
        return new_path
    except:
        return None

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
        return mzlib.SpectrumLibrary(filename = filepath)

def read_mgf(
    filepath: str,
) -> mgf.IndexedMGF:
    return mgf.read(filepath)
