from pyopenms import MSExperiment, MzMLFile
from pandas import DataFrame, read_csv
from fastaparser import Reader as FASTAReader, Writer as FASTAWriter
from .pivot import PivotingIntervalPair, pivot_from_data

def write_pivots_to_csv(path: str, pivots: list[PivotingIntervalPair]):
    pivot_data_strings = [f"{x1},{x2},{x3},{x4}\n" for (x1,x2,x3,x4) in map(lambda x: x.data(),pivots)]
    with open(path, 'w') as pivots_csv:
        pivots_csv.writelines(pivot_data_strings)

def read_csv_to_pivots(path: str):
    pivots = []
    with open(path) as pivots_csv:
        for pivot_line in pivots_csv.readlines():
            pivot_data = [float(x) for x in pivot_line.split(',')]
            pivots.append(PivotingIntervalPair(pivot_data))
    return pivots

def read_fasta_to_list(path: str):
    sequences = list[str]()
    with open(path) as fasta:
        parser = FASTAReader(fasta,parse_method="quick")
        for seq in parser:
            sequences.append(seq)
    return sequences

def write_list_to_fasta(path: str, sequences: list[str], headers = None):
    if headers == None:
        headers = [str(i) for i in range(len(sequences))]
    with open(path, 'w') as fasta:  
        writer = FASTAWriter(fasta)
        for header_and_sequence in zip(headers,sequences):
            writer.writefasta(header_and_sequence)

def read_mzml_to_df(path: str):
    data = MSExperiment()
    MzMLFile().load(path,data)
    return data.get_df()
    
# for now these two just wrap the pandas methods in functional notation
def read_csv_as_df(path: str):
    return read_csv(path)
def write_df_to_csv(path: str, df: DataFrame):
    df.to_csv(path)