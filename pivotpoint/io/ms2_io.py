from pyopenms import MSExperiment, MzMLFile
from pandas import DataFrame, read_csv

def read_mzml_to_df(path: str):
    data = MSExperiment()
    MzMLFile().load(path,data)
    return data.get_df()
    
# for now these two just wrap the pandas methods in functional notation
def read_csv_as_df(path: str):
    return read_csv(path)
def write_df_to_csv(path: str, df: DataFrame):
    df.to_csv(path)