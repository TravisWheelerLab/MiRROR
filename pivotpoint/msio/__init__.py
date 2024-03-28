from pyopenms import MSExperiment, MzMLFile
from pandas import DataFrame

def read_mzml_to_df(path: str):
    data = MSExperiment()
    MzMLFile().load(path,data)
    return data.get_df()

def write_df_to_csv(path: str,df: DataFrame):
    df.to_csv(path)