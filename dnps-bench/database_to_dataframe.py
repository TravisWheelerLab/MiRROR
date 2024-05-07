import sys
import pandas as pd
import sqlite3

MSF_AMINO_TABLES = [
    "AminoAcids",
    "AminoAcidModifications",
    "AminoAcidModificationsAminoAcids",
    "AminoAcidModificationsAminoAcidsNL",
    "AminoAcidModificationsNeutralLosses",
]

MSF_PEPTIDE_TABLES = [
    "Peptides",
    "PeptideScores",
    "PeptidesAminoAcidModifications",
    "PeptidesTerminalModifications",
]

MSF_DECOY_TABLES = [peptide_table_name + "_decoy" for peptide_table_name in MSF_PEPTIDE_TABLES]

MSF_METADATA_TABLES = [
    "FileInfos",
    "MassPeaks",
    "SpectrumHeaders",
    "SpectrumScores",
]

MSF_TABLES = MSF_AMINO_TABLES + MSF_PEPTIDE_TABLES + MSF_DECOY_TABLES + MSF_METADATA_TABLES

#JOIN_PEPTIDE_TABLES = '''SELECT Peptides.PeptideID, SpectrumID, TotalIonsCount, MatchedIonsCount, ConfidenceLevel, Sequence, Annotation, MissedCleavages
#FROM Peptides
#JOIN PeptideScores
#ON Peptides.PeptideID = PeptideScores.PeptideID'''
#
#def peptide_frame(cnx: sqlite3.Connection):
#    query = JOIN_PEPTIDE_TABLES
#    peptide_join = pd.read_sql_query(query,cnx)
#    return peptide_join

def load_tables(cnx: sqlite3.Connection, table_names):
    n_names = len(table_names)
    queries = ["SELECT * FROM " + name for name in table_names]
    tables = [pd.read_sql_query(query,cnx) for query in queries]
    return tables

def load_table_dict(cnx: sqlite3.Connection, table_names):
    tables = load_tables(cnx,table_names)
    n_tables = len(tables)
    return {table_names[i]: tables[i] for i in range(n_tables)}

def drop_columns(tables, keys: list[str]):
    n_tables = len(tables)
    for i in range(n_tables):
        tables[i] = tables[i].drop(keys, axis=1)
    return tables

def merge_tables(tables, key: str):
    n_tables = len(tables)
    accumulator = tables[0]
    for i in range(1,n_tables):
        accumulator = accumulator.merge(tables[i], on = key)
    return accumulator

def print_table_columns(tables,names=[]):
    n_tables = len(tables)
    if len(names) == 0:
        names = ["" for _ in range(n_tables)]
    for i in range(n_tables):
        print(names[i])
        print("\t",tables[i].columns)
        print()

def print_tables_dict(tables_dict):
    values = list(tables_dict.values())
    keys = list(tables_dict.keys())
    print_table_columns(values,keys)
    
if __name__ == "__main__":
    dbpath,dfpath = sys.argv[1:3]
    cnx = sqlite3.connect(dbpath)
    db_tables_dict = load_table_dict(cnx,MSF_TABLES)
    print("all table columns:")
    print_tables_dict(db_tables_dict)
    peptide_tables = drop_columns(load_tables(cnx,MSF_PEPTIDE_TABLES),"ProcessingNodeNumber")
    print("peptide table columns:")
    print_table_columns(peptide_tables,MSF_PEPTIDE_TABLES)
    bigtable = merge_tables(peptide_tables,"PeptideID")
    bigtable = merge_tables([bigtable,db_tables_dict["AminoAcidModifications"]],"AminoAcidModificationID")
    bigtable = merge_tables([bigtable,db_tables_dict["SpectrumHeaders"],db_tables_dict["SpectrumScores"]],"SpectrumID")
    print(bigtable.columns)
    cnx.close()
    
    for _,x in bigtable.iterrows():
        print(x)
        input()
