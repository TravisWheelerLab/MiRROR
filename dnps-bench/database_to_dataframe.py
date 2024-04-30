import sys
import pandas as pd
import sqlite3


def extract_tables(cnx: sqlite3.Connection, table_names):
    n_names = len(table_names)
    queries = ["SELECT * FROM " + name for name in table_names]
    frames = [pd.read_sql_query(query,cnx) for query in queries]
    return {table_names[i]: frames[i] for i in range(n_names)}

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
    "PeptidesTerminalModifications",]

MSF_DECOY_TABLES = [table_name + "_decoy" for table_name in MSF_PEPTIDE_TABLES]

MSF_METADATA_TABLES = [
    "FileInfos",
    "MassPeaks",
    "SpectrumHeaders",
    "SpectrumScores",
]

MSF_TABLES = MSF_AMINO_TABLES + MSF_PEPTIDE_TABLES + MSF_DECOY_TABLES + MSF_METADATA_TABLES

if __name__ == "__main__":
    dbpath,dfpath = sys.argv[1:3]
    cnx = sqlite3.connect(dbpath)
    db_tables = extract_tables(cnx, MSF_TABLES)
    cnx.close()
    for name in MSF_TABLES:
        print(name)
        print(db_tables[name])
        print()
