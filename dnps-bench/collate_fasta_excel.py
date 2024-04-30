from fastaparser import Reader as FASTAReader
import pandas as pd
import sys

def read_fasta_to_tuples(inpath: str):
    sequences = list[tuple[str,str]]()
    with open(inpath) as fasta:
        parser = FASTAReader(fasta,parse_method="quick")
        for record in parser:
            sequences.append((record.header,record.sequence))
    return sequences

def collate(fastapath: str, excelpath: str):
    records = read_fasta_to_tuples(fastapath)
    n_records = len(records)
    record_names = record_names = [rec[0].split('|')[0][1:-1] for rec in records]
    name_to_seq = {record_names[i]: records[i][1] for i in range(n_records)}
    sheet = pd.read_excel(excelpath)
    seqkey = "Sequence"
    sheet[seqkey] = ""
    for i in range(len(sheet)):
        accession = sheet["Accession"][i]
        try:
            sheet.loc[i,seqkey] = name_to_seq[accession]
        except KeyError:
            sheet.loc[i,seqkey] = "N/A"
    return sheet

if __name__ == "__main__":
    fastapath, excelpath, outpath = sys.argv[1:4]
    sheet = collate(fastapath,excelpath)
    sheet.to_csv(outpath)