import xml.dom.minidom
import xml.etree.ElementTree as ET
from pyteomics import mzid,pepxml
from pyopenms import MascotXMLFile, ProteinIdentification, PeptideIdentification, SpectrumMetaDataLookup, PeptideHit
import pandas as pd
import sys

def explore_xml(paths: list[str]):
    for xmlpath in paths:
        try:
            print(xmlpath)
            tree = ET.parse(xmlpath)
            for elem in tree.iter():
                print(elem)
                input()
        except KeyboardInterrupt:
            print("\t[skipping",xmlpath,']')
            continue

def display_xml(path):
    dom = xml.dom.minidom.parse(path)
    print(dom.toprettyxml())

def iterate_mzid(path):
    return mzid.MzIdentML(path)

def iterate_pepxml(path):
    return pepxml.PepXML(path)

def unpack_peptide_identification(pepid: PeptideIdentification):
    return {
        "mz": pepid.getMZ(),
        "hits": [
            {
                "charge": pephit.getCharge(),
                "sequence": str(pephit.getSequence()),
                "score": pephit.getScore()
            } for pephit in pepid.getHits()]
    }

def iterate_mascotxml(path):
    mascot_xml = MascotXMLFile()
    peptide_identifications = list[PeptideIdentification]() # = []
    mascot_xml.load(path,ProteinIdentification(),peptide_identifications,SpectrumMetaDataLookup())
    return map(unpack_peptide_identification,peptide_identifications)

if __name__ == "__main__":
    #display_xml(sys.argv[1])
    #explore_xml(sys.argv[1:])
    mode = sys.argv[1]
    path = sys.argv[2]
    data = None
    if mode == "mascot":
        data = iterate_mascotxml(path)
    elif mode == "pepxml":
        data = iterate_pepxml(path)
    elif mode == "mzid":
        data = iterate_mzid(path)
    else:
        print("format mode", mode, "is not implemented.")
        data = []
    
    for x in data:
        print(x)
        input()