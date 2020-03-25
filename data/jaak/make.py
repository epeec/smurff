#!/usr/bin/env python

import matrix_io as mio
import urllib.request
import scipy.io as sio
import os
from hashlib import sha256

urls = [
        (
            "http://homes.esat.kuleuven.be/~jsimm/chembl-IC50-346targets.mm",
            "10c3e1f989a7a415a585a175ed59eeaa33eff66272d47580374f26342cddaa88",
            "chembl-IC50-346targets.mm",
            ),
        (
            "http://homes.esat.kuleuven.be/~jsimm/chembl-IC50-compound-feat.mm",
            "f9fe0d296272ef26872409be6991200dbf4884b0cf6c96af8892abfd2b55e3bc",
            "chembl-IC50-compound-feat.mm", 
            ),
        (
            "http://homes.esat.kuleuven.be/~jsimm/chembl-IC50-compounds.csv",
            "e8f045a67ee149c6100684e07920036de72583366596eb5748a79be6e3b96f7c",
            "chembl-IC50-compounds.csv",
            ),
        (
            "http://homes.esat.kuleuven.be/~jsimm/chembl-IC50-proteins-uniprot.csv",
            "224b1b44abcab8448b023874f4676af30d64fe651754144f9cbdc67853b76ea8",
            "chembl-IC50-proteins-uniprot.csv",
            ),
        ]

for url, expected_sha, output in urls:
    if os.path.isfile(output):
        actual_sha = sha256(open(output, "rb").read()).hexdigest()
        if (expected_sha == actual_sha):
            continue

    print("download %s" % output)
    urllib.request.urlretrieve(url, output)

ic50 = sio.mmread("chembl-IC50-346targets.mm")
feat = sio.mmread("chembl-IC50-compound-feat.mm")

def make_ic50_sdm():
    mio.write_matrix("chembl-IC50-346targets.sdm", ic50)

def make_feat_sdm():
    mio.write_matrix("chembl-IC50-compound-feat.sdm", feat)

def make_01():
    # 0,1 binary for probit
    ic50_01 = ic50.copy()
    ic50_01.data = (ic50_01.data >= 6) * 1.
    mio.write_matrix("chembl-IC50-346targets-01.sdm", ic50_01)

def make_11():
    # -1,+1
    ic50_11 = ic50.copy()
    ic50_11.data = ((ic50.data >= 6) * 2.) - 1.
    mio.write_matrix("chembl-IC50-346targets-11.sdm", ic50_11)

def make_100compounds():
    ic50_100c = ic50.tocsr()[0:100,:]
    mio.write_matrix("chembl-IC50-346targets-100compounds.sdm", ic50_100c)

def feat_100():
    feat_100 = feat.tocsr()[0:100,:]
    feat_100 = feat_100[:,feat_100.getnnz(0)>0]
    return feat_100

def make_feat():
    mio.write_matrix("chembl-IC50-100compounds-feat.sdm", feat_100())

def make_feat_dense():
    mio.write_matrix("chembl-IC50-100compounds-feat-dense.ddm", feat_100().todense())

generated_files = [
        ( "f0d2ad6cf8173a64e12b48821e683b642b593555c552f4abf1f10ba255af78fc", "chembl-IC50-100compounds-feat-dense.ddm", make_feat_dense,),
        ( "0dd148a0da1a11ce6c6c3847d0cc2820dc9c819868f964a653a0d42063ce5c42", "chembl-IC50-100compounds-feat.sdm", make_feat,),
        ( "973074474497b236bf75fecfe9cc17471783fd40dbdda158b81e0ebbb408d30b", "chembl-IC50-346targets-01.sdm", make_01,),
        ( "5d7c821cdce02b4315a98a94cba5747e82d423feb1a2158bf03a7640aa82625d", "chembl-IC50-346targets-100compounds.sdm", make_100compounds,),
        ( "bcf5cee9702e318591b76f064859c1d0769158d0b0f5c44057392c2f9385a591", "chembl-IC50-346targets-11.sdm", make_11,),
        ( "1defd1c82ac3243ad60a23a753287df494d3b50f2fd5ff7f4a074182b07e3318", "chembl-IC50-346targets.sdm", make_ic50_sdm, ),
        ( "badfa23abb83e0b731e969e1117fd4269f2df16e1faf14eb54c53c60465e87f1", "chembl-IC50-compound-feat.sdm", make_feat_sdm, ),
        ]

for expected_sha, output, func in generated_files:
    if os.path.isfile(output):
        actual_sha = sha256(open(output, "rb").read()).hexdigest()
        if (expected_sha == actual_sha):
            continue

    print("make %s" % output)
    func()

    actual_sha = sha256(open(output, "rb").read()).hexdigest()
    if (expected_sha != actual_sha):
        print("Checksum mismatch for %s: expected %s, got %s" % (output, expected_sha, actual_sha))


