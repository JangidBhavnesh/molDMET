
from pyscf import gto
import os
from molDMET.my_pyscf.gto.basis import readbasis

def x2cbasis(molbasis):
    elements = list(molbasis.keys())
    mydict = {}
    for elem in elements:
        mydict = readbasis(elem, molbasis, mydict)
    assert len(molbasis) == len(mydict)
    return mydict
