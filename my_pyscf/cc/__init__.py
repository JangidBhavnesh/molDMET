import pyscf
import numpy as np
from pyscf import cc, lib
from molDMET.my_pyscf.cc.eomipcvs import get_eomip_child_class
from pyscf import __config__


def _ipcvsccsd(cc_class, nroots=1, ncvs=1, corehole=0, conv_tol=1e-6, left=False, 
        koopmans=False, guess=None, partition=None, eris=None):
    # If core orbital are freezed during the CCSD, those can be taken as the
    # MP2 guess before passing it to EOMIP.
    if isinstance(cc_class.frozen, np.int64):
        cc_class = set_frozen_Re(cc_class)
    if isinstance(cc_class.frozen, list):
        cc_class = frozen_lst_Re(cc_class)
    myeomip = get_eomip_child_class(cc_class, ncvs=ncvs, corehole=corehole)
    myeomip.partition = partition
    myeomip.conv_tol = conv_tol
    return myeomip.ipccsd(nroots, left, koopmans, guess,
                              partition, eris)


def _ipcvsccsd_star(cc_class, nroots=1, ncvs=1, corehole=0, koopmans=False, right_guess=None,
                       left_guess=None, eris=None, partition=None, conv_tol=1e-7):
    # If core orbital are freezed during the CCSD, those can be taken as the
    # MP2 guess before passing it to EOMIP.
    if isinstance(cc_class.frozen, np.int64):
        cc_class = set_frozen_Re(cc_class)
    if isinstance(cc_class.frozen, list):
        cc_class = frozen_lst_Re(cc_class)

    myeomip = get_eomip_child_class(cc_class, ncvs=ncvs, corehole=corehole)
    myeomip.partition = partition
    myeomip.conv_tol = conv_tol
    return myeomip.ipccsd_star(nroots, koopmans, right_guess, left_guess, eris)


def EOMIP(cc_class, nroots=1, ncvs=1, corehole=0, conv_tol=1e-6, left=False,
        koopmans=True, guess=None,partition=None, eris=None):
    return _ipcvsccsd(cc_class, nroots=nroots, ncvs=ncvs, corehole=corehole, conv_tol=conv_tol, 
            left=left,koopmans=koopmans, guess=guess,partition=partition, eris=eris)


def EOMIPStar(cc_class, nroots=1, ncvs=1, corehole=0,  conv_tol=1e-6, right_guess=None,
            left_guess=None, koopmans=True, partition=None, eris=None):
    return _ipcvsccsd_star(cc_class, nroots=nroots, ncvs=ncvs, conv_tol=conv_tol, corehole=corehole, 
            right_guess=right_guess,left_guess=left_guess,koopmans=koopmans, 
            partition=partition, eris=eris) 

def set_frozen_Re(cc_class):
    frozen = cc_class.frozen
    mycc = cc.CCSD(cc_class._scf)
    mycc.t1, mycc.t2 = mycc.get_init_guess()
    nocc, nvir = np.shape(mycc.t1)
    mycc.t1[frozen:, :] = cc_class.t1
    max_memory = cc_class.max_memory - lib.current_memory()[0]
    BLKMIN = getattr(__config__, 'cc_ccsd_blkmin', 4)
    blksize = int(min(nvir, max(BLKMIN, max_memory*.3e6/8/(nocc**2*nvir+1))))
    for p0, p1 in lib.prange(0, nvir, blksize):
        mycc.t2[frozen:, frozen:, p0:p1] = cc_class.t2[:, :, p0:p1]
    return mycc


def frozen_lst_Re(cc_class):
    frozen = cc_class.frozen
    assert isinstance(frozen, list)
    mycc = cc.CCSD(cc_class._scf)
    mycc.t1, mycc.t2 = mycc.get_init_guess()
    nocc = mycc.nocc
    if any(x > nocc for x in frozen):
        raise NotImplementedError("This feature is not yet implemented.")
    else:
        frozen = len(frozen)
        mycc.t1[frozen:, :] = cc_class.t1
        mycc.t2[frozen:, frozen:, :, :] = cc_class.t2
    return mycc
