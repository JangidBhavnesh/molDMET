import numpy as np
from pyscf import ao2mo, gto, scf
from pyscf.cc import ccsd
from molDMET.my_pyscf.cc import EOMIP, EOMIPStar

def solve(frag, guess_1RDM, chempot_imp, nroots=1, ncvs=1, corehole=0):
    # Augment OEI with the chemical potential
    OEI = frag.impham_OEI_C - chempot_imp
    # Get the RHF solution
    mol = gto.Mole()
    mol.build(verbose=0)
    mol.atom.append(('C', (0, 0, 0)))
    mol.nelectron = frag.nelec_imp
    mol.incore_anyway = True
    mol.max_memory = frag.ints.max_memory
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: OEI
    mf.get_ovlp = lambda *args: np.eye(frag.norbs_imp)
    mf._eri = ao2mo.restore(8, frag.impham_TEI, frag.norbs_imp)
    mf.energy_nuc = lambda *args: frag.impham_CONST
    mf.scf(guess_1RDM)

    if not mf.converged:
        mf = mf.newton()
        mf.kernel()

    ccsolver = ccsd.CCSD(mf)
    ccsolver.verbose = 4
    ccsolver.ccsd()
    EOMIPStar(ccsolver, nroots=nroots, ncvs=ncvs, corehole=corehole, koopmans=True)
    return None
