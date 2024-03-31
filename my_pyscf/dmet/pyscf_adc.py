import numpy as np
from pyscf import gto, scf, ao2mo, mp, adc


def solve(frag, guess_1RDM, chempot_imp, nroots=1, ncvs=1, corehole=0, method='adc(2)'):

    assert method == 'adc(2)' or 'adc(3)'
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
    mf.scf(guess_1RDM)

    if not mf.converged:
        mf = mf.newton()
        mf.kernel()

    # Get the MP2 solution
    mp2 = mp.MP2(mf)
    mp2.kernel()

    myadc = adc.ADC(mf)
    myadc.verbose = 4
    myadc.ncvs = ncvs
    myadc.compute_properties = True
    myadc.method = method
    myadc.method_type = "ip"
    myadc.kernel(nroots=nroots)
    return None
