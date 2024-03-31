import numpy as np
from pyscf import gto, scf, ao2mo, mp, adc
from mrh.util.basis import represent_operator_in_basis
from mrh.util.rdm import get_2CDM_from_2RDM
from mrh.util.tensors import symmetrize_tensor


def solve(frag, guess_1RDM, chempot_imp):
    # Augment OEI with the chemical potential
    OEI = frag.impham_OEI_C - chempot_imp
    # Get the RHF solution
    mol = gto.Mole()
    mol.max_memory = frag.ints.max_memory
    mol.build(verbose=0)
    mol.atom.append(('C', (0, 0, 0)))
    mol.nelectron = frag.nelec_imp
    mol.incore_anyway = True
    mol.verbose = 4
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: OEI
    mf.get_ovlp = lambda *args: np.eye(frag.norbs_imp)
    mf._eri = ao2mo.restore(8, frag.impham_TEI, frag.norbs_imp)
    mf.energy_nuc = lambda *args: frag.impham_CONST
    mf.scf(guess_1RDM)

    if not mf.converged:
        mf = mf.newton()
        mf.kernel()

    # Get the MP2 solution
    mp2 = mp.MP2(mf)
    mp2.kernel()

    imp2mo = mf.mo_coeff
    mo2imp = imp2mo.conjugate().T
    oneRDMimp_imp = mf.make_rdm1()
    twoRDMimp_mo = mp2.make_rdm2()
    twoRDMimp_imp = represent_operator_in_basis(twoRDMimp_mo, mo2imp)
    twoCDM_imp = get_2CDM_from_2RDM(twoRDMimp_imp, oneRDMimp_imp)
    frag.nelec_frag = np.trace(oneRDMimp_imp)

    # General impurity data
    frag.oneRDM_loc = symmetrize_tensor(frag.oneRDMfroz_loc + represent_operator_in_basis(oneRDMimp_imp, frag.imp2loc))
    frag.twoCDM_imp = symmetrize_tensor(twoCDM_imp)
    frag.E_imp = frag.impham_CONST + mp2.e_tot + np.einsum('ab,ab->', oneRDMimp_imp, chempot_imp)
    return None
