#--------
import numpy as np
from functools import reduce
from pyscf import ao2mo, gto, scf, lib
from mrh.util.basis import represent_operator_in_basis, project_operator_into_subspace, measure_basis_olap
from mrh.util.tensors import symmetrize_tensor


def solve(frag, guess_1RDM, chempot_imp):
    # Augment OEI with the chemical potential
    OEI = frag.impham_OEI_C - chempot_imp
    sign_MS = np.sign(frag.target_MS) or 1

    # Get the RHF solution
    mol = gto.Mole()
    mol.spin = abs(int(round(2 * frag.target_MS)))
    mol.verbose = 0
    mol.build()

    if frag.mol_stdout is None:
        frag.mol_stdout = mol.stdout
    else:
        mol.stdout = frag.mol_stdout
        mol.verbose = 0 if frag.mol_output is None else lib.logger.DEBUG

    mol.atom.append(('C', (0, 0, 0)))
    mol.nelectron = frag.nelec_imp
    mol.incore_anyway = True
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: OEI
    mf.get_ovlp = lambda *args: np.eye(frag.norbs_imp)
    mf.energy_nuc = lambda *args: frag.impham_CONST
    if frag.quasidirect:
        mf.get_jk = frag.impham_get_jk
    else:
        mf._eri = ao2mo.restore(8, frag.impham_TEI, frag.norbs_imp)
    mf = fix_my_RHF_for_nonsinglet_env(mf, sign_MS * frag.impham_OEI_S)
    mf.__dict__.update(frag.mf_attr)
    mf.scf(guess_1RDM)

    if not mf.converged:
        if np.any(np.abs(frag.impham_OEI_S) > 1e-8) and mol.spin != 0:
            raise NotImplementedError(
                'Gradient and Hessian fixes for nonsinglet environment of Newton-descent ROHF algorithm')
        mf = mf.newton()
        mf.kernel()

    # Instability check and repeat
    for i in range(frag.num_mf_stab_checks):
        if np.any(np.abs(frag.impham_OEI_S) > 1e-8) and mol.spin != 0:
            raise NotImplementedError('ROHF stability-check fixes for nonsinglet environment')
        new_mo = mf.stability()[0]
        guess_1RDM = reduce(np.dot, (new_mo, np.diag(mf.mo_occ), new_mo.conjugate().T))
        mf = scf.RHF(mol)
        mf.get_hcore = lambda *args: OEI
        mf.get_ovlp = lambda *args: np.eye(frag.norbs_imp)
        if frag.quasidirect:
            mf.get_jk = frag.impham_get_jk
        else:
            mf._eri = ao2mo.restore(8, frag.impham_TEI, frag.norbs_imp)
        mf = fix_my_RHF_for_nonsinglet_env(mf, sign_MS * frag.impham_OEI_S)
        mf.scf(guess_1RDM)
        if (mf.converged == False):
            mf = mf.newton()
            mf.kernel()

    oneRDM_imp = mf.make_rdm1()
    if np.asarray(oneRDM_imp).ndim == 3:
        oneSDM_imp = oneRDM_imp[0] - oneRDM_imp[1]
        oneRDM_imp = oneRDM_imp[0] + oneRDM_imp[1]
    else:
        oneSDM_imp = np.zeros_like(oneRDM_imp)
    print("Maximum distance between oneRDM_imp and guess_1RDM: {}".format(np.amax(np.abs(oneRDM_imp - guess_1RDM))))

    frag.oneRDM_loc = symmetrize_tensor(frag.oneRDMfroz_loc + represent_operator_in_basis(oneRDM_imp, frag.imp2loc))
    frag.oneSDM_loc = symmetrize_tensor(frag.oneSDMfroz_loc + represent_operator_in_basis(oneSDM_imp, frag.imp2loc))
    frag.twoCDM_imp = None
    frag.E_imp = mf.e_tot + np.einsum('ab,ab->', oneRDM_imp, chempot_imp)
    frag.loc2mo = np.dot(frag.loc2imp, mf.mo_coeff)
    return None







