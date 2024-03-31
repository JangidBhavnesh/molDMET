import numpy as np
from functools import reduce
from pyscf import ao2mo, gto, scf, lib
from molDMET.util.basis import represent_operator_in_basis, project_operator_into_subspace, measure_basis_olap
from molDMET.util.tensors import symmetrize_tensor


def solve(frag, guess_1RDM, chempot_imp):
    # Augment OEI with the chemical potential
    OEI = frag.impham_OEI_C - chempot_imp
    sign_MS = np.sign(frag.target_MS) or 1

    # Get the RHF solution
    mol = gto.Mole()
    mol.spin = abs(int(round(2 * frag.target_MS)))
    mol.verbose = 0

    if frag.mol_stdout is None:
        mol.output = frag.mol_output
        mol.verbose = 0 if frag.mol_output is None else lib.logger.DEBUG
    mol.build()

    if frag.mol_stdout is None:
        frag.mol_stdout = mol.stdout
    else:
        mol.stdout = frag.mol_stdout
        mol.verbose = 0 if frag.mol_output is None else lib.logger.DEBUG

    mol.atom.append(('C', (0, 0, 0)))
    mol.nelectron = frag.nelec_imp
    mol.max_memory = frag.ints.max_memory
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


def fix_my_RHF_for_nonsinglet_env(mf, h1e_s):
    if h1e_s is None or mf.mol.spin == 0: return mf

    class fixed_RHF(mf.__class__):

        def __init__(self, my_mf):
            self.__dict__.update(my_mf.__dict__)

        def get_fock(self, h1e=None, s1e=None, vhf=None, dm=None, cycle=1, diis=None,
                     diis_start_cycle=None, level_shift_factor=None, damp_factor=None):

            if vhf is None: vhf = self.get_veff(self.mol, dm)
            vhf[0] += h1e_s
            vhf[1] -= h1e_s
            return super().get_fock(h1e=h1e, s1e=s1e, vhf=vhf, dm=dm, cycle=cycle, diis=diis,
                                    diis_start_cycle=diis_start_cycle, level_shift_factor=level_shift_factor,
                                    damp_factor=damp_factor)

        def energy_elec(self, dm=None, h1e=None, vhf=None):
            if dm is None: dm = self.make_rdm1()
            e_elec, e_coul = super().energy_elec(dm=dm, h1e=h1e, vhf=vhf)
            e_elec += (h1e_s * (dm[0] - dm[1])).sum()
            return e_elec, e_coul

    return fixed_RHF(mf)
