import scipy
import numpy as np
import itertools
import time, sys, gc
from functools import reduce, partial
from molDMET.util.rdm import get_1RDM_from_OEI
from molDMET.util.basis import represent_operator_in_basis, measure_basis_nonorthonormality, \
    orthonormalize_a_basis, get_subspace_symmetry_blocks,  is_operator_block_adapted, measure_operator_blockbreaking
from molDMET.util.basis import *
from molDMET.util.tensors import symmetrize_tensor
from molDMET.util.la import matrix_eigen_control_options, matrix_svd_control_options, is_matrix_eye
from molDMET.my_pyscf.dmet import rhf as wm_rhf
from pyscf import gto, scf, ao2mo, tools, lo, lib
from pyscf.lo import nao, orth, boys
from pyscf.x2c import x2c
from pyscf.lib import current_memory
from pyscf.lib.numpy_helper import tag_array
logger = lib.logger

class localintegrals(lib.StreamObject):
    """
    Class to localize the integrals
    """

    def __init__(self, the_mf, active_orbs, localizationtype, ao_rotation=None,):

        # Do I need this localization_threshold=1e-6 ?

        assert ((localizationtype == 'meta_lowdin') or (localizationtype == 'boys') or (
                localizationtype == 'lowdin')), 'Only meta_lowdin, boys or lowdin available'
        self.num_mf_stab_checks = 0
        self.ao_rotation = ao_rotation
        # Information on the full HF problem
        self.mol = the_mf.mol
        self._scf = the_mf
        self.max_memory = the_mf.max_memory
        self.get_jk_ao = partial(the_mf.get_jk, self.mol)
        self.get_veff_ao = partial(the_mf.get_veff, self.mol)
        self.get_k_ao = partial(the_mf.get_k, self.mol)
        self.fullovlpao = the_mf.get_ovlp
        self.fullEhf = the_mf.e_tot
        self.log = lib.logger.new_logger(lib.StreamObject, verbose=the_mf.verbose)
        self.fullRDM_ao = np.asarray(the_mf.make_rdm1())
        if self.fullRDM_ao.ndim == 3:
            self.fullSDM_ao = self.fullRDM_ao[0] - self.fullRDM_ao[1]
            self.fullRDM_ao = self.fullRDM_ao[0] + self.fullRDM_ao[1]
        else:
            self.fullSDM_ao = np.zeros_like(self.fullRDM_ao)

        self.fullJK_ao = self.get_veff_ao(dm=self.fullRDM_ao, dm_last=0, vhf_last=0, hermi=1)

        if self.fullJK_ao.ndim == 3:
            self.fullJK_ao = self.fullJK_ao[0]

        self.fullFOCK_ao = the_mf.get_hcore() + self.fullJK_ao
        self.e_tot = the_mf.e_tot
        self.x2c = isinstance(the_mf, x2c._X2C_SCF)

        # System information
        self._which = localizationtype
        self.active = np.zeros([self.mol.nao_nr()], dtype=int)
        self.active[active_orbs] = 1
        self.norbs_tot = np.sum(self.active)
        self.nelec_tot = int(np.rint(self.mol.nelectron - np.sum(
            the_mf.mo_occ[self.active == 0])))  # Total number of electrons minus frozen part

        self.kernel(the_mf, ao_rotation=ao_rotation)
        self.twoelewrapper(the_mf)
        self.symmetryinfo(the_mf)
        sys.stdout.flush()

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** Localization flags ********')
        log.info("number of atomic orbitals: %s", sum(self.active))
        log.info("x2c: %s", self.x2c)
        log.info("loc type: %s", self._which)
        return self

    def kernel(self, the_mf, ao_rotation=None):
        log = logger.new_logger(self, verbose=the_mf.verbose)
        self.dump_flags(log)
        # Localize the orbitals
        if (self._which == 'meta_lowdin') or (self._which == 'boys'):
            if self._which == 'meta_lowdin':
                assert (self.norbs_tot == self.mol.nao_nr())
            if self._which == 'boys':
                self.ao2loc = the_mf.mo_coeff[:, self.active == 1]
            if self.norbs_tot == self.mol.nao_nr():  # If you want the full active, do meta-Lowdin
                nao.AOSHELL[4] = ['1s0p0d0f', '2s1p0d0f']  # redefine the valence shell for Be ?
                self.ao2loc = orth.orth_ao(self.mol, 'meta_lowdin')
                if ao_rotation != None:
                    self.ao2loc = np.dot(self.ao2loc, ao_rotation.T)
            if self._which == 'boys':
                old_verbose = self.mol.verbose
                self.mol.verbose = 0
                loc = boys.Boys(self.mol, self.ao2loc)
                self.mol.verbose = old_verbose
                self.ao2loc = loc.kernel()

            self.TI_OK = False  # Check yourself if OK, then overwrite

        if self._which == 'lowdin':
            assert (self.norbs_tot == self.mol.nao_nr())  # Full active space required
            ovlp = self.mol.intor('cint1e_ovlp_sph')
            ovlp_eigs, ovlp_vecs = np.linalg.eigh(ovlp)
            assert (np.linalg.norm(np.dot(np.dot(ovlp_vecs, np.diag(ovlp_eigs)), ovlp_vecs.T) - ovlp) < 1e-10)
            self.ao2loc = np.dot(np.dot(ovlp_vecs, np.diag(np.power(ovlp_eigs, -0.5))), ovlp_vecs.T)
            self.TI_OK = False  # Check yourself if OK, then overwrite

        assert (self.loc_ortho() < 1e-8)

        # Stored inverse overlap matrix
        self.ao_ovlp_inv = np.dot(self.ao2loc, self.ao2loc.conjugate().T)
        self.ao_ovlp = the_mf.get_ovlp()

        assert (is_matrix_eye(np.dot(self.ao_ovlp, self.ao_ovlp_inv)))

        # Effective Hamiltonian due to frozen part
        self.frozenDM_mo = np.array(the_mf.mo_occ, copy=True)
        self.frozenDM_mo[self.active == 1] = 0  # Only the frozen MO occupancies nonzero
        self.frozenDM_ao = np.dot(np.dot(the_mf.mo_coeff, np.diag(self.frozenDM_mo)), the_mf.mo_coeff.T)
        self.frozenJK_ao = self.get_veff_ao(self.frozenDM_ao, 0, 0, 1)  # Last 3 numbers: dm_last, vhf_last, hermi
        if self.frozenJK_ao.ndim == 3:
            self.frozenJK_ao = self.frozenJK_ao[0]
            # Because I gave it a spin-summed 1-RDM, the two spins for JK will necessarily be identical
        self.frozenOEI_ao = self.fullFOCK_ao - self.fullJK_ao + self.frozenJK_ao

        # Localized OEI and ERI
        self.activeCONST = self.mol.energy_nuc()
        self.activeCONST += np.einsum('ij,ij->', self.frozenOEI_ao - 0.5 * self.frozenJK_ao,
                                      self.frozenDM_ao)
        self.activeOEI = represent_operator_in_basis(self.frozenOEI_ao, self.ao2loc)
        self.activeFOCK = represent_operator_in_basis(self.fullFOCK_ao, self.ao2loc)
        self.activeVSPIN = np.zeros_like(self.activeFOCK)  # FIXME: correct behavior for ROHF init!
        self.activeJKidem = self.activeFOCK - self.activeOEI
        self.activeJKcorr = np.zeros((self.norbs_tot, self.norbs_tot), dtype=self.activeOEI.dtype)
        self.oneRDM_loc = self.ao2loc.conjugate().T @ self.ao_ovlp @ self.fullRDM_ao @ self.ao_ovlp @ self.ao2loc
        self.oneSDM_loc = self.ao2loc.conjugate().T @ self.ao_ovlp @ self.fullSDM_ao @ self.ao_ovlp @ self.ao2loc
        self.oneRDMcorr_loc = np.zeros((self.norbs_tot, self.norbs_tot), dtype=self.activeOEI.dtype)
        self.loc2idem = np.eye(self.norbs_tot, dtype=self.activeOEI.dtype)
        self.nelec_idem = self.nelec_tot
        self._eri = None
        self.with_df = None
        assert (abs(np.trace(self.oneRDM_loc) - self.nelec_tot) < 1e-8), '{} {}'.format(np.trace(self.oneRDM_loc),
                                                                                        self.nelec_tot)

    def twoelewrapper(self, the_mf):
        log = logger.new_logger(self, the_mf.verbose)
        # Unfortunately, there is currently no way to do the integral transformation directly on the antisymmetrized
        # two-electron integrals, at least none already implemented in PySCF. Therefore the smallest possible memory
        # footprint involves two arrays of fourfold symmetry, which works out to roughly one half of an array with no
        # symmetry
        if hasattr(the_mf, 'with_df') and hasattr(the_mf.with_df, '_cderi') and the_mf.with_df._cderi is not None:
            log.info("Found density-fitting three-center integrals scf object")
            loc2ao = self.ao2loc.conjugate().T
            locOao = np.dot(loc2ao, self.ao_ovlp)
            self.with_df = the_mf.with_df
            self.with_df.loc2eri_bas = lambda x: np.dot(self.ao2loc, x)
            self.with_df.loc2eri_op = lambda x: reduce(np.dot, (self.ao2loc, x, loc2ao))
            self.with_df.eri2loc_bas = lambda x: np.dot(locOao, x)
            self.with_df.eri2loc_op = lambda x: reduce(np.dot, (loc2ao, x, self.ao2loc))
        elif the_mf._eri is not None:
            log.info("Found eris on scf object")
            loc2ao = self.ao2loc.conjugate().T
            locOao = np.dot(loc2ao, self.ao_ovlp)
            self._eri = the_mf._eri
            self._eri = tag_array(self._eri, loc2eri_bas=lambda x: np.dot(self.ao2loc, x))
            self._eri = tag_array(self._eri, loc2eri_op=lambda x: reduce(np.dot, (self.ao2loc, x, loc2ao)))
            self._eri = tag_array(self._eri, eri2loc_bas=lambda x: np.dot(locOao, x))
            self._eri = tag_array(self._eri, eri2loc_op=lambda x: reduce(np.dot, (loc2ao, x, self.ao2loc)))
        elif self._is_mem_enough():
            log.info("Storing eris in memory")
            self._eri = ao2mo.restore(8, ao2mo.outcore.full_iofree(self.mol, self.ao2loc, compact=True), self.norbs_tot)
            self._eri = tag_array(self._eri, loc2eri_bas=lambda x: x)
            self._eri = tag_array(self._eri, loc2eri_op=lambda x: x)
            self._eri = tag_array(self._eri, eri2loc_bas=lambda x: x)
            self._eri = tag_array(self._eri, eri2loc_op=lambda x: x)
        else:
            log.info("Direct calculation")
        sys.stdout.flush()
        return self

    def _is_mem_enough(self):
        return 2 * (self.norbs_tot ** 4) / 1e6 + current_memory()[0] < self.max_memory * 0.95

    def symmetryinfo(self, the_mf):
        """Symmetry information

        Returns:

        """
        log = logger.new_logger(self, the_mf.verbose)
        try:
            self.loc2symm = [orthonormalize_a_basis(scipy.linalg.solve(self.ao2loc, ao2ir)) for ao2ir in
                             self.mol.symm_orb]
            self.symmetry = self.mol.groupname
            self.wfnsym = the_mf.wfnsym
            self.ir_names = self.mol.irrep_name
            self.ir_ids = self.mol.irrep_id
            self.enforce_symmetry = True

        except (AttributeError, TypeError) as e:
            if self.mol.symmetry: raise (e)
            self.loc2symm = [np.eye(self.norbs_tot)]
            self.symmetry = False
            self.wfnsym = 'A'
            self.ir_names = ['A']
            self.ir_ids = [0]
            self.enforce_symmetry = False

        log.info("initial loc2symm nonorthonormality: {}".format(
            measure_basis_nonorthonormality(np.concatenate(self.loc2symm, axis=1))))
        for loc2ir1, loc2ir2 in itertools.combinations(self.loc2symm, 2):
            proj = loc2ir1 @ loc2ir1.conjugate().T
            loc2ir2[:, :] -= proj @ loc2ir2
        for loc2ir in self.loc2symm:
            loc2ir[:, :] = orthonormalize_a_basis(loc2ir)
        log.info("final loc2symm nonorthonormality: {}".format(
            measure_basis_nonorthonormality(np.concatenate(self.loc2symm, axis=1))))
        return self

    def loc_ortho(self):
        ShouldBeI = represent_operator_in_basis(self.fullovlpao(), self.ao2loc)
        return np.linalg.norm(ShouldBeI - np.eye(ShouldBeI.shape[0]))

    def const(self):
        return self.activeCONST

    def loc_oei(self):
        return self.activeOEI + self.activeJKcorr

    def loc_rhf_fock(self):
        return self.activeOEI + self.activeJKcorr + self.activeJKidem

    def loc_rhf_jk_bis(self, DMloc):
        '''
            DMloc must be the spin-summed density matrix
        '''
        DM_ao = represent_operator_in_basis(DMloc, self.ao2loc.T)
        JK_ao = self.get_veff_ao(DM_ao, 0, 0, 1)  # Last 3 numbers: dm_last, vhf_last, hermi
        if JK_ao.ndim == 3:
            JK_ao = JK_ao[0]
        JK_loc = represent_operator_in_basis(JK_ao, self.ao2loc)
        return JK_loc

    def loc_rhf_fock_bis(self, DMloc):
        return self.activeOEI + self.loc_rhf_jk_bis(DMloc)

    def loc_rhf_k_bis(self, DMloc):
        DM_ao = represent_operator_in_basis(DMloc, self.ao2loc.T)
        K_ao = self.get_k_ao(DM_ao, 1)
        K_loc = represent_operator_in_basis(K_ao, self.ao2loc)
        return K_loc

    def loc_tei(self):
        raise RuntimeError("localintegrals::loc_tei : ERI of the localized orbitals are not stored in memory.")

    # OEIidem means that the OEI is only used to determine the idempotent part of the 1RDM;
    # the correlated part, if it exists, is kept unchanged

    def get_wm_1RDM_from_OEI(self, OEI, nelec=None, loc2wrk=None):
        nelec = nelec or self.nelec_idem
        loc2wrk = loc2wrk if np.any(loc2wrk) else self.loc2idem
        nocc = nelec // 2
        oneRDM_loc = 2 * get_1RDM_from_OEI(OEI, nocc,subspace=loc2wrk)
        return oneRDM_loc + self.oneRDMcorr_loc

    def get_wm_1RDM_from_scf_on_OEI(self, OEI, nelec=None, loc2wrk=None, oneRDMguess_loc=None, output=None,
                                    working_const=0):

        nelec = nelec or self.nelec_idem
        loc2wrk = loc2wrk if np.any(loc2wrk) else self.loc2idem
        oneRDM_wrk = represent_operator_in_basis(oneRDMguess_loc, loc2wrk) if np.any(oneRDMguess_loc) else None
        nocc = nelec // 2
        # DON'T call self.get_wm_1RDM_from_OEIidem here because you need to hold oneRDMcorr_loc frozen until the end
        # of the scf!
        OEI_wrk = represent_operator_in_basis(OEI, loc2wrk)

        if oneRDM_wrk is None:
            oneRDM_wrk = 2 * get_1RDM_from_OEI(OEI_wrk, nocc)

        ao2wrk = np.dot(self.ao2loc, loc2wrk)
        wrk2symm = get_subspace_symmetry_blocks(loc2wrk, self.loc2symm)

        if self.enforce_symmetry:
            assert (is_operator_block_adapted(oneRDM_wrk, wrk2symm)), \
                measure_operator_blockbreaking(oneRDM_wrk, wrk2symm)

        oneRDM_wrk = wm_rhf.solve_JK(working_const, OEI_wrk, ao2wrk, oneRDM_wrk, nocc,
                                     self.num_mf_stab_checks, self.get_veff_ao, self.get_jk_ao,
                                     groupname=self.symmetry, symm_orb=wrk2symm, irrep_name=self.mol.irrep_name,
                                     irrep_id=self.mol.irrep_id, enforce_symmetry=self.enforce_symmetry,
                                     output=output)
        if self.enforce_symmetry:
            assert (is_operator_block_adapted(oneRDM_wrk, wrk2symm)), \
                measure_operator_blockbreaking(oneRDM_wrk, wrk2symm)

        oneRDM_loc = represent_operator_in_basis(oneRDM_wrk, loc2wrk.T)

        if self.enforce_symmetry:
            assert (is_operator_block_adapted(oneRDM_loc, self.loc2symm)), \
                measure_operator_blockbreaking(oneRDM_loc,self.loc2symm)

        return oneRDM_loc + self.oneRDMcorr_loc

    def dmet_oei(self, loc2dmet, numActive):
        OEIdmet = np.dot(np.dot(loc2dmet[:, :numActive].T, self.activeOEI), loc2dmet[:, :numActive])
        return symmetrize_tensor(OEIdmet)

    def dmet_fock(self, loc2dmet, numActive, coreDMloc):
        FOCKdmet = np.dot(np.dot(loc2dmet[:, :numActive].T, self.loc_rhf_fock_bis(coreDMloc)), loc2dmet[:, :numActive])
        return symmetrize_tensor(FOCKdmet)

    def dmet_k(self, loc2imp, norbs_imp, DMloc):
        k_imp = represent_operator_in_basis(self.loc_rhf_k_bis(DMloc), loc2imp[:, :norbs_imp])
        return symmetrize_tensor(k_imp)

    def dmet_init_guess_rhf(self, loc2dmet, numActive, numPairs, norbs_frag, chempot_imp):

        Fock_small = np.dot(np.dot(loc2dmet[:, :numActive].T, self.loc_rhf_fock()), loc2dmet[:, :numActive])
        if (chempot_imp != 0.0):
            Fock_small[np.diag_indices(norbs_frag)] -= chempot_imp
        eigvals, eigvecs = np.linalg.eigh(Fock_small)
        eigvecs = eigvecs[:, eigvals.argsort()]
        DMguess = 2 * np.dot(eigvecs[:, :numPairs], eigvecs[:, :numPairs].T)
        return DMguess

    def dmet_cderi(self, loc2dmet, numAct=None):
        t0 = time.process_time()
        w0 = time.time()
        norbs_aux = self.with_df.get_naoaux()
        numAct = loc2dmet.shape[1] if numAct == None else numAct
        loc2imp = loc2dmet[:, :numAct]
        assert (self.with_df is not None), "density fitting required"
        npair = numAct * (numAct + 1) // 2
        CDERI = np.empty((self.with_df.get_naoaux(), npair), dtype=loc2dmet.dtype)
        full_cderi_size = (norbs_aux * self.mol.nao_nr() * (self.mol.nao_nr() + 1) * CDERI.itemsize // 2) / 1e6
        imp_eri_size = (CDERI.itemsize * npair * (npair + 1) // 2) / 1e6
        imp_cderi_size = CDERI.size * CDERI.itemsize / 1e6
        print(
            "Size comparison: cderi is ({0},{1},{1})->{2:.0f} MB compacted; eri is ({1},{1},{1},{1})->{3:.0f} MB compacted".format(
                norbs_aux, numAct, imp_cderi_size, imp_eri_size))
        ao2imp = np.dot(self.ao2loc, loc2imp)
        ijmosym, mij_pair, moij, ijslice = ao2mo.incore._conc_mos(ao2imp, ao2imp, compact=True)
        b0 = 0
        for eri1 in self.with_df.loop():
            b1 = b0 + eri1.shape[0]
            eri2 = CDERI[b0:b1]
            eri2 = ao2mo._ao2mo.nr_e2(eri1, moij, ijslice, aosym='s2', mosym=ijmosym, out=eri2)
            b0 = b1
        t1 = time.process_time()
        w1 = time.time()
        print(("({0}, {1}) seconds to turn {2:.0f}-MB full"
               "cderi array into {3:.0f}-MP impurity cderi array").format(
            t1 - t0, w1 - w0, full_cderi_size, imp_cderi_size))

        return CDERI

    def dmet_tei(self, loc2dmet, numAct=None, symmetry=1):

        numAct = loc2dmet.shape[1] if numAct == None else numAct
        loc2imp = loc2dmet[:, :numAct]
        TEI = symmetrize_tensor(self.general_tei([loc2imp for i in range(4)], compact=True))
        return ao2mo.restore(symmetry, TEI, numAct)

    def dmet_const(self, loc2dmet, norbs_imp, oneRDMfroz_loc, oneSDMfroz_loc):
        norbs_core = self.norbs_tot - norbs_imp
        if norbs_core == 0:
            return 0.0
        loc2core = loc2dmet[:, norbs_imp:]
        GAMMA = represent_operator_in_basis(oneRDMfroz_loc, loc2core)
        OEI = self.dmet_oei(loc2core, norbs_core)
        OEI += self.dmet_fock(loc2core, norbs_core, oneRDMfroz_loc)
        CONST = (GAMMA * OEI).sum() / 2
        M = represent_operator_in_basis(oneSDMfroz_loc, loc2core)
        K = self.dmet_k(loc2core, norbs_core, oneSDMfroz_loc)
        CONST -= (M * K).sum() / 4
        return CONST

    def general_tei(self, loc2bas_list, compact=False):
        norbs = [loc2bas.shape[1] for loc2bas in loc2bas_list]
        print(
            "Formal max memory: {} MB; Current usage: {} MB; Maximal storage requirements of this TEI tensor: {} MB".format(
                self.max_memory, current_memory()[0], 8 * norbs[0] * norbs[1] * norbs[2] * norbs[3] / 1e6))
        sys.stdout.flush()

        if self.with_df is not None:
            a2b_list = [self.with_df.loc2eri_bas(l2b) for l2b in loc2bas_list]
            TEI = self.with_df.ao2mo(a2b_list, compact=compact)
        elif self._eri is not None:
            a2b_list = [self._eri.loc2eri_bas(l2b) for l2b in loc2bas_list]
            TEI = ao2mo.incore.general(self._eri, a2b_list, compact=compact)
        else:
            a2b_list = [np.dot(self.ao2loc, l2b) for l2b in loc2bas_list]
            TEI = ao2mo.outcore.general_iofree(self.mol, a2b_list, compact=compact)

        if not compact: TEI = TEI.reshape(*norbs)
        gc.collect()  # I guess the ao2mo module is messy because until I put this here I was randomly losing up to 3 GB for big stretches of a calculation

        return TEI

    def get_trial_nos(self, aobasis=False, loc2wmas=None, oneRDM_loc=None, fock=None, jmol_shift=False,
                      try_symmetrize=True):
        if oneRDM_loc is None: oneRDM_loc = self.oneRDM_loc
        if fock is None:
            fock = self.activeFOCK
        elif isinstance(fock, str) and fock == 'calculate':
            fock = self.loc_rhf_fock_bis(oneRDM_loc)
        if loc2wmas is None:
            loc2wmas = [np.zeros((self.norbs_tot, 0), dtype=self.ao2loc.dtype)]
        elif isinstance(loc2wmas, np.ndarray):
            if loc2wmas.ndim == 2: loc2wmas = loc2wmas[None, :, :]
            loc2wmas = [loc2amo for loc2amo in loc2wmas]
        occ_wmas = [np.zeros(0) for ix in loc2wmas]
        symm_wmas = [np.zeros(0) for ix in loc2wmas]
        for ix, loc2amo in enumerate(loc2wmas):
            occ_wmas[ix], loc2wmas[ix], symm_wmas[ix] = matrix_eigen_control_options(oneRDM_loc, symmetry=self.loc2symm,
                                                                                     subspace=loc2amo,
                                                                                     sort_vecs=-1,
                                                                                     only_nonzero_vals=False,
                                                                                     strong_symm=self.enforce_symmetry)
        occ_wmas = np.concatenate(occ_wmas)
        symm_wmas = np.concatenate(symm_wmas)
        loc2wmas = np.concatenate(loc2wmas, axis=-1)
        nelec_wmas = int(round(compute_nelec_in_subspace(oneRDM_loc, loc2wmas)))

        loc2wmcs = get_complementary_states(loc2wmas, symmetry=self.loc2symm, enforce_symmetry=self.enforce_symmetry)
        norbs_wmas = loc2wmas.shape[1]
        norbs_wmcs = loc2wmcs.shape[1]
        ene_wmcs, loc2wmcs, symm_wmcs = matrix_eigen_control_options(fock, symmetry=self.loc2symm, subspace=loc2wmcs,
                                                                     sort_vecs=1, only_nonzero_vals=False,
                                                                     strong_symm=self.enforce_symmetry)

        assert ((self.nelec_tot - nelec_wmas) % 2 == 0), 'Non-even number of unactive electrons {}'.format(
            self.nelec_tot - nelec_wmas)
        norbs_core = (self.nelec_tot - nelec_wmas) // 2
        norbs_virt = norbs_wmcs - norbs_core
        loc2wmis = loc2wmcs[:, :norbs_core]
        symm_wmis = symm_wmcs[:norbs_core]
        loc2wmxs = loc2wmcs[:, norbs_core:]
        symm_wmxs = symm_wmcs[norbs_core:]

        if self.mol.symmetry:
            symm_wmis = {self.mol.irrep_name[x]: np.count_nonzero(symm_wmis == x) for x in np.unique(symm_wmis)}
            err = measure_subspace_blockbreaking(loc2wmis, self.loc2symm)
            print("Trial wave function inactive-orbital irreps = {}, err = {}".format(symm_wmis, err))
            symm_wmas = {self.mol.irrep_name[x]: np.count_nonzero(symm_wmas == x) for x in np.unique(symm_wmas)}
            err = measure_subspace_blockbreaking(loc2wmas, self.loc2symm)
            print("Trial wave function active-orbital irreps = {}, err = {}".format(symm_wmas, err))
            symm_wmxs = {self.mol.irrep_name[x]: np.count_nonzero(symm_wmxs == x) for x in np.unique(symm_wmxs)}
            err = measure_subspace_blockbreaking(loc2wmxs, self.loc2symm)
            print("Trial wave function external-orbital irreps = {}, err = {}".format(symm_wmxs, err))

        loc2no = np.concatenate((loc2wmcs[:, :norbs_core], loc2wmas, loc2wmcs[:, norbs_core:]), axis=1)
        occ_no = np.concatenate((2 * np.ones(norbs_core), occ_wmas, np.zeros(norbs_virt)))
        ene_no = np.concatenate((ene_wmcs[:norbs_core], np.zeros(norbs_wmas), ene_wmcs[norbs_core:]))
        assert (len(occ_no) == len(ene_no) and loc2no.shape[1] == len(occ_no)), '{} {} {}'.format(loc2no.shape,
                                                                                                  len(ene_no),
                                                                                                  len(occ_no))
        norbs_occ = norbs_core + norbs_wmas
        if jmol_shift:
            print("Shifting natural-orbital energies so that jmol puts them in the correct order:")
            if ene_no[norbs_core - 1] > 0: ene_no[:norbs_core] -= ene_no[norbs_core - 1] + 1e-6
            if ene_no[norbs_occ] < 0: ene_no[norbs_occ:] -= ene_no[norbs_occ] - 1e-6
            assert (np.all(np.diff(ene_no) >= 0)), ene_no
        if aobasis:
            return self.ao2loc @ loc2no, ene_no, occ_no
        return loc2no, ene_no, occ_no
