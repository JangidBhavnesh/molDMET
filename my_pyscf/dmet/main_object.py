from molDMET.my_pyscf.dmet import localintegrals, qcdmethelper
import warnings
import numpy as np
from scipy import optimize, linalg
import time, ctypes
from pyscf import scf, mcscf, lib
from pyscf.lo import orth, nao
from pyscf.gto import mole, same_mol
from pyscf.tools import molden
from pyscf.scf.addons import project_mo_nr2nr, project_dm_nr2nr
from molDMET.util import params
from molDMET.util.io import prettyprint_ndarray as prettyprint
from molDMET.util.la import matrix_eigen_control_options, matrix_svd_control_options, assign_blocks_weakly
from molDMET.util.basis import represent_operator_in_basis, orthonormalize_a_basis, get_complementary_states, \
    project_operator_into_subspace
from molDMET.util.basis import is_matrix_eye, measure_basis_olap, is_basis_orthonormal_and_complete, \
    is_basis_orthonormal, \
    get_overlapping_states

from molDMET.util.basis import assign_blocks, align_states, measure_subspace_blockbreaking
from molDMET.util.rdm import get_2RDM_from_2CDM, get_2CDM_from_2RDM, get_1RDM_from_OEI
from mrh.my_dmet.debug import debug_ofc_oneRDM, debug_Etot, examine_ifrag_olap, examine_wmcs
from functools import reduce
from itertools import combinations, product
from molDMET.my_pyscf.dmet.fragments import make_fragment_atom_list, make_fragment_orb_list


class dmet:

    def __init__(self, theInts, fragments, solver='RHF', bath_tol=1e-5, isTranslationInvariant=False, SCmethod='BFGS',
                 enforce_symmetry=False, minFunc='FOCK_INIT', nelec_int_thresh=1e-6, chempot_init=0.0,
                 num_mf_stab_checks=0, corrpot_maxiter=50, orb_maxiter=50, chempot_tol=1e-6, conv_tol_grad=1e-4):

        # I will have to put this on hold
        self.fragmentInfo = None
        if isTranslationInvariant:
            raise RuntimeError("The translational invariance option doesn't work!  It needs to be completely rebuilt!")
            assert (theInts.TI_OK == True)
            assert (len(fragments) == 1)

        assert ((SCmethod == 'LSTSQ') or (SCmethod == 'BFGS') or (SCmethod == 'NONE'))

        self.ints = theInts
        self.norbs_tot = self.ints.norbs_tot
        self.fragments = [make_fragment_atom_list(theInts, fragments, solver, bath_tol, name='MyFragment'), ]
        self.NI_hack = False
        self.doSCF = False
        self.TransInv = isTranslationInvariant
        self.minFunc = minFunc
        self.nelec_int_thresh = nelec_int_thresh
        self.chempot = chempot_init
        self.num_mf_stab_checks = num_mf_stab_checks
        self.corrpot_maxiter = corrpot_maxiter
        self.orb_maxiter = orb_maxiter
        self.chempot_tol = chempot_tol
        self.ints.num_mf_stab_checks = num_mf_stab_checks
        if not self.ints.symmetry: enforce_symmetry = False
        self.enforce_symmetry = enforce_symmetry
        self.conv_tol_grad = conv_tol_grad
        self.checkexact = False
        self.verbose = self.ints.mol.verbose
        for frag in self.fragments:
            frag.debug_energy = False
            frag.num_mf_stab_checks = num_mf_stab_checks
            frag.groupname = self.ints.symmetry
            frag.loc2symm = self.ints.loc2symm
            frag.ir_names = self.ints.ir_names
            frag.ir_ids = self.ints.ir_ids

        self.examine_ifrag_olap = False
        self.examine_wmcs = False
        self.ofc_emb_init_ncycles = 3
        self.get_allcore_orbs()
        if (self.norbs_allcore > 0):  # One or more impurities which do not cover the entire system
            assert (self.TransInv == False)  # Make sure that you don't work translational invariant
            # Note on working with impurities which do no tile the entire system: they should be the first orbitals in the Hamiltonian!

        def umat_ftriu_mask(x, k=0):
            r = np.zeros(x.shape, dtype=np.bool_)
            for frag in self.fragments:
                ftriu = np.triu_indices(frag.norbs_frag)
                c = tuple(np.asarray([frag.frag_orb_list[i] for i in f]) for f in ftriu)
                r[c] = True
            return r

        self.umat_ftriu_idx = np.mask_indices(self.norbs_tot, umat_ftriu_mask)
        self.altcostfunc = None
        self.loc2fno = None
        self.umat = np.zeros([self.norbs_tot, self.norbs_tot])
        self.relaxation = 0.0
        self.energy = 0.0
        self.spin = 0.0
        self.helper = qcdmethelper.qcdmethelper(self.ints, self.makelist_H1(), self.altcostfunc, self.minFunc)
        np.set_printoptions(precision=3, linewidth=160)

    def acceptable_errvec_check(obj):
        errcheck = (
            ("bath in errvec incompatible with DET", obj.incl_bath_errvec and obj.doDET),
            ("constrained opt incompatible with DET", obj.altcostfunc and obj.doDET),
            ("bath in errvec incompatible with constrained opt", obj.altcostfunc and obj.incl_bath_errvec),
            ("natural orbital basis not meaningful for DMET", obj.doDET_NO and (not obj.doDET))
        )
        for errstr, err in errcheck:
            if err:
                raise RuntimeError(errstr)

    def get_allcore_orbs(self):
        # I guess this just determines whether every orbital has a fragment or not    

        quicktest = np.zeros([self.norbs_tot], dtype=int)
        for frag in self.fragments:
            quicktest += np.abs(frag.is_frag_orb.astype(int))
        assert (np.all(quicktest >= 0))
        assert (np.all(quicktest <= 1))
        self.is_allcore_orb = np.logical_not(quicktest.astype(bool))
        self.oneRDMallcore_loc = np.zeros_like(self.ints.activeOEI)

    @property
    def loc2allcore(self):
        return np.eye(self.norbs_tot)[:, self.is_allcore_orb]

    @property
    def Pallcore(self):
        return np.diag(self.is_allcore_orb).astype(int)

    @property
    def norbs_allcore(self):
        return np.count_nonzero(self.is_allcore_orb)

    @property
    def allcore_orb_list(self):
        return np.flatnonzero(self.is_allcore_orb)

    @property
    def nelec_wma(self):
        return int(sum(frag.active_space[0] for frag in self.fragments))

    @property
    def norbs_wma(self):
        return int(sum(frag.active_space[1] for frag in self.fragments))

    @property
    def norbs_wmc(self):
        return self.norbs_tot - self.norbs_wma

    def makelist_H1(self):
        H1start = []
        H1row = []
        H1col = []
        H1start.append(0)
        totalsize = 0

        def sparsify(list_H1, totalsize, H1start, H1row, H1col):
            rowco, colco = np.where(list_H1 == 1)
            totalsize += len(rowco)
            H1start.append(totalsize)
            for count2 in range(len(rowco)):
                H1row.append(rowco[count2])
                H1col.append(colco[count2])
            return totalsize, H1start, H1row, H1col

        if (self.TransInv == True):  # Translational invariance assumed
            # Same as above
            for row in range(self.fragments[0].norbs_frag):
                for col in range(row, self.fragments[0].norbs_frag):
                    H1 = np.zeros([self.norbs_tot, self.norbs_tot], dtype=int)
                    for jumper in range(self.norbs_tot // self.fragments[0].norbs_frag):
                        jumpsquare = self.fragments[0].norbs_frag * jumper
                        H1[jumpsquare + row, jumpsquare + col] = 1
                        H1[jumpsquare + col, jumpsquare + row] = 1
                    # theH1.append( H1 )
                    totalsize, H1start, H1row, H1col = sparsify(H1, totalsize, H1start, H1row, H1col)
        else:  # NO translational invariance assumed
            # same as above
            jumpsquare = 0
            for frag in self.fragments:
                for row in range(frag.norbs_frag):
                    for col in range(row, frag.norbs_frag):
                        H1 = np.zeros([self.norbs_tot, self.norbs_tot], dtype=int)
                        H1[jumpsquare + row, jumpsquare + col] = 1
                        H1[jumpsquare + col, jumpsquare + row] = 1
                        # theH1.append( H1 )
                        totalsize, H1start, H1row, H1col = sparsify(H1, totalsize, H1start, H1row, H1col)
                jumpsquare += frag.norbs_frag
        H1start = np.array(H1start, dtype=ctypes.c_int)
        H1row = np.array(H1row, dtype=ctypes.c_int)
        H1col = np.array(H1col, dtype=ctypes.c_int)
        return (H1start, H1row, H1col)

    def kernel(self, chempot_frag=0.0):
        oneRDM_loc = self.helper.construct1RDM_loc(self.doSCF, self.umat)
        self.energy = 0.0
        self.spin = 0.0
        for frag in self.fragments:
            frag.solve_impurity_problem(chempot_frag)
            self.energy += frag.E_frag
            self.spin += frag.S2_frag

        Nelectrons = sum((frag.nelec_frag for frag in self.fragments))

        if self.TransInv:
            Nelectrons = Nelectrons * len(self.fragments)
            self.energy = self.energy * len(self.fragments)

        # When an incomplete impurity tiling is used for the Hamiltonian, self.energy should be augmented with the
        # remaining HF part
        if self.norbs_allcore > 0:
            from pyscf import scf
            from types import MethodType
            mol_ = self.ints.mol
            mol_.verbose = 0
            mf_ = scf.RHF(mol_)

            xorb = np.dot(mf_.get_ovlp(), self.ints.ao2loc)
            hc = -self.chempot * np.dot(xorb[:, self.is_allcore_orb], xorb[:, self.is_allcore_orb].T)
            dm0 = np.dot(self.ints.ao2loc, np.dot(oneRDM_loc, self.ints.ao2loc.T))

            def mf_hcore(self, mol=None):
                if mol is None: mol = self.mol
                return scf.hf.get_hcore(mol) + hc

            mf_.get_hcore = MethodType(mf_hcore, mf_)
            mf_.scf(dm0)
            assert (mf_.converged)

            rdm1 = mf_.make_rdm1()
            jk = mf_.get_veff(dm=rdm1)

            xorb = np.dot(mf_.get_ovlp(), self.ints.ao2loc)
            rdm1 = np.dot(xorb.T, np.dot(rdm1, xorb))
            oei = np.dot(self.ints.ao2loc.T, np.dot(mf_.get_hcore() - hc, self.ints.ao2loc))
            jk = np.dot(self.ints.ao2loc.T, np.dot(jk, self.ints.ao2loc))
            oei_eff = oei + (0.5 * jk)

            AllcoreEnergy = 0.5 * np.einsum('ij,ij->', rdm1[:, self.is_allcore_orb], oei_eff[:, self.is_allcore_orb]) \
                            + 0.5 * np.einsum('ij,ij->', rdm1[self.is_allcore_orb, :], oei_eff[self.is_allcore_orb, :])
            self.energy += AllcoreEnergy
            Nelectrons += np.trace(rdm1[np.ix_(self.is_allcore_orb, self.is_allcore_orb)])
            self.oneRDMallcore_loc = rdm1

        self.energy += self.ints.const()
        #print("Current sum of fragment energies: {0:.6f}".format(self.energy))
        #print("Current sum of fragment spins: {0:.6f}".format(self.spin))
        return Nelectrons

    def constructloc2fno(self):

        myloc2fno = np.zeros((self.norbs_tot, self.norbs_tot))
        for frag in self.fragments:
            myloc2fno[:, frag.frag_orb_list] = frag.loc2fno
        if self.TransInv:
            raise RuntimeError("fix constructloc2fno before you try to do translationally-invariant NO-basis things")
            norbs_frag = self.fragments[0].norbs_frag
            for it in range(1, self.norbs_tot / norbs_frag):
                myloc2fno[it * norbs_frag:(it + 1) * norbs_frag, it * norbs_frag:(it + 1) * norbs_frag] = myloc2fno[
                                                                                                          0:norbs_frag,
                                                                                                          0:norbs_frag]
        '''if True:
            assert ( linalg.norm( np.dot( myloc2fno.T, myloc2fno ) - np.eye( self.umat.shape[0] ) ) < 1e-10 )
            assert ( linalg.norm( np.dot( myloc2fno, myloc2fno.T ) - np.eye( self.umat.shape[0] ) ) < 1e-10 )
        elif (jumpsquare != frag.norbs_tot):
            myloc2fno = mrh.util.basis.complete_a_basis (myloc2fno)'''
        return myloc2fno

    def numeleccostfunction(self, chempot_imp):
        Nelec_dmet = self.kernel(chempot_imp)
        Nelec_target = self.ints.nelec_tot
        self.log.note("Number of electrons = (%.6f)", Nelec_target)
        self.log.note("Number of electrons at DMET = (%.6f)", Nelec_dmet)
        #self.log.note("chemical potential , number of electrons = (%.6f, %.6f)\n", chempot_imp, Nelec_dmet)
        return Nelec_dmet - Nelec_target

    def oneshot(self):
        loc2wmas_old = np.concatenate([frag.loc2amo for frag in self.fragments], axis=1)

        try:
            loc2wmcs_old = get_complementary_states(loc2wmas_old, symmetry=self.ints.loc2symm,
                                                    enforce_symmetry=self.enforce_symmetry)
        except linalg.LinAlgError as e:
            print(np.dot(loc2wmas_old.T, loc2wmas_old))
            raise e

        for frag in self.fragments:
            frag.restore_default_embedding_basis()
        oneRDM_loc = self.helper.construct1RDM_loc(self.doSCF, self.umat)
        old_energy = self.energy
        self.energy = 0.0
        self.spin = 0.0
        for frag in self.fragments:
            frag.do_Schmidt(oneRDM_loc, self.fragments, loc2wmcs_old)
            frag.construct_impurity_hamiltonian()

        if self.examine_ifrag_olap:
            examine_ifrag_olap(self)

        if self.examine_wmcs:
            examine_wmcs(self)

        log = lib.logger.new_logger(self.ints.mol)
        self.log = log
        # Iterate on chemical potential
        if self.checkexact:
            self.chempot = 0.0
            self.kernel(self.chempot)
        else:
            #self.chempot = optimize.newton(self.numeleccostfunction, self.chempot, tol=self.chempot_tol)
            print("one-Shot DMET Initiated")

        return self

    def CoreIP(self, method='adc(2)', nroots=1, ncvs=1, corehole=0):

        for frag in self.fragments:
            frag.solve_excited_problem(self.chempot, method=method, nroots=nroots, ncvs=ncvs, corehole=corehole)

        # loc2wmas_new = np.concatenate([frag.loc2amo for frag in self.fragments], axis=1)
        # loc2wmcs_new = get_complementary_states(loc2wmas_new, symmetry=self.ints.loc2symm,
        #                                         enforce_symmetry=self.enforce_symmetry)
        #
        # try:
        #     orb_diff = measure_basis_olap(loc2wmas_new, loc2wmcs_old)[0] / max(1, loc2wmas_new.shape[1])
        # except:
        #     raise RuntimeError("what?\n{0}\n{1}".format(loc2wmas_new.shape, loc2wmcs_old.shape))
        #
        # oneRDM_locs = np.stack([f.oneRDM_loc for f in self.fragments], axis=-1)
        # energies = np.asarray([f.E_imp for f in self.fragments])
        # oneRDM_stdev = np.std(oneRDM_locs, axis=2)
        # oneRDM_diff = np.linalg.norm(oneRDM_stdev)
        # Eimp_stdev = np.std(energies)
        # oneRDM_avg = self.transform_ed_1rdm(get_od=True)
        # no_occs, no_evecs = matrix_eigen_control_options(oneRDM_avg, sort_vecs=-1, only_nonzero_vals=False)
        # orb_diff = oneRDM_diff = Eimp_stdev = Eiter = norm_gorb = norm_gci = 0  # Do only 1 iteration
        # # else:
        # #     self.energy = e_tot
        # #     Eiter = self.energy - old_energy
        # #     norm_gorb = linalg.norm(np.concatenate([grads[0], grads[2]]))
        # #     norm_gci = linalg.norm(grads[1])
        # #     print("Whole-molecule orbital gradient norm = {}".format(norm_gorb))
        # #     print("Whole-molecule CI gradient norm = {}".format(norm_gci))
        #
        # print("Whole-molecule energy difference = {}".format(Eiter))
        #
        # # Safety until I figure out how to deal with this degenerate-orbital thing
        # # if abs (Eiter) < 1e-7 and np.all (np.abs (energies - self.energy) < 1e-7):
        # #    print ("Energies all converged to 100 nanoEh threshold; punking out of 1-RDM and orbital convergence")
        # #    orb_diff = oneRDM_diff = 0

        return

    def transform_ed_1rdm(self, get_od=False):

        norbs_frag = [f.loc2frag.shape[1] for f in self.fragments]
        result = np.zeros([sum(norbs_frag), sum(norbs_frag)], dtype=np.float64)
        loc2frag = np.concatenate([f.loc2frag for f in self.fragments], axis=1)
        assert (sum(norbs_frag) == loc2frag.shape[1])
        frag_ranges = [sum(norbs_frag[:i]) for i in range(len(norbs_frag) + 1)]
        for idx, f1 in enumerate(self.fragments):
            i = frag_ranges[idx]
            j = frag_ranges[idx + 1]
            result[i:j, i:j] = f1.get_oneRDM_frag()
            if get_od and idx + 1 < len(self.fragments):
                for idx2, f2 in enumerate(self.fragments[idx + 1:]):
                    k = frag_ranges[idx2 + idx + 1]
                    l = frag_ranges[idx2 + idx + 2]
                    result[i:j, k:l] = represent_operator_in_basis(f1.oneRDM_loc, f1.loc2frag, f2.loc2frag)
                    result[i:j, k:l] += represent_operator_in_basis(f2.oneRDM_loc, f1.loc2frag, f2.loc2frag)
                    result[i:j, k:l] /= 2
                    result[k:l, i:j] = result[i:j, k:l].T
        return represent_operator_in_basis(result, loc2frag.conjugate().T)

    def dump_bath_orbs(self, filename, frag_idx=0):
        from pyscf.tools import molden
        with open(filename, 'w') as thefile:
            molden.header(self.ints.mol, thefile)
            molden.orbital_coeff(self.ints.mol, thefile, np.dot(self.ints.ao2loc, self.fragment[frag_idx].loc2imp))

    def onedm_solution_rhf(self):
        return self.helper.construct1RDM_loc(self.doSCF, self.umat)

    def refragmentation(self, loc2wmas, loc2wmcs, oneRDM_loc):
        ''' Separately ortho-localize the whole-molecule active orbitals and whole-molecule core orbitals and adjust the RDM and CDMs. '''

        # Don't waste time if there are no active orbitals
        if loc2wmas.shape[1] == 0:
            return oneRDM_loc

        # Examine the states that turn into I relocalize if the debug is specified.
        if self.debug_reloc:
            ao2before = np.dot(self.ints.ao2loc, np.append(loc2wmas, loc2wmcs, axis=1))

        # Separately localize and assign the inactive and active orbitals as fragment orbitals
        loc2wmas = self.refrag_lowdin_active(loc2wmas, oneRDM_loc)
        loc2wmcs = self.refrag_lowdin_external(loc2wmcs, loc2wmas)

        # Evaluate how many electrons are in each active subspace
        interrs = []
        for loc2amo, frag in zip(loc2wmas, self.fragments):
            nelec_amo = np.einsum("ip,ij,jp->", loc2amo.conjugate(), oneRDM_loc, loc2amo)
            interr = nelec_amo - round(nelec_amo)
            interrs.append(interr)
            print("{0} fragment has {1:.5f} active electrons (integer error: {2:.2e})".format(frag.frag_name, nelec_amo,
                                                                                              interr))
            # assert (interr < self.nelec_int_thresh), "Fragment with non-integer number of electrons appears"
            interr = np.eye(loc2amo.shape[1]) * interr / loc2amo.shape[1]
            oneRDM_loc -= reduce(np.dot, [loc2amo, interr, loc2amo.conjugate().T])
        if ((not self.doLASSCF) or self.oldLASSCF):
            assert (all((i < self.nelec_int_thresh for i in
                         interrs))), "Fragment with non-integer number of electrons appears"

        # Evaluate the entanglement of the active subspaces
        for (o1, f1), (o2, f2) in combinations(zip(loc2wmas, self.fragments), 2):
            if o1.shape[1] > 0 and o2.shape[1] > 0:
                RDM_12 = reduce(np.dot, [o1.conjugate().T, oneRDM_loc, o2])
                RDM_12_sq = np.dot(RDM_12.conjugate().T, RDM_12)
                evals, evecs = linalg.eigh(RDM_12_sq)
                entanglement = sum(evals)
                print("Fragments {} and {} have an active-space entanglement trace of {:.2e}".format(f1.frag_name,
                                                                                                     f2.frag_name,
                                                                                                     entanglement))

        # Examine localized states if necessary
        # Put this at the end so if I get all the way to another orbiter and assert-fails on the integer thing, I can look at the ~last~ set of relocalized orbitals
        if self.debug_reloc:
            fock_loc = self.ints.loc_rhf_fock_bis(oneRDM_loc)
            loc2c = np.concatenate(loc2wmcs, axis=1)
            loc2a = np.concatenate(loc2wmas, axis=1)
            lo_ene, evecs_c = matrix_eigen_control_options(represent_operator_in_basis(fock_loc, loc2c), sort_vecs=1,
                                                           only_nonzero_vals=False)
            loc2c = np.dot(loc2c, evecs_c)
            lo_ene = np.append(-999 * np.ones(loc2a.shape[1]), lo_ene)
            loc2after = np.append(loc2a, loc2c, axis=1)
            lo_occ = np.einsum('ip,ij,jp->p', loc2after, oneRDM_loc, loc2after)
            ao2after = np.dot(self.ints.ao2loc, loc2after)
            occ = np.zeros(ao2before.shape[1])
            occ[:sum([l.shape[1] for l in loc2wmas])] = 1
            molden.from_mo(self.ints.mol, "main_object.refragmentation.before.molden", ao2before, occ=occ)
            molden.from_mo(self.ints.mol, "main_object.refragmentation.after.molden", ao2after, occ=lo_occ, ene=lo_ene)

        for loc2imo, loc2amo, frag in zip(loc2wmcs, loc2wmas, self.fragments):
            frag.set_new_fragment_basis(np.append(loc2imo, loc2amo, axis=1))
            if frag.imp_solver_name != 'dummy RHF':
                assert (is_basis_orthonormal(loc2imo))
                assert (is_basis_orthonormal(loc2amo))
                assert (is_basis_orthonormal(frag.loc2frag)), linalg.norm(loc2imo.conjugate().T @ loc2amo)

        # self.ints.setup_wm_core_scf (self.fragments, self.calcname)
        e_tot, grads = self.lasci_(oneRDM_loc, loc2wmas=loc2wmas)
        for loc2imo, frag in zip(loc2wmcs, self.fragments):
            frag.set_new_fragment_basis(np.append(loc2imo, frag.loc2amo, axis=1))
            assert (is_basis_orthonormal(frag.loc2amo))
            assert (is_basis_orthonormal(frag.loc2frag)), linalg.norm(loc2imo.conjugate().T @ frag.loc2amo)
        return e_tot, grads  # delete self.ints. if you take away the self.lasci_() above

    def refrag_lowdin_active(self, loc2wmas, oneRDM_loc):

        ovlp_wmas = np.dot(loc2wmas.conjugate().T, loc2wmas)
        oneRDM_amo = represent_operator_in_basis(oneRDM_loc, loc2wmas)
        print("Trace of oneRDM_amo = {}".format(np.trace(oneRDM_amo)))

        # print ("Active orbital overlap matrix:\n{}".format (prettyprint (ovlp_wmas)))

        orth_idx = np.isclose(np.sum(ovlp_wmas - np.eye(ovlp_wmas.shape[0]), axis=1), 0)
        print("{} active orbitals with overlap and {} active orbitals with no overlap".format(np.sum(~orth_idx),
                                                                                              np.sum(orth_idx)))

        evecs = orth.lowdin(ovlp_wmas)
        # print ("Orthonormal active orbitals eigenvectors:\n{}".format (prettyprint (evecs)))
        loc2wmas = np.dot(loc2wmas, evecs)
        wmas_labels = np.asarray(['A' for ix in range(loc2wmas.shape[1])])
        if self.ints.mol.symmetry:
            wmas_labels = assign_blocks_weakly(loc2wmas, self.ints.loc2symm)
            wmas_labels = np.asarray(self.ints.mol.irrep_name)[wmas_labels]

        for f in self.fragments:
            if f.loc2amo.shape[1] == 0:
                continue
            proj = represent_operator_in_basis(np.dot(f.loc2amo, f.loc2amo.conjugate().T), loc2wmas)
            # print ("Projector matrix of {} active orbitals in orthonormal active basis:\n{}".format (
            #    f.frag_name, prettyprint (proj, fmt='{:5.2f}')))
        # frag_weights = np.stack ([np.einsum ('ip,ij,jp->p', loc2wmas.conjugate (), np.dot (f.loc2amo, f.loc2amo.conjugate ().T), loc2wmas) for f in self.fragments], axis=-1)
        frag_weights = [
            f.loc2amo @ f.loc2amo.conjugate().T if f.norbs_as > 0 else np.zeros((self.norbs_tot, self.norbs_tot)) for f
            in self.fragments]
        frag_weights = np.stack([((p @ loc2wmas) * loc2wmas.conjugate()).sum(0) for p in frag_weights], axis=-1)
        # print ("Fragment weights of orthonormal amos:\n{}".format (prettyprint (frag_weights, fmt='{:5.2f}')))
        idx = np.argsort(frag_weights, axis=1)[:, -1]

        loc2amo = [loc2wmas[:, (idx == i)] for i in range(len(self.fragments))]
        amo_labels = [wmas_labels[idx == i] for i in range(len(self.fragments))]
        nelec_amo = [((oneRDM_loc @ l) * l.conjugate()).sum() for l in loc2amo]
        if self.ints.mol.symmetry:
            print("Approximate irreps of active orbitals in each fragment:")
            for f, l, l2a in zip(self.fragments, amo_labels, loc2amo):
                labeldict = dict(zip(*np.unique(l, return_counts=True)))
                err = measure_subspace_blockbreaking(l2a, self.ints.loc2symm)
                print("{}: {}, err = {}".format(f.frag_name, labeldict, err))
        print("Number of electrons in each fragment:")
        for f, n in zip(self.fragments, nelec_amo):
            print("{}: {}".format(f.frag_name, n))

        print("Entanglement norms:")
        for (idx1, f1), (idx2, f2) in combinations(enumerate(self.fragments), 2):
            print("{}-{}: {}".format(f1.frag_name, f2.frag_name, np.linalg.norm(
                np.einsum('ip,ij,iq->pq', loc2amo[idx1], oneRDM_loc, loc2amo[idx2]))))

        dm = represent_operator_in_basis(oneRDM_loc, np.concatenate(loc2amo, axis=1))
        # print ("Density matrix in symmetrically-orthogonalized active-orbital basis:\n{}".format (prettyprint (dm)))
        for l in loc2amo:
            assert (is_basis_orthonormal(l))

        return loc2amo

    def refrag_lowdin_external(self, loc2wmcs, loc2wmas):

        proj_gfrag = np.stack(
            [np.dot(f.get_true_loc2frag(), f.get_true_loc2frag().conjugate().T) for f in self.fragments], axis=-1)
        loc2x = loc2wmcs.copy()
        loc2wmcs = [np.zeros((self.norbs_tot, 0), dtype=loc2x.dtype) for ix in range(len(self.fragments))]
        wmcs_labels = [[] for ix in range(len(self.fragments))]
        # Can't be too careful when diagonalizing 1-body projection operators (massive degeneracy -> numerical problems)
        # Best to label the symmetry of the unactive subspace vectors explicitly and exactly
        x_labels = None
        if self.enforce_symmetry:
            loc2x = align_states(loc2x, self.ints.loc2symm)
            x_labels = assign_blocks(loc2x, self.ints.loc2symm)

        # First pass: Only retain the highest Mfk - Mak eigenvalues, for which the eigenvalue is larger than 1/2.
        # Do dummy fragments last so that distributing orbitals to them doesn't fuck up symmetry
        Mxk_tot = np.asarray([f.norbs_frag - loc2wmas[ix].shape[1] for ix, f in enumerate(self.fragments)])
        for ix, f in enumerate(self.fragments):
            if f.imp_solver_name == 'dummy RHF': Mxk_tot[ix] = 0
        it = 0
        assign_thresh = 1 / 2
        Mxk_rem = Mxk_tot
        max_eval = np.ones(len(self.fragments))
        # Do NOT attempt to enforce symmetry here YET. It may be necessary in future, but for now hold off!
        while np.any(Mxk_rem):  # loc2x.shape[1] > 0:
            assert (it < 20)
            if np.all(max_eval < assign_thresh):
                print(("Warning: external orbital assignment threshold lowered from {} to {}"
                       " after failing to assign any orbitals in iteration {}").format(assign_thresh,
                                                                                       np.max(max_eval) * 0.99, it - 1))
                assign_thresh = np.max(max_eval) * 0.99
            Mxk = np.asarray([w.shape[1] for w in loc2wmcs])
            Mxk_rem = Mxk_tot - Mxk
            ix_rem = Mxk_rem > 0
            max_eval = np.zeros(len(self.fragments))
            proj_gfrag_sum = proj_gfrag[:, :, ix_rem].sum(2)
            for ix_frag, f in enumerate(self.fragments):
                if loc2x.shape[1] == 0 or Mxk_rem[ix_frag] == 0:
                    continue
                print("it {}\nMxk = {}\nMxk_rem = {}\nix_rem = {}\nloc2x.shape = {}".format(it, Mxk, Mxk_rem, ix_rem,
                                                                                            loc2x.shape))
                evals, loc2evecs, labels = matrix_eigen_control_options(proj_gfrag[:, :, ix_frag], subspace=loc2x,
                                                                        symmetry=self.ints.loc2symm,
                                                                        strong_symm=self.enforce_symmetry,
                                                                        subspace_symmetry=x_labels, sort_vecs=-1,
                                                                        only_nonzero_vals=False)
                ix_zero = np.abs(evals) < 1e-8
                # Scale the eigenvalues to evaluate their comparison to the sum of projector expt vals of unfull fragments
                # exptvals = np.einsum ('ip,ijk,k,jp->p', loc2evecs.conjugate (), proj_gfrag, ix_rem.astype (int), loc2evecs)
                exptvals = ((proj_gfrag_sum @ loc2evecs) * loc2evecs).sum(0)
                evals = evals / exptvals
                evals[ix_zero] = 0
                max_eval[ix_frag] = np.max(evals)
                ix_loc = evals >= assign_thresh
                ix_loc[Mxk_rem[ix_frag]:] = False
                print("{} fragment, iteration {}: {} eigenvalues above 1/2 found of {} total sought:\n{}".format(
                    f.frag_name, it, np.count_nonzero(ix_loc), Mxk_rem[ix_frag], evals))
                loc2x = loc2evecs[:, ~ix_loc]
                x_labels = labels[~ix_loc]
                loc2wmcs[ix_frag] = np.append(loc2wmcs[ix_frag], loc2evecs[:, ix_loc], axis=1)
                wmcs_labels[ix_frag] = np.append(wmcs_labels[ix_frag], labels[ix_loc]).astype(int)
            it += 1

        if self.ints.mol.symmetry:
            for loc2frag, frag_labels, frag in zip(loc2wmcs, wmcs_labels, self.fragments):
                if frag.imp_solver_name == 'dummy RHF':
                    continue
                err = measure_subspace_blockbreaking(loc2frag, self.ints.loc2symm)
                labeldict = {self.ints.mol.irrep_name[x]: np.count_nonzero(frag_labels == x) for x in
                             np.unique(frag_labels)}
                print("New unactive {} fragment orbital irreps = {}, err = {}".format(frag.frag_name, labeldict, err))

        # Just assign (potentially symmetry-breaking!) orbitals to dummy fragments randomly because it cannot possibly matter
        for ix, f in enumerate(self.fragments):
            if f.imp_solver_name != 'dummy RHF':
                continue
            evals, loc2x = matrix_eigen_control_options(proj_gfrag[:, :, ix], subspace=loc2x, sort_vecs=-1,
                                                        only_nonzero_vals=False)
            loc2wmcs[ix] = loc2x[:, :f.norbs_frag]
            loc2x = loc2x[:, f.norbs_frag:]

        for ix, loc2frag in enumerate(loc2wmcs):
            ovlp = loc2frag.conjugate().T
            ovlp = np.dot(ovlp, loc2frag)
            err = linalg.norm(ovlp - np.eye(ovlp.shape[0]))
            it = 0
            while (abs(err) > 1e-8):
                umat = orth.lowdin(ovlp)
                loc2frag = np.dot(loc2frag, umat)
                ovlp = loc2frag.conjugate().T
                ovlp = np.dot(ovlp, loc2frag)
                err = linalg.norm(ovlp - np.eye(ovlp.shape[0]))
                it += 1
                if it > 100:
                    raise RuntimeError("I tried 100 times to orthonormalize this and failed")
            loc2wmcs[ix] = loc2frag

        # for loc2, f in zip (loc2wmcs, self.fragments):
        #    print ("Projector eigenvalues of {} external fragment orbitals:".format (f.frag_name))
        #    frag_weights = np.einsum ('ip,ijq,jp->pq', loc2.conjugate (), proj_gfrag, loc2)
        #    print (prettyprint (frag_weights))

        for l in loc2wmcs:
            assert (is_basis_orthonormal(l))
        return loc2wmcs

    def generate_frag_cas_guess(self, mo_coeff=None, caslst=None, cas_irrep_nocc=None, cas_irrep_ncore=None,
                                replace=False, force_imp=False, confine_guess=True, guess_somos=[]):
        ''' Generate initial active-orbital guesses for fragments from the whole-molecule MOs. Either add active orbitals
        or replace them. If you are getting errors about integer electron numbers and you are performing a calculation from an ROHF initial guess, try setting confine_guess=False'''

        if replace:
            print("Deleting existing amos...")
            for f in self.fragments:
                f.loc2amo = np.zeros((self.ints.norbs_tot, 0), dtype=self.ints.ao2loc.dtype)
                f.oneRDMas_loc = np.zeros_like(self.ints.oneRDM_loc)
                f.twoCDMimp_amo = np.zeros((0, 0, 0, 0), dtype=self.ints.ao2loc.dtype)
                f.ci_as = None
        loc2wmas_current = np.concatenate([f.loc2amo for f in self.fragments if f.active_space is not None], axis=1)
        if mo_coeff is None:
            if self.doLASSCF:
                mo_coeff = self.ints.get_trial_nos(ao_basis=True, loc2wmas=loc2wmas_current)[0]
            else:
                mo_coeff = self.ints.get_trial_nos(ao_basis=True, loc2wmas=None)[0]
        if force_imp and confine_guess: raise RuntimeError(
            "force_imp==True and confine_guess==True are incompatible and will lead to broken integers")

        nelec_wmas_target = sum([f.active_space[0] for f in self.fragments if f.active_space is not None])
        norbs_wmas_target = sum([f.active_space[1] for f in self.fragments if f.active_space is not None])
        assert ((
                        self.ints.nelec_tot - nelec_wmas_target) % 2 == 0), 'parity problem: {} of {} electrons active in target'.format(
            nelec_wmas_target, self.ints.nelec_tot)
        ncore_target = (self.ints.nelec_tot - nelec_wmas_target) // 2
        nocc_target = ncore_target + norbs_wmas_target
        print("ncore_target = {}; nocc_target = {}".format(ncore_target, nocc_target))

        nelec_wmas_current = sum([f.nelec_as for f in self.fragments if f.norbs_as > 0])
        norbs_wmas_current = sum([f.norbs_as for f in self.fragments if f.norbs_as > 0])
        if norbs_wmas_current > 0:
            assert ((
                            self.ints.nelec_tot - nelec_wmas_current) % 2 == 0), 'parity problem: {} of {} electrons active in currently'.format(
                nelec_wmas_current, self.ints.nelec_tot)
            ncore_current = (self.ints.nelec_tot - nelec_wmas_current) // 2
            nocc_current = ncore_current + norbs_wmas_current
            print("ncore_current = {}; nocc_current = {}".format(ncore_current, nocc_current))
            wmas2ao_current = (self.ints.ao2loc @ loc2wmas_current).conjugate().T
            amo_coeff = mo_coeff[:, ncore_current:nocc_current]
            ovlp = wmas2ao_current @ self.ints.ao_ovlp @ amo_coeff
            ovlp = np.trace(ovlp.conjugate().T @ ovlp) / norbs_wmas_current
            assert (np.isclose(ovlp,
                               1)), 'unless replace=True, mo_coeff must contain current active orbitals at range {}:{} (err={})'.format(
                ncore_current, nocc_current, ovlp - 1)
            if caslst is not None and len(caslst) > 0:
                assert (np.all(np.isin(list(range(ncore_current + 1, nocc_current + 1)),
                                       caslst))), 'caslst must contain range {}:{} inclusive ({})'.format(
                    ncore_current + 1, nocc_current, caslst)
        else:
            ncore_current = nocc_current = (self.ints.nelec_tot + 1) // 2

        if caslst is not None and len(caslst) == 0: caslst = None
        if caslst is not None and cas_irrep_nocc is not None:
            print("Warning: caslst overrides cas_irrep_nocc!")
        if caslst is not None:
            mf = scf.RHF(self.ints.mol)
            mf.mo_coeff = mo_coeff
            cas = mcscf.CASCI(mf, norbs_wmas_target, nelec_wmas_target)
            mo_coeff = cas.sort_mo(caslst)
        elif cas_irrep_nocc is not None:
            mf = scf.RHF(self.ints.mol)
            mf.mo_coeff = mo_coeff
            cas = mcscf.CASCI(mf, norbs_wmas_target, nelec_wmas_target)
            mo_coeff = cas.sort_mo_by_irrep(cas_irrep_nocc, cas_irrep_ncore=cas_irrep_ncore)

        amo_new_coeff = np.append(mo_coeff[:, ncore_target:ncore_current], mo_coeff[:, nocc_current:nocc_target],
                                  axis=1)

        loc2amo_new = orthonormalize_a_basis(linalg.solve(self.ints.ao2loc, amo_new_coeff))
        occ_new, loc2amo_new = matrix_eigen_control_options(self.ints.oneRDM_loc, subspace=loc2amo_new, sort_vecs=-1,
                                                            symmetry=self.ints.loc2symm,
                                                            strong_symm=self.enforce_symmetry)[:2]
        assert (np.all(reduce(np.logical_or, (occ_new < params.num_zero_rtol, np.isclose(occ_new, 1),
                                              np.isclose(occ_new, 2))))), 'New amos not integer-occupied: {}'.format(
            occ_new)
        occ_new = np.round(occ_new).astype(int)

        for f in self.fragments:
            if f.active_space is not None:
                my_new_orbs = f.active_space[1] - f.norbs_as
                if confine_guess:
                    proj_amo = np.dot(loc2amo_new, loc2amo_new.conjugate().T)
                    loc2amo_guess = \
                        matrix_eigen_control_options(proj_amo, subspace=f.get_true_loc2frag(),
                                                     symmetry=self.ints.loc2symm,
                                                     strong_symm=self.enforce_symmetry, sort_vecs=-1)[1]
                    loc2amo_guess = loc2amo_guess[:, :my_new_orbs]
                else:
                    proj_frag = np.dot(f.get_true_loc2frag(), f.get_true_loc2frag().conjugate().T)
                    wgts = np.zeros(len(occ_new))
                    if self.doLASSCF and force_imp:
                        # Prevent mixing singly-, doubly- and un-occupied guess orbitals with each other so guess has good integers
                        loc2amo_guess = loc2amo_new.copy()
                        for myocc in range(3):
                            idx = occ_new == myocc
                            evals, loc2amo_guess[:, idx] = matrix_eigen_control_options(proj_frag,
                                                                                        subspace=loc2amo_new[:, idx],
                                                                                        symmetry=self.ints.loc2symm,
                                                                                        strong_symm=self.enforce_symmetry,
                                                                                        sort_vecs=-1)[:2]
                            wgts[idx] = evals
                    else:
                        # Doesn't matter if not LASSCF; bias instead towards well-localized guess
                        wgts, loc2amo_guess = matrix_eigen_control_options(proj_frag, subspace=loc2amo_new,
                                                                           symmetry=self.ints.loc2symm,
                                                                           strong_symm=self.enforce_symmetry,
                                                                           sort_vecs=-1)[:2]
                    idx_sort = np.argsort(wgts)[::-1]
                    loc2amo_guess = loc2amo_guess[:, idx_sort]
                    loc2amo_other = loc2amo_guess[:, my_new_orbs:]
                    occ_other = occ_new[idx_sort][my_new_orbs:]
                    loc2amo_guess = loc2amo_guess[:, :my_new_orbs]
                # Set up somo twoCDM for accurate initial guess energy
                evals, loc2amo_guess = matrix_eigen_control_options(self.ints.oneRDM_loc, subspace=loc2amo_guess,
                                                                    symmetry=self.ints.loc2symm,
                                                                    strong_symm=self.enforce_symmetry)[:2]
                idx_somo = (evals > 0.8) & (evals < 1.2)
                nsomo = np.count_nonzero(idx_somo)
                print("Found {} singly-occupied orbitals; assuming they are occupied by alpha-electrons.".format(nsomo))
                if nsomo > 0:
                    loc2somo = loc2amo_guess[:, idx_somo]
                    dma = np.eye(nsomo)
                    dmb = np.zeros((nsomo, nsomo), dtype=dma.dtype)
                    twoCDM_somo = get_2CDM_from_2RDM(
                        get_2RDM_from_2CDM(np.zeros([nsomo, ] * 4, dtype=dma.dtype), [dma, dmb]), dma + dmb)
                loc2amo_guess = np.append(f.loc2amo, loc2amo_guess, axis=1)
                f.loc2amo_guess = \
                    matrix_eigen_control_options(self.ints.activeFOCK, subspace=loc2amo_guess,
                                                 symmetry=self.ints.loc2symm,
                                                 sort_vecs=1, strong_symm=self.enforce_symmetry)[1]
                if force_imp:
                    print("force_imp!")
                    old2new_amo = f.loc2amo.conjugate().T @ f.loc2amo_guess
                    f.loc2amo = f.loc2amo_guess.copy()
                    f.oneRDMas_loc = project_operator_into_subspace(self.ints.oneRDM_loc, f.loc2amo)
                    f.twoCDMimp_amo = represent_operator_in_basis(f.twoCDMimp_amo, old2new_amo)
                    # Assume all singly-occupied orbitals are alpha-spin
                    if nsomo > 0:
                        f.oneSDMas_loc = project_operator_into_subspace(f.oneRDMas_loc, loc2somo)
                        somo2amo = loc2somo.conjugate().T @ f.loc2amo
                        f.twoCDMimp_amo += represent_operator_in_basis(twoCDM_somo, somo2amo)
                    f.ci_as = None
                    # Take the just-added active orbitals out of consideration for the next fragment
                    loc2amo_new = loc2amo_other
                    occ_new = occ_other

        loc2amo = np.concatenate([f.loc2amo_guess for f in self.fragments if f.loc2amo_guess is not None], axis=1)
        if len(guess_somos) == len(self.fragments):
            assert (not force_imp), "Don't force_imp and guess_somos at the same time"
            loc2wmcs = get_complementary_states(loc2amo, symmetry=self.ints.loc2symm,
                                                enforce_symmetry=self.enforce_symmetry)
            ene_wmcs, loc2wmcs = matrix_eigen_control_options(self.ints.activeFOCK, subspace=loc2wmcs,
                                                              symmetry=self.ints.loc2symm,
                                                              strong_symm=self.enforce_symmetry, sort_vecs=1,
                                                              only_nonzero_vals=False)[:2]
            ncore = (self.ints.nelec_tot - sum(
                [f.active_space[0] for f in self.fragments if f.active_space is not None])) // 2
            oneRDMcore_loc = 2 * loc2wmcs[:, :ncore] @ loc2wmcs[:, :ncore].conjugate().T
            self.ints.oneRDM_loc = oneRDMcore_loc
            self.ints.oneSDM_loc = np.zeros_like(oneRDMcore_loc)
            self.ints.nelec_idem = ncore * 2
            self.ints.loc2idem = loc2wmcs
            # construct rohf-like density matrices and set loc2amo_guess -> loc2amo
            for nsomo, f, in zip(guess_somos, self.fragments):
                if f.active_space is None:
                    continue
                assert (nsomo % 2 == f.active_space[0] % 2)
                assert (nsomo <= f.active_space[1])
                neleca = (f.active_space[0] + nsomo) // 2
                nelecb = (f.active_space[0] - nsomo) // 2
                if f.target_MS < 0: neleca, nelecb = nelecb, neleca
                occa = np.zeros(f.active_space[1], dtype=np.float64)
                occb = np.zeros(f.active_space[1], dtype=np.float64)
                occa[:neleca] += 1
                occb[:nelecb] += 1
                dma, dmb = np.diag(occa), np.diag(occb)
                # This is to cajole my ROHF-like pseudo-cumulant decomposition
                twoCDM = np.zeros([f.active_space[1] for i in range(4)], dtype=np.float64)
                twoRDM = get_2RDM_from_2CDM(twoCDM, [dma, dmb])
                dm = dma + dmb
                f.loc2amo = f.loc2amo_guess.copy()
                f.twoCDMimp_amo = get_2CDM_from_2RDM(twoRDM, dm)
                f.oneRDMas_loc = represent_operator_in_basis(dma + dmb, f.loc2amo.conjugate().T)
                f.oneSDMas_loc = represent_operator_in_basis(dma - dmb, f.loc2amo.conjugate().T)
                f.ci_as = None
                self.ints.oneRDM_loc += f.oneRDMas_loc
                self.ints.oneSDM_loc += f.oneSDMas_loc
            for f in self.fragments:
                f.oneRDM_loc = self.ints.oneRDM_loc
                f.oneSDM_loc = self.ints.oneSDM_loc

        # Linear dependency check
        evals = matrix_eigen_control_options(1, subspace=loc2amo, symmetry=self.ints.loc2symm, only_nonzero_vals=False,
                                             strong_symm=self.enforce_symmetry)[0]
        lindeps = np.count_nonzero(evals < 1e-6)
        errstr = "{} linear dependencies found among {} active orbitals in guess construction".format(lindeps,
                                                                                                      loc2amo.shape[-1])
        if lindeps:
            if self.force_imp or len(guess_somos) == len(self.fragments):
                RuntimeError(errstr)
            else:
                warnings.warn(errstr, RuntimeWarning)

        return

    def save_checkpoint(self, fname):
        ''' Data array structure: nao_nr, chempot, 1RDM or umat, norbs_amo in frag 1, loc2amo of frag 1, oneRDM_amo of frag 1, twoCDMimp_amo of frag 1, norbs_amo of frag 2, ... '''
        nao = self.ints.mol.nao_nr()
        if self.doLASSCF:
            chkdata = self.helper.construct1RDM_loc(self.doSCF, self.umat)
            chkdata = represent_operator_in_basis(chkdata, self.ints.ao2loc.conjugate().T).flatten(order='C')
        else:
            chkdata = represent_operator_in_basis(self.umat, self.ints.ao2loc.conjugate().T).flatten(order='C')
        chkdata = np.append(np.asarray([nao, self.chempot]), chkdata)
        for f in self.fragments:
            chkdata = np.append(chkdata, [f.norbs_as])
            chkdata = np.append(chkdata, np.dot(self.ints.ao2loc, f.loc2amo).flatten(order='C'))
            chkdata = np.append(chkdata, represent_operator_in_basis(f.oneRDM_loc, f.loc2amo).flatten(order='C'))
            chkdata = np.append(chkdata, f.twoCDMimp_amo.flatten(order='C'))
        np.save(fname, chkdata)
        return

    def load_checkpoint(self, fname, prev_mol=None):
        nelec_amo = sum((f.active_space[0] for f in self.fragments if f.active_space is not None))
        norbs_amo = sum((f.active_space[1] for f in self.fragments if f.active_space is not None))
        norbs_cmo = (self.ints.mol.nelectron - nelec_amo) // 2
        norbs_omo = norbs_cmo + norbs_amo
        chkdata = np.load(fname)
        nao, self.chempot, chkdata = int(round(chkdata[0])), chkdata[1], chkdata[2:]
        print(
            "{} atomic orbital basis functions reported in checkpoint file, as opposed to {} in integral object".format(
                nao, self.ints.mol.nao_nr()))
        assert (prev_mol is not None or nao == self.ints.mol.nao_nr())
        locSao = np.dot(self.ints.ao_ovlp, self.ints.ao2loc).conjugate().T
        if prev_mol:
            oldSnew = mole.intor_cross('int1e_ovlp', prev_mol, self.ints.mol)
            aoSloc = np.dot(oldSnew, self.ints.ao2loc)
            if same_mol(prev_mol, self.ints.mol, cmp_basis=False):  # basis set expansion
                locSao = np.dot(self.ints.ao_ovlp, self.ints.ao2loc).conjugate().T
            else:  # geometry change
                locSao = aoSloc.conjugate().T
        else:
            aoSloc = np.dot(self.ints.ao_ovlp, self.ints.ao2loc)
            locSao = aoSloc.conjugate().T

        mat, chkdata = chkdata[:nao ** 2].reshape(nao, nao, order='C'), chkdata[nao ** 2:]
        mat = represent_operator_in_basis(mat, aoSloc)
        if self.doLASSCF:
            self.ints.oneRDM_loc = mat.copy()
            if prev_mol and not same_mol(prev_mol, self.ints.mol, cmp_basis=False):
                assert (abs(np.trace(
                    self.ints.oneRDM_loc) - self.ints.nelec_tot) < 1e-8), "checkpoint oneRDM trace = {}; nelec_tot = {}".format(
                    np.trace(self.ints.oneRDM_loc), self.ints.nelec_tot)
        else:
            self.umat = mat.copy()

        for f in self.fragments:
            if self.doLASSCF: f.oneRDM_loc = self.ints.oneRDM_loc
            namo, chkdata = int(round(chkdata[0])), chkdata[1:]
            print("{} active orbitals reported in checkpoint file for fragment {}".format(namo, f.frag_name))
            if namo > 0:
                f.loc2amo, chkdata = chkdata[:nao * namo].reshape(nao, namo, order='C'), chkdata[nao * namo:]
                f.oneRDMas_loc, chkdata = chkdata[:namo ** 2].reshape(namo, namo, order='C'), chkdata[namo ** 2:]
                print("{} fragment oneRDM_amo (trace = {}):\n{}".format(
                    f.frag_name, np.trace(f.oneRDMas_loc), prettyprint(f.oneRDMas_loc, fmt='{:6.3f}')))
                f.twoCDMimp_amo, chkdata = chkdata[:namo ** 4].reshape(namo, namo, namo, namo, order='C'), chkdata[
                                                                                                           namo ** 4:]
                if prev_mol and same_mol(prev_mol, self.ints.mol, cmp_basis=False): f.loc2amo = project_mo_nr2nr(
                    prev_mol, f.loc2amo, self.ints.mol)
                f.loc2amo = np.dot(locSao, f.loc2amo)
                # Normalize
                amo_norm = (f.loc2amo * f.loc2amo).sum(0)
                f.loc2amo /= np.sqrt(amo_norm)[None, :]
                # orthogonalize and natorbify
                ovlp = np.dot(f.loc2amo.conjugate().T, f.loc2amo)
                print("{} fragment amo overlap matrix (trace = {}):\n{}".format(
                    f.frag_name, np.trace(ovlp), prettyprint(ovlp, fmt='{:6.3f}')))
                no_occ, no_evecs = matrix_eigen_control_options(f.oneRDMas_loc, b_matrix=ovlp, sort_vecs=-1)
                f.loc2amo = f.loc2amo @ no_evecs
                f.oneRDMas_loc = (no_occ[None, :] * f.loc2amo) @ f.loc2amo.conjugate().T
                f.twoCDMimp_amo = represent_operator_in_basis(f.twoCDMimp_amo, no_evecs)
                print("{} fragment oneRDM_amo (trace = {}):\n{}".format(
                    f.frag_name, np.trace(f.oneRDMas_loc),
                    prettyprint(represent_operator_in_basis(f.oneRDMas_loc, f.loc2amo), fmt='{:6.3f}')))

                if np.amax(np.abs(f.twoCDMimp_amo)) > 1e-10:
                    tei = self.ints.dmet_tei(f.loc2amo)
                    f.E2_cum = np.tensordot(tei, f.twoCDMimp_amo, axes=4) / 2
        assert (chkdata.shape == tuple((0,))), chkdata.shape

        if self.doLASSCF: self.ints.setup_wm_core_scf(self.fragments, self.calcname)

    def expand_active_space(self, callables):
        ''' Add occupied or vitual orbitals from the whole molecule to one or several fragments' active spaces using functions in list "callables"
            Callables should have signature (self.ints.mol, mo, myamo, ncore, ncas) where mo is mo coefficients and myamo is the active orbital
            coefficients of the current fragment. They should return the expanded set of active orbital coefficients.
            All orbital coefficients interacting with the callables are in the ao basis (not the orthonormal localized DMET/LASSCF basis)

            Keeping the additional orbitals localized is your responsibility '''

        loc2wmas = np.concatenate([frag.loc2amo for frag in self.fragments], axis=1)
        mo = self.ints.get_trial_nos(aobasis=True, loc2wmas=loc2wmas)[0]

        for frag, cal in zip(self.fragments, callables):
            if frag.active_space is None:
                continue
            myamo = cal(self.ints.mol, mo, np.dot(self.ints.ao2loc, frag.loc2amo), norbs_cmo, norbs_amo)
            old2new_amo = frag.loc2amo.conjugate().T.copy()
            frag.loc2amo = reduce(np.dot, (self.ints.ao2loc.conjugate().T, self.ints.ao_ovlp, myamo))
            old2new_amo = np.dot(old2new_amo, frag.loc2amo)
            frag.oneRDMas_loc = project_operator_into_subspace(self.ints.oneRDM_loc, frag.loc2amo)
            frag.twoCDMimp_amo = represent_operator_in_basis(frag.twoCDMimp_amo, old2new_amo)

    def get_las_nos(self, **kwargs):
        ''' Save MOs in a form that can be loaded for a CAS calculation on a npy file '''
        kwargs['aobasis'] = True
        if not 'loc2wmas' in kwargs or kwargs['loc2wmas'] is None:
            kwargs['loc2wmas'] = [frag.loc2amo for frag in self.fragments]
        return self.ints.get_trial_nos(**kwargs)

    def void_symmetry(self):
        print("Voiding symmetry")
        for f in self.fragments:
            f.groupname = 'C1'
            f.loc2symm = [np.eye(self.norbs_tot)]
            f.ir_names = ['A']
            f.ir_ids = [0]
            f.wfnsym = 'A'
        self.ints.loc2symm = [np.eye(self.norbs_tot)]
        self.ints.symmetry = False
        self.ints.wfnsym = 'A'
        self.ints.ir_names = ['A']
        self.ints.ir_ids = [0]

    def check_fragment_symmetry_breaking(self, verbose=True, do_break=False):
        if not self.ints.mol.symmetry:
            return False
        symmetry = self.ints.mol.groupname

        def symm_analysis(mo, mo_name, symmetry):
            if verbose: print("Analyzing {} fragment orbitals:".format(mo_name))
            irrep_norbs = []
            for irrep, loc2ir in zip(self.ints.mol.irrep_name, self.ints.loc2symm):
                ovlp = mo.conjugate().T @ loc2ir
                irrep_norbs.append(np.trace(ovlp @ ovlp.conjugate().T))
                if verbose: print("{}: {}".format(irrep, irrep_norbs[-1]))
            if verbose: print("Total (sanity check!): {}".format(sum(irrep_norbs)))
            adapted = np.allclose(irrep_norbs, np.around(irrep_norbs))
            if not adapted: symmetry = False
            return symmetry, irrep_norbs

        try:
            loc2dummy = np.concatenate([f.loc2frag for f in self.fragments if f.imp_solver_name == 'dummy RHF'], axis=1)
            symmetry = symm_analysis(loc2dummy, 'dummy', symmetry)[0]
        except ValueError as e:
            pass
            # This isn't important.
        irrep_norbs = np.zeros((len(self.fragments), len(self.ints.mol.irrep_name)), dtype=np.int32)
        for idx, f in enumerate(self.fragments):
            if f.imp_solver_name == 'dummy RHF':
                continue
            symmetry, irrep_norbs[idx, :] = symm_analysis(f.loc2frag, f.frag_name, symmetry)

        if (not symmetry) and do_break and np.any(f.symmetry for f in self.fragments):
            if self.enforce_symmetry: raise RuntimeError("Fragmentation pattern of molecule breaks symmetry!")
            print("Fragmentation pattern of molecule broke point-group symmetry!")
            self.void_symmetry()
        elif symmetry and do_break:
            for f in self.fragments:
                f.enforce_symmetry = False if f.imp_solver_name == 'dummy RHF' else self.enforce_symmetry
                f.groupname = self.ints.mol.groupname
                f.loc2symm = self.ints.loc2symm
                f.ir_names = self.ints.ir_names
                f.ir_ids = self.ints.ir_ids
                if f.wfnsym is None: f.wfnsym = self.ints.wfnsym
        return symmetry

    def examine_symmetry(self, verbose=True):
        if not self.ints.mol.symmetry:
            return False

        symmetry = self.check_fragment_symmetry_breaking(verbose=verbose)

        nelec_symm = [np.trace(represent_operator_in_basis(self.ints.oneRDM_loc, loc2ir)) for loc2ir in
                      self.ints.loc2symm]
        if verbose:
            print("Number of electrons in each irrep of trial wvfn:")
            for ir, n in zip(self.ints.mol.irrep_name, nelec_symm):
                print("{}: {}".format(ir, n))
        # I think these are actually allowed to be non-integers. It's the coupling elements that have to be zero because an element
        # of the density operator that couples orbitals of two irreps necessarily couples CI vector elements of two irreps
        # (something something Abelian groups)
        for (ir1, loc2ir1), (ir2, loc2ir2) in combinations(zip(self.ints.mol.irrep_name, self.ints.loc2symm), 2):
            off = np.amax(np.abs(represent_operator_in_basis(self.ints.oneRDM_loc, loc2ir1, loc2ir2)))
            if verbose: print("1RDM maximum element coupling {} to {}: {}".format(ir1, ir2, off))
            if off > 1e-6: symmetry = False

        # In principle I should also check the twoCDM, but it ends up not mattering because the twoCDM only contributes
        # a constant to the energy!

        if symmetry and verbose:
            print("This system retains {} symmetry!".format(symmetry))
        elif verbose:
            print("This system loses its symmetry")
        return symmetry
