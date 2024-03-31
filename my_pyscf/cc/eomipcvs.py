#!/usr/bin/env python
# Copyright 2014-2023 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Bhavnesh Jangid <jangidbhavnesh@gmail.com>
#        !!? Matthew R. Hermes <MatthewRobertHermes@gmail.com>
#

"""
IP-CVS-EOM-CCSD, IP-CVS-EOM-CCSD(*), IP-CVS-EOM-CCSD(T)(*)
"""

# Done: IP-CVS-EOM-CCSD
# TODO: IP-CVS-EOM-CCSD(*), IP-CVS-EOM-CCSD(T)(*)
# TODO: This is one of the CVS implementation



import numpy as np
from pyscf import lib, scf, cc
from pyscf.lib import logger
from pyscf import __config__
from pyscf.cc.eom_rccsd import EOMIP

# These functions are there to make it consistent
# with the EOM-CCSD

def vector_to_amplitudes_ip(vector, nmo, nocc):
    nvir = nmo - nocc
    r1 = vector[:nocc].copy()
    r2 = vector[nocc:].copy().reshape(nocc, nocc, nvir)
    return r1, r2


def amplitudes_to_vector_ip(r1, r2):
    vector = np.hstack((r1, r2.ravel()))
    return vector


class _EOMIPCVS(EOMIP):
    """
    IP-CVS-EOM energy using restricted CCSD
    """

    def get_init_guess(self, nroots=1, koopmans=True, diag=None):
        ''' Initial guess vectors for the IP-CVS-EOM kernel
        Args:
            nroots: int
                Maximum roots. Default is 1
            koopmans : bool
                Calculate Koopmans'-like (quasi-particle) excitations only, targeting via
                overlap.
            diag:
                List containing the R1 and R2
        Returns:
            guess : list of ndarray
                List of guess vectors to use for targeting via overlap.

        '''
        size = self.vector_size()
        assert nroots <= size, "More than maximum root available"
        guess = []
        if koopmans:

            dtype = getattr(diag, 'dtype', np.double)
            for n in range(nroots):
                corehole = self.corehole
                g = np.zeros(int(size), dtype)
                g[n+corehole] = 1.0
                guess.append(g)
        else:
            if diag is None:
                diag = self.get_diag()
            dtype = getattr(diag, 'dtype', np.double)
            idx = np.argsort(diag)[::-1]
            guess = np.zeros((diag.shape[0], nroots))
            min_shape = min(diag.shape[0], nroots)
            guess[:min_shape, :min_shape] = np.identity(min_shape)
            g = np.zeros((diag.shape[0], nroots), dtype)
            g[idx] = guess.copy()
            guess = []
            for p in range(g.shape[1]):
                guess.append(g[:, p])
        return guess

    def dump_flags(self, verbose=None):
        logger.info(self, '')
        logger.info(self, '******** %s ********', self.__class__)
        logger.info(self, 'max_space = %d', self.max_space)
        logger.info(self, 'max_cycle = %d', self.max_cycle)
        logger.info(self, 'conv_tol = %s', self.conv_tol)
        logger.info(self, 'partition = %s', self.partition)
        logger.info(self, 'nocc = %d', self.nocc)
        logger.info(self, 'nmo = %d', self.nmo)
        logger.info(self, 'ncvs = %d', self.ncvs)
        logger.info(self, 'cvs-space = %d - %d', self.CVSMIN, self.CVSMAX)
        logger.info(self, 'max_memory %d MB (current use %d MB)',
                    self.max_memory, lib.current_memory()[0])
        return self

    def lipccsd_matvec(self, vector, imds=None, diag=None):
        '''For left eigenvector'''
        # Note this is not the same left EA equations used by Nooijen and Bartlett.
        # Small changes were made so that the same type L2 basis was used for both the
        # left EA and left IP equations.  You will note more similarity for these
        # equations to the left IP equations than for the left EA equations by Nooijen.
        if imds is None: imds = self.make_imds()
        nocc = self.nocc
        nmo = self.nmo
        r1, r2 = self.vector_to_amplitudes(vector, nmo, nocc)

        # 1h-1h block
        Hr1 = -1. * lib.einsum('ki,i->k', imds.Loo, r1)
        # 1h-2h1p block
        Hr1 -= lib.einsum('kbij,ijb->k', imds.Wovoo, r2)

        # 2h1p-1h block
        Hr2 = -1. * lib.einsum('kd,l->kld', imds.Fov, r1)
        Hr2 += 2. * lib.einsum('ld,k->kld', imds.Fov, r1)
        Hr2 -= lib.einsum('klid,i->kld', 2. * imds.Wooov - imds.Wooov.transpose(1, 0, 2, 3), r1)
        # 2h1p-2h1p block
        if self.partition == 'mp':
            fock = imds.eris.fock
            foo = fock[:nocc, :nocc]
            fvv = fock[nocc:, nocc:]
            Hr2 += lib.einsum('bd,klb->kld', fvv, r2)
            Hr2 -= lib.einsum('ki,ild->kld', foo, r2)
            Hr2 -= lib.einsum('lj,kjd->kld', foo, r2)
        elif self.partition == 'full':
            diag_matrix2 = self.vector_to_amplitudes(diag, nmo, nocc)[1]
            Hr2 += diag_matrix2 * r2
        else:
            Hr2 += lib.einsum('bd,klb->kld', imds.Lvv, r2)
            Hr2 -= lib.einsum('ki,ild->kld', imds.Loo, r2)
            Hr2 -= lib.einsum('lj,kjd->kld', imds.Loo, r2)
            Hr2 += lib.einsum('lbdj,kjb->kld', 2. * imds.Wovvo - imds.Wovov.transpose(0, 1, 3, 2), r2)
            Hr2 -= lib.einsum('kbdj,ljb->kld', imds.Wovvo, r2)
            Hr2 += lib.einsum('klij,ijd->kld', imds.Woooo, r2)
            Hr2 -= lib.einsum('kbid,ilb->kld', imds.Wovov, r2)
            tmp = lib.einsum('ijcb,ijb->c', imds.t2, r2)
            Hr2 -= lib.einsum('lkdc,c->kld', 2. * imds.Woovv - imds.Woovv.transpose(1, 0, 2, 3), tmp)

        # CVS Implementaiton
        ncvs = self.ncvs  # Consistent with python indexing
        CVSMIN = self.CVSMIN
        CVSMAX = self.CVSMAX

        # Hr1_i = 0, if i!=coreorbtial
        indices = np.arange(nocc)
        condition = ((indices < CVSMIN) | (indices > CVSMAX))
        Hr1 = np.where(condition, 0.0, Hr1)

        # Hr2_ija = 0, if i!=coreorbtial & j!=coreorbital
        indices = np.arange(nocc)
        condition = (indices > ncvs) & (indices < CVSMIN) | (indices > CVSMAX)
        Hr2[np.ix_(condition, condition, np.arange(Hr2.shape[2]))] = 0.0

        vector = self.amplitudes_to_vector(Hr1, Hr2)
        return vector

    def ipccsd_matvec(self, vector, imds=None, diag=None):
        # Ref: Nooijen and Snijders, J. Chem. Phys. 102, 1681 (1995) Eqs.(8)-(9)
        if imds is None: imds = self.make_imds()
        nocc = self.nocc
        nmo = self.nmo
        r1, r2 = self.vector_to_amplitudes(vector, nmo, nocc)

        # 1h-1h block
        Hr1 = -1. * lib.einsum('ki,k->i', imds.Loo, r1)
        # 1h-2h1p block
        Hr1 += 2 * lib.einsum('ld,ild->i', imds.Fov, r2)
        Hr1 -= lib.einsum('kd,kid->i', imds.Fov, r2)
        Hr1 -= 2 * lib.einsum('klid,kld->i', imds.Wooov, r2)
        Hr1 += lib.einsum('lkid,kld->i', imds.Wooov, r2)

        # 2h1p-1h block
        Hr2 = -1. * lib.einsum('kbij,k->ijb', imds.Wovoo, r1)
        # 2h1p-2h1p block
        if self.partition == 'mp':
            fock = imds.eris.fock
            foo = fock[:nocc, :nocc]
            fvv = fock[nocc:, nocc:]
            Hr2 += lib.einsum('bd,ijd->ijb', fvv, r2)
            Hr2 -= lib.einsum('ki,kjb->ijb', foo, r2)
            Hr2 -= lib.einsum('lj,ilb->ijb', foo, r2)
        elif self.partition == 'full':
            diag_matrix2 = self.vector_to_amplitudes(diag, nmo, nocc)[1]
            Hr2 += diag_matrix2 * r2
        else:
            Hr2 += lib.einsum('bd,ijd->ijb', imds.Lvv, r2)
            Hr2 -= lib.einsum('ki,kjb->ijb', imds.Loo, r2)
            Hr2 -= lib.einsum('lj,ilb->ijb', imds.Loo, r2)
            Hr2 += lib.einsum('klij,klb->ijb', imds.Woooo, r2)
            Hr2 += 2 * lib.einsum('lbdj,ild->ijb', imds.Wovvo, r2)
            Hr2 -= lib.einsum('kbdj,kid->ijb', imds.Wovvo, r2)
            Hr2 -= lib.einsum('lbjd,ild->ijb', imds.Wovov, r2)  # typo in Ref
            Hr2 -= lib.einsum('kbid,kjd->ijb', imds.Wovov, r2)
            tmp = 2 * lib.einsum('lkdc,kld->c', imds.Woovv, r2)
            tmp -= lib.einsum('kldc,kld->c', imds.Woovv, r2)
            Hr2 -= lib.einsum('c,ijcb->ijb', tmp, imds.t2)

        # CVS Implementaiton
        ncvs = self.ncvs # Consistent with python indexing
        CVSMIN = self.CVSMIN
        CVSMAX = self.CVSMAX
        # Hr1_i = 0, if i!=coreorbtial
        indices = np.arange(nocc)
        condition = ((indices < CVSMIN) | (indices > CVSMAX))
        Hr1 = np.where(condition, 0.0, Hr1)

        # Hr2_ija = 0, if i!=coreorbtial & j!=coreorbital
        indices = np.arange(nocc)
        condition = (indices > ncvs) & (indices < CVSMIN) | (indices > CVSMAX)
        Hr2[np.ix_(condition, condition, np.arange(Hr2.shape[2]))] = 0.0

        vector = self.amplitudes_to_vector(Hr1, Hr2)
        return vector

def get_eomip_child_class(cc, ncvs=0, corehole=0):
    '''

    Args:
        EOMIP:
        ncvs:
        **kwargs:

    Returns:

    '''

    class EOMIPCVS(_EOMIPCVS, EOMIP):
        _EOMIP_class = EOMIP.__class__
        setattr(_EOMIPCVS, 'ncvs', ncvs)
        setattr(_EOMIPCVS, 'corehole', corehole)
        setattr(_EOMIPCVS, 'CVSMIN', None)
        setattr(_EOMIPCVS, 'CVSMAX', None)

        # Setting Up the corehole creation site and corresponding CVS space
        CVSMIN = corehole
        CVSMAX = corehole + ncvs - 1
        _EOMIPCVS.CVSMAX = CVSMAX
        _EOMIPCVS.CVSMIN = CVSMIN

        def get_init_guess(self, nroots=1, koopmans=True, diag=None):
            return _EOMIPCVS.get_init_guess(self, nroots=nroots, koopmans=koopmans, diag=diag)
        
        def dump_flags(self, verbose=None):
            return _EOMIPCVS.dump_flags(self, verbose=verbose)
        
        def gen_matvec(self, imds=None, left=False):
            if imds is None: imds = self.make_imds()
            diag = self.get_diag(imds)
            if left:
                matvec = lambda xs: [_EOMIPCVS.lipccsd_matvec(self, x, imds, diag) for x in xs]
            else:
                matvec = lambda xs: [_EOMIPCVS.ipccsd_matvec(self, x, imds, diag) for x in xs]
            return matvec, diag

    eomip = EOMIPCVS(cc)
    _keys = eomip._keys.copy()
    eomip.__dict__.update(cc.__dict__)
    eomip._keys = eomip._keys.union(_keys)
    return eomip
