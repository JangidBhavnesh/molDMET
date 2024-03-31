'''
An example to show how to calculate the core
Ionization of oxygen (1s) for water using EOM-CCSD
Method.
For K-edge use the scalar relativistic effect. Currently
it can added to H via the X2C. While using X2C use the
corresponding contracted basis functions.
'''

from pyscf import gto, dft, mp, scf, mcscf, adc, lib, cc
from molDMET.my_pyscf.gto import x2cbasis
from molDMET.my_pyscf.cc import EOMIP, EOMIPStar

# Molecule Information
mol = gto.Mole()
mol.atom = '''
   H        0.61473       -0.02651        0.47485
   O        0.13157        0.02998       -0.34219
   H       -0.79537       -0.00348       -0.13266
   '''
mol.basis=x2cbasis({'O':'augccpcvtzx2c', 'H':'augccpvtzx2c'})
mol.symmetry=False
mol.verbose=4
mol.build()

# Mean Field Calculations
mf = scf.RHF(mol)
mf.x2c().density_fit()
mf.kernel()


# CCSD Solver
ccsolver = cc.ccsd.CCSD(mf).run()

# EOM-CCSD Solver
coreip = EOMIP(ccsolver, nroots=1, ncvs=1, koopmans=True)
coreip_star = EOMIPStar(ccsolver, nroots=1, ncvs=1, koopmans=True)


print("The core-ionization energy for O(1s) in water (eV)")
print(f"Experimental Value: {539.90:.2f}") # Reference: J.Chem.TheoryComput.2019, 15, 1642−1651
print(f"IP-EOM-CCSD Value : {27.21139*coreip[0]:.2f}")
print(f"IP-EOM-CCSD* Value: {27.21139*coreip_star[0]:.2f}")

