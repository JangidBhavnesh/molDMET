from pyscf import gto, dft, mp, scf, mcscf, adc, lib
from molDMET.my_pyscf.dmet import localintegrals, main_object, fragments
from molDMET.my_pyscf.gto import x2cbasis

# Molecule Information
mol = gto.Mole()
mol.atom = 'azobenzene.xyz'
mol.spin = 0
mol.charge = 0
mol.basis = x2cbasis({'N':'ccpcvdzx2c', 'C':'ccpcvdzx2c', 'H':'ccpvdzx2c'})
mol.verbose = 4
mol.build()

# Mean Field Calculation
mf = scf.RHF(mol)
mf.x2c().density_fit()
mf.max_memory = 10000
mf.kernel()

# Localization
myInts = localintegrals.localintegrals(mf, range(mol.nao_nr()), 'meta_lowdin')

# Fragment Information
Fragment_List = [0, 1, 12, 13] 

# DMET
mydmet = main_object.dmet(myInts, Fragment_List, 'MP2', bath_tol=1e-6)
mydmet.oneshot()
mydmet.CoreIP(method='IPCVSEOM', nroots=2, ncvs=2)
