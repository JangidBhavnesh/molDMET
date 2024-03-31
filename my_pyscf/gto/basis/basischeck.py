
import pyscf 
from pyscf import gto, scf
from molDMET.my_pyscf.gto import x2cbasis

Hbasislst = ["augccpv5zx2c", "augccpv6zx2c", "augccpvdzx2c","augccpvqzx2c","augccpvtzx2c",
                                "ccpv5zx2c","ccpv6zx2c","ccpvdzx2c","ccpvqzx2c","ccpvtzx2c"]

for b in Hbasislst:
    mol = gto.Mole()
    mol.atom = '''
    H 0 0 0
    '''
    mol.basis = x2cbasis({'H':b})
    mol.spin = 1
    mol.verbose = 0
    mol.build()

    mf = scf.HF(mol).x2c().density_fit().run()
    print("Done for Hydrogen" ,  b)


print("Done for Hydrogen" , len(Hbasislst))

Bbasislst =  [
    "augccpcv5zx2c",
    "augccpcvdzx2c",
    "augccpcvtzx2c",
    "augccpv5zx2c",
    "augccpv6zx2c",
    "augccpvdzx2c",
    "augccpvqzx2c",
    "augccpvtzx2c",
    "ccpcv5zx2c",
    "ccpcv6zx2c",
    "ccpcvdzx2c",
    "ccpcvqzx2c",
    "ccpv5zx2c",
    "ccpv6zx2c",
    "ccpvdzx2c",
    "ccpvqzx2c",
    "ccpvtzx2c"]



for b in Bbasislst:
    mol = gto.Mole()
    mol.atom = '''
    B 0 0 0
    '''
    mol.basis = x2cbasis({'B':b})
    mol.spin = 1
    mol.verbose = 0
    mol.build()

    mf = scf.HF(mol).x2c().density_fit().run()
    print("Done for Boron" ,  b)


print("Done for Boron", len(Bbasislst), '\n')


Cbasislst = [
    "augccpcv5zx2c",
    "augccpcvdzx2c",
    "augccpcvqzx2c",
    "augccpcvtzx2c",
    "augccpv5zx2c",
    "augccpv6zx2c",
    "augccpvdzx2c",
    "augccpvqzx2c",
    "augccpvtzx2c",
    "ccpcvdzx2c",
    "ccpcvqzx2c",
    "ccpcvtzx2c",
    "ccpv5zx2c",
    "ccpv6zx2c",
    "ccpvdzx2c",
    "ccpvqzx2c",
    "ccpvtzx2c"
]

for b in Cbasislst:
    mol = gto.Mole()
    mol.atom = '''
    C 0 0 0
    '''
    mol.basis = x2cbasis({'C':b})
    mol.spin = 2
    mol.verbose = 0
    mol.build()

    mf = scf.HF(mol).x2c().density_fit().run()
    print("Done for Carbon" ,  b)


print("Done for Carbon", len(Cbasislst), '\n')

Nbasislst = ["augccpcv5zx2c","augccpcvdzx2c","augccpcvqzx2c","augccpcvtzx2c","augccpv5zx2c","augccpv6zx2c","augccpvdzx2c",
            "augccpvqzx2c","augccpvtzx2c","ccpcvdzx2c","ccpcvqzx2c","ccpcvtzx2c","ccpv5zx2c","ccpv6zx2c",
            "ccpvdzx2c","ccpvqzx2c","ccpvtzx2c"]
                                                                 
for b in Nbasislst:
    mol = gto.Mole()
    mol.atom = '''
        N 0 0 0
        '''
    mol.basis = x2cbasis({'N':b})
    mol.spin = 3
    mol.verbose = 0
    mol.build()

    mf = scf.HF(mol).x2c().density_fit().run()
    print("Done for Nitrogen" ,  b)
print("Done for Nitrogen", len(Nbasislst), '\n')

Obasislst =[
    "augccpcv5zx2c",
    "augccpcvdzx2c",
    "augccpcvqzx2c",
    "augccpcvtzx2c",
    "augccpv5zx2c",
    "augccpv6zx2c",
    "augccpvdzx2c",
    "augccpvqzx2c",
    "augccpvtzx2c",
    "ccpcv5zx2c",
    "ccpcv6zx2c",
    "ccpcvdzx2c",
    "ccpcvqzx2c",
    "ccpcvtzx2c",
    "ccpv5zx2c",
    "ccpv6zx2c",
    "ccpvdzx2c",
    "ccpvqzx2c",
    "ccpvtzx2c"]


                                                                 
for b in Obasislst:
    mol = gto.Mole()
    mol.atom = '''
        O 0 0 0
        '''
    mol.basis = x2cbasis({'O':b})
    mol.spin = 2
    mol.verbose = 0
    mol.build()

    mf = scf.HF(mol).x2c().density_fit().run()
    print("Done for Oxygen" ,  b)


print("Done for Oxygen", len(Obasislst), '\n')

Fbasislst = [
    "augccpcv5zx2c",
    "augccpcvdzx2c",
    "augccpcvtzx2c",
    "augccpv5zx2c",
    "augccpv6zx2c",
    "augccpvdzx2c",
    "augccpvqzx2c",
    "augccpvtzx2c",
    "ccpcv5zx2c",
    "ccpcv6zx2c",
    "ccpcvdzx2c",
    "ccpcvqzx2c",
    "ccpcvtzx2c",
    "ccpv5zx2c",
    "ccpv6zx2c",
    "ccpvdzx2c",
    "ccpvqzx2c",
    "ccpvtzx2c"]

                                                                 
for b in Fbasislst:
    mol = gto.Mole()
    mol.atom = '''
        F 0 0 0
        '''
    mol.basis = x2cbasis({'F':b})
    mol.spin = 1
    mol.verbose = 0
    mol.build()

    mf = scf.HF(mol).x2c().density_fit().run()
    print("Done for Flourine" ,  b)


print("Done for Flourine", len(Fbasislst), '\n')


Sbasislst = ['augccpcv5zx2c', 'augccpcvdzx2c', 'augccpcvqzx2c', 'augccpcvtzx2c', 'augccpv(5+d)zx2c', 'augccpv(6+d)zx2c', 'augccpv(d+d)zx2c', \
'augccpv(q+d)zx2c', 'augccpv(t+d)zx2c', 'augccpv5zx2c', 'augccpv6zx2c', 'augccpvdzx2c', 'augccpvqzx2c', 'augccpvtzx2c', 'ccpcv5zx2c', 'ccpcvdzx2c',\
 'ccpcvqzx2c', 'ccpcvtzx2c', 'ccpv(5+d)zx2c', 'ccpv(6+d)zx2c', 'ccpv(d+d)zx2c', 'ccpv(q+d)zx2c', 'ccpv(t+d)zx2c', 'ccpv5zx2c', 'ccpv6zx2c', \
 'ccpvdzx2c', 'ccpvqzx2c', 'ccpvtzx2c']
                                                                 
for b in Sbasislst:
    mol = gto.Mole()
    mol.atom = '''
        S 0 0 0
        '''
    mol.basis = x2cbasis({'S':b})
    mol.spin = 2
    mol.verbose = 0
    mol.build()

    mf = scf.HF(mol).x2c().density_fit().run()
    print("Done for Sulfur" ,  b)


print("Done for Sulfur", len(Sbasislst), '\n')
