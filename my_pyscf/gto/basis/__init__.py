# Reference: CFOUR X2C contracted
# Converted to nwchem format


from pyscf import gto
import os


def readbasis(element, molbasis, mydict):

    element_dict = {
        'H': 'Hydrogen',
        'He': 'Helium',
        'Li': 'Lithium',
        'Be': 'Beryllium',
        'B': 'Boron',
        'C': 'Carbon',
        'N': 'Nitrogen',
        'O': 'Oxygen',
        'F': 'Fluorine',
        'Ne': 'Neon',
        'Na': 'Sodium',
        'Mg': 'Magnesium',
        'Al': 'Aluminum',
        'Si': 'Silicon',
        'P': 'Phosphorus',
        'S': 'Sulfur',
        'Cl': 'Chlorine',
        'Ar': 'Argon'}


    foldername = element_dict.get(element)
    x2cbasis = molbasis.get(element)

    filename = (foldername + '_' + x2cbasis+'.dat').lower()
    if os.path.exists(os.path.join(os.path.dirname(__file__), filename)):
        with open(os.path.join(os.path.dirname(__file__), filename), 'r') as f:
            file_content = f.read()
        mydict[element] = gto.parse(f'''{file_content}''')
        return mydict
    else:
        print("Recontracted basis for this molecule is not present", element, x2cbasis)





