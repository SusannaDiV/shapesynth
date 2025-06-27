from .brics_fragmenizer import BRICS_Fragmenizer
from .ring_r_fragmenizer import RING_R_Fragmenizer
from rdkit import Chem

# from brics_fragmenizer import BRICS_Fragmenizer
# from ring_r_fragmenizer import RING_R_Fragmenizer

class BRICS_RING_R_Fragmenizer():
    def __init__(self):
        self.type = 'BRICS_RING_R_Fragmenizer'
        self.brics_fragmenizer = BRICS_Fragmenizer()
        self.ring_r_fragmenizer = RING_R_Fragmenizer()
    
    def fragmenize(self, mol, dummyStart=1):
        # First, identify bonds to break
        brics_bonds = self.brics_fragmenizer.get_bonds(mol)
        ring_r_bonds = self.ring_r_fragmenizer.get_bonds(mol)
        
        # Filter out C-H bonds and bonds involving hydrogens
        bonds = []
        for bond in brics_bonds + ring_r_bonds:
            atom1 = mol.GetAtomWithIdx(bond[0])
            atom2 = mol.GetAtomWithIdx(bond[1])
            # Skip if either atom is hydrogen
            if atom1.GetSymbol() == 'H' or atom2.GetSymbol() == 'H':
                continue
            bonds.append(bond)

        if len(bonds) != 0:
            # Get unique bond IDs
            bond_ids = [mol.GetBondBetweenAtoms(x, y).GetIdx() for x, y in bonds]
            bond_ids = list(set(bond_ids))
            
            # Create dummy labels for fragmentation
            dummyLabels = [(i + dummyStart, i + dummyStart) for i in range(len(bond_ids))]
            
            # Fragment the molecule
            break_mol = Chem.FragmentOnBonds(mol, bond_ids, dummyLabels=dummyLabels)
            dummyEnd = dummyStart + len(dummyLabels) - 1
        else:
            break_mol = mol
            dummyEnd = dummyStart - 1

        return break_mol, dummyEnd
