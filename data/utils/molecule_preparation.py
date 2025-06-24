import os
import sys
from collections import OrderedDict

from openbabel import openbabel as ob

from meeko.atomtyper import AtomTyper
from meeko.bondtyper import BondTyperLegacy
from meeko.hydrate import HydrateMoleculeLegacy
from meeko.macrocycle import FlexMacrocycle
from meeko.writer import PDBQTWriterLegacy
from meeko.molsetup import MoleculeSetup
from meeko.flexibility import FlexibilityBuilder
from rdkit import Chem
import numpy as np
from rdkit.Chem import rdMolTransforms
from copy import deepcopy
import re
import random
from functools import cmp_to_key

PLACE_HOLDER_ATOM = 80 # Hg

def find_parts_bonds(mol, parts):
    ret_bonds = []
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            i_part = parts[i]
            j_part = parts[j]
            for i_atom_idx in i_part:
                for j_atom_idx in j_part:
                    bond = mol.GetBondBetweenAtoms(i_atom_idx, j_atom_idx)
                    if bond is None:
                        continue
                    ret_bonds.append((i_atom_idx, j_atom_idx))
    return ret_bonds

def get_other_atom_idx(mol, atom_idx_list):
    ret_atom_idx = []
    for atom in mol.GetAtoms():
        if atom.GetIdx() not in atom_idx_list:
            ret_atom_idx.append(atom.GetIdx())
    return ret_atom_idx

def get_rings(mol):
    rings = []
    for ring in list(Chem.GetSymmSSSR(mol)):
        ring = list(ring)
        rings.append(ring)
    return rings

def get_bonds(mol, bond_type):
    bonds = []
    for bond in mol.GetBonds():
        if bond.GetBondType() is bond_type:
            bonds.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    return bonds

def get_center(mol, confId=-1):
    conformer = mol.GetConformer(confId)
    center = np.mean(conformer.GetPositions(), axis=0)
    return center

def trans(x, y, z):
    translation = np.eye(4)
    translation[:3, 3] = [x, y, z]
    return translation

def centralize(mol, confId=-1):
    mol = deepcopy(mol)
    conformer = mol.GetConformer(confId)
    center = get_center(mol, confId)
    translation = trans(-center[0], -center[1], -center[2])  
    rdMolTransforms.TransformConformer(conformer, translation)
    return mol

def canonical_frag_smi(frag_smi):
    frag_smi = re.sub(r'\[\d+\*\]', '[*]', frag_smi)
    canonical_frag_smi = Chem.CanonSmiles(frag_smi)
    return canonical_frag_smi

def get_surrogate_frag(frag):
    frag = deepcopy(frag)
    m_frag = Chem.RWMol(frag)
    for atom in m_frag.GetAtoms():
        if atom.GetSymbol() == '*':
            atom_idx = atom.GetIdx()
            m_frag.ReplaceAtom(atom_idx, Chem.Atom(PLACE_HOLDER_ATOM))
    Chem.SanitizeMol(m_frag)
    return m_frag

def get_align_points(frag1, frag2):
    align_point1 = np.zeros((frag1.GetNumAtoms(), 3))
    align_point2 = np.zeros((frag2.GetNumAtoms(), 3))
    frag12frag2 = dict()
    frag22farg1 = dict()
    order1 = list(Chem.CanonicalRankAtoms(frag1, breakTies=True))
    order2 = list(Chem.CanonicalRankAtoms(frag2, breakTies=True))
    con1 = frag1.GetConformer()
    con2 = frag2.GetConformer()
    for i in range(len(order1)):
        frag_idx1 = order1.index(i)
        frag_idx2 = order2.index(i)
        assert frag1.GetAtomWithIdx(frag_idx1).GetSymbol() == frag2.GetAtomWithIdx(frag_idx2).GetSymbol()
        atom_pos1 = list(con1.GetAtomPosition(frag_idx1))
        atom_pos2 = list(con2.GetAtomPosition(frag_idx2))
        align_point1[i] = atom_pos1
        align_point2[i] = atom_pos2
        frag12frag2[frag_idx1] = frag_idx2
        frag22farg1[frag_idx2] = frag_idx1
    return align_point1, align_point2, frag12frag2, frag22farg1

def get_atom_mapping_between_frag_and_surrogate(frag, surro):
    con1 = frag.GetConformer()
    con2 = surro.GetConformer()
    pos2idx1 = dict()
    pos2idx2 = dict()
    for atom in frag.GetAtoms():
        pos2idx1[tuple(con1.GetAtomPosition(atom.GetIdx()))] = atom.GetIdx()
    for atom in surro.GetAtoms():
        pos2idx2[tuple(con2.GetAtomPosition(atom.GetIdx()))] = atom.GetIdx()
    frag2surro = dict()
    surro2frag = dict()
    for key in pos2idx1.keys():
        frag_idx = pos2idx1[key]
        surro_idx = pos2idx2[key]
        frag2surro[frag_idx] = surro_idx
        surro2frag[surro_idx] = frag_idx
    return frag2surro, surro2frag

def get_tree(adj_dict, start_idx, visited, iter_num):
    ret = [start_idx]
    visited.append(start_idx)
    for i in range(iter_num):
        if (not i in visited) and ((start_idx, i) in adj_dict):
            ret.append(get_tree(adj_dict, i, visited, iter_num))
    visited.pop()
    return ret

def get_tree_high(tree):
    if len(tree) == 1:
        return 1
    
    subtree_highs = []
    for subtree in tree[1:]:
        subtree_high = get_tree_high(subtree)
        subtree_highs.append(subtree_high)
    
    return 1 + max(subtree_highs)

def tree_sort_cmp(a_tree, b_tree):
    a_tree_high = get_tree_high(a_tree)
    b_tree_high = get_tree_high(b_tree)

    if a_tree_high < b_tree_high:
        return -1
    if a_tree_high > b_tree_high:
        return 1
    return random.choice([-1, 1])

def tree_linearize(tree, res):
    res.append(tree[0])
    
    subtrees = tree[1:]
    subtrees.sort(key=cmp_to_key(tree_sort_cmp))
    
    for subtree in subtrees:
        if subtree != subtrees[-1]:
            res.append('b')
            tree_linearize(subtree, res)
            res.append('e')
        else:
            tree_linearize(subtree, res)

class MoleculePreparation:
    def __init__(self, merge_hydrogens=True, hydrate=False, macrocycle=False, amide_rigid=True):
        self._merge_hydrogens = merge_hydrogens
        self._add_water = hydrate
        self._break_macrocycle = macrocycle
        self._keep_amide_rigid = amide_rigid
        self._mol = None

        self._atom_typer = AtomTyper()
        self._bond_typer = BondTyperLegacy()
        self._macrocycle_typer = FlexMacrocycle(min_ring_size=7, max_ring_size=33) #max_ring_size=26, min_ring_size=8)
        self._flex_builder = FlexibilityBuilder()
        self._water_builder = HydrateMoleculeLegacy()
        self._writer = PDBQTWriterLegacy()

    def prepare(self, mol, freeze_bonds=None):
        """ """

        if mol.NumAtoms() == 0:
            raise ValueError('Error: no atoms present in the molecule')

        self._mol = mol
        MoleculeSetup(mol)

        # 1.  assign atom types (including HB types, vectors and stuff)
        # DISABLED TODO self.atom_typer.set_parm(mol)
        self._atom_typer.set_param_legacy(mol)

        # 2a. add pi-model + merge_h_pi (THIS CHANGE SOME ATOM TYPES)

        # 2b. merge_h_classic
        if self._merge_hydrogens:
            mol.setup.merge_hydrogen()

        # 3.  assign bond types by using SMARTS...
        #     - bonds should be typed even in rings (but set as non-rotatable)
        #     - if macrocycle is selected, they will be enabled (so they must be typed already!)
        self._bond_typer.set_types_legacy(mol)

        # 4 . hydrate molecule
        if self._add_water:
            self._water_builder.hydrate(mol)

        # 5.  scan macrocycles
        if self._break_macrocycle:
            # calculate possible breakable bonds
            self._macrocycle_typer.search_macrocycle(mol)

        # 6.  build flexibility...
        # 6.1 if macrocycles typed:
        #     - walk the setup graph by skipping proposed closures
        #       and score resulting flex_trees basing on the lenght
        #       of the branches generated
        #     - actually break the best closure bond (THIS CHANGES SOME ATOM TYPES)
        # 6.2  - walk the graph and build the flextree
        # 7.  but disable all bonds that are in rings and not
        #     in flexible macrocycles
        # TODO restore legacy AD types for PDBQT
        #self._atom_typer.set_param_legacy(mol)
        if freeze_bonds == 'All': 
            freeze_bonds = []
            for b in ob.OBMolBondIter(mol):
                begin = b.GetBeginAtomIdx()
                end = b.GetEndAtomIdx()
                bond_id = mol.setup.get_bond_id(begin, end)
                freeze_bonds.append(bond_id)
        # print(freeze_bonds)

        self._flex_builder.process_mol(mol, freeze_bonds=freeze_bonds)
        # TODO re-run typing after breaking bonds
        # self.bond_typer.set_types_legacy(mol, exclude=[macrocycle_bonds])
    
    def show_setup(self):
        if self._mol is not None:
            tot_charge = 0

            print("Molecule setup\n")
            print("==============[ ATOMS ]===================================================")
            print("idx  |          coords            | charge |ign| atype    | connections")
            print("-----+----------------------------+--------+---+----------+--------------- . . . ")
            for k, v in list(self._mol.setup.coord.items()):
                print("% 4d | % 8.3f % 8.3f % 8.3f | % 1.3f | %d" % (k, v[0], v[1], v[2],
                      self._mol.setup.charge[k], self._mol.setup.atom_ignore[k]),
                      "| % -8s |" % self._mol.setup.atom_type[k],
                      self._mol.setup.graph[k])
                tot_charge += self._mol.setup.charge[k]
            print("-----+----------------------------+--------+---+----------+--------------- . . . ")
            print("  TOT CHARGE: %3.3f" % tot_charge)

            print("\n======[ DIRECTIONAL VECTORS ]==========")
            for k, v in list(self._mol.setup.coord.items()):
                if k in self._mol.setup.interaction_vector:
                    print("% 4d " % k, self._mol.setup.atom_type[k], end=' ')

            print("\n==============[ BONDS ]================")
            # For sanity users, we won't show those keys for now
            keys_to_not_show = ['bond_order', 'type']
            for k, v in list(self._mol.setup.bond.items()):
                t = ', '.join('%s: %s' % (i, j) for i, j in v.items() if not i in keys_to_not_show)
                print("% 8s - " % str(k), t)

            self._macrocycle_typer.show_macrocycle_scores()

            print('')
    
    def write_pdbqt_string(self):
        if self._mol is not None:
            return self._writer.write_string(self._mol)
        else:
            raise RuntimeError('Cannot generate PDBQT file, the molecule is not prepared.')

    def write_pdbqt_file(self, pdbqt_filename):
        try:
            with open(pdbqt_filename,'w') as w:
                w.write(self.write_pdbqt_string())
        except:
            raise RuntimeError('Cannot write PDBQT file %s.' % pdbqt_filename)
