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
    #print("conformer", conformer)
    center = np.mean(conformer.GetPositions(), axis=0)
    #print("center", center)
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
    return Chem.CanonSmiles(re.sub(r'\[\d+\*\]', '[*]', frag_smi))

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
    """Get points for alignment between two fragments.
    Simple matching based on atom positions and types.
    """
    points1 = []
    points2 = []
    frag1_to_frag2_atom_mapping = {}
    frag2_to_frag1_atom_mapping = {}
    
    # Get conformers
    conf1 = frag1.GetConformer()
    conf2 = frag2.GetConformer()
    
    # First match attachment points and their neighbors
    matched_frag2_indices = set()
    for idx1, atom1 in enumerate(frag1.GetAtoms()):
        # Don't skip attachment points or their neighbors
        if atom1.GetSymbol() == '*' or any(n.GetSymbol() == '*' for n in atom1.GetNeighbors()):
            pos1 = conf1.GetAtomPosition(idx1)
            
            # Find closest matching atom in frag2
            best_idx2 = None
            best_dist = float('inf')
            for idx2, atom2 in enumerate(frag2.GetAtoms()):
                if (idx2 in matched_frag2_indices or
                    atom1.GetSymbol() != atom2.GetSymbol()):  # Must be same element
                    continue
                
                pos2 = conf2.GetAtomPosition(idx2)
                dist = sum((p1 - p2) * (p1 - p2) for p1, p2 in zip(pos1, pos2))
                if dist < best_dist:
                    best_dist = dist
                    best_idx2 = idx2
            
            if best_idx2 is not None:
                points1.append(pos1)
                points2.append(conf2.GetAtomPosition(best_idx2))
                frag1_to_frag2_atom_mapping[idx1] = best_idx2
                frag2_to_frag1_atom_mapping[best_idx2] = idx1
                matched_frag2_indices.add(best_idx2)
    
    # Then match remaining atoms (except hydrogens)
    for idx1, atom1 in enumerate(frag1.GetAtoms()):
        if (idx1 in frag1_to_frag2_atom_mapping or  # Skip already matched
            atom1.GetSymbol() == 'H'):  # Skip regular hydrogens
            continue
            
        pos1 = conf1.GetAtomPosition(idx1)
        
        # Find closest matching atom in frag2
        best_idx2 = None
        best_dist = float('inf')
        for idx2, atom2 in enumerate(frag2.GetAtoms()):
            if (atom2.GetSymbol() == 'H' or  # Skip hydrogens
                idx2 in matched_frag2_indices or
                atom1.GetSymbol() != atom2.GetSymbol()):  # Must be same element
                continue
            
            pos2 = conf2.GetAtomPosition(idx2)
            dist = sum((p1 - p2) * (p1 - p2) for p1, p2 in zip(pos1, pos2))
            if dist < best_dist:
                best_dist = dist
                best_idx2 = idx2
        
        if best_idx2 is not None:
            points1.append(pos1)
            points2.append(conf2.GetAtomPosition(best_idx2))
            frag1_to_frag2_atom_mapping[idx1] = best_idx2
            frag2_to_frag1_atom_mapping[best_idx2] = idx1
            matched_frag2_indices.add(best_idx2)
    
    if not points1 or not points2:
        raise ValueError("No matching atoms found between fragments")
    
    return np.array(points1), np.array(points2), frag1_to_frag2_atom_mapping, frag2_to_frag1_atom_mapping

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
