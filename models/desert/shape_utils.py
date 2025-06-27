import numpy as np
from rdkit.Chem import rdMolTransforms, AllChem
from rdkit import Chem
from math import ceil, pi
import random
import copy
from skimage.util import view_as_blocks
import pickle as pkl
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired
import math
from rdkit.Chem.AllChem import AlignMol
from vina import Vina
import os
import sys
from collections import OrderedDict

from openbabel import openbabel as ob

meeko_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'data', 'Meeko')
sys.path.append(meeko_path)

from meeko.setup import MoleculeSetup
from meeko.atomtyper import AtomTyperLegacy
from meeko.bondtyper import BondTyperLegacy
from meeko.hydrate import HydrateMoleculeLegacy
from meeko.macrocycle import FlexMacrocycle
from meeko.flexibility import FlexibilityBuilder
from meeko.writer import PDBQTWriterLegacy


class MoleculePreparation:
    def __init__(self, merge_hydrogens=True, hydrate=False, macrocycle=False, amide_rigid=True):
        self._merge_hydrogens = merge_hydrogens
        self._add_water = hydrate
        self._break_macrocycle = macrocycle
        self._keep_amide_rigid = amide_rigid
        self._mol = None

        self._atom_typer = AtomTyperLegacy()
        self._bond_typer = BondTyperLegacy(self._keep_amide_rigid)
        self._macrocycle_typer = FlexMacrocycle(min_ring_size=7, max_ring_size=33, min_score=50) #max_ring_size=26, min_ring_size=8)
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
# fragmenizer = BRICS_RING_R_Fragmenizer()

def get_mol_centroid(mol, confId=-1):
    conformer = mol.GetConformer(confId)
    centroid = np.mean(conformer.GetPositions(), axis=0)
    return centroid

def trans(x, y, z):
    translation = np.eye(4)
    translation[:3, 3] = [x, y, z]
    return translation

def centralize(mol, confId=-1):
    conformer = mol.GetConformer(confId)
    centroid = get_mol_centroid(mol, confId)
    translation = trans(-centroid[0], -centroid[1], -centroid[2])  
    rdMolTransforms.TransformConformer(conformer, translation)
    return mol

def get_atom_stamp(grid_resolution, max_dist):
    # atom stamp is a sphere which radius equal to atom van der Waals radius
    def _get_atom_stamp(symbol):
        box_size = ceil(2 * max_dist // grid_resolution + 1)

        x, y, z = np.indices((box_size, box_size, box_size))
        x = x * grid_resolution + grid_resolution / 2
        y = y * grid_resolution + grid_resolution / 2
        z = z * grid_resolution + grid_resolution / 2

        mid = (box_size // 2, box_size // 2, box_size // 2)
        mid_x = x[mid]
        mid_y = y[mid]
        mid_z = z[mid]

        sphere = (x - mid_x)**2 + (y - mid_y)**2 + (z - mid_z)**2 \
            <= ATOM_RADIUS[symbol]**2
        sphere = sphere.astype(int)
        sphere[sphere > 0] = ATOMIC_NUMBER[symbol]
        return sphere

    atom_stamp = {}
    for symbol in ATOM_RADIUS:
        atom_stamp[symbol] = _get_atom_stamp(symbol)
    return atom_stamp

def get_atom_stamp_with_noise(grid_resolution, max_dist, mu, sigma):
    def _get_atom_stamp_with_noise(symbol, mu, sigma):
        box_size = ceil(2 * max_dist // grid_resolution + 1)

        x, y, z = np.indices((box_size, box_size, box_size))
        x = x * grid_resolution + grid_resolution / 2
        y = y * grid_resolution + grid_resolution / 2
        z = z * grid_resolution + grid_resolution / 2

        mid = (box_size // 2, box_size // 2, box_size // 2)
        mid_x = x[mid]
        mid_y = y[mid]
        mid_z = z[mid]

        noise = np.random.normal(loc=mu, scale=sigma)
        if noise < 0:
            noise = 0 

        sphere = (x - mid_x)**2 + (y - mid_y)**2 + (z - mid_z)**2 \
            <= (ATOM_RADIUS[symbol] + noise)**2
        sphere = sphere.astype(int)
        sphere[sphere > 0] = ATOMIC_NUMBER[symbol]
        return sphere
    
    atom_stamp = {}
    for symbol in ATOM_RADIUS:
        atom_stamp[symbol] = _get_atom_stamp_with_noise(symbol, mu, sigma)
    return atom_stamp

def get_binary_features(mol, confId):
    coords = []
    features = []
    confermer = mol.GetConformer(confId)
    for atom in mol.GetAtoms():
        if atom.HasProp('mask') and get_atom_prop(atom, 'mask') == 'true':
            continue
        idx = atom.GetIdx()
        coord = list(confermer.GetAtomPosition(idx))
        coords.append(coord)
        features.append(atom.GetAtomicNum())
    coords = np.array(coords)
    features = np.array(features)
    features = np.expand_dims(features, axis=1)
    return coords, features

def get_shape(mol, atom_stamp, grid_resolution, max_dist, confId=-1):
    # expand each atom point to a sphere
    coords, features = get_binary_features(mol, confId)
    grid, atomic2grid = make_grid(coords, features, grid_resolution, max_dist)
    shape = np.zeros(grid[0, :, :, :, 0].shape)
    for tup in atomic2grid:
        atomic_number = int(tup[0])
        stamp = atom_stamp[ATOMIC_NUMBER_REVERSE[atomic_number]]
        for grid_ijk in atomic2grid[tup]:
            i = grid_ijk[0]
            j = grid_ijk[1]
            k = grid_ijk[2]

            x_left = i - stamp.shape[0] // 2 if i - stamp.shape[0] // 2 > 0 else 0
            x_right = i + stamp.shape[0] // 2 if i + stamp.shape[0] // 2 < shape.shape[0] else shape.shape[0] - 1
            x_l = i - x_left
            x_r = x_right - i

            y_left = j - stamp.shape[1] // 2 if j - stamp.shape[1] // 2 > 0 else 0
            y_right = j + stamp.shape[1] // 2 if j + stamp.shape[1] // 2 < shape.shape[1] else shape.shape[1] - 1
            y_l = j - y_left
            y_r = y_right - j

            z_left = k - stamp.shape[2] // 2 if k - stamp.shape[2] // 2 >0 else 0
            z_right = k + stamp.shape[2] // 2 if k + stamp.shape[2] // 2 < shape.shape[2] else shape.shape[2] - 1
            z_l = k - z_left
            z_r = z_right - k

            mid = stamp.shape[0] // 2
            shape_part =  shape[x_left: x_right + 1, y_left: y_right + 1, z_left: z_right + 1]
            stamp_part = stamp[mid - x_l: mid + x_r + 1, mid - y_l: mid + y_r + 1, mid - z_l: mid + z_r + 1]

            shape_part += stamp_part
    shape[shape > 0] = 1
    return shape

def sample_augment(sample, rotation_bin, max_translation, confId=-1):
    sample = copy.deepcopy(sample)
    confermer = sample['mol'].GetConformer(confId)

    rot = random.choice(range(rotation_bin))
    rotation_mat = ROTATIONS[rot]

    # rotation the molecule
    rotation = np.zeros((4, 4))
    rotation[:3, :3] = rotation_mat
    rdMolTransforms.TransformConformer(confermer, rotation)

    # rotation fragments
    for fragment in sample['frag_list']:
        frag_rotation_mat = fragment['rotate_mat']
        frag_trans_vec = fragment['trans_vec']
        
        frag_rotation_translation = np.zeros((4, 4))
        frag_rotation_translation[:3, :3] = frag_rotation_mat
        frag_rotation_translation[:3, 3] = frag_trans_vec

        frag_rotation_translation_rotation = np.dot(rotation, frag_rotation_translation)

        fragment['rotate_mat'] = frag_rotation_translation_rotation[:3, :3]
        fragment['trans_vec'] = frag_rotation_translation_rotation[:3, 3]

    tr = max_translation * np.random.rand(3)

    # translate the molecule
    translate = trans(tr[0], tr[1], tr[2])
    rdMolTransforms.TransformConformer(confermer, translate)

    # translate fragments
    for fragment in sample['frag_list']:
        frag_trans_vec = fragment['trans_vec']
        frag_trans_vec = frag_trans_vec + tr
        fragment['trans_vec'] = frag_trans_vec

    return sample

def get_shape_patches(shape, patch_size):
    assert shape.shape[0] % patch_size == 0
    shape_patches = view_as_blocks(shape, (patch_size, patch_size, patch_size))
    return shape_patches

def time_shift(s):
    return s[:-1], s[1:]

def get_grid_coords(coords, max_dist, grid_resolution):
    grid_coords = (coords + max_dist) / grid_resolution
    grid_coords = grid_coords.round().astype(int)
    return grid_coords

def get_rotation_bins(sp, rp):
    mid = sp // 2
    sr = 1.0 / sp

    face1 = []
    for y in range(sp):
        for z in range(sp):
            face1.append(np.array([0.5, (y - mid) * sr, (z - mid) * sr]))
    face2 = []
    for x in range(sp):
        for y in range(sp):
            face2.append(np.array([(x - mid) * sr, (y - mid) * sr, 0.5]))
    face3 = []
    for x in range(sp):
        for z in range(sp):
            face3.append(np.array([(x - mid) * sr, 0.5, (z - mid) * sr]))
    
    face_point = face1 + face2 + face3
    
    rotation_mat_bin = [rotation_matrix(np.array((1, 1, 1)), 0)]
    for p in face_point:
        for t in range(1, rp):
            axis = p
            theta = t * pi / (rp / 2)
            rotation_mat_bin.append(rotation_matrix(axis, theta))
    rotation_mat_bin = np.stack(rotation_mat_bin, axis=0)

    return rotation_mat_bin

def set_atom_prop(atom, prop_name, prop_value):
    atom.SetProp(prop_name, prop_value)

def get_atom_prop(atom, prop_name):
    if atom.HasProp(prop_name):
        return atom.GetProp(prop_name)
    else:
        return None

def real_tree_len(tree, special_tokens):
    cnt = 0
    for item in tree:
        if isinstance(item[0], str) and item[0].lower() in special_tokens:
            continue
        cnt += 1
    return cnt

def get_partial_tree(tree, mask_len):
    mask_len = mask_len + 1 # EOS will take one spot
    partial_tree = tree[:-mask_len]
    partial_tree.append(('EOS', None, None))
    remove_parts = tree[-mask_len:]
    return partial_tree, remove_parts

def mask_frags(mol, frag_list, mask_frags_idx):
    curr_frags, _ = fragmenizer.fragmenize(mol)
    curr_frags = Chem.GetMolFrags(curr_frags, asMols=True)
    curr_cen_f_idx_mapping = {}
    for f_idx, frag in enumerate(curr_frags):
        curr_cen = tuple(get_mol_centroid(frag).round(2))
        curr_cen_f_idx_mapping[curr_cen] = f_idx
    
    list_cen_f_idx_mapping = {}
    for f_idx, item in enumerate(frag_list):
        cen = tuple(item['trans_vec'].round(2))
        list_cen_f_idx_mapping[cen] = f_idx

    assert len(curr_cen_f_idx_mapping) == len(list_cen_f_idx_mapping)

    list_curr_mapping = {}
    for cen in list_cen_f_idx_mapping:
        try:
            assert cen in curr_cen_f_idx_mapping
            list_curr_mapping[list_cen_f_idx_mapping[cen]] = curr_cen_f_idx_mapping[cen]
        except:
            list_curr_mapping[list_cen_f_idx_mapping[cen]] = list_cen_f_idx_mapping[cen]
    
    for m_f_idx in mask_frags_idx:
        curr_frag = curr_frags[list_curr_mapping[m_f_idx]]
        for atom in curr_frag.GetAtoms():
            origin_atom_idx = get_atom_prop(atom, 'origin_atom_idx')
            if origin_atom_idx is None:
                continue
            origin_atom_idx = int(origin_atom_idx)
            origin_atom = mol.GetAtomWithIdx(origin_atom_idx)
            set_atom_prop(origin_atom, 'mask', 'true')
    
    return mol

# vocab_path = '/opt/tiger/shape_based_pretraining/data/vocab/BRICS_RING_R.100000000.pkl'
# with open(vocab_path, 'rb') as fr:
#         vocab = pkl.load(fr)
# vocab_r = {v[2]: k for k, v in vocab.items()}

# rotation_bin_path = '/opt/tiger/shape_based_pretraining/evaluation/rotation_bin.pkl'
# with open(rotation_bin_path, 'rb') as fr:
#     rotation_bin = pkl.load(fr)

def bin_to_grid_coords(bin, box_size):
    if bin == 0:
        return None
    if bin == 1:
        return float('inf'), float('inf'), float('inf')

    bin = bin - 2
    z = bin % box_size
    bin = bin - z
    y = bin % (box_size ** 2) / box_size
    bin = bin - (bin % (box_size ** 2))
    x = bin / (box_size ** 2)

    z = int(z)
    y = int(y)
    x = int(x)
    return x, y, z

def grid_coords_to_real_coords(grid_coords, box_size, grid_resolution):
    if grid_coords is None:
        return None

    if box_size % 2 == 0:
        mid = box_size / 2
        x = (grid_coords[0] - mid) * grid_resolution + grid_resolution / 2
        y = (grid_coords[1] - mid) * grid_resolution + grid_resolution / 2
        z = (grid_coords[2] - mid) * grid_resolution + grid_resolution / 2
    
    return x, y, z

def bin_to_rotation_mat(bin, rotation_bin):
    if bin == 0:
        return None
    bin = bin - 1
    return rotation_bin[bin]

def get_3d_frags(frags):
    ret_frags = []
    for unit in frags:
        idx = unit[0]
        tr = unit[1]
        rm = unit[2]

        key = vocab_r[idx]
        if key in ['UNK', 'BOS', 'BOB', 'EOB', 'PAD']:
            continue
        if key == 'EOS':
            break

        frag = copy.deepcopy(vocab[key][0])

        conformer = frag.GetConformer()

        newU = np.zeros((4, 4))
        newU[:3,:3] = rm
        rdMolTransforms.TransformConformer(conformer, newU)

        trans_mat = trans(tr[0], tr[1], tr[2])
        rdMolTransforms.TransformConformer(conformer, trans_mat)
        
        ret_frags.append(frag)
    return ret_frags

def connect_fragments(frags):
    """
        input: 
            frags (List): a list of fragments need to be connect
        output:
            mol (rdkit.Chem.rdchem.Mol): connected molecule
    """
    def get_star_info(frags):
        star_info = []
        for f_idx, frag in enumerate(frags):
            con = frag.GetConformer()
            for atom in frag.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    neighbours = atom.GetNeighbors()
                    assert len(neighbours) == 1
                    nei_idx = neighbours[0].GetIdx()
                    atom_idx = atom.GetIdx()

                    atom_pos = con.GetAtomPosition(atom_idx)
                    nei_pos = con.GetAtomPosition(nei_idx)

                    if atom_pos.x == float('inf'):
                        continue

                    info = {
                        'f_idx': f_idx,
                        'atom_idx': atom_idx,
                        'nei_idx': nei_idx,
                        'atom_pos': np.array([atom_pos.x, atom_pos.y, atom_pos.z]),
                        'nei_pos': np.array([nei_pos.x, nei_pos.y, nei_pos.z])
                    }

                    star_info.append(info)
        return star_info
    
    def distance(x, y):
        return sum((x - y) ** 2) ** 0.5
    
    def connectMols(mol1, mol2, atom1, atom2):
        """function copied from here https://github.com/molecularsets/moses/blob/master/moses/baselines/combinatorial.py"""
        combined = Chem.CombineMols(mol1, mol2)
        emol = Chem.EditableMol(combined)
        neighbor1_idx = atom1.GetNeighbors()[0].GetIdx()
        neighbor2_idx = atom2.GetNeighbors()[0].GetIdx()
        atom1_idx = atom1.GetIdx()
        atom2_idx = atom2.GetIdx()
        bond_order = atom2.GetBonds()[0].GetBondType()
        emol.AddBond(neighbor1_idx,
                     neighbor2_idx + mol1.GetNumAtoms(),
                     order=bond_order)
        emol.RemoveAtom(atom2_idx + mol1.GetNumAtoms())
        emol.RemoveAtom(atom1_idx)
        mol = emol.GetMol()
        return mol
    
    frags = copy.deepcopy(frags)
    while True:
        if len(frags) == 1:
            break

        star_info = get_star_info(frags)
        if len(star_info) <= 1:
            break
        
        d_mat = np.zeros((len(star_info), len(star_info))) + float('inf')
        for i in range(len(star_info)):
            for j in range(i + 1, len(star_info)):
                i_star = star_info[i]
                j_star = star_info[j]

                if i_star['f_idx'] == j_star['f_idx']:
                    continue

                dis = distance(i_star['atom_pos'], j_star['nei_pos']) + \
                    distance(i_star['nei_pos'], j_star['atom_pos'])
                
                d_mat[i][j] = dis
        
        if d_mat.min() == float('inf'):
            break

        index = np.where(d_mat == d_mat.min())
        
        fa = frags[star_info[index[0][0]]['f_idx']]
        fb = frags[star_info[index[1][0]]['f_idx']]

        fragIndex1, fragIndex2 = star_info[index[0][0]]['atom_idx'], star_info[index[0][0]]['nei_idx']
        molIndex1, molIndex2 = star_info[index[1][0]]['atom_idx'], star_info[index[1][0]]['nei_idx']

        finalMol = connectMols(fb, fa, fb.GetAtomWithIdx(molIndex1), fa.GetAtomWithIdx(fragIndex1))

        frags.remove(fa)
        frags.remove(fb)
        frags.append(finalMol)
    
    if len(frags) > 1:
        frag_weight = []
        for f in frags:
            frag_weight.append(f.GetNumAtoms())
        frag_weight = np.array(frag_weight)
        max_idx = frag_weight.argmax()
        frags = [frags[max_idx]]
    
    if len(frags) == 0:
        return None

    mol = frags[0]
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == '*':
            atom_idx = atom.GetIdx()
            m_mol = Chem.RWMol(mol)
            m_mol.ReplaceAtom(atom_idx, Chem.Atom(6))
            mol = m_mol
    
    return mol

def hypos_to_mols(token_hypos, trans_hypos, rotat_hypos):
    mols = []
    for token_hypo, trans_hypo, rotat_hypo in zip(token_hypos, trans_hypos, rotat_hypos):
        m = []
        for token, trans, rotat in zip(token_hypo, trans_hypo, rotat_hypo):
            tups = []
            for to, tr, ro in zip(token, trans, rotat):
                if to == 3: # EOS -- end of sequence
                    break
                if to in [0, 1, 2, 4, 5]: # [PAD, UNK, BOS, BOB, EOB]
                    continue
                tups.append((
                    to, 
                    grid_coords_to_real_coords(bin_to_grid_coords(tr, 28), 28, 0.5), # 2 * 6.75 / 0.5 + 1 = 28
                    bin_to_rotation_mat(ro, rotation_bin))
                )
            curr_m = connect_fragments(get_3d_frags(tups))
            m.append(curr_m)
        mols.append(m)
    return mols

preparator = MoleculePreparation()
def dock(op_results, receptor, ref):
    ref_mol = Chem.MolFromMolFile(ref)
    ref_mol = Chem.AddHs(ref_mol,addCoords=True)
    pos = ref_mol.GetConformer().GetPositions()
    x_min, x_max = min(pos[:, 0]), max(pos[:, 0])
    y_min, y_max = min(pos[:, 1]), max(pos[:, 1])
    z_min, z_max = min(pos[:, 2]), max(pos[:, 2])
    pocket_center = [int((x_min + x_max)/2), 
                     int((y_min + y_max)/2), 
                     int((z_min + z_max)/2)]
    box_size = [20, 20, 20]
    ref_center = get_mol_centroid(ref_mol)

    dock_results = []
    for mol, idx in op_results:
        try:
            con = mol.GetConformer()
            translation = trans(ref_center[0], ref_center[1], ref_center[2])
            rdMolTransforms.TransformConformer(con, translation)

            mol_block = Chem.MolToMolBlock(mol)
            preparator.prepare(obmol, freeze_bonds=None)
            pdbqt_string = preparator.write_pdbqt_string()
            v = Vina(sf_name='vina', verbosity=0)
            v.set_receptor(receptor)
            v.set_ligand_from_string(pdbqt_string)
            v.compute_vina_maps(center=pocket_center, box_size=box_size)
            score = -v.optimize()[0]
            dock_results.append((score, idx))
        except:
            dock_results.append((float('-inf'), idx))
    return dock_results

def mmff(mol):
    Chem.SanitizeMol(mol)
    molH = Chem.AddHs(mol)
    AllChem.EmbedMolecule(molH)
    while True:
        flag = AllChem.MMFFOptimizeMolecule(molH)
        if flag == 0:
            break
    molNoH = Chem.RemoveAllHs(molH)
    tmp_0, tmp_1 = GetBestRMSD(molNoH, mol)
    return molNoH

def MMFF(mols):
    op_results = []
    with ProcessPool() as pool:
        future = pool.map(mmff, mols, timeout=10.0)
        iterator = future.result()
        cnt = -1
        while True:
            cnt += 1
            try:
                result = next(iterator)
                op_results.append((result, cnt))
            except StopIteration:
                break
            except TimeoutError:
                op_results.append((None, cnt))
            except ProcessExpired:
                op_results.append((None, cnt))
            except Exception:
                op_results.append((None, cnt))
    return op_results

def GetBestRMSD(probe, ref, refConfId=-1, probConfId=-1, maps=None):
    def orginXYZ(mol, ConfId):
        mol_pos={}
        for i in range(0, mol.GetNumAtoms()):
            pos = mol.GetConformer(ConfId).GetAtomPosition(i)
            mol_pos[i] = pos
        return mol_pos
    
    def dist_2(atoma_xyz, atomb_xyz):
        dis2 = 0.0
        for i, j  in zip(atoma_xyz,atomb_xyz):
            dis2 += (i -j)**2
        return dis2
    
    def RMSD(probe, ref, amap):
        rmsd = 0.0
        atomNum = ref.GetNumAtoms() + 0.0
        for (pi,ri) in amap:
            posp = probe.pos[pi]
            posf = ref.pos[ri]
            rmsd += dist_2(posp,posf)
        rmsd = math.sqrt(rmsd/atomNum)
        return rmsd
    
    ref.pos = orginXYZ(ref, refConfId)
    probe.pos = orginXYZ(probe, probConfId)

    if not maps:
        matches = ref.GetSubstructMatches(probe, uniquify=False)
        if not matches:
            raise ValueError('mol %s does not match mol %s'%(ref.GetProp('_Name'), probe.GetProp('_Name')))
        if len(matches) > 1e6:
            warnings.warn("{} matches detected for molecule {}, this may lead to a performance slowdown.".format(len(matches), probe.GetProp('_Name')))
        maps = [list(enumerate(match)) for match in matches]
    
    bestRMS=1000.0
    bestRMSD = 1000.0
    bestMap = None
    finalMap = None
    for amap in maps:
        rms = AlignMol(probe, ref, probConfId, refConfId, atomMap=amap)
        rmsd = RMSD(probe, ref, amap)
        if rmsd < bestRMSD:
            bestRMSD = rmsd
        if rms < bestRMS:
            bestRMS = rms
            bestMap = amap
        finalMap = amap
    
    if (bestMap is not None) and (finalMap is not None) and (bestMap != finalMap):
        AlignMol(probe, ref, probConfId,refConfId, atomMap=bestMap)
    return bestRMS, bestRMSD

def hypo_to_mol(token, tran, rotat):
    tups = []
    for to, tr, ro in zip(token, tran, rotat):
        if to == 3: # EOS -- end of sequence
            break
        if to in [0, 1, 2, 4, 5]: # [PAD, UNK, BOS, BOB, EOB]
            continue
        tups.append(
            (
                to, 
                grid_coords_to_real_coords(bin_to_grid_coords(tr, 28), 28, 0.5), # 2 * 6.75 / 0.5 + 1 = 28
                bin_to_rotation_mat(ro, rotation_bin)
            )
        )
    return connect_fragments(get_3d_frags(tups))

def dock_one(mol, receptor, ref):
    ref_mol = Chem.MolFromMolFile(ref)
    ref_mol = Chem.AddHs(ref_mol, addCoords=True)
    pos = ref_mol.GetConformer().GetPositions()
    x_min, x_max = min(pos[:, 0]), max(pos[:, 0])
    y_min, y_max = min(pos[:, 1]), max(pos[:, 1])
    z_min, z_max = min(pos[:, 2]), max(pos[:, 2])
    pocket_center = [int((x_min + x_max)/2), 
                     int((y_min + y_max)/2), 
                     int((z_min + z_max)/2)]
    box_size = [20, 20, 20]
    ref_center = get_mol_centroid(ref_mol)
    
    curr_center = get_mol_centroid(mol)
    con = mol.GetConformer()
    translation = trans(ref_center[0] - curr_center[0], ref_center[1] - curr_center[1], ref_center[2] - curr_center[2])
    rdMolTransforms.TransformConformer(con, translation)

    mol_block = Chem.MolToMolBlock(mol)
    preparator.prepare(obmol, freeze_bonds=None)
    pdbqt_string = preparator.write_pdbqt_string()
    v = Vina(sf_name='vina', verbosity=0)
    v.set_receptor(receptor)
    v.set_ligand_from_string(pdbqt_string)
    v.compute_vina_maps(center=pocket_center, box_size=box_size)
    score = -v.optimize()[0]
    return score

def get_dock_one(token, tran, rotat, receptor, ref):
    mol = hypo_to_mol(token, tran, rotat)
    op_mol = mmff(mol)
    dock_result = dock_one(op_mol, receptor, ref)
    return dock_result

def get_dock_one_with_smiles(token, tran, rotat, receptor, ref):
    mol = hypo_to_mol(token, tran, rotat)
    op_mol = mmff(mol)
    dock_result = dock_one(op_mol, receptor, ref)
    smi = Chem.MolToSmiles(op_mol)
    smi = Chem.CanonSmiles(smi)
    return dock_result, smi

def get_dock_one_with_smiles_with_mol(mol, receptor, ref):
    op_mol = mmff(mol)
    dock_result = dock_one(op_mol, receptor, ref)
    smi = Chem.MolToSmiles(op_mol)
    smi = Chem.CanonSmiles(smi)
    return dock_result, smi

def get_dock_fast(tokens, trans, rotat, receptor, ref):
    dock_results = []
    with ProcessPool() as pool:
        future = pool.map(get_dock_one, tokens, trans, rotat, [receptor] * len(tokens), [ref] * len(tokens), timeout=60.0)
        iterator = future.result()
        cnt = 0
        while True:
            try:
                dock_result = next(iterator)
                dock_results.append((dock_result, cnt))
            except StopIteration:
                break
            except TimeoutError:
                dock_results.append((float('-inf'), cnt))
            except ProcessExpired:
                dock_results.append((float('-inf'), cnt))
            except Exception:
                dock_results.append((float('-inf'), cnt))
            cnt += 1
    return dock_results

def get_dock_fast_with_smiles(tokens, trans, rotat, receptor, ref):
    dock_results = []
    with ProcessPool() as pool:
        future = pool.map(get_dock_one_with_smiles, tokens, trans, rotat, [receptor] * len(tokens), [ref] * len(tokens), timeout=60.0)
        iterator = future.result()
        cnt = 0
        while True:
            try:
                result = next(iterator)
                dock_result = result[0]
                smi = result[1]
                dock_results.append((dock_result, cnt, smi))
            except StopIteration:
                break
            except TimeoutError:
                dock_results.append((float('-inf'), cnt, ''))
            except ProcessExpired:
                dock_results.append((float('-inf'), cnt, ''))
            except Exception:
                dock_results.append((float('-inf'), cnt, ''))
            cnt += 1
    return dock_results

def get_dock_fast_with_smiles_with_mol(mols, receptor, ref):
    dock_results = []
    with ProcessPool() as pool:
        future = pool.map(get_dock_one_with_smiles_with_mol, mols, [receptor] * len(mols), [ref] * len(mols), timeout=600.0)
        iterator = future.result()
        cnt = 0
        while True:
            try:
                result = next(iterator)
                dock_result = result[0]
                smi = result[1]
                dock_results.append((dock_result, cnt, smi))
            except StopIteration:
                break
            except TimeoutError:
                dock_results.append((float('-inf'), cnt, ''))
            except ProcessExpired:
                dock_results.append((float('-inf'), cnt, ''))
            except Exception:
                dock_results.append((float('-inf'), cnt, ''))
            cnt += 1
    return dock_results