import random
import numpy as np
import torch
from pytransform3d.rotations import quaternion_from_matrix
from math import ceil
from .utils.shape_utils import get_atom_stamp, get_shape, get_shape_patches, time_shift, get_grid_coords, get_rotation_bins
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import pickle
from typing import Optional

class ShapePretrainingTaskNoRegression:
    def __init__(self,
                 grid_resolution=0.5,
                 max_dist_stamp=4.0,
                 max_dist=6.75,
                 rotation_bin=24,
                 max_translation=1.0,
                 max_seq_len=20,
                 patch_size=4,
                 delta_input=False,
                 teacher_force_inference=False,
                 shape_noise_mu=0.0,
                 shape_noise_sigma=0.0,
                 rotation_bin_direction=11,
                 rotation_bin_angle=24,
                 vocab_path=None):
        
        self._grid_resolution = grid_resolution
        self._max_dist_stamp = max_dist_stamp
        self._max_dist = max_dist
        self._rotation_bin = rotation_bin # for data augmentation
        self._max_translation = max_translation
        self._atom_stamp = get_atom_stamp(grid_resolution, max_dist_stamp)
        self._max_seq_len = max_seq_len
        self._patch_size = patch_size
        self._delta_input = delta_input
        self._teacher_force_inference = teacher_force_inference
        self._box_size = ceil(2 * max_dist // grid_resolution + 1)
        self._rotation_bins = get_rotation_bins(rotation_bin_direction, rotation_bin_angle) # for mapping a fragment rotation of a bin
        self._shape_noise_mu = shape_noise_mu
        self._shape_noise_sigma = shape_noise_sigma
        
        # Load vocabulary if path provided
        if vocab_path:
            import pickle
            with open(vocab_path, 'rb') as f:
                self._vocab = pickle.load(f)
                self._common_special_tokens = {
                    'bos': self._vocab['BOS'][2],
                    'eos': self._vocab['EOS'][2],
                    'bob': self._vocab['BOB'][2],
                    'eob': self._vocab['EOB'][2],
                    'pad': self._vocab['PAD'][2],
                    'unk': self._vocab['UNK'][2]
                }

    def process_samples(self, samples, training=True):
        """Process a batch of samples."""
        self._training = training
        return self._collate(samples)

    def _collate(self, samples):
        shape = []
        shape_patches = []
        seq_len = []
        input_frag_idx = []
        input_frag_idx_mask = []
        input_frag_trans = []
        input_frag_trans_mask = []
        input_frag_r_mat = []
        input_frag_r_mat_mask = []

        if not hasattr(self, '_infering') or (hasattr(self, '_infering') and self._teacher_force_inference):
            for sample in samples:
                if len(sample['tree_list']) == 0:
                    raise Exception('No tree list!')
                
                curr_atom_stamp = get_atom_stamp(self._grid_resolution, 
                                                          self._max_dist_stamp)
                curr_shape = get_shape(sample['mol'], 
                                     curr_atom_stamp, 
                                     self._grid_resolution, 
                                     self._max_dist)
                shape.append(curr_shape)

                # Calculate patches of the shape
                curr_shape_patches = get_shape_patches(curr_shape, self._patch_size)
                curr_shape_patches = curr_shape_patches.reshape(curr_shape.shape[0] // self._patch_size,
                                                              curr_shape.shape[0] // self._patch_size,
                                                              curr_shape.shape[0] // self._patch_size, -1)
                curr_shape_patches = curr_shape_patches.reshape(-1, self._patch_size**3)
                shape_patches.append(curr_shape_patches)

                # Create training sequence
                #print(f"Amount of possible trees: {len(sample['tree_list'])}")
                random_tree = sample['tree_list'][0]
                
                curr_idx = [] # fragment idx in vocab
                curr_idx_mask = []
                curr_trans = [] # fragment centroid position
                curr_trans_mask = []
                curr_r_mat = [] # fragment roation in quaternion
                curr_r_mat_mask = []
                curr_seq_len = 0

                # Process each unit in the tree
                for unit in random_tree:
                    curr_seq_len += 1
                    # Not a special token
                    if unit[0] not in ['BOS', 'EOS', 'BOB', 'EOB']:
                        curr_frag = sample['frag_list'][unit[0]]
                        curr_idx.append(curr_frag['vocab_id'])
                        curr_idx_mask.append(1)
                        # Known fragment
                        if curr_frag['vocab_id'] != self._vocab['UNK'][2]:
                            curr_trans_grid_coords = get_grid_coords(curr_frag['trans_vec'], 
                                                                   self._max_dist, 
                                                                   self._grid_resolution)

                            # Map position to bin
                            if (curr_trans_grid_coords[0] < 0 or curr_trans_grid_coords[0] >= self._box_size) or \
                               (curr_trans_grid_coords[1] < 0 or curr_trans_grid_coords[1] >= self._box_size) or \
                               (curr_trans_grid_coords[2] < 0 or curr_trans_grid_coords[2] >= self._box_size):
                                curr_trans.append(1) # out of the box
                                curr_trans_mask.append(1)
                            else:
                                pos_bin = curr_trans_grid_coords[0] * self._box_size**2 + \
                                        curr_trans_grid_coords[1] * self._box_size + \
                                        curr_trans_grid_coords[2] + 2 # plus 2, because 0 for not a fragment, 1 for out of box
                                curr_trans.append(pos_bin)
                                curr_trans_mask.append(1)

                            # Map rotation to bin
                            tmp = self._rotation_bins - curr_frag['rotate_mat']
                            tmp = abs(tmp).sum(axis=-1).sum(axis=-1)
                            min_index = np.argmin(tmp)
                            curr_r_mat.append(min_index + 1) # 0 for not a fragment
                            curr_r_mat_mask.append(1)
                        # Unknown fragment
                        else:
                            curr_trans.append(0)
                            curr_trans_mask.append(0)
                            curr_r_mat.append(0)
                            curr_r_mat_mask.append(0)
                    # Special tokens
                    else:
                        curr_idx.append(self._vocab[unit[0]][2])
                        curr_idx_mask.append(1)
                        curr_trans.append(0)
                        curr_trans_mask.append(0)
                        curr_r_mat.append(0)
                        curr_r_mat_mask.append(0)

                # Create shifted sequence
                input_curr_idx, _ = time_shift(curr_idx)
                input_curr_idx_mask, _ = time_shift(curr_idx_mask)

                # Create delta translation
                if self._delta_input:
                    delta = []
                    pre_trans = np.zeros(1)
                    for tr, tr_m in zip(curr_trans, curr_trans_mask):
                        if tr_m != 0:
                            delta.append(tr - pre_trans)
                            pre_trans = tr
                        else:
                            delta.append(np.zeros(1))
                    curr_trans = delta

                input_curr_trans, _ = time_shift(curr_trans)
                input_curr_trans_mask, _ = time_shift(curr_trans_mask)

                input_curr_r_mat, _ = time_shift(curr_r_mat)
                input_curr_r_mat_mask, _ = time_shift(curr_r_mat_mask)

                curr_seq_len -= 1

                # Create truncated sequence
                if self._training:
                    input_curr_idx = input_curr_idx[:self._max_seq_len]
                    input_curr_idx_mask = input_curr_idx_mask[:self._max_seq_len]

                    input_curr_trans = input_curr_trans[:self._max_seq_len]
                    input_curr_trans_mask = input_curr_trans_mask[:self._max_seq_len]

                    input_curr_r_mat = input_curr_r_mat[:self._max_seq_len]
                    input_curr_r_mat_mask = input_curr_r_mat_mask[:self._max_seq_len]

                    curr_seq_len = min(curr_seq_len, self._max_seq_len)

                input_frag_idx.append(np.array(input_curr_idx))
                input_frag_idx_mask.append(np.array(input_curr_idx_mask))

                input_frag_trans.append(np.array(input_curr_trans))
                input_frag_trans_mask.append(np.array(input_curr_trans_mask))

                input_frag_r_mat.append(np.array(input_curr_r_mat))
                input_frag_r_mat_mask.append(np.array(input_curr_r_mat_mask))

                seq_len.append(curr_seq_len)

            # Create padded sequence
            max_seq_len = max(seq_len)
            for i in range(len(input_frag_idx)):
                pad_input_frag_idx = np.zeros(max_seq_len) # pad 0
                pad_input_frag_idx[:len(input_frag_idx[i])] = input_frag_idx[i]
                pad_input_frag_idx_mask = np.zeros(max_seq_len) # pad 0
                pad_input_frag_idx_mask[:len(input_frag_idx_mask[i])] = input_frag_idx_mask[i]

                pad_input_frag_trans = np.zeros((max_seq_len,)) # pad 0
                pad_input_frag_trans[:len(input_frag_trans[i])] = input_frag_trans[i]
                pad_input_frag_trans_mask = np.zeros(max_seq_len) # pad 0
                pad_input_frag_trans_mask[:len(input_frag_trans_mask[i])] = input_frag_trans_mask[i]

                pad_input_frag_r_mat = np.zeros((max_seq_len, )) # pad 0
                pad_input_frag_r_mat[:len(input_frag_r_mat[i])] = input_frag_r_mat[i]
                pad_input_frag_r_mat_mask = np.zeros(max_seq_len) # pad 0
                pad_input_frag_r_mat_mask[:len(input_frag_r_mat_mask[i])] = input_frag_r_mat_mask[i]

                input_frag_idx[i] = pad_input_frag_idx
                input_frag_idx_mask[i] = pad_input_frag_idx_mask
     
                input_frag_trans[i] = pad_input_frag_trans
                input_frag_trans_mask[i] = pad_input_frag_trans_mask

                input_frag_r_mat[i] = pad_input_frag_r_mat
                input_frag_r_mat_mask[i] = pad_input_frag_r_mat_mask

            shape = torch.tensor(np.array(shape), dtype=torch.long)
            shape_patches = torch.tensor(np.array(shape_patches), dtype=torch.float)

            input_frag_idx = torch.tensor(np.array(input_frag_idx), dtype=torch.long)
            input_frag_idx_mask = torch.tensor(np.array(input_frag_idx_mask), dtype=torch.float)

            input_frag_trans = torch.tensor(np.array(input_frag_trans), dtype=torch.long)
            input_frag_trans_mask = torch.tensor(np.array(input_frag_trans_mask), dtype=torch.float)


            input_frag_r_mat = torch.tensor(np.array(input_frag_r_mat), dtype=torch.long)
            input_frag_r_mat_mask = torch.tensor(np.array(input_frag_r_mat_mask), dtype=torch.float)

            if not hasattr(self, '_infering'):
                batch = {
                    'net_input': {
                        'shape': shape,
                        'shape_patches': shape_patches,
                        'input_frag_idx': input_frag_idx,
                        'input_frag_idx_mask': input_frag_idx_mask,
                        'input_frag_trans': input_frag_trans, 
                        'input_frag_trans_mask': input_frag_trans_mask,
                        'input_frag_r_mat': input_frag_r_mat,
                        'input_frag_r_mat_mask': input_frag_r_mat_mask,
                    }

                }
            elif hasattr(self, '_infering') and self._teacher_force_inference:
                net_input = {
                    'encoder': (shape_patches,),
                    'decoder': ((input_frag_idx, input_frag_trans, input_frag_r_mat, True),),
                }
                batch = {'net_input': net_input}
            else:
                raise Exception('please make sure [self._infering is False] or [(self._infering and self._teacher_force_inference) is True]!')
            '''
            print("\nFragment Sequences:")
            print("Input  idx:", batch['net_input']['input_frag_idx'][0].tolist())
            
            print("\nActual Fragments:")
            print("Input  seq:", end=" ")
            '''
            for idx in batch['net_input']['input_frag_idx'][0].tolist():
                fragment_name = "UNK"
                for token, (smiles, _, token_id) in self._vocab.items():
                    if token_id == idx:
                        fragment_name = token
                        break
                #print(f"{fragment_name}", end=" ")
            
            '''
            
            print("\nTranslations:")
            print("Input  trans:", batch['net_input']['input_frag_trans'][0].tolist())
            
            print("\nRotations:")
            print("Input  r_mat:", batch['net_input']['input_frag_r_mat'][0].tolist())
            
            print("\nMasks:")
            print("Input  idx mask:", batch['net_input']['input_frag_idx_mask'][0].tolist())
            print("Input  trans mask:", batch['net_input']['input_frag_trans_mask'][0].tolist())
            print("Input  r_mat mask:", batch['net_input']['input_frag_r_mat_mask'][0].tolist())

            # Simple assertions for shape consistency
            '''
            batch_size = batch['net_input']['shape'].size(0)
            for batch_idx in range(1, batch_size):
                assert torch.equal(batch['net_input']['shape'][batch_idx], 
                                 batch['net_input']['shape'][0]), \
                    f"Shape mismatch in batch {batch_idx}"
                assert torch.equal(batch['net_input']['shape_patches'][batch_idx], 
                                 batch['net_input']['shape_patches'][0]), \
                    f"Shape patches mismatch in batch {batch_idx}"

            return batch

def visualize_batch(batch, vocab=None, save_dir='visualizations'):
    """Visualize both input and output batch data and save as PNGs."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import seaborn as sns
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create two figures side by side for input and output
    fig, ((ax1_in, ax1_out), (ax2_in, ax2_out), (ax3_in, ax3_out)) = plt.subplots(3, 2, figsize=(20, 24))
    
    # 1. Visualize fragment sequences (input and output)
    def plot_sequence(ax, frags, mask, title):
        seq_data = []
        for i, (frag, m) in enumerate(zip(frags, mask)):
            if m == 1:
                if vocab:
                    frag_smiles = "UNK"
                    for k, v in vocab.items():
                        if v[2] == frag.item():
                            frag_smiles = k
                            break
                    seq_data.append(f"{i}: {frag_smiles}")
                else:
                    seq_data.append(f"{i}: Frag_{frag.item()}")
        ax.text(0.1, 0.5, '\n'.join(seq_data), fontsize=8)
        ax.axis('off')
        ax.set_title(title)
    
    plot_sequence(ax1_in, 
                 batch['net_input']['input_frag_idx'][0],
                 batch['net_input']['input_frag_idx_mask'][0],
                 'Input Fragment Sequence')
    
    # 2. Visualize translation coordinates (input and output)
    def plot_translations(ax, trans, mask, title):
        box_size = int(round(trans.max().item() ** (1/3)))
        coords = []
        for t, m in zip(trans, mask):
            if m == 1 and t > 1:  # Skip padding and out-of-box
                t = t.item() - 2  # Adjust for special tokens
                x = t // (box_size**2)
                y = (t % (box_size**2)) // box_size
                z = t % box_size
                coords.append((x, y, z))
        
        if coords:
            coords = np.array(coords)
            scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                               c=range(len(coords)), cmap='viridis')
            plt.colorbar(scatter, ax=ax, label='Fragment Order')
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    ax2_in = fig.add_subplot(323, projection='3d')
    plot_translations(ax2_in,
                     batch['net_input']['input_frag_trans'][0],
                     batch['net_input']['input_frag_trans_mask'][0],
                     'Input Fragment Positions')
    
   
    
    # 3. Visualize masks (input and output)
    def plot_masks(ax, idx_mask, trans_mask, rot_mask, title):
        masks = {
            'frag': idx_mask,
            'trans': trans_mask,
            'rot': rot_mask
        }
        mask_matrix = np.stack([m.numpy() for m in masks.values()])
        sns.heatmap(mask_matrix, ax=ax, cmap='Blues',
                   xticklabels=range(mask_matrix.shape[1]),
                   yticklabels=masks.keys())
        ax.set_title(title)
    
    plot_masks(ax3_in,
              batch['net_input']['input_frag_idx_mask'][0],
              batch['net_input']['input_frag_trans_mask'][0],
              batch['net_input']['input_frag_r_mat_mask'][0],
              'Input Attention Masks')
    
    
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'input_output_comparison.png'))
    plt.close(fig)
    
    # Additional shape-related plots in a separate figure
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Shape middle slice
    shape = batch['net_input']['shape'][0]
    middle_slice = shape[shape.shape[0]//2, :, :]
    sns.heatmap(middle_slice, ax=ax1, cmap='viridis')
    ax1.set_title('3D Shape (Middle Slice)')
    
    # Shape patches distribution
    patches = batch['net_input']['shape_patches'][0]
    patch_means = patches.mean(dim=1)
    sns.histplot(patch_means.numpy(), ax=ax2, bins=30)
    ax2.set_title('Patch Value Distribution')
    
    plt.tight_layout()
    fig2.savefig(os.path.join(save_dir, 'shape_visualization.png'))
    plt.close(fig2)
    
    print(f"Saved visualizations to {save_dir}/")
    return save_dir








import torch
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
import pickle
from copy import deepcopy
import re
from collections import defaultdict
from .fragmenizer import BRICS_RING_R_Fragmenizer
import numpy as np
import rmsd
import os
from datetime import datetime
import time 
from .utils import get_tree, tree_linearize, get_atom_mapping_between_frag_and_surrogate
import torch.nn as nn

PLACE_HOLDER_ATOM = 80     
class SingleSmilesProcessor:
    def __init__(self, vocab_path):
        # Load vocabulary
        with open(vocab_path, 'rb') as fr:
            self._vocab = pickle.load(fr)
        self.fragmenizer = BRICS_RING_R_Fragmenizer()
    
    def generate_3d_structure(self, smiles: str) -> Chem.Mol:
        """Generate a 3D conformer for a given SMILES."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, useRandomCoords=True) != 0:
            raise ValueError(f"Failed to generate 3D conformer")

        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
        except Exception as e:
            print(f"MMFF optimization failed ({e})")
        return mol

    def process_smiles(self, smiles):
        """Process a single SMILES string through the entire pipeline."""
        # Generate 3D structure
        mol = self.generate_3d_structure(smiles)
        
        # Fragment the molecule
        frags, _ = self.fragmenizer.fragmenize(mol)
        frags = Chem.GetMolFrags(frags, asMols=True)
        
        if not frags:
            print(f"Warning: No fragments generated for SMILES {smiles}")
            return {'tree_list': [], 'mol': mol, 'frag_list': []}
            
        
        # Process fragments
        frags_list = []
        start_frag_idx = []
        cluster_dict = defaultdict(list)
        
        # Calculate centroid once
        mol_conformer = mol.GetConformer(-1)
        centroid = np.mean(mol_conformer.GetPositions(), axis=0)
        translation_matrix = np.eye(4)
        translation_matrix[:3, 3] = -centroid
        rdMolTransforms.TransformConformer(mol_conformer, translation_matrix)
        
        for idx, frag in enumerate(frags):
            # Cache conformer and positions
            frag_conformer = frag.GetConformer(-1)
            frag_positions = frag_conformer.GetPositions()
            
            # Calculate canonical SMILES once
            frag_smi = Chem.CanonSmiles(re.sub(r'\[\d+\*\]', '[*]', Chem.MolToSmiles(frag)))
            
            # Handle unknown fragments
            if frag_smi not in self._vocab:
                frags_list.append((self._vocab['UNK'][2], 'UNK', frag_smi, None, None, None, None))
                for atom in frag.GetAtoms():
                    if atom.GetSymbol() == '*':
                        cluster_dict[atom.GetSmarts()].append(idx)
                if frag_smi.count('*') == 1:
                    start_frag_idx.append(idx)
                continue
            
            # Process known fragments
            frag_idx = self._vocab[frag_smi][2]
            frag_center = np.mean(frag_positions, axis=0)
            v_frag = self._vocab[frag_smi][0]
            trans_vec = frag_center - centroid
            
            # Create surrogate fragments once
            c_frag = deepcopy(frag)
            c_frag_conformer = c_frag.GetConformer(-1)
            center = np.mean(c_frag_conformer.GetPositions(), axis=0)
            translation = np.eye(4)
            translation[:3, 3] = -center
            rdMolTransforms.TransformConformer(c_frag_conformer, translation)
            
            # Cache surrogate fragments
            c_frag_sur = self._get_surrogate_fragment(c_frag)
            v_frag_sur = self._get_surrogate_fragment(v_frag)
            
            # Get points for alignment between fragments (using cached conformers)
            points1, points2, c2v_atom_mapping, v2c_atom_mapping = self._get_alignment_points(
                c_frag_sur, v_frag_sur
            )
            
            # Calculate rotation matrix and RMSD
            r_matrix = rmsd.kabsch(points2, points1)
            r_v_points = np.dot(points2, r_matrix)
            diff = rmsd.rmsd(r_v_points, points1)
            r_matrix = r_matrix.T
            
            # Cache atom mappings
            c_frag2surro_atom_mapping, c_surro2frag_atom_mapping = get_atom_mapping_between_frag_and_surrogate(c_frag, c_frag_sur)
            v_frag2surro_atom_mapping, v_surro2frag_atom_mapping = get_atom_mapping_between_frag_and_surrogate(v_frag, v_frag_sur)
            v_frag_attach_mapping = self._vocab[frag_smi][1]
            
            # Process attachment points
            c_frag_attach_mapping = self._process_attachments(
                c_frag, c_frag2surro_atom_mapping, c2v_atom_mapping,
                v_surro2frag_atom_mapping, v_frag, v_frag_attach_mapping
            )
            
            frags_list.append((frag_idx, frag_smi, frag_smi, trans_vec, r_matrix, diff, c_frag_attach_mapping))
            
            # Update cluster dict
            for atom in frag.GetAtoms():
                if atom.GetSymbol() == '*':
                    cluster_dict[atom.GetSmarts()].append(idx)
            if frag_smi.count('*') == 1:
                start_frag_idx.append(idx)
        
        # Generate only one linear tree
        adj_dict = defaultdict(str)
        for key, indices in cluster_dict.items():
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    adj_dict[(indices[i], indices[j])] = key
                    adj_dict[(indices[j], indices[i])] = key
        
        # Process only the first starting fragment
        linear_trees = []
        if start_frag_idx:
            start_idx = start_frag_idx[0]
            tree = get_tree(adj_dict, start_idx, [], len(frags))
            linear_tree = []
            tree_linearize(tree, linear_tree)
            new_linear_tree = self._process_tree(linear_tree, father_stack=[], b_stack=[], frags_list=frags_list, adj_dict=adj_dict)
            linear_trees.append(new_linear_tree)
        
        # Create processed fragments list
        processed_frags = [
            {
                'vocab_id': frag_idx,
                'vocab_key': vocab_key,
                'frag_smi': frag_smi,
                'trans_vec': trans_vec,
                'rotate_mat': rot_mat
            }
            for frag_idx, vocab_key, frag_smi, trans_vec, rot_mat, _, _ in frags_list
        ]
        
        return {
            'mol': mol,
            'frag_list': processed_frags,
            'tree_list': linear_trees
        }

    def _get_surrogate_fragment(self, frag):
        """Helper method to create surrogate fragment with placeholder atoms."""
        surrogate = deepcopy(frag)
        m_surrogate = Chem.RWMol(surrogate)
        for atom in m_surrogate.GetAtoms():
            if atom.GetSymbol() == '*':
                m_surrogate.ReplaceAtom(atom.GetIdx(), Chem.Atom(PLACE_HOLDER_ATOM))
        Chem.SanitizeMol(m_surrogate)
        return m_surrogate

    def _process_attachments(self, c_frag, c_frag2surro_atom_mapping, c2v_atom_mapping,
                           v_surro2frag_atom_mapping, v_frag, v_frag_attach_mapping):
        """Helper method to process attachment points."""
        c_frag_attach_mapping = {}
        for c_atom in c_frag.GetAtoms():
            if c_atom.GetSymbol() == '*':
                c_smarts = c_atom.GetSmarts()
                c_surro_idx = c_frag2surro_atom_mapping[c_atom.GetIdx()]
                v_surro_idx = c2v_atom_mapping[c_surro_idx]
                v_frag_idx = v_surro2frag_atom_mapping[v_surro_idx]
                v_smarts = v_frag.GetAtomWithIdx(v_frag_idx).GetSmarts()
                attach_idx = v_frag_attach_mapping[v_smarts]
                c_frag_attach_mapping[c_smarts] = attach_idx
                c_frag_attach_mapping[attach_idx] = c_smarts
        return c_frag_attach_mapping

    def _process_tree(self, linear_tree, father_stack, b_stack, frags_list, adj_dict):
        """Helper method to process tree with attachment points."""
        new_linear_tree = [('BOS', (None, None))]
        
        for node in linear_tree:
            if node not in ['b', 'e']:
                if len(father_stack) == 0:
                    new_linear_tree.append((node, (None, None)))
                else:
                    father = father_stack[-1]
                    attach = adj_dict[(father, node)]
                    father_attach = frags_list[father][-1][attach] if frags_list[father][-1] else None
                    son_attach = frags_list[node][-1][attach] if frags_list[node][-1] else None
                    new_linear_tree.append((node, (father_attach, son_attach)))
                father_stack.append(node)
            elif node == 'b':
                b_stack.append(len(father_stack))
                new_linear_tree.append(('BOB', (None, None)))
            elif node == 'e':
                father_stack = father_stack[:b_stack.pop()]
                new_linear_tree.append(('EOB', (None, None)))
        
        new_linear_tree.append(('EOS', (None, None)))
        return new_linear_tree

    def _get_alignment_points(self, c_frag_sur, v_frag_sur):
        """Get points for alignment between fragments."""
        points1 = []
        points2 = []
        c2v_atom_mapping = {}
        v2c_atom_mapping = {}
        
        # Get conformers
        conf1 = c_frag_sur.GetConformer()
        conf2 = v_frag_sur.GetConformer()
        
        # First match attachment points and their neighbors
        matched_frag2_indices = set()
        for idx1, atom1 in enumerate(c_frag_sur.GetAtoms()):
            # Don't skip attachment points or their neighbors
            if atom1.GetSymbol() == '*' or any(n.GetSymbol() == '*' for n in atom1.GetNeighbors()):
                pos1 = conf1.GetAtomPosition(idx1)
                
                # Find closest matching atom in frag2
                best_idx2 = None
                best_dist = float('inf')
                for idx2, atom2 in enumerate(v_frag_sur.GetAtoms()):
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
                    c2v_atom_mapping[idx1] = best_idx2
                    v2c_atom_mapping[best_idx2] = idx1
                    matched_frag2_indices.add(best_idx2)
        
        # Then match remaining atoms (except hydrogens)
        for idx1, atom1 in enumerate(c_frag_sur.GetAtoms()):
            if (idx1 in c2v_atom_mapping or  # Skip already matched
                atom1.GetSymbol() == 'H'):  # Skip regular hydrogens
                continue
                
            pos1 = conf1.GetAtomPosition(idx1)
            
            # Find closest matching atom in frag2
            best_idx2 = None
            best_dist = float('inf')
            for idx2, atom2 in enumerate(v_frag_sur.GetAtoms()):
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
                c2v_atom_mapping[idx1] = best_idx2
                v2c_atom_mapping[best_idx2] = idx1
                matched_frag2_indices.add(best_idx2)
        
        if not points1 or not points2:
            raise ValueError("No matching atoms found between fragments")
        
        return np.array(points1), np.array(points2), c2v_atom_mapping, v2c_atom_mapping









import enum
from collections.abc import Sequence
from typing import TypedDict

import torch

from synformer.chem.fpindex import FingerprintIndex
from synformer.chem.mol import Molecule
from synformer.chem.reaction import Reaction
from synformer.chem.stack import Stack
from synformer.utils.image import draw_text, make_grid
vocab_path = '/home/luost_local/sdivita/zerosynth/chemprojector/data/fragment/preparation/fragmenizer/vocab.pkl'


class TokenType(enum.IntEnum):
    END = 0
    START = 1
    REACTION = 2
    REACTANT = 3


class ProjectionData(TypedDict, total=False):
    # Original fields
    atoms: torch.Tensor
    bonds: torch.Tensor
    atom_padding_mask: torch.Tensor
    smiles: torch.Tensor
    token_types: torch.Tensor
    rxn_indices: torch.Tensor
    reactant_fps: torch.Tensor
    token_padding_mask: torch.Tensor
    mol_seq: Sequence[Molecule]
    rxn_seq: Sequence[Reaction | None]
    # DESERT fields without prefix
    input_frag_idx: torch.Tensor
    input_frag_idx_mask: torch.Tensor
    input_frag_trans: torch.Tensor
    input_frag_trans_mask: torch.Tensor
    input_frag_r_mat: torch.Tensor
    input_frag_r_mat_mask: torch.Tensor
    shape: torch.Tensor
    shape_patches: torch.Tensor


class ProjectionBatch(TypedDict, total=False):
    # Encoder
    atoms: torch.Tensor
    bonds: torch.Tensor
    atom_padding_mask: torch.Tensor
    smiles: torch.Tensor
    # Decoder
    token_types: torch.Tensor
    rxn_indices: torch.Tensor
    reactant_fps: torch.Tensor
    token_padding_mask: torch.Tensor
    # Auxilliary
    mol_seq: Sequence[Sequence[Molecule]]
    rxn_seq: Sequence[Sequence[Reaction | None]]


def featurize_stack_actions(
    mol_idx_seq: Sequence[int | None],
    rxn_idx_seq: Sequence[int | None],
    end_token: bool,
    fpindex: FingerprintIndex,
) -> dict[str, torch.Tensor]:
    seq_len = len(mol_idx_seq) + 1  # Plus START token
    if end_token:
        seq_len += 1
    fp_dim = fpindex.fp_option.dim
    feats = {
        "token_types": torch.zeros([seq_len], dtype=torch.long),
        "rxn_indices": torch.zeros([seq_len], dtype=torch.long),
        "reactant_fps": torch.zeros([seq_len, fp_dim], dtype=torch.float),
        "token_padding_mask": torch.zeros([seq_len], dtype=torch.bool),
    }
    feats["token_types"][0] = TokenType.START
    for i, (mol_idx, rxn_idx) in enumerate(zip(mol_idx_seq, rxn_idx_seq), start=1):
        if rxn_idx is not None:
            feats["token_types"][i] = TokenType.REACTION
            feats["rxn_indices"][i] = rxn_idx
        elif mol_idx is not None:
            feats["token_types"][i] = TokenType.REACTANT
            _, mol_fp = fpindex[mol_idx]
            feats["reactant_fps"][i] = torch.from_numpy(mol_fp)
    return feats


def featurize_stack(stack: Stack, end_token: bool, fpindex: FingerprintIndex) -> dict[str, torch.Tensor]:
    return featurize_stack_actions(
        mol_idx_seq=stack.get_mol_idx_seq(),
        rxn_idx_seq=stack.get_rxn_idx_seq(),
        end_token=end_token,
        fpindex=fpindex,
    )


def create_data(
    product: Molecule,
    mol_seq: Sequence[Molecule],
    mol_idx_seq: Sequence[int | None],
    rxn_seq: Sequence[Reaction | None],
    rxn_idx_seq: Sequence[int | None],
    fpindex: FingerprintIndex,
    pretrained_model_path: Optional[str] = None,
):
    atom_f, bond_f = product.featurize_simple()
    stack_feats = featurize_stack_actions(
        mol_idx_seq=mol_idx_seq,
        rxn_idx_seq=rxn_idx_seq,
        end_token=True,
        fpindex=fpindex,
    )

    processor = SingleSmilesProcessor(vocab_path)
    
    processed_data = processor.process_smiles(product._smiles)
    
    encoder = ShapePretrainingTaskNoRegression(vocab_path=vocab_path)
    desert_batch = encoder.process_samples([processed_data], training=True)
    '''
    if pretrained_model_path:
        print("\nLoading pretrained encoder...")
        pretrained_encoder = ShapeEncoder.from_pretrained(pretrained_model_path)
        with torch.no_grad():
            encoder_output = pretrained_encoder(desert_batch['net_input']['shape_patches'])
        desert_batch['net_input']['shape_embeddings'] = encoder_output[0]
    
    desert_data = {
        k: (v[0] if isinstance(v, torch.Tensor) and v.dim() > 0 else v)
        for k, v in desert_batch["net_input"].items()
    }
    '''
    '''
    print("\nAfter extraction, desert_data keys and shapes:")
    for k, v in desert_data.items():
        if isinstance(v, (torch.Tensor, np.ndarray)):
            print(f"{k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"{k}: type={type(v)}")
    '''
    data: "ProjectionData" = {
        "mol_seq": mol_seq,
        "rxn_seq": rxn_seq,
        "atoms": atom_f,
        "bonds": bond_f,
        "smiles": product.tokenize_csmiles(),
        "atom_padding_mask": torch.zeros([atom_f.size(0)], dtype=torch.bool),
        "token_types": stack_feats["token_types"],
        "rxn_indices": stack_feats["rxn_indices"],
        "reactant_fps": stack_feats["reactant_fps"],
        "token_padding_mask": stack_feats["token_padding_mask"],
        "input_frag_idx": desert_data["input_frag_idx"],
        "input_frag_idx_mask": desert_data["input_frag_idx_mask"],
        "input_frag_trans": desert_data["input_frag_trans"],
        "input_frag_trans_mask": desert_data["input_frag_trans_mask"],
        "input_frag_r_mat": desert_data["input_frag_r_mat"],
        "input_frag_r_mat_mask": desert_data["input_frag_r_mat_mask"],
        "shape": desert_data["shape"],
        "shape_patches": desert_data["shape_patches"]
    }

    return data


def draw_data(data: ProjectionData):
    im_list = [draw_text("START")]
    for m, r in zip(data["mol_seq"], data["rxn_seq"]):
        if r is not None:
            im_list.append(r.draw())
        else:
            im_list.append(m.draw())
    im_list.append(draw_text("END"))
    return make_grid(im_list)


def draw_batch(batch: ProjectionBatch):
    bsz = len(batch["mol_seq"])
    for b in range(bsz):
        im_list = [draw_text("START")]
        for m, r in zip(batch["mol_seq"][b], batch["rxn_seq"][b]):
            if r is not None:
                im_list.append(r.draw())
            else:
                im_list.append(m.draw())
        im_list.append(draw_text("END"))
        yield make_grid(im_list)
