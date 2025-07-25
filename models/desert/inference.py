import torch
import torch.nn as nn
import numpy as np
import pickle
import os
import argparse
import copy
import math
from rdkit import Chem
from rdkit.Chem import AllChem
import subprocess 
import tempfile 
import uuid 
import shutil
import dataclasses 
import pandas as pd
from rdkit.Chem import QED 

from synformer.data.common import process_smiles
from synformer.models.encoder.shape import ShapePretrainingEncoder
from synformer.models.desert.fulldecoder import ShapePretrainingDecoderIterativeNoRegression
from synformer.data.utils.shape_utils import bin_to_grid_coords, grid_coords_to_real_coords, bin_to_rotation_mat, get_3d_frags, connect_fragments, get_atom_stamp, get_shape, get_rotation_bins, get_shape_patches, ATOM_RADIUS, ATOMIC_NUMBER, ATOMIC_NUMBER_REVERSE
from synformer.models.transformer.positional_encoding import PositionalEncoding 
D_MODEL = 1024 #This should be num_heads * head_dim = 8 * 128 = 1024

SHAPE_UTILS_VOCAB = None
SHAPE_UTILS_VOCAB_R = None
SHAPE_UTILS_ROTATION_BIN = None

def load_desert_encoder(model_path, device='cuda'):
    """Load a pre-trained DESERT encoder."""
    print(f"Loading encoder from {model_path}")

    encoder = ShapePretrainingEncoder(
        patch_size=4,
        num_layers=12,
        d_model=D_MODEL,
        n_head=8,
        dim_feedforward=4096,
        dropout=0.1,
        attention_dropout=0.1,
        activation='relu',
        learn_pos=True
    )


    dummy_embed = nn.Embedding(2, D_MODEL) # Vocab size 2, embedding dim D_MODEL
    encoder.build(
        embed=dummy_embed,
        special_tokens={'pad': 0}
    )

    state_dict = torch.load(model_path, map_location=device)
    if 'model' in state_dict:
        state_dict = state_dict['model']
            
    encoder_state_dict = {k.replace('_encoder.', ''): v for k, v in state_dict.items() 
                                if k.startswith('_encoder.')}
    
    filtered_encoder_state_dict = {}
    for k, v in encoder_state_dict.items():
        if k in encoder.state_dict():
            filtered_encoder_state_dict[k] = v
        elif k.startswith('_patch_ffn._fc') and k.replace('_patch_ffn.','_patch_ffn.ffn.') in encoder.state_dict() : 
             filtered_encoder_state_dict[k.replace('_patch_ffn.','_patch_ffn.ffn.')] = v

    if not filtered_encoder_state_dict:
        print("Warning: Encoder state dict is empty after filtering. Check prefix and keys.")
    else:
        missing_keys, unexpected_keys = encoder.load_state_dict(filtered_encoder_state_dict, strict=False)
        if missing_keys:
            print(f"Encoder missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Encoder unexpected keys: {unexpected_keys}")

    encoder.to(device)
    encoder.eval()
    return encoder

def load_desert_decoder(model_path, device='cuda'):
    """Load a pre-trained DESERT decoder."""
    print(f"Loading decoder from {model_path}")

    vocab_path = os.path.join(os.path.dirname(model_path), "vocab.pkl")
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    tgt_vocab_size = len(vocab)
    tgt_special_tokens = {'pad': vocab['PAD'][2]}

    tgt_embed = nn.Embedding(tgt_vocab_size, D_MODEL, padding_idx=tgt_special_tokens['pad'])
    
    class IterativeBlock: 
        def __init__(self, config):
            self.config = config
            self._mode = 'train'
            self.embed = None
            self.special_tokens = None
            self.trans_vocab_size = None
            self.rot_vocab_size = None

        def build(self, embed, special_tokens, trans_vocab_size, rot_vocab_size):
            self.embed = embed
            self.special_tokens = special_tokens
            self.trans_vocab_size = trans_vocab_size
            self.rot_vocab_size = rot_vocab_size
            
        def __call__(self, logits, trans, r_mat, padding_mask):
            return logits, trans, r_mat 
            
        def reset(self, mode):
            self._mode = mode

    iterative_block = IterativeBlock({
        'num_layers': 3,
        'd_model': D_MODEL,
        'n_head': 8,
        'dim_feedforward': 4096,
        'dropout': 0.1,
        'activation': 'relu',
        'learn_pos': True
    })
    
    decoder = ShapePretrainingDecoderIterativeNoRegression(
        num_layers=12,
        d_model=D_MODEL,
        n_head=8,
        dim_feedforward=4096,
        dropout=0.1,
        activation='relu',
        learn_pos=True,
        iterative_num=1,
        max_dist=6.75,
        grid_resolution=0.5,
        rotation_bin_direction=11,
        rotation_bin_angle=24,
        iterative_block=iterative_block
    )
    
    decoder.build(embed=tgt_embed, special_tokens=tgt_special_tokens, out_proj=nn.Linear(D_MODEL, tgt_vocab_size, bias=False))

    state_dict = torch.load(model_path, map_location=device)
    if 'model' in state_dict:
        state_dict = state_dict['model']
    
    decoder_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_decoder.'):
            decoder_key = k.replace('_decoder.', '')
            decoder_state_dict[decoder_key] = v
            
    missing_keys, unexpected_keys = decoder.load_state_dict(decoder_state_dict, strict=False)
    if missing_keys:
        print(f"Decoder missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Decoder unexpected keys: {unexpected_keys}")

    decoder.to(device)
    decoder.eval()
    return decoder

def load_desert_model(model_path, device='cuda'):
    """Load pre-trained DESERT encoder and decoder."""
    print(f"Loading DESERT model (encoder and decoder) from {model_path}")
    encoder = load_desert_encoder(model_path, device)
    decoder = load_desert_decoder(model_path, device)
    return encoder, decoder

def run_desert_inference(model_path, shape_patches_path, device='cuda', max_length=50, use_zero_memory=False):
    """Run full DESERT inference with separate encoder and decoder."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"DESERT model file not found: {model_path}")
    
    if not os.path.exists(shape_patches_path):
        raise FileNotFoundError(f"Shape patches file not found: {shape_patches_path}")
        
    print(f"Loading shape patches from: {shape_patches_path}")
    with open(shape_patches_path, 'rb') as f:
        loaded_data = pickle.load(f)
        if isinstance(loaded_data, list) and len(loaded_data) > 0 and isinstance(loaded_data[0], (torch.Tensor, np.ndarray)):
            print("Pickle file contained a list; using the first element.")
            shape_patches = loaded_data[0]
        elif isinstance(loaded_data, (torch.Tensor, np.ndarray)):
            shape_patches = loaded_data
        else:
            raise ValueError("Shape patches file has an unexpected format or is empty.")

    if isinstance(shape_patches, np.ndarray):
        shape_patches = torch.tensor(shape_patches, dtype=torch.float)
    
    print(f"Initial loaded shape_patches dimension: {shape_patches.ndim}, shape: {shape_patches.shape}")

    shape_patches = shape_patches.to(device)
    
    encoder, decoder = load_desert_model(model_path, device)

    if use_zero_memory:
        print("DEBUG: Bypassing encoder and using ZERO memory tensor for decoder.")
        batch_size_mem = 1 

        if shape_patches.ndim == 3 and shape_patches.shape[0] == 1: # Already (1, num_patches, features)
             num_patches_for_mem = shape_patches.shape[1]
        elif shape_patches.ndim == 2: # Potentially (num_patches, features)
             num_patches_for_mem = shape_patches.shape[0]
        else:
            grid_dim = shape_patches.shape[0] if shape_patches.ndim == 3 else 28 # Default if not obvious
            temp_patch_size = 4
            num_patches_for_mem = (grid_dim // temp_patch_size)**3 if grid_dim % temp_patch_size == 0 else shape_patches.shape[0] * shape_patches.shape[1] # Fallback if not divisible
            print(f"DEBUG: Zero memory mode, derived num_patches_for_mem: {num_patches_for_mem} from grid_dim {grid_dim} and patch_size {temp_patch_size}")

        memory = torch.zeros((num_patches_for_mem, batch_size_mem, D_MODEL), device=device)
        memory_padding_mask = torch.zeros((batch_size_mem, num_patches_for_mem), dtype=torch.bool, device=device)
        print(f"DEBUG: Using ZERO memory of shape {memory.shape} and padding mask shape {memory_padding_mask.shape}")
    else:
        if encoder is None:
            raise RuntimeError("Encoder was not loaded by load_desert_model, but use_zero_memory is False.")
        
        print(f"Shape of raw loaded shape_patches: {shape_patches.shape}")
        if shape_patches.ndim == 2 and shape_patches.shape[0]==1: # (1, N) needs to be (N) for get_shape_patches if it expects a single grid element
            print("Raw loaded shape_patches is 2D (1,N), squeezing to 1D for get_shape_patches if it is a single grid element")
            

        if shape_patches.ndim == 4 and shape_patches.shape[0] == 1: # (1, D1, D2, D3)
            print(f"Input is 4D with shape {shape_patches.shape}, squeezing to 3D grid.")
            shape_patches = shape_patches.squeeze(0)
        elif shape_patches.ndim == 2: 
             print(f"Warning: Loaded shape_patches is 2D {shape_patches.shape}. This might not be a raw 3D grid for get_shape_patches.")
             # If it was (num_patches, features) already, this is wrong. 
             # The pkl is assumed to contain a single 3D shape grid like (28,28,28)

        # At this point, shape_patches should be the raw 3D grid, e.g., (28, 28, 28)
        if shape_patches.ndim != 3:
            raise ValueError(f"Shape patches must be a 3D grid (e.g., HxWxD) before calling get_shape_patches, but got {shape_patches.shape}")

        print(f"Processing with get_shape_patches. Input grid shape: {shape_patches.shape}, Encoder patch_size: {encoder._patch_size}")
        # Convert to NumPy array for get_shape_patches (which uses skimage.view_as_blocks)
        shape_patches_numpy = shape_patches.cpu().numpy() # Ensure it's on CPU before converting to numpy
        processed_patches_view = get_shape_patches(shape_patches_numpy, patch_size=encoder._patch_size)
        # processed_patches_view is (num_blocks_dim0, num_blocks_dim1, num_blocks_dim2, patch_size, patch_size, patch_size)
        # e.g., for (28,28,28) input and patch_size=4, it's (7,7,7,4,4,4)
        
        # Reshape to (num_blocks_dim0, num_blocks_dim1, num_blocks_dim2, patch_size**3)
        num_blocks_dim0 = shape_patches_numpy.shape[0] // encoder._patch_size
        num_blocks_dim1 = shape_patches_numpy.shape[1] // encoder._patch_size
        num_blocks_dim2 = shape_patches_numpy.shape[2] // encoder._patch_size
        features_per_patch_dim = encoder._patch_size**3
        
        reshaped_patches = processed_patches_view.reshape(num_blocks_dim0, 
                                                           num_blocks_dim1, 
                                                           num_blocks_dim2, 
                                                           features_per_patch_dim)
        # Now it's (e.g., 7,7,7,64)
        
        # Reshape to (num_patches, features_per_patch)
        final_patches_numpy = reshaped_patches.reshape(-1, features_per_patch_dim)
        # Now it's (e.g., 343, 64)

        # Convert back to tensor and move to the correct device
        shape_patches = torch.tensor(final_patches_numpy, dtype=torch.float).to(device)

        # get_shape_patches returns (num_patches, patch_size**3)
        print(f"Shape after get_shape_patches and reshaping (num_patches, features_per_patch): {shape_patches.shape}")
        
        shape_patches = shape_patches.unsqueeze(0) 
        print(f"Shape after adding batch dimension (batch_size, num_patches, features_per_patch): {shape_patches.shape}")

        # Ensure the feature dimension matches encoder's d_model (it should if get_shape_patches worked correctly with patch_size**3)
        # The encoder's internal projection will handle mapping from patch_size**3 to d_model.
       
        
        with torch.no_grad():
            memory, memory_padding_mask = encoder(shape_patches)
        print(f"Memory shape from encoder: {memory.shape}, Padding mask shape: {memory_padding_mask.shape}")

    vocab_path = os.path.join(os.path.dirname(model_path), "vocab.pkl")
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    BOS_token = vocab['BOS'][2]
    EOS_token = vocab['EOS'][2] if 'EOS' in vocab else None
    
    # Batch size for decoder input should now consistently be 1 if logic is correct
    batch_size_decoder = memory.shape[1]
    input_frag_idx = torch.tensor([[BOS_token]] * batch_size_decoder, dtype=torch.long, device=device)
    input_frag_trans = torch.zeros((batch_size_decoder, 1), dtype=torch.long, device=device)
    input_frag_r_mat = torch.zeros((batch_size_decoder, 1), dtype=torch.long, device=device)
    
    all_pred_idx_batch = [[] for _ in range(batch_size_decoder)]
    
    for step in range(max_length):
        with torch.no_grad():
            memory_padding_mask = memory_padding_mask.bool()
            logits_list, trans_list, r_mat_list = decoder(
                input_frag_idx=input_frag_idx,
                input_frag_trans=input_frag_trans,
                input_frag_r_mat=input_frag_r_mat,
                memory=memory,
                memory_padding_mask=memory_padding_mask
            )
            
            logits = logits_list[-1] 
            trans = trans_list[-1]
            r_mat = r_mat_list[-1]
            
            pred_idx = torch.argmax(logits[:, -1:], dim=-1)
            pred_trans = torch.argmax(trans[:, -1:], dim=-1) 
            pred_rot = torch.argmax(r_mat[:, -1:], dim=-1)

            finished_batch = True
            for i in range(batch_size_decoder):
                next_idx_item = pred_idx[i, 0].item()
                if not all_pred_idx_batch[i] or (EOS_token is None or all_pred_idx_batch[i][-1][0] != EOS_token) :
                    all_pred_idx_batch[i].append((next_idx_item, pred_trans[i, 0].item(), pred_rot[i, 0].item()))
                    if EOS_token is None or next_idx_item != EOS_token:
                        finished_batch = False
                elif all_pred_idx_batch[i][-1][0] == EOS_token: 
                    pass 

            if finished_batch and step > 0: 
                 print("All sequences in batch finished with EOS.")
                 break
                
            input_frag_idx = torch.cat([input_frag_idx, pred_idx], dim=1)
            input_frag_trans = torch.cat([input_frag_trans, pred_trans], dim=1)
            input_frag_r_mat = torch.cat([input_frag_r_mat, pred_rot], dim=1)

            if input_frag_idx.size(1) > max_length:
                input_frag_idx = input_frag_idx[:, -max_length:]
                input_frag_trans = input_frag_trans[:, -max_length:]
                input_frag_r_mat = input_frag_r_mat[:, -max_length:]

    output_sequences = []
    for i in range(batch_size_decoder):
        sequence = all_pred_idx_batch[i]
        output_sequences.append(sequence)
        
    return output_sequences


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def get_rotation_bins_local(sp, rp):
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
            theta = t * math.pi / (rp / 2)
            rotation_mat_bin.append(rotation_matrix(axis, theta))
    rotation_mat_bin = np.stack(rotation_mat_bin, axis=0)
    return rotation_mat_bin

def distance(pos1, pos2): return np.sqrt(np.sum((pos1 - pos2)**2))

def _get_3d_frags_local(frags_sequence):
    ret_frags = []
    if SHAPE_UTILS_VOCAB is None or SHAPE_UTILS_VOCAB_R is None or SHAPE_UTILS_ROTATION_BIN is None:
        print("Error: Vocab or rotation_bin for _get_3d_frags_local not initialized."); return ret_frags
        
    for unit in frags_sequence:
        idx, tr_bin, rm_bin = unit
        key = SHAPE_UTILS_VOCAB_R.get(idx)
        if key is None or key in ['UNK', 'BOS', 'BOB', 'EOB', 'PAD', 'EOS']:
            if key == 'EOS': break
            continue

        frag_mol_template_container = SHAPE_UTILS_VOCAB.get(key)
        frag_mol_template = None
        if frag_mol_template_container:
            if hasattr(frag_mol_template_container, 'GetConformer'): 
                frag_mol_template = frag_mol_template_container
            elif isinstance(frag_mol_template_container, (list, tuple)) and len(frag_mol_template_container) > 0 and hasattr(frag_mol_template_container[0], 'GetConformer'):
                frag_mol_template = frag_mol_template_container[0]
        
        if frag_mol_template is None: continue

        frag = copy.deepcopy(frag_mol_template)
        conformer = frag.GetConformer() 
        if conformer is None: 
            AllChem.Compute2DCoords(frag)
            if AllChem.EmbedMolecule(frag, AllChem.ETKDG()) == -1: continue
            conformer = frag.GetConformer()
            if conformer is None: continue 

        grid_coords = bin_to_grid_coords(tr_bin, 28)
        tr_real_coords = grid_coords_to_real_coords(grid_coords, 28, 0.5)
        rm_matrix = bin_to_rotation_mat(rm_bin, SHAPE_UTILS_ROTATION_BIN)

        if tr_real_coords is None or rm_matrix is None: continue

        transform_matrix_rot = np.eye(4)
        transform_matrix_rot[:3,:3] = rm_matrix
        Chem.rdMolTransforms.TransformConformer(conformer, transform_matrix_rot)

        transform_matrix_trans = np.eye(4)
        transform_matrix_trans[0,3] = tr_real_coords[0]
        transform_matrix_trans[1,3] = tr_real_coords[1]
        transform_matrix_trans[2,3] = tr_real_coords[2]
        Chem.rdMolTransforms.TransformConformer(conformer, transform_matrix_trans)
        
        ret_frags.append(frag)
    return ret_frags

def _get_star_info_robust_local(current_frags):
    star_info_list = []
    for f_idx, frag_mol in enumerate(current_frags):
        if frag_mol is None: continue 
        con = frag_mol.GetConformer()
        if con is None: continue 
        for atom in frag_mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                neighbours = atom.GetNeighbors()
                if not neighbours: continue
                assert len(neighbours) == 1, f"Star atom {atom.GetIdx()} in frag {f_idx} has {len(neighbours)} neighbors, expected 1"
                nei_idx = neighbours[0].GetIdx()
                atom_idx = atom.GetIdx()
                try:
                    atom_pos_rdk = con.GetAtomPosition(atom_idx)
                    nei_pos_rdk = con.GetAtomPosition(nei_idx)
                    atom_pos = np.array([atom_pos_rdk.x, atom_pos_rdk.y, atom_pos_rdk.z])
                    nei_pos = np.array([nei_pos_rdk.x, nei_pos_rdk.y, nei_pos_rdk.z])
                except RuntimeError: continue
                if np.isinf(atom_pos).any() or np.isinf(nei_pos).any(): continue
                star_info_list.append({
                    'f_idx': f_idx, 'atom_idx': atom_idx, 'nei_idx': nei_idx,
                    'atom_pos': atom_pos, 'nei_pos': nei_pos,
                    'breakpoint_id_self': (f_idx, atom_idx)
                })
    return star_info_list

def _connectMols_helper_local(mol1, mol2, atom_idx_mol1_star, atom_idx_mol2_star):
    try:
        if not mol1 or not mol2: return None
        dummy_atom_mol1 = mol1.GetAtomWithIdx(atom_idx_mol1_star)
        if dummy_atom_mol1.GetAtomicNum() != 0: return None
        neighbors_mol1 = dummy_atom_mol1.GetNeighbors()
        if not neighbors_mol1: return None
        attach_atom_idx_mol1 = neighbors_mol1[0].GetIdx()

        dummy_atom_mol2 = mol2.GetAtomWithIdx(atom_idx_mol2_star)
        if dummy_atom_mol2.GetAtomicNum() != 0: return None
        neighbors_mol2 = dummy_atom_mol2.GetNeighbors()
        if not neighbors_mol2: return None
        attach_atom_idx_mol2 = neighbors_mol2[0].GetIdx()

        combo_mol = Chem.RWMol(mol1)
        mol2_atom_map = {}
        for atom_mol2 in mol2.GetAtoms():
            new_idx = combo_mol.AddAtom(atom_mol2)
            mol2_atom_map[atom_mol2.GetIdx()] = new_idx
        
        for bond_mol2 in mol2.GetBonds():
            begin_atom_orig_idx = bond_mol2.GetBeginAtomIdx()
            end_atom_orig_idx = bond_mol2.GetEndAtomIdx()
            combo_mol.AddBond(mol2_atom_map[begin_atom_orig_idx], 
                              mol2_atom_map[end_atom_orig_idx], 
                              bond_mol2.GetBondType())

        new_attach_atom_idx_mol2 = mol2_atom_map[attach_atom_idx_mol2]
        combo_mol.AddBond(attach_atom_idx_mol1, new_attach_atom_idx_mol2, Chem.BondType.SINGLE)
        
        dummy1_final_idx = atom_idx_mol1_star
        dummy2_final_idx = mol2_atom_map[atom_idx_mol2_star]
        
        indices_to_remove = sorted([dummy1_final_idx, dummy2_final_idx], reverse=True)
        
        for idx_to_remove in indices_to_remove:
            if idx_to_remove < combo_mol.GetNumAtoms() and combo_mol.GetAtomWithIdx(idx_to_remove).GetAtomicNum() == 0:
                combo_mol.RemoveAtom(idx_to_remove)
            else:
                return None

        result_mol = combo_mol.GetMol()
        Chem.SanitizeMol(result_mol)
        return result_mol
    except Exception as e:
        return None

def connect_fragments_local(frags_list_input):
    if not frags_list_input: return None
    current_frags_list = [f for f in copy.deepcopy(frags_list_input) if f is not None]
    if not current_frags_list: return None
    
    failed_connection_breakpoint_pairs = set()

    while True: 
        made_a_connection_in_this_pass = False
        if len(current_frags_list) <= 1: break 

        all_star_info = _get_star_info_robust_local(current_frags_list)
        if len(all_star_info) <= 1: break 

        potential_connections = []
        for i in range(len(all_star_info)):
            for j in range(i + 1, len(all_star_info)):
                star1_data = all_star_info[i]
                star2_data = all_star_info[j]

                if star1_data['f_idx'] == star2_data['f_idx']: continue

                bp_id1 = star1_data['breakpoint_id_self']
                bp_id2 = star2_data['breakpoint_id_self']
                current_pair_key = tuple(sorted((bp_id1, bp_id2)))

                if current_pair_key in failed_connection_breakpoint_pairs: continue 
                dist_val = distance(star1_data['atom_pos'], star2_data['nei_pos']) + \
                           distance(star1_data['nei_pos'], star2_data['atom_pos'])
                
                potential_connections.append({
                    'dist': dist_val,
                    'star1_data': star1_data,
                    'star2_data': star2_data,
                    'pair_key': current_pair_key
                })
        
        if not potential_connections: break 
        potential_connections.sort(key=lambda x: x['dist'])

        for attempt in potential_connections:
            star1_data_to_connect = attempt['star1_data']
            star2_data_to_connect = attempt['star2_data']
            
            if not (star1_data_to_connect['f_idx'] < len(current_frags_list) and \
                    star2_data_to_connect['f_idx'] < len(current_frags_list)):
                failed_connection_breakpoint_pairs.add(attempt['pair_key'])
                continue

            mol_obj1 = current_frags_list[star1_data_to_connect['f_idx']]
            mol_obj2 = current_frags_list[star2_data_to_connect['f_idx']]
            
            connected_mol = _connectMols_helper_local(mol_obj1, mol_obj2, 
                                              star1_data_to_connect['atom_idx'], 
                                              star2_data_to_connect['atom_idx'])

            if connected_mol:
                indices_to_remove_from_list = sorted([star1_data_to_connect['f_idx'], star2_data_to_connect['f_idx']], reverse=True)
                temp_list = []
                for k_idx, frag in enumerate(current_frags_list):
                    if k_idx not in indices_to_remove_from_list:
                        temp_list.append(frag)
                temp_list.append(connected_mol)
                current_frags_list = temp_list
                failed_connection_breakpoint_pairs.clear() 
                made_a_connection_in_this_pass = True
                break 
            else:
                failed_connection_breakpoint_pairs.add(attempt['pair_key'])
        
        if not made_a_connection_in_this_pass: break

    if not current_frags_list: return None
    if len(current_frags_list) > 1:
        current_frags_list.sort(key=lambda m: m.GetNumAtoms() if m else 0, reverse=True)
    
    final_mol = current_frags_list[0]
    discarded_frags = current_frags_list[1:] # capture discarded fragments
    if final_mol is None: return None, discarded_frags # return discarded even if final_mol is None

    rw_final_mol = Chem.RWMol(final_mol)
    needs_another_pass = True
    passes = 0
    max_passes = 10 # limit capping passes to prevent infinite loops

    while needs_another_pass and passes < max_passes:
        needs_another_pass = False
        passes += 1
        atoms_to_replace_this_pass = []
        for atom in rw_final_mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atoms_to_replace_this_pass.append(atom.GetIdx())
        
        if atoms_to_replace_this_pass:
            for atom_idx in sorted(atoms_to_replace_this_pass, reverse=True):
                try:
                    atom_to_check = rw_final_mol.GetAtomWithIdx(atom_idx)
                    if atom_to_check and atom_to_check.GetAtomicNum() == 0:
                        rw_final_mol.ReplaceAtom(atom_idx, Chem.Atom(6)) 
                        needs_another_pass = True 
                except RuntimeError: pass 
            if needs_another_pass: 
                try: Chem.SanitizeMol(rw_final_mol) 
                except Exception: final_mol = rw_final_mol.GetMol(); return final_mol, discarded_frags 
        else: pass

    final_mol = rw_final_mol.GetMol()
    try: Chem.SanitizeMol(final_mol) 
    except Exception: pass 
    return final_mol, discarded_frags 

@dataclasses.dataclass
class QVinaOption:
    """Options for QVina2 docking"""
    center_x: float
    center_y: float
    center_z: float
    size_x: float = 20.0
    size_y: float = 20.0
    size_z: float = 20.0
    exhaustiveness: int = 8
    num_modes: int = 1

def prepare_ligand_pdbqt_local(mol, obabel_path="obabel"):
    """
    Prepare a ligand for docking by converting it to PDBQT format using OpenBabel
    """
    unique_id = str(uuid.uuid4())[:8]
    temp_dir = tempfile.gettempdir()
    temp_mol_file = os.path.join(temp_dir, f"temp_ligand_{unique_id}.mol")
    temp_pdbqt_file = os.path.join(temp_dir, f"temp_ligand_{unique_id}.pdbqt")

    try:
        Chem.MolToMolFile(mol, temp_mol_file)
        cmd = [
            obabel_path,
            "-imol", temp_mol_file,
            "-opdbqt", "-O", temp_pdbqt_file,
            "--partialcharge", "gasteiger",
            "--gen3d", "best"
        ]
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        if process.returncode != 0:
            print(f"    Error converting molecule to PDBQT: {process.stderr}")
            return None
        with open(temp_pdbqt_file, "r") as f:
            pdbqt_content = f.read()
        if "ATOM" not in pdbqt_content and "HETATM" not in pdbqt_content:
            print("    Error: Generated PDBQT file does not contain valid atom entries")
            return None
        return pdbqt_content
    except Exception as e:
        print(f"    Error preparing ligand: {str(e)}")
        return None
    finally:
        for f_path in [temp_mol_file, temp_pdbqt_file]:
            if os.path.exists(f_path):
                try:
                    os.remove(f_path)
                except OSError:
                    print(f"    Warning: Could not remove temporary file {f_path}")

def dock_best_molecule_local(mol, receptor_path, receptor_center):
    """Dock the molecule against receptor target"""
    temp_ligand_file = None
    output_file = None
    try:
        smiles = Chem.MolToSmiles(mol)
        print(f"  Attempting docking for: {smiles}")
        unique_id = str(uuid.uuid4())[:8]
        temp_dir = tempfile.gettempdir()
        
        center = receptor_center
        box_size = [22.5, 22.5, 22.5] 
        qvina_path = shutil.which("qvina2.1")
        if qvina_path is None:
            local_qvina_path = os.path.join(os.getcwd(), "bin", "qvina2.1")
            if os.path.exists(local_qvina_path) and os.access(local_qvina_path, os.X_OK):
                 qvina_path = local_qvina_path
            else:
                user_specific_qvina_path = "/workspace/synformer/bin/qvina2.1"
                if os.path.exists(user_specific_qvina_path) and os.access(user_specific_qvina_path, os.X_OK):
                    qvina_path = user_specific_qvina_path
                else:
                    print("    Error: QVina2 executable (qvina2.1) not found or not executable in PATH, ./bin/, or /workspace/synformer/bin/.")
                    return None
        
        obabel_path = shutil.which("obabel")
        if obabel_path is None:
            print("    Error: OpenBabel (obabel) not found in PATH.")
            return None
            
        if not os.path.exists(receptor_path):
            print(f"    Error: Receptor file not found at {receptor_path}")
            return None
            
        ligand_pdbqt = prepare_ligand_pdbqt_local(mol, obabel_path)
        if ligand_pdbqt is None:
            print("    Failed to prepare ligand for docking.")
            return None
            
        temp_ligand_file = os.path.join(temp_dir, f"temp_ligand_dock_{unique_id}.pdbqt")
        with open(temp_ligand_file, "w") as f:
            f.write(ligand_pdbqt)
            
        options = QVinaOption(
            center_x=center[0], center_y=center[1], center_z=center[2],
            size_x=box_size[0], size_y=box_size[1], size_z=box_size[2]
        )
        
        output_file = os.path.join(temp_dir, f"temp_ligand_dock_out_{unique_id}.pdbqt")
        cmd = [
            qvina_path, "--receptor", receptor_path, "--ligand", temp_ligand_file,
            "--center_x", str(options.center_x), "--center_y", str(options.center_y), "--center_z", str(options.center_z),
            "--size_x", str(options.size_x), "--size_y", str(options.size_y), "--size_z", str(options.size_z),
            "--exhaustiveness", str(options.exhaustiveness), "--num_modes", str(options.num_modes),
            "--out", output_file
        ]
        
        process = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300, check=False
        )
        
        if process.returncode != 0:
            print(f"    QVina2 execution error for {smiles}: {process.stderr}")

        score = None
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                for line in f:
                    if "REMARK VINA RESULT" in line:
                        try:
                            score = float(line.split()[3])
                            print(f"    Docking score for {smiles}: {score:.3f}")
                            break
                        except (IndexError, ValueError) as e_parse:
                            print(f"    Could not parse score from VINA output line: '{line.strip()}'. Error: {e_parse}")
                            pass 
        if score is None and process.returncode != 0:
             print(f"    No docking score found and QVina2 reported an error for {smiles}.")
             return None 
        elif score is None:
             print(f"    No docking score found in output for {smiles}, but QVina2 ran (exit code {process.returncode}).")
             

        return score
        
    except subprocess.TimeoutExpired:
        print(f"    Docking timed out for {smiles}")
        return None
    except Exception as e:
        print(f"    Error during docking for {smiles}: {str(e)}")
        return None
    finally:
        for f_path in [temp_ligand_file, output_file]:
            if f_path and os.path.exists(f_path):
                try:
                    os.remove(f_path)
                except OSError:
                     print(f"    Warning: Could not remove temporary file {f_path}")

def main():
    global SHAPE_UTILS_VOCAB, SHAPE_UTILS_VOCAB_R, SHAPE_UTILS_ROTATION_BIN
    parser = argparse.ArgumentParser(description='Run full DESERT inference with encoder and decoder.')
    parser.add_argument('--model_path', type=str, 
                        default="/workspace/data/desert/1WW_30W_5048064.pt",
                        help='Path to the model checkpoint (containing both encoder and decoder weights)')
    parser.add_argument('--shape_patches_path', type=str,
                        default="/workspace/data/3tym_A_shapes.pkl",
                        help='Path to the shape patches file (e.g., .pkl)')
    parser.add_argument('--vocab_path', type=str, 
                        default=None, # Default to None, will derive from model_path if not set
                        help='Path to the vocabulary pkl file (for fragment connection and names). If None, uses vocab.pkl in model_path dir.')
    parser.add_argument('--rotation_bin_path', type=str, 
                        default="/workspace/data/desert/rotation_bin.pkl",
                        help='Path to the rotation_bin.pkl file. Will be generated if not found.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run on (cuda or cpu)')
    parser.add_argument('--max_length', type=int, default=50, help='Maximum length of generated sequence')
    parser.add_argument('--use-zero-memory', action='store_true', help='Bypass encoder and use zero memory for decoder (for debugging).')
    parser.add_argument('--receptor_info_csv', type=str, 
                        default="/workspace/data/TestCrossDocked2020/receptors/test.csv", 
                        help='Path to CSV file with receptor PDB IDs and binding site center coordinates (pdb,c1,c2,c3).')
    parser.add_argument('--receptors_dir', type=str, 
                        default="/workspace/data/TestCrossDocked2020/receptors/", 
                        help='Directory where receptor PDBQT files are stored.')

    args = parser.parse_args()
    
    try:
        print(f"Using device: {args.device}")
        print(f"Loading model from: {args.model_path}")
        print(f"Loading shape patches from: {args.shape_patches_path}")
        if args.use_zero_memory:
            print("Running in ZERO MEMORY mode for decoder.")
        
       
        actual_vocab_path = args.vocab_path
        if actual_vocab_path is None:
            actual_vocab_path = os.path.join(os.path.dirname(args.model_path), "vocab.pkl")

        if not os.path.exists(actual_vocab_path):
            print(f"Error: Vocabulary file for fragment connection not found at {actual_vocab_path}. Cannot proceed with connection.")
            return
        
        print(f"Loading vocabulary for fragment connection from: {actual_vocab_path}")
        with open(actual_vocab_path, 'rb') as f_vocab_su:
            SHAPE_UTILS_VOCAB = pickle.load(f_vocab_su)
        SHAPE_UTILS_VOCAB_R = {v_item[2]: k_item for k_item, v_item in SHAPE_UTILS_VOCAB.items() if isinstance(v_item, (list, tuple)) and len(v_item) >=3}
        print("Vocabulary for fragment connection loaded.")

        if not os.path.exists(args.rotation_bin_path):
            print(f"Rotation bin file not found at {args.rotation_bin_path}. Generating it...")
            try:
                sp_param = 11 
                rp_param = 24
                print(f"Generating rotation_bin.pkl with sp={sp_param}, rp={rp_param}...")
                generated_rotation_bin = get_rotation_bins_local(sp=sp_param, rp=rp_param)
                with open(args.rotation_bin_path, 'wb') as f_rot_bin:
                    pickle.dump(generated_rotation_bin, f_rot_bin)
                print(f"Successfully generated and saved rotation_bin.pkl to {args.rotation_bin_path}")
                SHAPE_UTILS_ROTATION_BIN = generated_rotation_bin
            except Exception as e_rot_gen:
                print(f"Error generating rotation_bin.pkl: {e_rot_gen}. Cannot proceed with connection.")
                return
        else:
            print(f"Loading existing rotation_bin.pkl from {args.rotation_bin_path}")
            with open(args.rotation_bin_path, 'rb') as f_rot_bin_su:
                SHAPE_UTILS_ROTATION_BIN = pickle.load(f_rot_bin_su)
        print("Rotation bin for fragment connection loaded.")
        receptor_centers_df = None
        if os.path.exists(args.receptor_info_csv):
            try:
                receptor_centers_df = pd.read_csv(args.receptor_info_csv)
                if 'pdb' not in receptor_centers_df.columns:
                    print(f"Warning: 'pdb' column not found in {args.receptor_info_csv}. Docking might fail to find centers.")
                    receptor_centers_df = None
                else:
                    receptor_centers_df.set_index('pdb', inplace=True)
                    print(f"Successfully loaded receptor center information from {args.receptor_info_csv}")
            except Exception as e_csv:
                print(f"Error loading receptor center CSV {args.receptor_info_csv}: {e_csv}. Docking may be skipped.")
        else:
            print(f"Receptor center CSV not found at {args.receptor_info_csv}. Docking will be skipped if centers cannot be determined.")

        generated_sequences = run_desert_inference(
            model_path=args.model_path,
            shape_patches_path=args.shape_patches_path,
            device=args.device,
            max_length=args.max_length,
            use_zero_memory=args.use_zero_memory
        )
        
        print("\nSuccessfully ran full DESERT inference!")
        
        if len(generated_sequences) == 1:
            print("\nGenerated fragment sequence:")
        else:
            print("\nGenerated fragment sequences (multiple items detected in input/memory - check processing logic if unexpected):")

        vocab_path = os.path.join(os.path.dirname(args.model_path), "vocab.pkl")
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        id_to_token = {idx: token for token, (_, _, idx) in vocab.items()}
        EOS_token_id = vocab['EOS'][2] if 'EOS' in vocab else None

        for i, sequence in enumerate(generated_sequences):
            if len(generated_sequences) > 1:
                print(f"\nSequence for batch item {i} (length {len(sequence)}):")
            else:
                 print(f"\nSequence (length {len(sequence)}):")

            if sequence and EOS_token_id is not None and sequence[-1][0] == EOS_token_id:
                 print("  Sequence properly terminated with EOS token.")
            elif not sequence:
                 print("  Sequence is empty.")

            for frag_idx, (frag_id, trans_id, rot_id) in enumerate(sequence):
                token_name = id_to_token.get(frag_id, f"UnknownID({frag_id})")
                print(f"  Fragment {frag_idx + 1}: ID={frag_id} ({token_name}), Translation={trans_id}, Rotation={rot_id}")
                if EOS_token_id is not None and frag_id == EOS_token_id:
                    break 
            
            if sequence: 
                print("\n  Attempting to connect generated fragments...")
                rdkit_frags = _get_3d_frags_local(sequence)
                if rdkit_frags:
                    print(f"    Successfully converted {len(rdkit_frags)} fragments to RDKit Mol objects.")
                    connected_mol, discarded_frags = connect_fragments_local(rdkit_frags)
                    if connected_mol:
                        try:
                            connected_smiles = Chem.MolToSmiles(connected_mol)
                            print(f"  Connected Molecule SMILES: {connected_smiles}")

                            current_receptor_path = None
                            current_receptor_center = None
                            docking_score_val = "NotAttempted"
                            qed_score_val = "NotCalculated"

                            try:
                                qed_score = QED.qed(connected_mol)
                                qed_score_val = f"{qed_score:.3f}"
                                print(f"  QED Score: {qed_score_val}")
                            except Exception as e_qed:
                                qed_score_val = "QEDError"
                                print(f"  Error calculating QED: {e_qed}")

                            try:
                                shape_basename = os.path.basename(args.shape_patches_path)
                                receptor_id_from_shape = shape_basename.split('_shapes.pkl')[0].split('.pkl')[0].split('.patch')[0]
                                
                                current_receptor_path = os.path.join(args.receptors_dir, f"{receptor_id_from_shape}.pdbqt")
                                
                                if receptor_centers_df is not None and receptor_id_from_shape in receptor_centers_df.index:
                                    center_info = receptor_centers_df.loc[receptor_id_from_shape]
                                    if all(c in center_info and pd.notna(center_info[c]) for c in ['c1', 'c2', 'c3']):
                                        current_receptor_center = [float(center_info['c1']), float(center_info['c2']), float(center_info['c3'])]
                                    else:
                                        print(f"    Warning: Center coordinates (c1,c2,c3) missing or invalid for receptor ID '{receptor_id_from_shape}' in {args.receptor_info_csv}.")
                                else:
                                    print(f"    Warning: Center information for receptor ID '{receptor_id_from_shape}' not found in loaded CSV data (index: {receptor_centers_df.index if receptor_centers_df is not None else 'None'}).")

                            except Exception as e_receptor_info:
                                print(f"    Error determining receptor path/center for {args.shape_patches_path}: {e_receptor_info}")

                            if current_receptor_path and os.path.exists(current_receptor_path) and current_receptor_center:
                                print(f"  Receptor for docking: {current_receptor_path}, Center: {current_receptor_center}")
                                score = dock_best_molecule_local(connected_mol, current_receptor_path, current_receptor_center)
                                if score is not None:
                                    docking_score_val = f"{score:.3f}"
                                    print(f"  Docking Score: {docking_score_val}")
                                else:
                                    docking_score_val = "DockingFailed"
                                    print(f"  Docking failed for {connected_smiles}")
                            else:
                                print("  Skipping docking: Receptor PDBQT file not found, not configured, or center coordinates missing/invalid.")
                                if not current_receptor_path or not os.path.exists(current_receptor_path):
                                     print(f"    Receptor path check: {current_receptor_path} (Exists: {os.path.exists(current_receptor_path) if current_receptor_path else 'N/A'})")
                                if not current_receptor_center:
                                     print(f"    Receptor center check: {current_receptor_center}")
                                docking_score_val = "DockingSkipped (Receptor/Center Missing)"

                        except Exception as e_smiles:
                            print(f"  Error converting connected molecule to SMILES: {e_smiles}")
                        
                        if discarded_frags:
                            print("  Discarded fragments during connection:")
                            for i, d_frag in enumerate(discarded_frags):
                                if d_frag:
                                    try:
                                        d_smiles = Chem.MolToSmiles(d_frag)
                                        orig_name_approx = f"(approx. {d_frag.GetNumHeavyAtoms()} heavy atoms)"
                                        print(f"    Discarded {i+1}: {d_smiles} {orig_name_approx}")
                                    except Exception as e_d_smiles:
                                        print(f"    Discarded {i+1}: Could not get SMILES (Error: {e_d_smiles})")
                                else:
                                    print(f"    Discarded {i+1}: (None object)")
                    else:
                        print("  Fragment connection resulted in None (failed to connect).")
                        if discarded_frags:
                            print("  Fragments present before connection attempt failed:")
                            for i, d_frag in enumerate(discarded_frags):
                                if d_frag:
                                    try: d_smiles = Chem.MolToSmiles(d_frag); print(f"    Fragment {i+1}: {d_smiles}")
                                    except: print(f"    Fragment {i+1}: Could not get SMILES")
                                else: print(f"    Fragment {i+1}: (None object)")
                else:
                    print("  Could not convert generated sequence to RDKit fragments for connection.")
            else:
                print("\n  Skipping fragment connection as the generated sequence is empty.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except RuntimeError as e:
        print(f"Runtime Error: {e}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main() 