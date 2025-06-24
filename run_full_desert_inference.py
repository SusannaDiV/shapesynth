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
import subprocess # Added for docking
import tempfile # Added for docking
import uuid # Added for docking
import shutil # Added for docking
import dataclasses # Added for QVinaOption
import pandas as pd # Added for receptor info CSV
from rdkit.Chem import QED # Added for QED calculation

from synformer.data.common import process_smiles
from synformer.models.encoder.shape import ShapePretrainingEncoder
from synformer.models.desert.fulldecoder import ShapePretrainingDecoderIterativeNoRegression, ShapePretrainingIteratorNoRegression
from synformer.data.utils.shape_utils import bin_to_grid_coords, grid_coords_to_real_coords, bin_to_rotation_mat, get_3d_frags, connect_fragments, get_atom_stamp, get_shape, get_rotation_bins, get_shape_patches, ATOM_RADIUS, ATOMIC_NUMBER, ATOMIC_NUMBER_REVERSE
from synformer.models.transformer.positional_encoding import PositionalEncoding # Added this import

# Define d_model globally or pass as an argument
D_MODEL = 1024 #This should be num_heads * head_dim = 8 * 128 = 1024

# --- BEGIN: Functions for rotation bin generation (kept from original) ---
# These are needed to load/generate rotation_bin.pkl for fragment orientation
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
# --- END: Functions for rotation bin generation ---

def load_desert_encoder(model_path, device='cuda'):
    """Load a pre-trained DESERT encoder."""
    print(f"Loading encoder from {model_path}")

    # Initialize encoder with correct parameters
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

    # Build the encoder (dummy embedding and special_tokens, as they are not strictly needed for state_dict loading)
    # Ensure the embedding dimension matches d_model
    dummy_embed = nn.Embedding(2, D_MODEL) # Vocab size 2, embedding dim D_MODEL
    encoder.build(
        embed=dummy_embed,
        special_tokens={'pad': 0}
    )

    # Load checkpoint
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
    
    # Instantiate ShapePretrainingIteratorNoRegression based on provided YAML config
    # These are the parameters for ShapePretrainingIteratorNoRegression itself.
    # It inherits from TransformerEncoder, which will take these.
    iterative_block = ShapePretrainingIteratorNoRegression(
        num_layers=3,
        d_model=D_MODEL, 
        n_head=8,
        dim_feedforward=4096,
        dropout=0.1,
        # attention_dropout is often a separate param in Transformer Encoders, 
        # if not specified in YAML for ShapePretrainingIteratorNoRegression, assume it might default or use dropout value.
        # For now, let's rely on **kwargs or internal defaults of TransformerEncoder if not explicitly in YAML for this specific class.
        activation='relu',
        learn_pos=True
    )
    
    # These are the parameters for the main decoder
    decoder_max_dist = 6.75
    decoder_grid_resolution = 0.5
    decoder_rot_bin_direction = 11
    decoder_rot_bin_angle = 24

    decoder = ShapePretrainingDecoderIterativeNoRegression(
        num_layers=12,
        d_model=D_MODEL,
        n_head=8,
        dim_feedforward=4096,
        dropout=0.1,
        activation='relu',
        learn_pos=True,
        iterative_num=1, 
        max_dist=decoder_max_dist,
        grid_resolution=decoder_grid_resolution,
        rotation_bin_direction=decoder_rot_bin_direction,
        rotation_bin_angle=decoder_rot_bin_angle,
        iterative_block=iterative_block # Pass the instantiated correct block
    )

    # Calculate trans_size and rot_size for iterative_block.build()
    # Use the same values that define the decoder's operating space for translations/rotations.
    trans_size = int((2 * decoder_max_dist / decoder_grid_resolution + 1)**3)
    rotation_bins_for_size = get_rotation_bins_local(sp=decoder_rot_bin_direction, rp=decoder_rot_bin_angle)
    rot_size = rotation_bins_for_size.shape[0]

    print(f"DEBUG: IterativeBlock build params: trans_size={trans_size}, rot_size={rot_size}")

    iterative_block.build(
        embed=tgt_embed, 
        special_tokens=tgt_special_tokens,
        trans_size=trans_size, 
        rotat_size=rot_size
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

def _generate_sequences_from_full_model(model_path, shape_patches_path, device='cuda', max_length=50, use_zero_memory=False):
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
    
    decoder = load_desert_decoder(model_path, device)

    if use_zero_memory:
        print("DEBUG: Bypassing encoder and using ZERO memory tensor for decoder.")
        batch_size_mem = 1 
        # Determine num_patches for zero memory. If using get_shape_patches, it would be (grid_dim/patch_size)**3
        # For a typical 28x28x28 grid and patch_size 4, this is (28/4)^3 = 7^3 = 343
        # We need a placeholder if shape_patches wasn't processed by get_shape_patches yet.
        # Assuming a default based on common grid size if shape_patches isn't yet in patch format.
        if shape_patches.ndim == 3 and shape_patches.shape[0] == 1: # Already (1, num_patches, features)
             num_patches_for_mem = shape_patches.shape[1]
        elif shape_patches.ndim == 2: # Potentially (num_patches, features)
             num_patches_for_mem = shape_patches.shape[0]
        else: # Assuming raw grid (e.g. 28,28,28), calculate num_patches
            # This part is tricky if encoder patch_size isn't easily accessible here
            # For now, let's assume a common case or require shape_patches to be pre-processed for zero_mem path
            # Or, we load encoder just to get patch_size, which is a bit inefficient for zero_mem
            # For simplicity, let's use a typical value if encoder isn't loaded.
            # A better way would be to pass patch_size or calculate num_patches more robustly.
            grid_dim = shape_patches.shape[0] if shape_patches.ndim == 3 else 28 # Default if not obvious
            temp_patch_size = 4 # Assuming default patch size for DESERT encoder
            num_patches_for_mem = (grid_dim // temp_patch_size)**3 if grid_dim % temp_patch_size == 0 else shape_patches.shape[0] * shape_patches.shape[1] # Fallback if not divisible
            print(f"DEBUG: Zero memory mode, derived num_patches_for_mem: {num_patches_for_mem} from grid_dim {grid_dim} and patch_size {temp_patch_size}")

        memory = torch.zeros((num_patches_for_mem, batch_size_mem, D_MODEL), device=device)
        memory_padding_mask = torch.zeros((batch_size_mem, num_patches_for_mem), dtype=torch.bool, device=device)
        print(f"DEBUG: Using ZERO memory of shape {memory.shape} and padding mask shape {memory_padding_mask.shape}")
    else:
        encoder = load_desert_encoder(model_path, device)
        
        # --- BEGIN: Process shape_patches using get_shape_patches --- 
        print(f"Shape of raw loaded shape_patches: {shape_patches.shape}")
        if shape_patches.ndim == 2 and shape_patches.shape[0]==1: # (1, N) needs to be (N) for get_shape_patches if it expects a single grid element
            print("Raw loaded shape_patches is 2D (1,N), squeezing to 1D for get_shape_patches if it is a single grid element")
            # This case is unlikely for a 3D grid. Assuming it is a batch of 1 of 2D patches already.
            # If shape_patches was (1, 784), this indicates it's likely already processed or a different format.
            # Let's assume if ndim is 3 and first dim is 1, it might be (1, grid, grid)
            # The original code handled (grid, grid, features) then reshaped. 
            # We need to ensure shape_patches is the raw 3D grid (e.g., 28x28x28) before get_shape_patches

        # Ensure shape_patches is a 3D grid (e.g., 28x28x28) before get_shape_patches
        # The original logic: loaded_data -> shape_patches (tensor)
        # if ndim == 2 -> unsqueeze(0) -> (1, D1, D2) -> then assumed (1, num_patches, features) for old reshape.
        # if ndim == 3 -> (D1, D2, D3) -> then assumed (grid1, grid2, features) for old reshape.
        # For get_shape_patches, we need it to be the raw grid, e.g. (28,28,28)

        if shape_patches.ndim == 4 and shape_patches.shape[0] == 1: # (1, D1, D2, D3)
            print(f"Input is 4D with shape {shape_patches.shape}, squeezing to 3D grid.")
            shape_patches = shape_patches.squeeze(0)
        elif shape_patches.ndim == 2: # Example (784, 28) in old logic.
                                      # This needs to be re-evaluated. If it's (N, M) it's not a single 3D grid.
                                      # Assuming the pickle stores a single 3D grid, this path shouldn't be common.
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
        
        # Add batch dimension for the encoder
        shape_patches = shape_patches.unsqueeze(0) 
        print(f"Shape after adding batch dimension (batch_size, num_patches, features_per_patch): {shape_patches.shape}")

        # Ensure the feature dimension matches encoder's d_model (it should if get_shape_patches worked correctly with patch_size**3)
        # The encoder's internal projection will handle mapping from patch_size**3 to d_model.
        # So, no explicit check against D_MODEL here for shape_patches last dim.
        # The check is that features_per_patch from get_shape_patches (patch_size**3) is what the encoder's 1st linear layer expects.

        # Remove old padding logic:
        # expected_patch_feature_dim = encoder._patch_size**3 
        # loaded_patch_feature_dim = shape_patches.shape[-1]
        # if loaded_patch_feature_dim != expected_patch_feature_dim: ... padding ...
        # --- END: Process shape_patches using get_shape_patches --- 
        
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
            # This loop should only run once if batch_size_decoder is 1
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
    # This loop should only run once if batch_size_decoder is 1
    for i in range(batch_size_decoder):
        sequence = all_pred_idx_batch[i]
        output_sequences.append(sequence)
        
    return output_sequences

# --- BEGIN: User-provided connect_fragments function ---
EOS_TOKEN = "EOS" # Define EOS_TOKEN, or get from vocab if available

# User-provided connect_fragments function (with potential optimizations)
def connect_fragments(frags):
    if not frags:
        return None, [] # Return None for molecule and empty list for discarded

    active_frags = [f for f in copy.deepcopy(frags) if f is not None]
    if not active_frags:
        return None, []

    # Helper to get star atom information
    def get_star_info_list(current_frags_list):
        star_info = []
        for i, mol in enumerate(current_frags_list):
            if mol is None: # Should not happen if active_frags is filtered
                continue
            try:
                conf = mol.GetConformer()
                if conf is None: # Ensure conformer exists
                    # print(f"Warning: Fragment {i} has no conformer. Skipping.")
                    continue
                for atom in mol.GetAtoms():
                    if atom.GetAtomicNum() == 0: # Star atom
                        neighbors = atom.GetNeighbors()
                        if not neighbors:
                            # print(f"Warning: Star atom {atom.GetIdx()} in frag {i} has no neighbors.")
                            continue
                        # Assuming star atom has only one neighbor as per typical DESERT fragment definition
                        if len(neighbors) != 1:
                            # print(f"Warning: Star atom {atom.GetIdx()} in frag {i} has {len(neighbors)} neighbors, expected 1. Taking first.")
                            pass # Or continue, depending on strictness

                        neighbor_atom = neighbors[0]
                        
                        # Ensure positions can be retrieved
                        try:
                            star_pos_rdk = conf.GetAtomPosition(atom.GetIdx())
                            nei_pos_rdk = conf.GetAtomPosition(neighbor_atom.GetIdx())
                            star_pos = np.array([star_pos_rdk.x, star_pos_rdk.y, star_pos_rdk.z])
                            nei_pos = np.array([nei_pos_rdk.x, nei_pos_rdk.y, nei_pos_rdk.z])
                        except RuntimeError:
                            # print(f"Warning: Could not get atom positions for star/neighbor in frag {i}. Skipping this star atom.")
                            continue
                        
                        star_info.append({
                            'frag_idx_in_list': i, # Index in the current active_frags list
                            'mol_obj': mol,        # Direct reference to the RDKit mol object
                            'star_atom_idx': atom.GetIdx(),
                            'neighbor_atom_idx': neighbor_atom.GetIdx(),
                            'star_pos': star_pos,
                            'neighbor_pos': nei_pos
                        })
            except Exception as e_star_info:
                # print(f"Error processing fragment {i} for star info: {e_star_info}")
                pass # Continue to next fragment
        return star_info

    # Helper for distance calculation (original logic from user's function)
    def distance_metric(s1_data, s2_data):
        # Original distance: dist(star1_pos, star2_neighbor_pos) + dist(star1_neighbor_pos, star2_pos)
        # Simplified from the original d_mat calculation, which was more complex.
        # This needs to match the intended metric from the user's version for fair comparison/optimization.
        # The user's d_mat was (N, N, N_star1, N_star2), with a specific indexing.
        # Let's use a common metric for now: distance between the two star atoms.
        # More sophisticated metrics can be reintroduced if this isn't representative.
        dist_star1_nei2 = np.sqrt(np.sum((s1_data['star_pos'] - s2_data['neighbor_pos'])**2))
        dist_nei1_star2 = np.sqrt(np.sum((s1_data['neighbor_pos'] - s2_data['star_pos'])**2))
        return dist_star1_nei2 + dist_nei1_star2


    # Helper to connect two molecules (user's original connectMols)
    def connectMols_user_version(mol1, mol2, star_atom_idx_mol1, star_atom_idx_mol2):
        try:
            if not mol1 or not mol2: return None

            dummy_atom_mol1 = mol1.GetAtomWithIdx(star_atom_idx_mol1)
            if dummy_atom_mol1.GetAtomicNum() != 0: return None
            neighbors_mol1 = dummy_atom_mol1.GetNeighbors()
            if not neighbors_mol1: return None
            attach_atom_idx_mol1 = neighbors_mol1[0].GetIdx()

            dummy_atom_mol2 = mol2.GetAtomWithIdx(star_atom_idx_mol2)
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
            
            dummy1_final_idx = star_atom_idx_mol1
            dummy2_final_idx = mol2_atom_map[star_atom_idx_mol2]
            
            indices_to_remove = sorted([dummy1_final_idx, dummy2_final_idx], reverse=True)
            
            for idx_to_remove in indices_to_remove:
                if idx_to_remove < combo_mol.GetNumAtoms() and combo_mol.GetAtomWithIdx(idx_to_remove).GetAtomicNum() == 0:
                    combo_mol.RemoveAtom(idx_to_remove)
                else:
                    return None # Failed to remove dummy atom properly

            result_mol = combo_mol.GetMol()
            Chem.SanitizeMol(result_mol)
            return result_mol
        except Exception as e:
            # print(f"Error in connectMols_user_version: {e}")
            return None

    made_a_connection_in_overall_process = True # Loop control
    connection_attempts_max_passes = len(active_frags) * 2 # Heuristic to prevent infinite loops if logic error

    current_pass = 0
    while len(active_frags) > 1 and made_a_connection_in_overall_process and current_pass < connection_attempts_max_passes:
        made_a_connection_in_overall_process = False # Reset for this pass
        current_pass +=1
        
        all_star_atoms_info = get_star_info_list(active_frags)

        if len(all_star_atoms_info) < 2: # Not enough star atoms to make a connection
            break

        potential_connections = []
        # Generate all unique pairs of star atoms from different fragments
        for i in range(len(all_star_atoms_info)):
            for j in range(i + 1, len(all_star_atoms_info)):
                star1_data = all_star_atoms_info[i]
                star2_data = all_star_atoms_info[j]

                # Ensure they are from different original fragments
                # frag_idx_in_list is the key here, as mol_obj references might change
                if star1_data['frag_idx_in_list'] == star2_data['frag_idx_in_list']:
                    continue
                
                # Check if the fragments themselves are still valid (not None)
                # This is more of a safeguard; active_frags should not contain None
                if active_frags[star1_data['frag_idx_in_list']] is None or \
                   active_frags[star2_data['frag_idx_in_list']] is None:
                    continue

                dist = distance_metric(star1_data, star2_data)
                # Threshold from original code was implicitly part of d_mat logic,
                # here we might need an explicit one if desired, e.g. if dist < MAX_CONNECTION_DISTANCE:
                potential_connections.append({
                    'dist': dist,
                    's1_data': star1_data, # Contains mol_obj, star_atom_idx, frag_idx_in_list
                    's2_data': star2_data  # Contains mol_obj, star_atom_idx, frag_idx_in_list
                })
        
        if not potential_connections:
            break # No potential connections found in this pass

        potential_connections.sort(key=lambda x: x['dist'])

        for attempt in potential_connections:
            s1_details = attempt['s1_data']
            s2_details = attempt['s2_data']

            # Retrieve original molecule objects using their *indices* from when all_star_atoms_info was created.
            # These indices refer to the active_frags list *at the beginning of this while loop iteration*.
            # Need to be careful if active_frags is modified in place. It's safer to get fresh objects or re-index.

            # Let's get the actual Mol objects from the current active_frags list using the indices.
            # This is critical because after a merge, indices shift.
            # However, star1_data['mol_obj'] and star2_data['mol_obj'] should be direct references
            # to the Mol objects in active_frags *at the time get_star_info_list was called*.
            # The critical check is whether these Mol objects are still part of the *current* active_frags.

            mol1_to_connect = s1_details['mol_obj']
            mol2_to_connect = s2_details['mol_obj']
            
            # Check if these specific mol objects are still in active_frags
            # (they might have been consumed by a previous connection in an earlier iteration)
            # This check also implicitly handles cases where an index might now be out of bounds
            # if we were relying on sX_details['frag_idx_in_list'] against a modified active_frags.
            
            # A more robust way: find the current indices of mol1_to_connect and mol2_to_connect in active_frags
            try:
                current_idx_mol1 = active_frags.index(mol1_to_connect)
                current_idx_mol2 = active_frags.index(mol2_to_connect)
            except ValueError:
                # One or both fragments are no longer in active_frags (already merged)
                # print("    Skipping connection attempt: one or both fragments already merged.")
                continue 
            
            # Ensure they are truly different fragments in the *current* list
            if current_idx_mol1 == current_idx_mol2: # Should not happen if ValueError check passes and logic is correct
                continue


            connected_mol = connectMols_user_version(mol1_to_connect, mol2_to_connect,
                                                     s1_details['star_atom_idx'], 
                                                     s2_details['star_atom_idx'])

            if connected_mol:
                # print(f"    Successfully connected fragment (orig list idx {s1_details['frag_idx_in_list']}) with (orig list idx {s2_details['frag_idx_in_list']})")
                
                # Create a new list for active_frags
                new_active_frags = [connected_mol]
                for i, frag in enumerate(active_frags):
                    if i != current_idx_mol1 and i != current_idx_mol2:
                        new_active_frags.append(frag)
                active_frags = new_active_frags
                
                made_a_connection_in_overall_process = True
                break # Break from iterating through potential_connections; rebuild and re-sort
        
        # If the inner loop (potential_connections) completed without making a connection,
        # made_a_connection_in_overall_process will remain False, and the outer while loop will terminate.

    # Finalization: Select largest component and cap remaining star atoms
    final_mol = None
    discarded_frags_after_connection = []

    if not active_frags:
        return None, frags # Return original frags as discarded if all processing failed

    if len(active_frags) > 1:
        # print(f"  Multiple ({len(active_frags)}) disconnected components remain. Selecting the largest by atom count.")
        active_frags.sort(key=lambda m: m.GetNumAtoms() if m else 0, reverse=True)
        final_mol = active_frags[0]
        discarded_frags_after_connection.extend(active_frags[1:])
    elif active_frags: # Exactly one fragment remains
        final_mol = active_frags[0]
    
    if final_mol is None:
        # This case means active_frags was empty or became empty, original frags are all discarded
        return None, frags 

    # Capping logic (from user's original function)
    rw_final_mol = Chem.RWMol(final_mol)
    atoms_to_remove = [] # Dummy atoms to remove after replacing with Carbon
    for atom in rw_final_mol.GetAtoms():
        if atom.GetAtomicNum() == 0: # Star atom
            neighbors = atom.GetNeighbors()
            if not neighbors: # Dangling star atom, replace with Carbon directly.
                rw_final_mol.ReplaceAtom(atom.GetIdx(), Chem.Atom(6)) # Replace with Carbon
            else:
                # This part of the original logic was a bit unclear if it intended to cap or remove.
                # Given typical DESERT, star atoms are connection points. If they remain, they usually mean
                # an unformed bond. Capping with Carbon is a common strategy.
                # The original code was complex here with remove_idx and bond_to_remove.
                # Simpler: if a star atom is left, it means it didn't connect. Cap it.
                rw_final_mol.ReplaceAtom(atom.GetIdx(), Chem.Atom(6)) # Cap with Carbon
    
    try:
        # Get the molecule from RWMol before sanitization
        final_mol_capped = rw_final_mol.GetMol()
        if final_mol_capped:
            Chem.SanitizeMol(final_mol_capped)
            # print(f"Final connected and capped SMILES: {Chem.MolToSmiles(final_mol_capped)}")
            return final_mol_capped, discarded_frags_after_connection
        else:
            # print("Warning: Capping resulted in an invalid molecule.")
            return final_mol, discarded_frags_after_connection # Return pre-cap version

    except Exception as e_sanitize:
        # print(f"    Warning: Sanitization of final capped molecule failed: {e_sanitize}. Returning pre-cap or pre-sanitize molecule.")
        # Try to return the molecule from rw_final_mol even if sanitization fails
        try:
            return rw_final_mol.GetMol(), discarded_frags_after_connection
        except: # If GetMol itself fails after bad edits
            return final_mol, discarded_frags_after_connection # Fallback to molecule before capping attempt

    # Fallback if all else fails
    return None, frags
# --- END: User-provided connect_fragments function ---

# --- BEGIN: Helper function to convert sequence to RDKit Mols ---
def _convert_sequence_to_mols(frag_sequence, fragment_vocab, rotation_bin_data, device='cpu'):
    """
    Converts a generated fragment sequence into a list of RDKit Mol objects.
    Args:
        frag_sequence (list): List of (frag_id, trans_id, rot_id) tuples.
        fragment_vocab (dict): Vocabulary mapping fragment names/IDs to templates.
        rotation_bin_data (np.ndarray): Loaded rotation bin matrices.
        device (str): Device for temporary tensors if any.
    Returns:
        list: A list of RDKit Mol objects.
    """
    ret_frags = []
    if not fragment_vocab or rotation_bin_data is None:
        print("Error (_convert_sequence_to_mols): Vocab or rotation_bin_data not initialized.")
        return ret_frags
    id_to_key_map = {val[2]: key for key, val in fragment_vocab.items() if isinstance(val, (list, tuple)) and len(val) >= 3}

    for unit in frag_sequence:
        idx, tr_bin, rm_bin = unit
        token_key = id_to_key_map.get(idx)
        if token_key is None or token_key in ['UNK', 'BOS', 'BOB', 'EOB', 'PAD', 'EOS']:
            if token_key == 'EOS': break
            continue
        frag_mol_template_container = fragment_vocab.get(token_key)
        frag_mol_template = None
        if frag_mol_template_container:
            if isinstance(frag_mol_template_container, (list, tuple)) and len(frag_mol_template_container) > 0 and hasattr(frag_mol_template_container[0], 'GetConformer'):
                frag_mol_template = frag_mol_template_container[0]
            elif hasattr(frag_mol_template_container, 'GetConformer'):
                frag_mol_template = frag_mol_template_container
        if frag_mol_template is None: continue
        frag = copy.deepcopy(frag_mol_template)
        conformer = frag.GetConformer() 
        if conformer is None: 
            AllChem.Compute2DCoords(frag)
            if AllChem.EmbedMolecule(frag, AllChem.ETKDG()) == -1: continue
            try: AllChem.UFFOptimizeMolecule(frag) 
            except Exception: pass
            conformer = frag.GetConformer()
            if conformer is None: continue 
        box_size_for_coords = 28 
        grid_res_for_coords = 0.5
        grid_coords = bin_to_grid_coords(tr_bin, box_size_for_coords) 
        tr_real_coords = grid_coords_to_real_coords(grid_coords, box_size_for_coords, grid_res_for_coords)
        rm_matrix = bin_to_rotation_mat(rm_bin, rotation_bin_data)
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
# --- END: Helper function to convert sequence to RDKit Mols ---

# --- BEGIN: Docking functions (copied/adapted from run_all_desert_inference.py) ---
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
            # print(f"    Error converting molecule to PDBQT: {process.stderr}")
            return None
        with open(temp_pdbqt_file, "r") as f:
            pdbqt_content = f.read()
        if "ATOM" not in pdbqt_content and "HETATM" not in pdbqt_content:
            # print("    Error: Generated PDBQT file does not contain valid atom entries")
            return None
        return pdbqt_content
    except Exception as e:
        # print(f"    Error preparing ligand: {str(e)}")
        return None
    finally:
        for f_path in [temp_mol_file, temp_pdbqt_file]:
            if os.path.exists(f_path):
                try:
                    os.remove(f_path)
                except OSError:
                    # print(f"    Warning: Could not remove temporary file {f_path}")
                    pass # Non-critical

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
                            # print(f"    Could not parse score from VINA output line: '{line.strip()}'. Error: {e_parse}")
                            pass 
        
        if score is None and process.returncode != 0:
             # print(f"    No docking score found and QVina2 reported an error for {smiles}.")
             return None 
        elif score is None:
             # print(f"    No docking score found in output for {smiles}, but QVina2 ran (exit code {process.returncode}).")
             pass # Keep score as None

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
                     # print(f"    Warning: Could not remove temporary file {f_path}")
                     pass # Non-critical
# --- END: Docking functions ---

# --- BEGIN: New Main Inference Function --- 
def run_desert_inference(
    model_path: str,
    shape_patches_path: str,
    vocab_path: str, # For fragment names and SHAPE_UTILS_VOCAB
    rotation_bin_path: str, # For SHAPE_UTILS_ROTATION_BIN
    receptor_info_csv: str | None = None, # Optional: for docking
    receptors_dir: str | None = None, # Optional: for docking
    device: str = 'cuda',
    max_length: int = 50,
    use_zero_memory: bool = False,
    # Parameters for get_rotation_bins_local if rotation_bin_path doesn't exist
    rot_bin_sp_param: int = 11, 
    rot_bin_rp_param: int = 24,
    # Parameters for _get_3d_frags_local (if different from decoder defaults)
    frag_conn_box_size: int = 28,
    frag_conn_grid_res: float = 0.5
):
    """
    Main function to run DESERT inference, including model loading, sequence generation,
    fragment connection, QED calculation, and optional docking.
    """
    global SHAPE_UTILS_VOCAB, SHAPE_UTILS_VOCAB_R, SHAPE_UTILS_ROTATION_BIN
    results = {
        'input_model_path': model_path,
        'input_shape_patches_path': shape_patches_path,
        'generated_sequence_raw': None,
        'rdkit_fragments_before_connection': [],
        'connected_smiles': None,
        'qed_score': None,
        'docking_score': None,
        'discarded_fragments_smiles': [],
        'error_message': None
    }

    try:
        print(f"--- Running DESERT Inference ---")
        print(f"Device: {device}")
        print(f"Model: {model_path}")
        print(f"Shape Patches: {shape_patches_path}")
        if use_zero_memory:
            print("Mode: ZERO MEMORY (encoder bypassed)")

        # 1. Initialize SHAPE_UTILS_VOCAB and SHAPE_UTILS_ROTATION_BIN for fragment connection
        if not os.path.exists(vocab_path):
            results['error_message'] = f"Vocabulary file for fragment connection not found: {vocab_path}"
            print(f"Error: {results['error_message']}")
            return results
        
        print(f"Loading SHAPE_UTILS vocabulary from: {vocab_path}")
        with open(vocab_path, 'rb') as f_vocab_su:
            SHAPE_UTILS_VOCAB = pickle.load(f_vocab_su)
        SHAPE_UTILS_VOCAB_R = {v_item[2]: k_item for k_item, v_item in SHAPE_UTILS_VOCAB.items() if isinstance(v_item, (list, tuple)) and len(v_item) >=3}
        print("SHAPE_UTILS vocabulary loaded.")

        if not os.path.exists(rotation_bin_path):
            print(f"Rotation bin file not found at {rotation_bin_path}. Generating...")
            try:
                generated_rotation_bin = get_rotation_bins_local(sp=rot_bin_sp_param, rp=rot_bin_rp_param)
                with open(rotation_bin_path, 'wb') as f_rot_bin_out:
                    pickle.dump(generated_rotation_bin, f_rot_bin_out)
                print(f"Generated and saved rotation_bin.pkl to {rotation_bin_path}")
                SHAPE_UTILS_ROTATION_BIN = generated_rotation_bin
            except Exception as e_rot_gen:
                results['error_message'] = f"Error generating rotation_bin.pkl: {e_rot_gen}"
                print(f"Error: {results['error_message']}")
                return results
        else:
            print(f"Loading SHAPE_UTILS rotation bin from: {rotation_bin_path}")
            with open(rotation_bin_path, 'rb') as f_rot_bin_in:
                SHAPE_UTILS_ROTATION_BIN = pickle.load(f_rot_bin_in)
        print("SHAPE_UTILS rotation bin loaded.")

        # 2. Generate sequences using the full model (encoder + decoder or decoder only)
        raw_sequences = _generate_sequences_from_full_model(
            model_path=model_path,
            shape_patches_path=shape_patches_path,
            device=device,
            max_length=max_length,
            use_zero_memory=use_zero_memory
        )
        results['generated_sequence_raw'] = raw_sequences

        if not raw_sequences or not raw_sequences[0]:
            results['error_message'] = "Model did not generate any sequence."
            print(f"Info: {results['error_message']}")
            return results
        
        # Assuming we process the first sequence if multiple are returned (e.g. batch_size > 1)
        # The _generate_sequences_from_full_model is set up to return a list of sequences for batch compatibility
        # but for single inference, it's a list containing one sequence.
        sequence_to_process = raw_sequences[0]
        print(f"Generated sequence (length {len(sequence_to_process)}): {sequence_to_process[:5]}...") # Print first 5 for brevity

        # 3. Convert sequence to RDKit Mol objects
        print("Converting sequence to RDKit Mol objects...")
        # _get_3d_frags_local uses the global SHAPE_UTILS_VOCAB and SHAPE_UTILS_ROTATION_BIN
        # It also needs box_size and grid_resolution for bin_to_grid_coords, bin_to_real_coords
        # These were hardcoded to 28 and 0.5 in _get_3d_frags_local, ensure it aligns or pass them.
        # For now, relying on its internal defaults or add params to _get_3d_frags_local if needed.
        rdkit_frags = _get_3d_frags_local(sequence_to_process) # Uses globals
        
        if not rdkit_frags:
            results['error_message'] = "Could not convert generated sequence to RDKit fragments."
            print(f"Error: {results['error_message']}")
            return results
        results['rdkit_fragments_before_connection'] = [Chem.MolToSmiles(f) if f else None for f in rdkit_frags]
        print(f"Successfully converted {len(rdkit_frags)} fragments to RDKit Mol objects.")

        # 4. Connect fragments
        print("Attempting to connect fragments...")
        connected_mol, discarded_frags_mols = connect_fragments_local(rdkit_frags)
        results['discarded_fragments_smiles'] = [Chem.MolToSmiles(f) if f else None for f in discarded_frags_mols]

        if not connected_mol:
            results['error_message'] = "Fragment connection resulted in None."
            print(f"Error: {results['error_message']}")
            # Still return other results like raw sequence and discarded frags
            return results
        
        try:
            connected_smiles = Chem.MolToSmiles(connected_mol)
            results['connected_smiles'] = connected_smiles
            print(f"Connected Molecule SMILES: {connected_smiles}")
        except Exception as e_smiles:
            results['error_message'] = f"Error converting connected molecule to SMILES: {e_smiles}"
            print(f"Error: {results['error_message']}")
            # Return even if SMILES conversion fails, connected_mol might still be useful
            return results

        # 5. Calculate QED
        try:
            qed_score = QED.qed(connected_mol)
            results['qed_score'] = round(qed_score, 3)
            print(f"QED Score: {results['qed_score']}")
        except Exception as e_qed:
            print(f"Error calculating QED: {e_qed}")
            results['qed_score'] = "QEDError"

        # 6. Perform Docking (optional)
        if receptor_info_csv and receptors_dir:
            print("Attempting docking...")
            receptor_centers_df = None
            if os.path.exists(receptor_info_csv):
                try:
                    receptor_centers_df = pd.read_csv(receptor_info_csv)
                    if 'pdb' in receptor_centers_df.columns:
                        receptor_centers_df.set_index('pdb', inplace=True)
                    else:
                        print(f"Warning: 'pdb' column not found in {receptor_info_csv}. Cannot map shape file to receptor center.")
                        receptor_centers_df = None
                except Exception as e_csv_dock:
                    print(f"Error loading receptor center CSV for docking {receptor_info_csv}: {e_csv_dock}")
            
            current_receptor_path = None
            current_receptor_center = None
            try:
                shape_basename = os.path.basename(shape_patches_path)
                receptor_id_from_shape = shape_basename.split('_shapes.pkl')[0].split('.pkl')[0].split('.patch')[0]
                current_receptor_path = os.path.join(receptors_dir, f"{receptor_id_from_shape}.pdbqt")
                
                if receptor_centers_df is not None and receptor_id_from_shape in receptor_centers_df.index:
                    center_info = receptor_centers_df.loc[receptor_id_from_shape]
                    if all(c in center_info and pd.notna(center_info[c]) for c in ['c1', 'c2', 'c3']):
                        current_receptor_center = [float(center_info['c1']), float(center_info['c2']), float(center_info['c3'])]
                    else:
                        print(f"Warning: Center coords missing/invalid for {receptor_id_from_shape}")
                else:
                    print(f"Warning: Center info for {receptor_id_from_shape} not in CSV.")
            except Exception as e_rec_info_dock:
                print(f"Error determining receptor path/center for docking: {e_rec_info_dock}")

            if current_receptor_path and os.path.exists(current_receptor_path) and current_receptor_center:
                print(f"Docking with Receptor: {current_receptor_path}, Center: {current_receptor_center}")
                docking_score = dock_best_molecule_local(connected_mol, current_receptor_path, current_receptor_center)
                if docking_score is not None:
                    results['docking_score'] = round(docking_score, 3)
                    print(f"Docking Score: {results['docking_score']}")
                else:
                    results['docking_score'] = "DockingFailed"
                    print("Docking failed.")
            else:
                results['docking_score'] = "DockingSkipped (Receptor/Center Missing or Invalid)"
                print(f"Skipping docking: Receptor path ({current_receptor_path}) or center ({current_receptor_center}) invalid.")
        else:
            print("Skipping docking: receptor_info_csv or receptors_dir not provided.")
            results['docking_score'] = "DockingNotAttempted (No Receptor Info)"

        print(f"--- DESERT Inference Finished ---")
        return results

    except Exception as e_main_run:
        results['error_message'] = f"Overall error in run_desert_inference: {str(e_main_run)}"
        print(f"Critical Error: {results['error_message']}")
        import traceback
        traceback.print_exc()
        return results
# --- END: New Main Inference Function --- 

def main():
    # global SHAPE_UTILS_VOCAB, SHAPE_UTILS_VOCAB_R, SHAPE_UTILS_ROTATION_BIN # Globals are handled within run_desert_inference
    parser = argparse.ArgumentParser(description='Run full DESERT inference with encoder and decoder.')
    parser.add_argument('--model_path', type=str, 
                        default="/workspace/data/desert/1WW_30W_5048064.pt",
                        help='Path to the model checkpoint (containing both encoder and decoder weights)')
    parser.add_argument('--shape_patches_path', type=str,
                        default="/workspace/data/3tym_A_shapes.pkl",
                        help='Path to the shape patches file (e.g., .pkl)')
    parser.add_argument('--vocab_path', type=str, 
                        default=None, 
                        help='Path to the vocabulary pkl file (for fragment names and SHAPE_UTILS_*). If None, uses vocab.pkl in model_path dir.')
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
    # Args for get_rotation_bins_local generation if file not found
    parser.add_argument('--rot_bin_sp', type=int, default=11, help='sp parameter for get_rotation_bins_local if generating file.')
    parser.add_argument('--rot_bin_rp', type=int, default=24, help='rp parameter for get_rotation_bins_local if generating file.')

    args = parser.parse_args()
    
    # Call the main inference function
    inference_results = run_desert_inference(
        model_path=args.model_path,
        shape_patches_path=args.shape_patches_path,
        vocab_path=args.vocab_path, # Pass the resolved vocab path
        rotation_bin_path=args.rotation_bin_path,
        receptor_info_csv=args.receptor_info_csv,
        receptors_dir=args.receptors_dir,
        device=args.device,
        max_length=args.max_length,
        use_zero_memory=args.use_zero_memory,
        rot_bin_sp_param=args.rot_bin_sp,
        rot_bin_rp_param=args.rot_bin_rp
    )

    print("\n--- Inference Results Summary ---")
    if inference_results:
        for key, value in inference_results.items():
            if key == 'generated_sequence_raw' and value and isinstance(value, list) and value[0]:
                print(f"{key}: First sequence (first 5 elements): {value[0][:5]}... (Length: {len(value[0])})")
            elif key == 'rdkit_fragments_before_connection' and value:
                 print(f"{key}: ({len(value)} fragments) First 5: {[f[:30] + '...' if f and len(f)>30 else f for f in value[:5]]}") # Truncate long SMILES
            elif key == 'discarded_fragments_smiles' and value:
                 print(f"{key}: ({len(value)} fragments) First 5: {[f[:30] + '...' if f and len(f)>30 else f for f in value[:5]]}")
            else:
                print(f"{key}: {value}")
    else:
        print("Inference did not return results.")

if __name__ == "__main__":
    main() 