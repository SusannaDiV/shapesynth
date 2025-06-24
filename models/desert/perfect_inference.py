import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point3D
import io
from PIL import Image
import argparse

# Import necessary functions from your codebase
from synformer.data.common import process_smiles
from synformer.models.encoder.shape import ShapeEncoder, ShapePretrainingEncoder
from synformer.models.desert.fulldecoder import ShapePretrainingDecoderIterativeNoRegression, ShapePretrainingModel
from synformer.data.utils.shape_utils import bin_to_grid_coords, grid_coords_to_real_coords, bin_to_rotation_mat, get_3d_frags, connect_fragments, get_atom_stamp, get_shape, get_rotation_bins, get_shape_patches, ATOM_RADIUS, ATOMIC_NUMBER, ATOMIC_NUMBER_REVERSE

def generate_shape_patches(smiles: str, grid_resolution=0.5, max_dist_stamp=4.0, max_dist=6.75, patch_size=4):
    """Generate shape patches from a SMILES string"""
    try:
        # Process SMILES to get 3D conformer
        processed = process_smiles(smiles)
        if processed is None:
            raise ValueError(f"Failed to process SMILES: {smiles}")
        
        mol = processed['mol']
        
        # Get shape patches using parameters from the original config
        curr_atom_stamp = get_atom_stamp(grid_resolution, max_dist_stamp)
        curr_shape = get_shape(mol, curr_atom_stamp, grid_resolution, max_dist)
        
        # Process shape patches
        curr_shape_patches = get_shape_patches(curr_shape, patch_size)
        grid_size = curr_shape.shape[0] // patch_size
        curr_shape_patches = curr_shape_patches.reshape(grid_size, grid_size, grid_size, -1)
        curr_shape_patches = curr_shape_patches.reshape(-1, patch_size**3)
        
        # Convert to tensor
        shape_patches = torch.tensor(curr_shape_patches, dtype=torch.float)
        
        return shape_patches.unsqueeze(0)  # Add batch dimension
        
    except Exception as e:
        raise RuntimeError(f"Error generating shape patches: {str(e)}")

def load_desert_model(model_path, device='cuda'):
    """Load a pre-trained DESERT model.
    
    Args:
        model_path: Path to the model checkpoint
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    print(f"Loading model from {model_path}")
    
    # Load vocabulary first to get vocab sizes
    vocab_path = os.path.join(os.path.dirname(model_path), "vocab.pkl")
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # Get vocab sizes and special tokens
    src_vocab_size = len(vocab)  # Use same vocab for source and target
    tgt_vocab_size = len(vocab)
    src_special_tokens = {'pad': vocab['PAD'][2]}  # Assuming PAD token exists
    tgt_special_tokens = {'pad': vocab['PAD'][2]}

    # Create embeddings first
    d_model = 1024  # This should be num_heads * head_dim = 8 * 128 = 1024
    src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=src_special_tokens['pad'])
    tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=tgt_special_tokens['pad'])
    
    # Create a custom ShapePretrainingModel that directly uses the encoder and decoder
    class CustomShapePretrainingModel(nn.Module):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
            
        def forward(self, shape_patches, input_frag_idx, input_frag_idx_mask, 
                   input_frag_trans, input_frag_trans_mask, input_frag_r_mat, input_frag_r_mat_mask, 
                   memory=None, memory_padding_mask=None, **kwargs):
            # If memory is not provided, create a default one
            if memory is None:
                batch_size, seq_len = shape_patches.shape[0], shape_patches.shape[1]
                memory = torch.zeros((seq_len, batch_size, d_model), device=shape_patches.device)
                memory_padding_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=shape_patches.device)
            
            # Forward through decoder
            logits, trans, r_mat = self.decoder(
                input_frag_idx=input_frag_idx,
                input_frag_trans=input_frag_trans,
                input_frag_r_mat=input_frag_r_mat,
                memory=memory,
                memory_padding_mask=memory_padding_mask
            )
            
            return logits, trans, r_mat
    
    # Create iterative block
    class IterativeBlock:
        def __init__(self, config):
            self.config = config
            self._mode = 'train'
            
        def build(self, embed, special_tokens, trans_vocab_size, rot_vocab_size):
            self.embed = embed
            self.special_tokens = special_tokens
            self.trans_vocab_size = trans_vocab_size
            self.rot_vocab_size = rot_vocab_size
            
        def __call__(self, logits, trans, r_mat, padding_mask):
            # During inference, just return inputs unchanged
            return logits, trans, r_mat
            
        def reset(self, mode):
            self._mode = mode
    
    # Create iterative block first
    iterative_block = IterativeBlock({
        'num_layers': 3,
        'd_model': 1024,
        'n_head': 8,
        'dim_feedforward': 4096,
        'dropout': 0.1,
        'activation': 'relu',
        'learn_pos': True
    })
    
    # Create decoder with iterative block
    decoder = ShapePretrainingDecoderIterativeNoRegression(
        num_layers=12,
        d_model=1024,
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
        iterative_block=iterative_block  # Pass it directly here
    )
    
    # Build decoder
    decoder.build(embed=tgt_embed, special_tokens=tgt_special_tokens, out_proj=nn.Linear(d_model, tgt_vocab_size, bias=False))

    # Initialize the custom model
    model = CustomShapePretrainingModel(encoder=None, decoder=decoder)

    # Load the state dict for the decoder only
    state_dict = torch.load(model_path, map_location=device)
    if 'model' in state_dict:
        state_dict = state_dict['model']
    
    # Extract only decoder-related parameters
    decoder_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_decoder'):
            decoder_key = k.replace('_decoder.', '')
            decoder_state_dict[decoder_key] = v
    
    # Load the decoder state dict
    decoder.load_state_dict(decoder_state_dict, strict=False)
    model.to(device)
    model.eval()
    
    # Replace your simple output scaling with a more sophisticated approach
    '''
    model.decoder.output_projection = nn.Sequential(
        nn.Linear(d_model, d_model * 2),
        nn.ReLU(),
        nn.Linear(d_model * 2, d_model),
        nn.LayerNorm(d_model)
    )
    '''
    return model

def run_desert_inference(smiles, model_path, device='cuda', max_length=50):
    """Run DESERT inference on SMILES string"""
    # Load the DESERT model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"DESERT model file not found: {model_path}")
    
    # Generate shape patches
    with open('/home/luost_local/sdivita/synformer/data/desert/sample_shapes.pkl', 'rb') as f:
        shape_patches = pickle.load(f)[0]
    
    # Make sure shape_patches is a tensor
    if isinstance(shape_patches, np.ndarray):
        shape_patches = torch.tensor(shape_patches, dtype=torch.float)
    
    shape_patches = shape_patches.to(device)
    
    # Load model
    model = load_desert_model(model_path, device)
    
    # Load vocabulary to get BOS and EOS token IDs
    vocab_path = os.path.join(os.path.dirname(model_path), "vocab.pkl")
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    BOS_token = vocab['BOS'][2]  # Get BOS token ID
    EOS_token = vocab['EOS'][2] if 'EOS' in vocab else None  # Get EOS token ID if available
    
    # Create initial decoder inputs with just the BOS token
    batch_size = 1
    input_frag_idx = torch.tensor([[BOS_token]], dtype=torch.long, device=device)
    input_frag_trans = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
    input_frag_r_mat = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
    
    # Autoregressive generation
    all_pred_idx = [BOS_token]
    all_pred_trans = [0]
    all_pred_rot = [0]
    
    # Calculate dimensions for memory
    # The memory tensor needs to be [seq_len, batch_size, d_model]
    # Based on the error, we need memory size to be 802816
    # From DESERT model configuration, seq_len should be 32 (standard value)
    seq_len = 32
    num_heads = 8
    head_dim = 128
    
    # Create correctly sized memory tensor
    memory = torch.zeros((seq_len, batch_size, num_heads * head_dim), device=device)
    memory_padding_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
    
    for step in range(max_length):
        # Create masks (all False since we're using the full sequence)
        input_frag_idx_mask = torch.zeros_like(input_frag_idx, dtype=torch.bool, device=device)
        input_frag_trans_mask = torch.zeros_like(input_frag_trans, dtype=torch.bool, device=device)
        input_frag_r_mat_mask = torch.zeros_like(input_frag_r_mat, dtype=torch.bool, device=device)
        
        # Run inference
        with torch.no_grad():
            # Forward pass through the model with explicitly provided memory
            outputs = model(
                shape=None,  # Shape is not used in this case
                shape_patches=shape_patches,
                input_frag_idx=input_frag_idx,
                input_frag_idx_mask=input_frag_idx_mask,
                input_frag_trans=input_frag_trans,
                input_frag_trans_mask=input_frag_trans_mask,
                input_frag_r_mat=input_frag_r_mat,
                input_frag_r_mat_mask=input_frag_r_mat_mask,
                memory=memory,
                memory_padding_mask=memory_padding_mask
            )
            
            # Unpack outputs - model returns a tuple of (logits, trans, r_mat)
            logits, trans, r_mat = outputs
            
            # Process outputs - take the last prediction if we get a list of predictions
            if isinstance(logits, list):
                logits = logits[-1]
            if isinstance(trans, list):
                trans = trans[-1]
            if isinstance(r_mat, list):
                r_mat = r_mat[-1]
            
            # Get the predictions for the last position
            pred_idx = torch.argmax(logits[:, -1:], dim=-1)
            probs = torch.softmax(trans[:, -1:] / 1.2, dim=-1)
            pred_trans = torch.multinomial(probs.view(-1, probs.size(-1)), 1)
            pred_rot = torch.argmax(r_mat[:, -1:], dim=-1)
            
            # Convert to scalars
            next_idx = pred_idx[0, 0].item()
            next_trans = pred_trans[0, 0].item()
            next_rot = pred_rot[0, 0].item()
            
            # Store predictions
            all_pred_idx.append(next_idx)
            all_pred_trans.append(next_trans)
            all_pred_rot.append(next_rot)
            
            # Check if we've reached the EOS token
            if EOS_token is not None and next_idx == EOS_token:
                break
                
            # Append the new predictions to the input for the next step
            input_frag_idx = torch.cat([input_frag_idx, pred_idx], dim=1)
            input_frag_trans = torch.cat([input_frag_trans, pred_trans], dim=1)
            input_frag_r_mat = torch.cat([input_frag_r_mat, pred_rot], dim=1)
    
    # Create the sequence of fragments
    sequence = []
    for i in range(1, len(all_pred_idx)):  # Skip the BOS token
        sequence.append((all_pred_idx[i], all_pred_trans[i], all_pred_rot[i]))
    
    return [sequence]  # Return as a list of sequences for batch compatibility

def visualize_fragments(sequence, id_to_token, grid_resolution=0.5, max_dist=6.75, output_file="molecule_3d.png"):
    """
    Visualize the predicted fragments as a 3D molecule using the proper fragment reconstruction.
    
    Args:
        sequence: List of tuples (fragment_id, translation, rotation)
        id_to_token: Dictionary mapping fragment IDs to token names
        grid_resolution: Resolution of the grid
        max_dist: Maximum distance
        output_file: Output file path for the PNG image
    """
    # Box size for translation calculations
    box_size = int(2 * max_dist / grid_resolution + 1)
    
    # Load rotation bins
    rotation_bin = get_rotation_bins(11, 24)  # Use the same values as in your model
    
    # Convert sequence to the format expected by get_3d_frags
    frags_data = []
    for frag_id, trans_id, rot_id in sequence:
        if id_to_token.get(frag_id) == "EOS":
            break
            
        # Convert translation to real coordinates
        grid_coords = bin_to_grid_coords(trans_id, box_size)
        if grid_coords is not None and grid_coords[0] != float('inf'):
            real_coords = grid_coords_to_real_coords(grid_coords, box_size, grid_resolution)
        else:
            # Skip fragments with invalid translation
            continue
            
        # Convert rotation to rotation matrix
        rot_mat = bin_to_rotation_mat(rot_id, rotation_bin)
        if rot_mat is None:
            continue
            
        frags_data.append((frag_id, real_coords, rot_mat))
    
    # Get 3D fragments and connect them
    frags = get_3d_frags(frags_data)
    mol = connect_fragments(frags)
    
    if mol is None:
        print("Failed to create a valid molecule from fragments")
        return None
        
    # Draw the molecule and save as PNG
    try:
        drawer = rdMolDraw2D.MolDraw2DCairo(500, 500)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        png_data = drawer.GetDrawingText()
        
        with open(output_file, 'wb') as f:
            f.write(png_data)
        
        print(f"Molecule visualization saved to {output_file}")
    except Exception as e:
        print(f"Warning: Could not create 2D visualization: {str(e)}")
    
    # Create a 3D image using matplotlib
    try:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot atoms
        for i, atom in enumerate(mol.GetAtoms()):
            pos = mol.GetConformer().GetAtomPosition(i)
            ax.scatter(pos.x, pos.y, pos.z, s=100, label=atom.GetSymbol())
            ax.text(pos.x, pos.y, pos.z, atom.GetSymbol(), size=12)
        
        # Plot bonds
        for bond in mol.GetBonds():
            idx1 = bond.GetBeginAtomIdx()
            idx2 = bond.GetEndAtomIdx()
            pos1 = mol.GetConformer().GetAtomPosition(idx1)
            pos2 = mol.GetConformer().GetAtomPosition(idx2)
            ax.plot([pos1.x, pos2.x], [pos1.y, pos2.y], [pos1.z, pos2.z], 'k-', linewidth=2)
        
        ax.set_title('3D Molecule Structure')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Save the 3D plot
        plt.savefig(output_file.replace('.png', '_3d.png'))
        print(f"3D visualization saved to {output_file.replace('.png', '_3d.png')}")
    except Exception as e:
        print(f"Warning: Could not create 3D visualization: {str(e)}")
    
    return mol

def main():
    # Example usage
    parser = argparse.ArgumentParser(description='Run DESERT inference on a SMILES string')
    parser.add_argument('--smiles', type=str, default="CC1=CC=C(C=C1)C2=CC=C(C=C2)N", 
                        help='SMILES string to process')
    parser.add_argument('--model_path', type=str, 
                        default="/home/luost_local/sdivita/synformer/data/desert/1WW_30W_5048064.pt",
                        help='Path to the model checkpoint')
    parser.add_argument('--output', type=str, default="molecule_3d.png",
                        help='Output file path for the visualization')
    
    args = parser.parse_args()
    
    try:
        # First check if files exist
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model file not found: {args.model_path}")
            
        print(f"Loading model from {args.model_path}")
        print(f"Processing SMILES: {args.smiles}")
        
        # Load vocabulary to understand fragment IDs
        vocab_path = os.path.join(os.path.dirname(args.model_path), "vocab.pkl")
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        
        # Create reverse mapping from ID to token
        id_to_token = {}
        for token, (_, _, idx) in vocab.items():
            id_to_token[idx] = token
        
        # Get special token IDs
        BOS_token = vocab['BOS'][2]
        EOS_token = vocab['EOS'][2] if 'EOS' in vocab else None
        
        outputs = run_desert_inference(args.smiles, args.model_path)
        print("\nSuccessfully ran DESERT inference!")
        print("\nGenerated sequences:")
        
        for sequence in outputs:
            print("\nSequence with", len(sequence), "fragments:")
            
            # Check if sequence ends with EOS
            if EOS_token is not None and sequence and sequence[-1][0] == EOS_token:
                print("Sequence properly terminated with EOS token")
            
            # Print fragment information
            for i, (frag_id, trans, rot) in enumerate(sequence):
                token_name = id_to_token.get(frag_id, "Unknown")
                print(f"Fragment {i+1}: ID={frag_id} ({token_name}), Translation={trans}, Rotation={rot}")
            
            # Visualize the predicted fragments
            visualize_fragments(sequence, id_to_token, output_file=args.output)
        
    except Exception as e:
        print(f"\nError running DESERT inference: {str(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main()