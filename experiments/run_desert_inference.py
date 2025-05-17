import os
import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from synformer.models.encoder.shape import ShapePretrainingEncoder
from synformer.data.utils.shape_utils import get_atom_stamp, get_shape, get_shape_patches, get_binary_features

class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

def load_desert_model(model_path):
    """Load the DESERT pretrained model."""
    print("Loading DESERT model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize encoder with correct parameters
    encoder = ShapePretrainingEncoder(
        patch_size=4,
        num_layers=12,
        d_model=1024,
        n_head=8,
        dim_feedforward=4096,
        dropout=0.1,
        attention_dropout=0.1,
        activation='relu',
        learn_pos=True
    )
    
    # Build the encoder
    encoder.build(
        embed=nn.Embedding(2, 1024),
        special_tokens={'pad': 0}
    )
    
    # Load state dict
    encoder_state_dict = {k[9:]: v for k, v in checkpoint['model'].items() 
                         if k.startswith('_encoder.')}
    encoder.load_state_dict(encoder_state_dict)
    
    # Move to device and set eval mode
    encoder.to(device)
    encoder.eval()
    
    return encoder

def process_smiles(smiles, grid_resolution=0.5, max_dist=6.75):
    """Process SMILES string to get shape representation."""
    print(f"Processing SMILES: {smiles}")
    
    # Create RDKit molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Failed to parse SMILES string")
    
    # Generate 3D conformer
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    
    # Center molecule
    centroid = get_mol_centroid(mol)
    coords = mol.GetConformer().GetPositions()
    coords -= centroid
    
    # Create atom stamp for shape encoding
    max_dist_stamp = 4.0  # Use a smaller max_dist for the atom stamp to avoid edge effects
    atom_stamp = get_atom_stamp(grid_resolution=grid_resolution, max_dist=max_dist_stamp)
    
    # Get shape using by_coords=True to use the centered coordinates
    shape = get_shape(mol, atom_stamp, grid_resolution, max_dist, by_coords=True, coords=coords, features=get_binary_features(mol, confId=0, without_H=False)[1])
    
    # Process shape into patches
    patch_size = 4
    shape_patches = get_shape_patches(shape, patch_size)
    grid_size = shape.shape[0] // patch_size
    
    print(f"Shape patches shape: {shape_patches.shape}")
    
    return {
        'mol': mol,
        'shape': shape,
        'shape_patches': shape_patches,
        'grid_size': grid_size
    }

def run_inference(encoder, shape_patches):
    """Run inference with the DESERT model."""
    with torch.no_grad():
        memory, memory_padding_mask = encoder(shape_patches)
    return memory, memory_padding_mask

def main():
    # Model path
    model_path = '/home/luost_local/sdivita/synformer/data/desert/1WW_30W_5048064.pt'
    
    # SMILES string to process
    smiles = 'NC(=O)c1ccc(Cl)c(CCN2CC[C@@H]3CCCC[C@H]3C2)c1'
    
    # Load model
    encoder = load_desert_model(model_path)
    
    # Process SMILES
    shape_patches = process_smiles(smiles)
    shape_patches = shape_patches['shape_patches'].to(encoder._patch_ffn.linear1.weight.device)
    
    # Run inference
    memory, memory_padding_mask = run_inference(encoder, shape_patches)
    
    # Print results
    print("Shape patches shape:", shape_patches.shape)
    print("Memory shape:", memory.shape)
    print("Memory padding mask shape:", memory_padding_mask.shape)

if __name__ == '__main__':
    main() 