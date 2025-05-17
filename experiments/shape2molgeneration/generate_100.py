import os
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
import pandas as pd
from tqdm import tqdm
import pickle as pkl
import copy
from random import sample

# Import Shape2Mol specific modules
from synformer.chem.mol import Molecule
from synformer.data.utils.shape_utils import get_atom_stamp, get_shape, get_shape_patches
from synformer.data.utils.shape_utils import ROTATIONS, centralize, get_mol_centroid, trans
from synformer.data.utils.shape_utils import get_binary_features
from synformer.data.utils.tfbio_data import make_grid
from synformer.sampler.analog.parallel import run_parallel_sampling_return_smiles


def process_binding_site(cavity_file_path, grid_resolution=0.5, max_dist_stamp=4.0, max_dist=6.75):
    """
    Process binding site from PDB file using the provided code.
    
    Args:
        cavity_file_path: Path to cavity PDB file
        grid_resolution: Resolution of the grid for shape encoding
        max_dist_stamp: Maximum distance for atom stamps
        max_dist: Maximum distance for shape encoding
        
    Returns:
        Dictionary containing processed binding site data
    """
    print(f"Processing binding site from {cavity_file_path}...")
    
    # Create atom stamp for shape encoding
    atom_stamp = get_atom_stamp(grid_resolution=grid_resolution, max_dist=max_dist_stamp)
    
    # Load cavity from PDB file
    cavity = Chem.MolFromPDBFile(cavity_file_path, proximityBonding=False)
    
    if cavity is None:
        raise ValueError("Failed to load cavity PDB file")
    
    # Center cavity
    cavity_centroid = get_mol_centroid(cavity)
    cavity = centralize(cavity)
    
    # Get cavity shape
    cavity_shape = get_shape(cavity, atom_stamp, grid_resolution, max_dist)
    
    # Process shape into patches
    patch_size = 4
    shape_patches = get_shape_patches(cavity_shape, patch_size)
    grid_size = cavity_shape.shape[0] // patch_size
    shape_patches = shape_patches.reshape(grid_size, grid_size, grid_size, -1)
    shape_patches = shape_patches.reshape(-1, patch_size**3)
    
    # Convert to tensor
    shape_patches_tensor = torch.tensor(shape_patches, dtype=torch.float)
    
    return {
        'cavity': cavity,
        'cavity_shape': cavity_shape,
        'shape_patches': shape_patches_tensor
    }


def generate_molecules_from_shape(shape_patches, model_path, num_candidates=100):
    """
    Generate molecules from shape patches using Shape2Mol.
    
    Args:
        shape_patches: Tensor of shape patches
        model_path: Path to pretrained Shape2Mol model
        num_candidates: Number of candidate molecules to generate
        
    Returns:
        List of generated molecules (as SMILES strings)
    """
    print(f"Generating {num_candidates} molecules from binding site shape...")
    
    # Create a dummy molecule object to hold the shape patches
    dummy_mol = Molecule("C")  # Placeholder SMILES
    dummy_mol.shape_patches = shape_patches
    
    # Run Shape2Mol sampling
    result_df = run_parallel_sampling_return_smiles(
        input=[dummy_mol] * num_candidates,  # Replicate to generate multiple candidates
        model_path=model_path,
        search_width=24,
        exhaustiveness=64,
        num_gpus=-1,  # Use all available GPUs
        num_workers_per_gpu=1,
        task_qsize=0,
        result_qsize=0,
        time_limit=300,  # 5 minutes time limit
        sort_by_scores=True,
    )
    
    if result_df is None or len(result_df) == 0:
        print("No molecules generated!")
        return []
    
    # Remove duplicates and return SMILES
    result_df.drop_duplicates(subset="target", inplace=True, keep="first")
    return result_df.smiles.to_list()


def process_generated_molecules(smiles_list, cavity_shape, grid_resolution=0.5, max_dist=6.75):
    """
    Process generated molecules to create 3D structures and score them based on
    shape compatibility with the binding site.
    
    Args:
        smiles_list: List of SMILES strings
        cavity_shape: Shape of the binding site cavity
        grid_resolution: Resolution of the grid
        max_dist: Maximum distance for shape encoding
        
    Returns:
        List of (molecule, score) tuples
    """
    print("Processing and scoring generated molecules...")
    atom_stamp = get_atom_stamp(grid_resolution=grid_resolution, max_dist=4.0)
    scored_mols = []
    
    for smiles in tqdm(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
                
            # Add hydrogens
            mol = Chem.AddHs(mol)
            
            # Generate 3D coordinates
            if AllChem.EmbedMolecule(mol, useRandomCoords=True) == -1:
                continue
                
            # Optimize geometry
            AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
            
            # Center molecule
            mol = centralize(mol)
            
            # Get molecule shape
            mol_shape = get_shape(mol, atom_stamp, grid_resolution, max_dist)
            
            # Calculate shape overlap with cavity
            shape_overlap = np.sum(mol_shape * cavity_shape) / np.sum(cavity_shape)
            
            # Calculate shape similarity (Tanimoto coefficient)
            shape_union = np.sum(np.logical_or(mol_shape > 0, cavity_shape > 0))
            shape_intersection = np.sum(np.logical_and(mol_shape > 0, cavity_shape > 0))
            if shape_union > 0:
                tanimoto = shape_intersection / shape_union
            else:
                tanimoto = 0
            
            # Combined score
            combined_score = 0.5 * shape_overlap + 0.5 * tanimoto
            
            # Add properties
            mol.SetProp("SMILES", smiles)
            mol.SetProp("ShapeOverlap", f"{shape_overlap:.4f}")
            mol.SetProp("ShapeTanimoto", f"{tanimoto:.4f}")
            mol.SetProp("CombinedScore", f"{combined_score:.4f}")
            
            scored_mols.append((mol, combined_score))
            
        except Exception as e:
            print(f"Error processing molecule {smiles}: {e}")
            continue
    
    # Sort by score (descending)
    scored_mols.sort(key=lambda x: x[1], reverse=True)
    return scored_mols


def main():
    # Hardcoded paths
    cavity_file_path = "/home/luost_local/sdivita/synformer/experiments/this_cavity_1.pdb"
    model_path = "/home/luost_local/sdivita/synformer/logs/shape_l/2502270533-90d9453@shapedeserencoder/2025_03_04__06_45_25/checkpoints/last.ckpt"
    output_file = "compatible_molecules.sdf"
    num_candidates = 100
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    # 1. Process binding site from PDB file
    binding_site_data = process_binding_site(cavity_file_path)
    cavity_shape = binding_site_data['cavity_shape']
    
    # 2. Generate molecules from shape using Shape2Mol
    generated_smiles = generate_molecules_from_shape(
        binding_site_data['shape_patches'], 
        model_path, 
        num_candidates=max(num_candidates * 2, 200)  # Generate extra for filtering
    )
    
    if not generated_smiles:
        print("Failed to generate molecules. Exiting.")
        return
    
    # 3. Process and score generated molecules
    scored_mols = process_generated_molecules(
        generated_smiles, 
        cavity_shape
    )
    
    if not scored_mols:
        print("No valid molecules after processing. Exiting.")
        return
    
    # 4. Select top candidates
    top_candidates = scored_mols[:num_candidates]
    
    # 5. Write output
    print(f"Writing {len(top_candidates)} compatible molecules to {output_file}")
    writer = Chem.SDWriter(output_file)
    
    # Also create a CSV file with SMILES and scores
    csv_data = []
    
    for mol, score in top_candidates:
        writer.write(mol)
        
        # Add to CSV data
        csv_data.append({
            "SMILES": mol.GetProp("SMILES"),
            "ShapeOverlap": mol.GetProp("ShapeOverlap"),
            "ShapeTanimoto": mol.GetProp("ShapeTanimoto"),
            "CombinedScore": mol.GetProp("CombinedScore")
        })
    
    writer.close()
    
    # Write CSV
    csv_file = os.path.splitext(output_file)[0] + ".csv"
    pd.DataFrame(csv_data).to_csv(csv_file, index=False)
    
    print(f"Results saved to {output_file} and {csv_file}")
    print("Done!")


if __name__ == "__main__":
    main()