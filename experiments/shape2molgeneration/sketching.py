import torch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from shape_utils import get_atom_stamp, get_shape, centralize
from experiments.shape2mol import ShapePretrainingModel

def process_cavity(cavity_file_path, grid_resolution=0.5, max_dist_stamp=4.0, max_dist=6.75):
    """Process binding site from PDB file to get shape representation."""
    print(f"Processing binding site from {cavity_file_path}...")
    
    # Create atom stamp for shape encoding
    atom_stamp = get_atom_stamp(grid_resolution=grid_resolution, max_dist=max_dist_stamp)
    
    # Load cavity from PDB file
    cavity = Chem.MolFromPDBFile(cavity_file_path, proximityBonding=False)
    if cavity is None:
        raise ValueError(f"Failed to load cavity PDB file: {cavity_file_path}")
    
    # Center cavity
    cavity = centralize(cavity)
    
    # Get cavity shape
    cavity_shape = get_shape(cavity, atom_stamp, grid_resolution, max_dist)
    
    return {
        'cavity': cavity,
        'cavity_shape': cavity_shape
    }

def generate_molecules(cavity_shape, model_path, num_candidates=100):
    """Generate molecules using ShapePretrainingModel."""
    print(f"Loading model from {model_path}...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the pretrained model
    checkpoint = torch.load(model_path, map_location=device)
    model = ShapePretrainingModel.from_pretrained(checkpoint)
    model = model.to(device)
    model.eval()
    
    print(f"Generating {num_candidates} molecules...")
    
    # Process cavity shape for the model
    # This depends on how the model expects the shape input
    # For example, we might need to convert it to patches or a specific format
    shape_input = prepare_shape_for_model(cavity_shape).to(device)
    
    generated_smiles = []
    
    with torch.no_grad():
        # Generate molecules
        for _ in range(num_candidates):
            # Use the model's generation method
            # This will vary based on the model's API
            outputs = model.generate(shape_input)
            
            # Convert outputs to SMILES
            smiles = model.decode_to_smiles(outputs)
            generated_smiles.append(smiles)
    
    # Remove duplicates
    unique_smiles = list(set(generated_smiles))
    print(f"Generated {len(unique_smiles)} unique molecules")
    
    return unique_smiles

def prepare_shape_for_model(cavity_shape):
    """
    Prepare the cavity shape in the format expected by the model.
    This function needs to be implemented based on the specific input format required by the model.
    """
    # Convert shape to patches if needed
    patch_size = 4
    grid_size = cavity_shape.shape[0] // patch_size
    
    # Reshape to get patches
    shape_patches = cavity_shape[:grid_size*patch_size, :grid_size*patch_size, :grid_size*patch_size]
    shape_patches = shape_patches.reshape(grid_size, patch_size, grid_size, patch_size, grid_size, patch_size)
    shape_patches = shape_patches.transpose(0, 2, 4, 1, 3, 5)
    shape_patches = shape_patches.reshape(grid_size**3, patch_size**3)
    
    # Convert to tensor and add batch dimension
    shape_patches_tensor = torch.tensor(shape_patches, dtype=torch.float).unsqueeze(0)
    
    return shape_patches_tensor

def process_and_score_molecules(smiles_list, cavity_shape):
    """Score generated molecules based on shape compatibility."""
    print("Processing and scoring molecules...")
    atom_stamp = get_atom_stamp(grid_resolution=0.5, max_dist=4.0)
    scored_mols = []
    
    for smiles in tqdm(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            mol = Chem.AddHs(mol)
            if AllChem.EmbedMolecule(mol, useRandomCoords=True) == -1:
                continue
            
            AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
            mol = centralize(mol)
            
            mol_shape = get_shape(mol, atom_stamp, 0.5, 6.75)
            shape_overlap = np.sum(mol_shape * cavity_shape) / np.sum(cavity_shape)
            
            mol.SetProp("SMILES", smiles)
            mol.SetProp("ShapeOverlap", f"{shape_overlap:.4f}")
            scored_mols.append((mol, shape_overlap))
            
        except Exception as e:
            print(f"Error processing molecule {smiles}: {e}")
            continue
    
    scored_mols.sort(key=lambda x: x[1], reverse=True)
    return scored_mols

def main():
    cavity_file_path = "/home/luost_local/sdivita/synformer/experiments/this_cavity_1.pdb"
    model_path = "/home/luost_local/sdivita/synformer/data/desert/1WW_30W_5048064.pt"
    output_file = "compatible_molecules.sdf"
    num_candidates = 100
    
    try:
        # Process cavity
        binding_site_data = process_cavity(cavity_file_path)
        
        # Generate molecules
        generated_smiles = generate_molecules(
            binding_site_data['cavity_shape'], 
            model_path, 
            num_candidates
        )
        
        if not generated_smiles:
            print("Failed to generate molecules.")
            return
        
        # Score molecules
        scored_mols = process_and_score_molecules(
            generated_smiles, 
            binding_site_data['cavity_shape']
        )
        
        if not scored_mols:
            print("No valid molecules after processing.")
            return
        
        # Save results
        writer = Chem.SDWriter(output_file)
        csv_data = []
        
        for mol, score in scored_mols[:num_candidates]:
            writer.write(mol)
            csv_data.append({
                "SMILES": mol.GetProp("SMILES"),
                "ShapeOverlap": mol.GetProp("ShapeOverlap")
            })
        
        writer.close()
        
        csv_file = os.path.splitext(output_file)[0] + ".csv"
        pd.DataFrame(csv_data).to_csv(csv_file, index=False)
        
        print(f"Results saved to {output_file} and {csv_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()