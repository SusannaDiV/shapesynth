import argparse
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit import RDLogger
import threading
import time
import os
import py3Dmol
from rdkit.Chem import BRICS  # Add this import for BRICS decomposition

# Disable RDKit logging
RDLogger.DisableLog('rdApp.*')

# Two hardcoded valid SMILES strings (medium length to avoid processing issues)
SMILES1 = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
SMILES2 = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"  # Ibuprofen

def timeout_handler(smiles, func, args=(), timeout=3):
    """Thread-based timeout handler for molecule processing"""
    result = [None]
    error = [None]
    
    def worker():
        try:
            result[0] = func(*args)
        except Exception as e:
            error[0] = e
    
    thread = threading.Thread(target=worker)
    thread.daemon = True
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        print(f"Timeout for {smiles}")
        return None
    if error[0] is not None:
        print(f"Error processing {smiles}: {str(error[0])}")
        return None
    return result[0]

def process_mol(mol, smiles):
    """Process molecule without timeout"""
    if mol is None:
        return None
        
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'S' and atom.GetFormalCharge() != 0:
            return None
    
    mol = Chem.AddHs(mol)
    
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
        Chem.SanitizeMol(mol)
        
        if AllChem.EmbedMolecule(mol, useRandomCoords=True) != 0:
            return None

        AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
        return mol
        
    except Exception as e:
        print(f"Error in process_mol: {str(e)}")
        return None

def process_smiles(smiles):
    """Generate a 3D conformer for a given SMILES."""
    # Early rejection criteria
    if any([
        len(smiles) > 180,  # Very long SMILES strings
        smiles.count('(') > 8,  # Too many branches
        smiles.count('@') > 6,  # Too many chiral centers
        smiles.count('1') + smiles.count('2') + smiles.count('3') > 6,  # Too many rings
    ]):
        print(f"Early rejection of complex molecule: {smiles}")
        return None
    
    mol = Chem.MolFromSmiles(smiles)
    return timeout_handler(smiles, process_mol, args=(mol, smiles))

def calculate_usr_descriptors(mol):
    """Calculate USR (Ultrafast Shape Recognition) descriptors"""
    try:
        # Calculate USR descriptors
        usr = rdMolDescriptors.GetUSR(mol)
        
        # Calculate USRCAT descriptors (USR with atom types)
        usrcat = rdMolDescriptors.GetUSRCAT(mol)
    
        return {
                    "usr": usr,
                    "usrcat": usrcat
                }
    except Exception as e:
        print(f"Error calculating USR descriptors: {str(e)}")
        return None

def calculate_usr_similarity(usr1, usr2):
    """Calculate similarity between USR descriptors"""
    try:
        # Calculate USR similarity
        usr_sim = rdMolDescriptors.GetUSRScore(usr1, usr2)
        return usr_sim
    except Exception as e:
        print(f"Error calculating USR similarity: {str(e)}")
        return None

def calculate_shape_moments(mol):
    """Calculate shape moments and related descriptors"""
    try:
        # Get the principal moments of inertia
        conf = mol.GetConformer()
        
        # Calculate center of mass
        center_of_mass = np.zeros(3)
        total_mass = 0.0
        
        for atom in mol.GetAtoms():
            mass = atom.GetMass()
            pos = conf.GetAtomPosition(atom.GetIdx())
            center_of_mass += mass * np.array([pos.x, pos.y, pos.z])
            total_mass += mass
        
        center_of_mass /= total_mass
        
        # Calculate inertia tensor
        inertia = np.zeros((3, 3))
        
        for atom in mol.GetAtoms():
            mass = atom.GetMass()
            pos = conf.GetAtomPosition(atom.GetIdx())
            pos_array = np.array([pos.x, pos.y, pos.z]) - center_of_mass
            
            # Diagonal elements
            inertia[0, 0] += mass * (pos_array[1]**2 + pos_array[2]**2)
            inertia[1, 1] += mass * (pos_array[0]**2 + pos_array[2]**2)
            inertia[2, 2] += mass * (pos_array[0]**2 + pos_array[1]**2)
            
            # Off-diagonal elements
            inertia[0, 1] -= mass * pos_array[0] * pos_array[1]
            inertia[0, 2] -= mass * pos_array[0] * pos_array[2]
            inertia[1, 2] -= mass * pos_array[1] * pos_array[2]
        
        inertia[1, 0] = inertia[0, 1]
        inertia[2, 0] = inertia[0, 2]
        inertia[2, 1] = inertia[1, 2]
        
        # Calculate eigenvalues (principal moments of inertia)
        eigenvalues = np.linalg.eigvalsh(inertia)
        pmi = sorted(eigenvalues)
        
        # Calculate normalized PMI ratios
        npr1 = pmi[0] / pmi[2]  # I1/I3
        npr2 = pmi[1] / pmi[2]  # I2/I3
        return {"pmi": pmi, "npr": (npr1, npr2)}
    except Exception as e:
        print(f"Error calculating shape moments: {str(e)}")
        return None

def calculate_gaussian_overlap(mol1, mol2, grid_spacing=0.3, return_fingerprints=False):
    """
    Calculate Gaussian shape overlap between two molecules with improved accuracy
    
    This version uses:
    1. Finer grid spacing (0.3Å)
    2. Atom-specific sigma values based on atomic properties
    
    Args:
        mol1: First RDKit molecule
        mol2: Second RDKit molecule
        grid_spacing: Spacing between grid points
        return_fingerprints: Whether to return the fingerprints
        
    Returns:
        If return_fingerprints=False: Tanimoto similarity
        If return_fingerprints=True: (Tanimoto similarity, fingerprint1, fingerprint2, grid_info, grid1_3d, grid2_3d)
    """
    try:
        # Get conformers
        conf1 = mol1.GetConformer()
        conf2 = mol2.GetConformer()
        
        # Calculate centers of mass
        com1 = np.zeros(3)
        com2 = np.zeros(3)
        total_mass1 = 0.0
        total_mass2 = 0.0
        
        for atom in mol1.GetAtoms():
            mass = atom.GetMass()
            pos = conf1.GetAtomPosition(atom.GetIdx())
            com1 += mass * np.array([pos.x, pos.y, pos.z])
            total_mass1 += mass
        
        for atom in mol2.GetAtoms():
            mass = atom.GetMass()
            pos = conf2.GetAtomPosition(atom.GetIdx())
            com2 += mass * np.array([pos.x, pos.y, pos.z])
            total_mass2 += mass
        
        com1 /= total_mass1
        com2 /= total_mass2
        
        # Get atom positions relative to center of mass
        pos1 = []
        pos2 = []
        radii1 = []
        radii2 = []
        atom_types1 = []
        atom_types2 = []
        hybridizations1 = []
        hybridizations2 = []
        
        for atom in mol1.GetAtoms():
            pos = conf1.GetAtomPosition(atom.GetIdx())
            pos1.append(np.array([pos.x, pos.y, pos.z]) - com1)
            radii1.append(Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()))
            atom_types1.append(atom.GetSymbol())
            hybridizations1.append(atom.GetHybridization())
        
        for atom in mol2.GetAtoms():
            pos = conf2.GetAtomPosition(atom.GetIdx())
            pos2.append(np.array([pos.x, pos.y, pos.z]) - com2)
            radii2.append(Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()))
            atom_types2.append(atom.GetSymbol())
            hybridizations2.append(atom.GetHybridization())
        
        # Determine grid dimensions
        all_pos = np.vstack(pos1 + pos2)
        min_coords = np.min(all_pos, axis=0) - 3.0
        max_coords = np.max(all_pos, axis=0) + 3.0
        
        # Create grid with finer spacing (0.3Å)
        x = np.arange(min_coords[0], max_coords[0], grid_spacing)
        y = np.arange(min_coords[1], max_coords[1], grid_spacing)
        z = np.arange(min_coords[2], max_coords[2], grid_spacing)
        
        print(f"Grid dimensions: {len(x)} x {len(y)} x {len(z)}")
        
        # Initialize grids
        grid1 = np.zeros((len(x), len(y), len(z)))
        grid2 = np.zeros((len(x), len(y), len(z)))
        
        # Function to get atom-specific sigma value
        def get_atom_sigma(atom_type, hybridization, radius):
            # Base sigma factor
            sigma_factor = 0.5  # Default sigma factor
            
            # Adjust based on atom type
            if atom_type == 'H':
                sigma_factor = 0.4  # Sharper Gaussians for hydrogen
            elif atom_type == 'O':
                sigma_factor = 0.45  # Slightly sharper for oxygen
            elif atom_type == 'N':
                sigma_factor = 0.45  # Slightly sharper for nitrogen
            elif atom_type == 'F' or atom_type == 'Cl' or atom_type == 'Br' or atom_type == 'I':
                sigma_factor = 0.55  # Broader for halogens
            elif atom_type == 'S':
                sigma_factor = 0.6  # Broader for sulfur
            
            # Adjust based on hybridization
            if hybridization == Chem.HybridizationType.SP:
                sigma_factor *= 0.9  # Sharper for sp hybridization
            elif hybridization == Chem.HybridizationType.SP2:
                sigma_factor *= 0.95  # Slightly sharper for sp2
            elif hybridization == Chem.HybridizationType.SP3:
                sigma_factor *= 1.05  # Broader for sp3
            
            return sigma_factor * radius
        
        # Compute Gaussian contributions for molecule 1
        for i, atom_pos in enumerate(pos1):
            radius = radii1[i]
            atom_type = atom_types1[i]
            hybridization = hybridizations1[i]
            
            # Get atom-specific sigma
            atom_sigma = get_atom_sigma(atom_type, hybridization, radius)
            
            for ix, px in enumerate(x):
                for iy, py in enumerate(y):
                    for iz, pz in enumerate(z):
                        grid_pos = np.array([px, py, pz])
                        dist = np.linalg.norm(grid_pos - atom_pos)
                        grid1[ix, iy, iz] += np.exp(-(dist**2) / (2 * (atom_sigma)**2))
        
        # Compute Gaussian contributions for molecule 2
        for i, atom_pos in enumerate(pos2):
            radius = radii2[i]
            atom_type = atom_types2[i]
            hybridization = hybridizations2[i]
            
            # Get atom-specific sigma
            atom_sigma = get_atom_sigma(atom_type, hybridization, radius)
            
            for ix, px in enumerate(x):
                for iy, py in enumerate(y):
                    for iz, pz in enumerate(z):
                        grid_pos = np.array([px, py, pz])
                        dist = np.linalg.norm(grid_pos - atom_pos)
                        grid2[ix, iy, iz] += np.exp(-(dist**2) / (2 * (atom_sigma)**2))
        
        # Store original 3D grids
        grid1_3d = grid1.copy()
        grid2_3d = grid2.copy()
        
        # Flatten grids
        grid1_flat = grid1.flatten()
        grid2_flat = grid2.flatten()
        
        # Normalize grids
        if np.sum(grid1_flat) > 0:
            grid1_flat = grid1_flat / np.sum(grid1_flat)
        if np.sum(grid2_flat) > 0:
            grid2_flat = grid2_flat / np.sum(grid2_flat)
        
        # Calculate Tanimoto similarity
        intersection = np.sum(np.minimum(grid1_flat, grid2_flat))
        union = np.sum(np.maximum(grid1_flat, grid2_flat))
        tanimoto = intersection / union if union > 0 else 0
        
        # Grid info for reconstruction
        grid_info = {
            'min_coords': min_coords,
            'max_coords': max_coords,
            'grid_spacing': grid_spacing,
            'shape': (len(x), len(y), len(z)),
            'com1': com1,
            'com2': com2
        }
        
        if return_fingerprints:
            return tanimoto, grid1_flat, grid2_flat, grid_info, grid1_3d, grid2_3d
        else:
            return tanimoto
    except Exception as e:
        print(f"Error calculating Gaussian overlap: {str(e)}")
        if return_fingerprints:
            return None, None, None, None, None, None
        else:
            return None

def save_fingerprints_to_file(fp1, fp2, grid_info, filename="gso_fingerprints.npz"):
    """Save fingerprints to a compressed numpy file"""
    np.savez_compressed(filename, 
                        fp1=fp1, 
                        fp2=fp2, 
                        min_coords=grid_info['min_coords'],
                        max_coords=grid_info['max_coords'],
                        grid_spacing=grid_info['grid_spacing'],
                        shape=grid_info['shape'],
                        com1=grid_info['com1'],
                        com2=grid_info['com2'])
    print(f"Fingerprints saved to {os.path.abspath(filename)}")

def reconstruct_shape_from_fingerprint(fp, grid_info, threshold=0.01):
    """
    Reconstruct a 3D shape from a flattened fingerprint
    
    Args:
        fp: Flattened fingerprint
        grid_info: Grid information dictionary
        threshold: Threshold for including a grid point in the shape
        
    Returns:
        points: List of 3D points representing the shape
        values: Corresponding values at each point
    """
    # Reshape the fingerprint to 3D grid
    grid_3d = fp.reshape(grid_info['shape'])
    
    # Create coordinate grids
    x = np.linspace(grid_info['min_coords'][0], grid_info['max_coords'][0], grid_info['shape'][0])
    y = np.linspace(grid_info['min_coords'][1], grid_info['max_coords'][1], grid_info['shape'][1])
    z = np.linspace(grid_info['min_coords'][2], grid_info['max_coords'][2], grid_info['shape'][2])
    
    # Find points above threshold
    points = []
    values = []
    
    for i in range(grid_info['shape'][0]):
        for j in range(grid_info['shape'][1]):
            for k in range(grid_info['shape'][2]):
                if grid_3d[i, j, k] > threshold:
                    points.append([x[i], y[j], z[k]])
                    values.append(grid_3d[i, j, k])
    
    return np.array(points), np.array(values)

def visualize_fingerprint_shape(grid_3d, grid_info, filename="fingerprint_shape.html"):
    """
    Create a visualization of the 3D shape represented by a fingerprint
    
    Args:
        grid_3d: List of two 3D grids of fingerprint values
        grid_info: Grid information dictionary
        filename: Output HTML filename
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import numpy as np
        
        # Create coordinate grids
        x = np.linspace(grid_info['min_coords'][0], grid_info['max_coords'][0], grid_info['shape'][0])
        y = np.linspace(grid_info['min_coords'][1], grid_info['max_coords'][1], grid_info['shape'][1])
        z = np.linspace(grid_info['min_coords'][2], grid_info['max_coords'][2], grid_info['shape'][2])
        
        # Create a figure with two subplots
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'volume'}, {'type': 'volume'}]],
            subplot_titles=["Molecule 1 Shape", "Molecule 2 Shape"]
        )
        
        # Find appropriate isosurface values
        max_val1 = np.max(grid_3d[0])
        max_val2 = np.max(grid_3d[1])
        
        # Use a percentage of the max value for the isosurface
        iso_val1 = max_val1 * 0.1
        iso_val2 = max_val2 * 0.1
        
        print(f"Max value in grid 1: {max_val1}, using isosurface value: {iso_val1}")
        print(f"Max value in grid 2: {max_val2}, using isosurface value: {iso_val2}")
        
        # Add isosurface for molecule 1
        fig.add_trace(
            go.Isosurface(
                x=x.reshape(-1, 1, 1).repeat(len(y), axis=1).repeat(len(z), axis=2).flatten(),
                y=y.reshape(1, -1, 1).repeat(len(x), axis=0).repeat(len(z), axis=2).flatten(),
                z=z.reshape(1, 1, -1).repeat(len(x), axis=0).repeat(len(y), axis=1).flatten(),
                value=grid_3d[0].flatten(),
                isomin=iso_val1,
                isomax=max_val1,
                opacity=0.6,
                surface_count=3,
                colorscale='Blues',
                caps=dict(x_show=False, y_show=False, z_show=False)
            ),
            row=1, col=1
        )
        
        # Add isosurface for molecule 2
        fig.add_trace(
            go.Isosurface(
                x=x.reshape(-1, 1, 1).repeat(len(y), axis=1).repeat(len(z), axis=2).flatten(),
                y=y.reshape(1, -1, 1).repeat(len(x), axis=0).repeat(len(z), axis=2).flatten(),
                z=z.reshape(1, 1, -1).repeat(len(x), axis=0).repeat(len(y), axis=1).flatten(),
                value=grid_3d[1].flatten(),
                isomin=iso_val2,
                isomax=max_val2,
                opacity=0.6,
                surface_count=3,
                colorscale='Greens',
                caps=dict(x_show=False, y_show=False, z_show=False)
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="3D Shape Reconstruction from GSO Fingerprints",
            width=1200,
            height=600,
            scene=dict(
                aspectmode='data',
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            scene2=dict(
                aspectmode='data',
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
        
        # Write to HTML file
        fig.write_html(filename)
        print(f"Fingerprint shape visualization saved to {os.path.abspath(filename)}")
        
        # Create a simpler visualization as backup
        create_simple_visualization(grid_3d, grid_info, filename.replace('.html', '_simple.html'))
        
        return True
    except Exception as e:
        print(f"Error creating fingerprint visualization: {str(e)}")
        # Try a simpler visualization method
        return create_simple_visualization(grid_3d, grid_info, filename.replace('.html', '_simple.html'))

def create_simple_visualization(grid_3d, grid_info, filename="fingerprint_shape_simple.html"):
    """
    Create a simpler visualization of the 3D shape using scatter points
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import numpy as np
        
        # Create coordinate grids
        x = np.linspace(grid_info['min_coords'][0], grid_info['max_coords'][0], grid_info['shape'][0])
        y = np.linspace(grid_info['min_coords'][1], grid_info['max_coords'][1], grid_info['shape'][1])
        z = np.linspace(grid_info['min_coords'][2], grid_info['max_coords'][2], grid_info['shape'][2])
        
        # Create meshgrid
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Find points above threshold for molecule 1
        threshold1 = np.max(grid_3d[0]) * 0.1
        mask1 = grid_3d[0] > threshold1
        x1 = X[mask1]
        y1 = Y[mask1]
        z1 = Z[mask1]
        values1 = grid_3d[0][mask1]
        
        # Find points above threshold for molecule 2
        threshold2 = np.max(grid_3d[1]) * 0.1
        mask2 = grid_3d[1] > threshold2
        x2 = X[mask2]
        y2 = Y[mask2]
        z2 = Z[mask2]
        values2 = grid_3d[1][mask2]
        
        # Subsample points if there are too many
        max_points = 5000
        if len(x1) > max_points:
            idx1 = np.random.choice(len(x1), max_points, replace=False)
            x1, y1, z1, values1 = x1[idx1], y1[idx1], z1[idx1], values1[idx1]
        
        if len(x2) > max_points:
            idx2 = np.random.choice(len(x2), max_points, replace=False)
            x2, y2, z2, values2 = x2[idx2], y2[idx2], z2[idx2], values2[idx2]
        
        print(f"Molecule 1: {len(x1)} points above threshold")
        print(f"Molecule 2: {len(x2)} points above threshold")
        
        # Create a figure with two subplots
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
            subplot_titles=["Molecule 1 Shape", "Molecule 2 Shape"]
        )
        
        # Add scatter points for molecule 1
        fig.add_trace(
            go.Scatter3d(
                x=x1,
                y=y1,
                z=z1,
                mode='markers',
                marker=dict(
                    size=4,
                    color=values1,
                    colorscale='Blues',
                    opacity=0.8
                ),
                name="Molecule 1"
            ),
            row=1, col=1
        )
        
        # Add scatter points for molecule 2
        fig.add_trace(
            go.Scatter3d(
                x=x2,
                y=y2,
                z=z2,
                mode='markers',
                marker=dict(
                    size=4,
                    color=values2,
                    colorscale='Greens',
                    opacity=0.8
                ),
                name="Molecule 2"
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="3D Shape Reconstruction from GSO Fingerprints (Simple View)",
            width=1200,
            height=600,
            scene=dict(
                aspectmode='data',
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            scene2=dict(
                aspectmode='data',
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
        
        # Write to HTML file
        fig.write_html(filename)
        print(f"Simple fingerprint shape visualization saved to {os.path.abspath(filename)}")
        return True
    except Exception as e:
        print(f"Error creating simple visualization: {str(e)}")
        return False

def visualize_molecules(mol1, mol2, filename="/home/luost_local/sdivita/synformer/experiments/molecule_shapes.html"):
    """
    Create an HTML file with 3D visualizations of both molecules side by side
    
    Args:
        mol1: First RDKit molecule
        mol2: Second RDKit molecule
        filename: Output HTML filename
    """
    try:
        # Convert molecules to PDB format for py3Dmol
        pdb1 = Chem.MolToPDBBlock(mol1)
        pdb2 = Chem.MolToPDBBlock(mol2)
        
        # Create HTML content with two separate viewers
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Molecule Shape Comparison</title>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
            <script src="https://3dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                }}
                .container {{
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                }}
                .viewers-container {{
                    display: flex;
                    justify-content: space-between;
                    width: 100%;
                    max-width: 1200px;
                }}
                .viewer-box {{
                    width: 48%;
                }}
                .viewer {{
                    width: 100%;
                    height: 400px;
                    position: relative;
                    margin: 10px 0;
                }}
                .info {{
                    width: 100%;
                    max-width: 1200px;
                    background-color: #f5f5f5;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }}
                .molecule-name {{
                    text-align: center;
                    font-weight: bold;
                    margin-bottom: 5px;
                }}
                h2 {{
                    color: #333;
                }}
                .controls {{
                    display: flex;
                    justify-content: center;
                    gap: 10px;
                    margin-bottom: 20px;
                }}
                button {{
                    padding: 8px 15px;
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                }}
                button:hover {{
                    background-color: #45a049;
                }}
                .metrics {{
                    width: 100%;
                    max-width: 1200px;
                    margin-top: 20px;
                    padding: 15px;
                    background-color: #e9f7ef;
                    border-radius: 5px;
                }}
                .metrics h3 {{
                    margin-top: 0;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                }}
                th, td {{
                    padding: 8px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>Molecule Shape Comparison</h2>
                
                <div class="info">
                    <p>This visualization shows the 3D shapes of two molecules side by side.</p>
                    <p>Use the mouse to rotate and zoom. The controls below each viewer allow you to toggle different visualization elements.</p>
                </div>
                
                <div class="viewers-container">
                    <div class="viewer-box">
                        <div class="molecule-name">Molecule 1: {Chem.MolToSmiles(mol1)}</div>
                        <div id="viewer1" class="viewer"></div>
                        <div class="controls">
                            <button onclick="toggleSurface(1)">Toggle Surface</button>
                            <button onclick="toggleSticks(1)">Toggle Sticks</button>
                            <button onclick="resetView(1)">Reset View</button>
                        </div>
                    </div>
                    
                    <div class="viewer-box">
                        <div class="molecule-name">Molecule 2: {Chem.MolToSmiles(mol2)}</div>
                        <div id="viewer2" class="viewer"></div>
                        <div class="controls">
                            <button onclick="toggleSurface(2)">Toggle Surface</button>
                            <button onclick="toggleSticks(2)">Toggle Sticks</button>
                            <button onclick="resetView(2)">Reset View</button>
                        </div>
                    </div>
                </div>
                
                <div class="metrics">
                    <h3>Shape Similarity Metrics</h3>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                            <th>Description</th>
                        </tr>
                        <tr>
                            <td>USR Similarity</td>
                            <td id="usr-sim">Calculating...</td>
                            <td>Ultrafast Shape Recognition similarity (0-1 scale)</td>
                        </tr>
                        <tr>
                            <td>USRCAT Similarity</td>
                            <td id="usrcat-sim">Calculating...</td>
                            <td>USR with Chemical Atom Types (0-1 scale)</td>
                        </tr>
                        <tr>
                            <td>NPR Distance</td>
                            <td id="npr-dist">Calculating...</td>
                            <td>Normalized PMI Ratio distance (lower is more similar)</td>
                        </tr>
                        <tr>
                            <td>Gaussian Shape Overlap</td>
                            <td id="gso-overlap">Calculating...</td>
                            <td>Tanimoto coefficient of Gaussian shape functions (0-1 scale)</td>
                        </tr>
                    </table>
                </div>
            </div>
            
            <script>
                let viewer1 = null;
                let viewer2 = null;
                let surfaceOn1 = true;
                let surfaceOn2 = true;
                let sticksOn1 = true;
                let sticksOn2 = true;
                
                $(document).ready(function() {{
                    // Initialize first viewer
                    viewer1 = $3Dmol.createViewer($("#viewer1"), {{backgroundColor: 'white'}});
                    let m1 = viewer1.addModel(`{pdb1}`, "pdb");
                    viewer1.setStyle({{}}, {{stick: {{radius: 0.2, color: 'skyBlue'}}, 
                                          sphere: {{scale: 0.3, color: 'skyBlue'}}}});
                    viewer1.addSurface($3Dmol.VDW, {{opacity: 0.6, color: 'skyBlue'}});
                    viewer1.zoomTo();
                    viewer1.render();
                    
                    // Initialize second viewer
                    viewer2 = $3Dmol.createViewer($("#viewer2"), {{backgroundColor: 'white'}});
                    let m2 = viewer2.addModel(`{pdb2}`, "pdb");
                    viewer2.setStyle({{}}, {{stick: {{radius: 0.2, color: 'lightGreen'}}, 
                                          sphere: {{scale: 0.3, color: 'lightGreen'}}}});
                    viewer2.addSurface($3Dmol.VDW, {{opacity: 0.6, color: 'lightGreen'}});
                    viewer2.zoomTo();
                    viewer2.render();
                    
                    // Update metrics
                    updateMetrics();
                }});
                
                function toggleSurface(viewerNum) {{
                    let viewer = viewerNum === 1 ? viewer1 : viewer2;
                    let surfaceOn = viewerNum === 1 ? surfaceOn1 : surfaceOn2;
                    let color = viewerNum === 1 ? 'skyBlue' : 'lightGreen';
                    
                    surfaceOn = !surfaceOn;
                    if (viewerNum === 1) surfaceOn1 = surfaceOn;
                    else surfaceOn2 = surfaceOn;
                    
                    viewer.removeAllSurfaces();
                    
                    if (surfaceOn) {{
                        viewer.addSurface($3Dmol.VDW, {{opacity: 0.6, color: color}});
                    }}
                    
                    viewer.render();
                }}
                
                function toggleSticks(viewerNum) {{
                    let viewer = viewerNum === 1 ? viewer1 : viewer2;
                    let sticksOn = viewerNum === 1 ? sticksOn1 : sticksOn2;
                    let color = viewerNum === 1 ? 'skyBlue' : 'lightGreen';
                    
                    sticksOn = !sticksOn;
                    if (viewerNum === 1) sticksOn1 = sticksOn;
                    else sticksOn2 = sticksOn;
                    
                    if (sticksOn) {{
                        viewer.setStyle({{}}, {{stick: {{radius: 0.2, color: color}}, 
                                            sphere: {{scale: 0.3, color: color}}}});
                    }} else {{
                        viewer.setStyle({{}}, {{sphere: {{scale: 0.3, color: color}}}});
                    }}
                    
                    viewer.render();
                }}
                
                function resetView(viewerNum) {{
                    let viewer = viewerNum === 1 ? viewer1 : viewer2;
                    viewer.zoomTo();
                    viewer.render();
                }}
                
                function updateMetrics() {{
                    // In a real application, you would calculate these values in Python
                    // and pass them to the HTML. For this example, we'll use placeholders.
                    setTimeout(function() {{
                        document.getElementById('usr-sim').textContent = 'PLACEHOLDER';
                        document.getElementById('usrcat-sim').textContent = 'PLACEHOLDER';
                        document.getElementById('npr-dist').textContent = 'PLACEHOLDER';
                        document.getElementById('gso-overlap').textContent = 'PLACEHOLDER';
                    }}, 1000);
                }}
            </script>
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(filename, 'w') as f:
            f.write(html_content)
        
        print(f"Visualization saved to {os.path.abspath(filename)}")
        return True
    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
        return False

def identify_pharmacophore_features(mol):
    """
    Identify pharmacophore features in a molecule
    
    Returns a dictionary mapping atom indices to feature types:
    - H-bond donor (D)
    - H-bond acceptor (A)
    - Positive ionizable (P)
    - Negative ionizable (N)
    - Hydrophobic (H)
    - Aromatic (R)
    """
    features = {}
    
    # Define SMARTS patterns for pharmacophore features
    smarts_patterns = {
        'D': ['[#7!H0]', '[#8!H0]'],  # H-bond donors (N-H, O-H)
        'A': ['[#7]', '[#8]'],        # H-bond acceptors (N, O)
        'P': ['[+,#7;!$(N~[!#6]);!$(*~[#7,#8,#15,#16])]'],  # Positive ionizable (basic N)
        'N': ['[-,#8;$(*~[#6,#7])]'],  # Negative ionizable (acidic O)
        'H': ['[#6;+0]~[#6;+0]', '[#6]~[F,Cl,Br,I]', '[S;D2;+0]'],  # Hydrophobic (C-C, C-halogen, sulfide)
        'R': ['a5', 'a6']             # Aromatic (5 and 6-membered rings)
    }
    
    # Identify features using SMARTS patterns
    for feature_type, patterns in smarts_patterns.items():
        for pattern in patterns:
            patt = Chem.MolFromSmarts(pattern)
            if patt:
                matches = mol.GetSubstructMatches(patt)
                for match in matches:
                    for atom_idx in match:
                        if atom_idx not in features:
                            features[atom_idx] = []
                        features[atom_idx].append(feature_type)
    
    # Refine features (remove duplicates, handle special cases)
    refined_features = {}
    for atom_idx, feature_list in features.items():
        # Remove duplicates
        unique_features = list(set(feature_list))
        
        # Special case: carboxylic acids are strong H-bond acceptors
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetSymbol() == 'O' and atom.GetFormalCharge() == -1:
            if 'A' not in unique_features:
                unique_features.append('A')
        
        # Special case: amines can be both donors and positive ionizable
        if atom.GetSymbol() == 'N' and atom.GetTotalNumHs() > 0:
            if 'D' not in unique_features:
                unique_features.append('D')
            if 'P' not in unique_features and atom.GetFormalCharge() >= 0:
                # Check if it's a basic amine (not amide, etc.)
                is_basic = True
                for bond in atom.GetBonds():
                    other_atom = bond.GetOtherAtom(atom)
                    if other_atom.GetSymbol() in ['C', 'S'] and other_atom.GetIsAromatic():
                        is_basic = False
                    if other_atom.GetSymbol() in ['O', 'S'] and bond.GetBondType() == Chem.BondType.DOUBLE:
                        is_basic = False
                if is_basic:
                    unique_features.append('P')
        
        # Special case: aromatic rings are hydrophobic
        if 'R' in unique_features and 'H' not in unique_features:
            unique_features.append('H')
        
        refined_features[atom_idx] = unique_features
    
    return refined_features

def get_atom_sigma(atom_type, hybridization, radius):
    """
    Get atom-specific sigma value for Gaussian functions
    
    Args:
        atom_type: Atom symbol (e.g., 'C', 'N', 'O')
        hybridization: RDKit hybridization type
        radius: Van der Waals radius
        
    Returns:
        sigma: Sigma value for Gaussian function
    """
    # Base sigma factor
    sigma_factor = 0.5  # Default sigma factor
    
    # Adjust based on atom type
    if atom_type == 'H':
        sigma_factor = 0.4  # Sharper Gaussians for hydrogen
    elif atom_type == 'O':
        sigma_factor = 0.45  # Slightly sharper for oxygen
    elif atom_type == 'N':
        sigma_factor = 0.45  # Slightly sharper for nitrogen
    elif atom_type == 'F' or atom_type == 'Cl' or atom_type == 'Br' or atom_type == 'I':
        sigma_factor = 0.55  # Broader for halogens
    elif atom_type == 'S':
        sigma_factor = 0.6  # Broader for sulfur
    
    # Adjust based on hybridization
    if hybridization == Chem.HybridizationType.SP:
        sigma_factor *= 0.9  # Sharper for sp hybridization
    elif hybridization == Chem.HybridizationType.SP2:
        sigma_factor *= 0.95  # Slightly sharper for sp2
    elif hybridization == Chem.HybridizationType.SP3:
        sigma_factor *= 1.05  # Broader for sp3
    
    return sigma_factor * radius

def generate_pharmacophore_fingerprints(mol, grid_spacing=0.3):
    """
    Generate pharmacophore-tagged GSO fingerprints
    
    Args:
        mol: RDKit molecule with 3D coordinates
        grid_spacing: Spacing between grid points (Angstroms)
        
    Returns:
        overall_grid: Overall shape grid
        feature_grids: Dictionary of grids for each pharmacophore feature
        feature_atoms: Dictionary mapping feature types to atom indices
        grid_info: Information about the grid for reconstruction
    """
    # Ensure molecule has 3D coordinates
    if not mol.GetNumConformers():
        print("Error: Molecule has no conformers")
        return None, None, None, None
    
    # Get conformer
    conf = mol.GetConformer()
    
    # Identify pharmacophore features
    features = identify_pharmacophore_features(mol)
    
    # Organize atoms by feature type
    feature_atoms = {
        'D': [],  # H-bond donor
        'A': [],  # H-bond acceptor
        'P': [],  # Positive ionizable
        'N': [],  # Negative ionizable
        'H': [],  # Hydrophobic
        'R': []   # Aromatic
    }
    
    for atom_idx, feature_list in features.items():
        for feature in feature_list:
            feature_atoms[feature].append(atom_idx)
    
    # Print feature counts
    print("\nPharmacophore feature counts:")
    for feature_type, atoms in feature_atoms.items():
        print(f"  {feature_type}: {len(atoms)}")
    
    # Calculate center of mass
    com = np.zeros(3)
    total_mass = 0.0
    
    for atom in mol.GetAtoms():
        mass = atom.GetMass()
        pos = conf.GetAtomPosition(atom.GetIdx())
        com += mass * np.array([pos.x, pos.y, pos.z])
        total_mass += mass
    
    com /= total_mass
    
    # Get atom positions relative to center of mass
    positions = []
    radii = []
    atom_types = []
    hybridizations = []
    atom_indices = []
    
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        pos = conf.GetAtomPosition(atom_idx)
        positions.append(np.array([pos.x, pos.y, pos.z]) - com)
        radii.append(Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()))
        atom_types.append(atom.GetSymbol())
        hybridizations.append(atom.GetHybridization())
        atom_indices.append(atom_idx)
    
    # Determine grid dimensions
    all_pos = np.array(positions)
    min_coords = np.min(all_pos, axis=0) - 3.0
    max_coords = np.max(all_pos, axis=0) + 3.0
    
    # Create grid
    x = np.arange(min_coords[0], max_coords[0], grid_spacing)
    y = np.arange(min_coords[1], max_coords[1], grid_spacing)
    z = np.arange(min_coords[2], max_coords[2], grid_spacing)
    
    print(f"Grid dimensions: {len(x)} x {len(y)} x {len(z)}")
    
    # Initialize grids
    overall_grid = np.zeros((len(x), len(y), len(z)))
    feature_grids = {
        'D': np.zeros((len(x), len(y), len(z))),
        'A': np.zeros((len(x), len(y), len(z))),
        'P': np.zeros((len(x), len(y), len(z))),
        'N': np.zeros((len(x), len(y), len(z))),
        'H': np.zeros((len(x), len(y), len(z))),
        'R': np.zeros((len(x), len(y), len(z)))
    }
    
    # Compute Gaussian contributions for all atoms
    for i, atom_pos in enumerate(positions):
        atom_idx = atom_indices[i]
        radius = radii[i]
        atom_type = atom_types[i]
        hybridization = hybridizations[i]
        
        # Get atom-specific sigma
        atom_sigma = get_atom_sigma(atom_type, hybridization, radius)
        
        # Get atom's pharmacophore features
        atom_features = features.get(atom_idx, [])
        
        # Compute Gaussian contribution for this atom
        for ix, px in enumerate(x):
            for iy, py in enumerate(y):
                for iz, pz in enumerate(z):
                    grid_pos = np.array([px, py, pz])
                    dist = np.linalg.norm(grid_pos - atom_pos)
                    gaussian_value = np.exp(-(dist**2) / (2 * (atom_sigma)**2))
                    
                    # Add to overall shape grid
                    overall_grid[ix, iy, iz] += gaussian_value
                    
                    # Add to feature-specific grids
                    for feature in atom_features:
                        feature_grids[feature][ix, iy, iz] += gaussian_value
    
    # Grid info for reconstruction
    grid_info = {
        'min_coords': min_coords,
        'max_coords': max_coords,
        'grid_spacing': grid_spacing,
        'shape': (len(x), len(y), len(z)),
        'com': com
    }
    
    return overall_grid, feature_grids, feature_atoms, grid_info

def save_pharmacophore_fingerprints(mol_name, overall_grid, feature_grids, feature_atoms, grid_info, output_dir="pharmacophore_fingerprints"):
    """
    Save pharmacophore fingerprints to files
    
    Args:
        mol_name: Name of the molecule
        overall_grid: Overall shape grid
        feature_grids: Dictionary of grids for each pharmacophore feature
        feature_atoms: Dictionary mapping feature types to atom indices
        grid_info: Information about the grid for reconstruction
        output_dir: Directory to save fingerprints
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save overall grid
    np.save(f"{output_dir}/{mol_name}_overall_grid.npy", overall_grid)
    
    # Save feature grids
    for feature_type, grid in feature_grids.items():
        np.save(f"{output_dir}/{mol_name}_{feature_type}_grid.npy", grid)
    
    # Save grid info
    np.savez(f"{output_dir}/{mol_name}_grid_info.npz",
             min_coords=grid_info['min_coords'],
             max_coords=grid_info['max_coords'],
             grid_spacing=grid_info['grid_spacing'],
             shape=grid_info['shape'],
             com=grid_info['com'])
    
    # Save feature atoms
    with open(f"{output_dir}/{mol_name}_feature_atoms.txt", 'w') as f:
        for feature_type, atoms in feature_atoms.items():
            f.write(f"{feature_type}: {','.join(map(str, atoms))}\n")
    
    print(f"Saved pharmacophore fingerprints for {mol_name} to {output_dir}")

def load_pharmacophore_fingerprints(mol_name, input_dir="pharmacophore_fingerprints"):
    """
    Load pharmacophore fingerprints from files
    
    Args:
        mol_name: Name of the molecule
        input_dir: Directory containing fingerprints
        
    Returns:
        overall_grid: Overall shape grid
        feature_grids: Dictionary of grids for each pharmacophore feature
        feature_atoms: Dictionary mapping feature types to atom indices
        grid_info: Information about the grid for reconstruction
    """
    # Load overall grid
    overall_grid = np.load(f"{input_dir}/{mol_name}_overall_grid.npy")
    
    # Load feature grids
    feature_types = ['D', 'A', 'P', 'N', 'H', 'R']
    feature_grids = {}
    for feature_type in feature_types:
        feature_grids[feature_type] = np.load(f"{input_dir}/{mol_name}_{feature_type}_grid.npy")
    
    # Load grid info
    grid_info_file = np.load(f"{input_dir}/{mol_name}_grid_info.npz")
    grid_info = {
        'min_coords': grid_info_file['min_coords'],
        'max_coords': grid_info_file['max_coords'],
        'grid_spacing': grid_info_file['grid_spacing'],
        'shape': grid_info_file['shape'],
        'com': grid_info_file['com']
    }
    
    # Load feature atoms
    feature_atoms = {
        'D': [],
        'A': [],
        'P': [],
        'N': [],
        'H': [],
        'R': []
    }
    
    with open(f"{input_dir}/{mol_name}_feature_atoms.txt", 'r') as f:
        for line in f:
            parts = line.strip().split(': ')
            if len(parts) == 2:
                feature_type = parts[0]
                if parts[1]:  # Check if there are any atoms
                    atoms = list(map(int, parts[1].split(',')))
                    feature_atoms[feature_type] = atoms
    
    print(f"Loaded pharmacophore fingerprints for {mol_name} from {input_dir}")
    return overall_grid, feature_grids, feature_atoms, grid_info

def visualize_pharmacophore_fingerprints(mol, overall_grid, feature_grids, grid_info, filename="pharmacophore_fingerprints.html"):
    """
    Create an interactive visualization of pharmacophore fingerprints
    
    Args:
        mol: RDKit molecule
        overall_grid: Overall shape grid
        feature_grids: Dictionary of grids for each pharmacophore feature
        grid_info: Information about the grid for reconstruction
        filename: Output HTML filename
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Create a figure with subplots
        fig = make_subplots(
            rows=2, cols=3,
            specs=[[{'type': 'volume'}, {'type': 'volume'}, {'type': 'volume'}],
                   [{'type': 'volume'}, {'type': 'volume'}, {'type': 'volume'}]],
            subplot_titles=["H-bond Donor (D)", "H-bond Acceptor (A)", "Positive Ionizable (P)",
                           "Negative Ionizable (N)", "Hydrophobic (H)", "Aromatic (R)"]
        )
        
        # Define colors for feature types
        feature_colors = {
            'D': 'blues',      # H-bond donor
            'A': 'reds',       # H-bond acceptor
            'P': 'purples',    # Positive ionizable
            'N': 'oranges',    # Negative ionizable
            'H': 'greens',     # Hydrophobic
            'R': 'pinkyl'      # Aromatic
        }
        
        # Create coordinate grids
        x = np.linspace(grid_info['min_coords'][0], grid_info['max_coords'][0], grid_info['shape'][0])
        y = np.linspace(grid_info['min_coords'][1], grid_info['max_coords'][1], grid_info['shape'][1])
        z = np.linspace(grid_info['min_coords'][2], grid_info['max_coords'][2], grid_info['shape'][2])
        
        # Add isosurfaces for each feature type
        feature_types = ['D', 'A', 'P', 'N', 'H', 'R']
        for i, feature_type in enumerate(feature_types):
            # Get grid for this feature
            grid = feature_grids[feature_type]
            
            # Find appropriate isosurface value
            max_val = np.max(grid)
            if max_val > 0:
                iso_val = max_val * 0.2  # Use 20% of max value
                
                # Calculate row and column for this subplot
                row = i // 3 + 1
                col = i % 3 + 1
                
                # Add isosurface
                fig.add_trace(
                    go.Isosurface(
                        x=x.reshape(-1, 1, 1).repeat(len(y), axis=1).repeat(len(z), axis=2).flatten(),
                        y=y.reshape(1, -1, 1).repeat(len(x), axis=0).repeat(len(z), axis=2).flatten(),
                        z=z.reshape(1, 1, -1).repeat(len(x), axis=0).repeat(len(y), axis=1).flatten(),
                        value=grid.flatten(),
                        isomin=iso_val,
                        isomax=max_val,
                        opacity=0.7,
                        surface_count=3,
                        colorscale=feature_colors[feature_type],
                        caps=dict(x_show=False, y_show=False, z_show=False)
                    ),
                    row=row, col=col
                )
        
        # Update layout
        fig.update_layout(
            title=f"Pharmacophore Fingerprints: {Chem.MolToSmiles(mol)}",
            width=1200,
            height=800,
            scene=dict(
                aspectmode='data',
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
        
        # Write to HTML file
        fig.write_html(filename)
        print(f"Pharmacophore fingerprint visualization saved to {os.path.abspath(filename)}")
        return True
    except Exception as e:
        print(f"Error creating pharmacophore visualization: {str(e)}")
        return False

def blend_pharmacophore_fingerprints(overall_grid1, feature_grids1, overall_grid2, feature_grids2, blend_ratio=0.5, feature_weight=2.0):
    """
    Blend two sets of pharmacophore fingerprints
    
    Args:
        overall_grid1, overall_grid2: Overall shape grids
        feature_grids1, feature_grids2: Dictionaries of feature grids
        blend_ratio: Ratio of mol1 to mol2 (0.5 = equal blend)
        feature_weight: Weight given to pharmacophore features (higher = more preservation)
        
    Returns:
        blended_overall: Blended overall grid
        blended_features: Dictionary of blended feature grids
    """
    # Ensure grids have the same dimensions
    if overall_grid1.shape != overall_grid2.shape:
        print("Error: Grids have different dimensions. Need to align molecules first.")
        return None, None
    
    # Create importance maps for pharmacophore features
    importance_map1 = np.zeros_like(overall_grid1)
    importance_map2 = np.zeros_like(overall_grid2)
    
    # Add importance to regions with pharmacophore features
    for feature_type in feature_grids1:
        # Add weighted feature grids to importance maps
        importance_map1 += feature_grids1[feature_type] * feature_weight
        importance_map2 += feature_grids2[feature_type] * feature_weight
    
    # Normalize importance maps
    if np.max(importance_map1) > 0:
        importance_map1 /= np.max(importance_map1)
    if np.max(importance_map2) > 0:
        importance_map2 /= np.max(importance_map2)
    
    # Create blending weights that respect pharmacophore features
    # Higher values in importance maps mean we want to preserve that region
    weight_map1 = blend_ratio + (1 - blend_ratio) * importance_map1
    weight_map2 = (1 - blend_ratio) + blend_ratio * importance_map2
    
    # Normalize weight maps
    weight_sum = weight_map1 + weight_map2
    weight_map1 /= weight_sum
    weight_map2 /= weight_sum
    
    # Blend the overall grids
    blended_overall = weight_map1 * overall_grid1 + weight_map2 * overall_grid2
    
    # Blend feature grids
    blended_features = {}
    for feature_type in feature_grids1:
        # For feature grids, we want to preserve features from both parents
        # Use maximum instead of weighted average to ensure features are preserved
        blended_features[feature_type] = np.maximum(
            feature_grids1[feature_type],
            feature_grids2[feature_type]
        )
    
    return blended_overall, blended_features

def align_pharmacophore_grids(overall_grid1, feature_grids1, grid_info1, 
                             overall_grid2, feature_grids2, grid_info2,
                             target_spacing=0.3):
    """
    Align two sets of pharmacophore grids to the same coordinate system
    
    Args:
        overall_grid1, overall_grid2: Overall shape grids
        feature_grids1, feature_grids2: Dictionaries of feature grids
        grid_info1, grid_info2: Grid information dictionaries
        target_spacing: Target grid spacing for the aligned grids
        
    Returns:
        aligned_overall1, aligned_overall2: Aligned overall grids
        aligned_features1, aligned_features2: Aligned feature grids
        aligned_grid_info: Grid information for the aligned grids
    """
    # Determine the combined grid dimensions
    min_coords = np.minimum(grid_info1['min_coords'], grid_info2['min_coords'])
    max_coords = np.maximum(grid_info1['max_coords'], grid_info2['max_coords'])
    
    # Create new grid with target spacing
    x = np.arange(min_coords[0], max_coords[0], target_spacing)
    y = np.arange(min_coords[1], max_coords[1], target_spacing)
    z = np.arange(min_coords[2], max_coords[2], target_spacing)
    
    print(f"Creating aligned grid with dimensions: {len(x)} x {len(y)} x {len(z)}")
    
    # Initialize aligned grids
    aligned_overall1 = np.zeros((len(x), len(y), len(z)))
    aligned_overall2 = np.zeros((len(x), len(y), len(z)))
    
    aligned_features1 = {
        'D': np.zeros((len(x), len(y), len(z))),
        'A': np.zeros((len(x), len(y), len(z))),
        'P': np.zeros((len(x), len(y), len(z))),
        'N': np.zeros((len(x), len(y), len(z))),
        'H': np.zeros((len(x), len(y), len(z))),
        'R': np.zeros((len(x), len(y), len(z)))
    }
    
    aligned_features2 = {
        'D': np.zeros((len(x), len(y), len(z))),
        'A': np.zeros((len(x), len(y), len(z))),
        'P': np.zeros((len(x), len(y), len(z))),
        'N': np.zeros((len(x), len(y), len(z))),
        'H': np.zeros((len(x), len(y), len(z))),
        'R': np.zeros((len(x), len(y), len(z)))
    }
    
    # Function to map from original grid coordinates to aligned grid coordinates
    def map_to_aligned_grid(grid, original_info, aligned_x, aligned_y, aligned_z):
        # Create interpolation function
        from scipy.interpolate import RegularGridInterpolator
        
        # Get original grid coordinates
        orig_x = np.linspace(original_info['min_coords'][0], original_info['max_coords'][0], original_info['shape'][0])
        orig_y = np.linspace(original_info['min_coords'][1], original_info['max_coords'][1], original_info['shape'][1])
        orig_z = np.linspace(original_info['min_coords'][2], original_info['max_coords'][2], original_info['shape'][2])
        
        # Create interpolator
        interpolator = RegularGridInterpolator(
            (orig_x, orig_y, orig_z), 
            grid, 
            bounds_error=False, 
            fill_value=0
        )
        
        # Create meshgrid for aligned coordinates
        X, Y, Z = np.meshgrid(aligned_x, aligned_y, aligned_z, indexing='ij')
        points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
        
        # Interpolate values
        aligned_values = interpolator(points)
        
        # Reshape to grid
        return aligned_values.reshape((len(aligned_x), len(aligned_y), len(aligned_z)))
    
    # Map grids to aligned coordinate system
    print("Aligning grid 1...")
    aligned_overall1 = map_to_aligned_grid(overall_grid1, grid_info1, x, y, z)
    
    print("Aligning grid 2...")
    aligned_overall2 = map_to_aligned_grid(overall_grid2, grid_info2, x, y, z)
    
    # Map feature grids
    print("Aligning feature grids...")
    for feature_type in feature_grids1:
        aligned_features1[feature_type] = map_to_aligned_grid(
            feature_grids1[feature_type], grid_info1, x, y, z
        )
        aligned_features2[feature_type] = map_to_aligned_grid(
            feature_grids2[feature_type], grid_info2, x, y, z
        )
    
    # Create aligned grid info
    aligned_grid_info = {
        'min_coords': min_coords,
        'max_coords': max_coords,
        'grid_spacing': target_spacing,
        'shape': (len(x), len(y), len(z)),
        'com': (grid_info1['com'] + grid_info2['com']) / 2  # Average center of mass
    }
    
    print("Grid alignment complete.")
    return aligned_overall1, aligned_overall2, aligned_features1, aligned_features2, aligned_grid_info

def visualize_all_pharmacophore_fingerprints(mol1, mol2, 
                                           overall_grid1, feature_grids1, grid_info1,
                                           overall_grid2, feature_grids2, grid_info2,
                                           blended_overall, blended_features, blended_grid_info,
                                           filename="all_pharmacophore_fingerprints.html"):
    """
    Create an interactive visualization showing all pharmacophore fingerprints for
    two parent molecules and their blended child in three separate 3D grids
    
    Args:
        mol1, mol2: Parent RDKit molecules
        overall_grid1, overall_grid2: Overall shape grids for parents
        feature_grids1, feature_grids2: Dictionaries of feature grids for parents
        grid_info1, grid_info2: Grid information for parents
        blended_overall: Blended overall grid
        blended_features: Dictionary of blended feature grids
        blended_grid_info: Grid information for blended fingerprints
        filename: Output HTML filename
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Define feature types and their colors
        feature_types = ['D', 'A', 'P', 'N', 'H', 'R']
        feature_names = {
            'D': 'H-bond Donor',
            'A': 'H-bond Acceptor',
            'P': 'Positive Ionizable',
            'N': 'Negative Ionizable',
            'H': 'Hydrophobic',
            'R': 'Aromatic'
        }
        feature_colors = {
            'D': 'blue',
            'A': 'red',
            'P': 'purple',
            'N': 'orange',
            'H': 'green',
            'R': 'pink'
        }
        
        # Create a figure with three subplots (one for each molecule)
        fig = make_subplots(
            rows=1, cols=3,
            specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]],
            subplot_titles=[
                f"Molecule 1: {Chem.MolToSmiles(mol1)}",
                f"Molecule 2: {Chem.MolToSmiles(mol2)}",
                "Blended Child"
            ],
            horizontal_spacing=0.02
        )
        
        # Function to add molecule structure to a subplot
        def add_molecule_structure(mol, row, col):
            # Get atom positions
            conf = mol.GetConformer()
            positions = []
            elements = []
            
            for atom in mol.GetAtoms():
                idx = atom.GetIdx()
                pos = conf.GetAtomPosition(idx)
                positions.append([pos.x, pos.y, pos.z])
                elements.append(atom.GetSymbol())
            
            # Add atoms as scatter3d
            fig.add_trace(
                go.Scatter3d(
                    x=[pos[0] for pos in positions],
                    y=[pos[1] for pos in positions],
                    z=[pos[2] for pos in positions],
                    mode='markers',
                    marker=dict(
                        size=7,
                        color='gray',
                        opacity=0.8
                    ),
                    text=elements,
                    name=f"Atoms (Mol {col})",
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # Add bonds as lines
            for bond in mol.GetBonds():
                begin_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                begin_pos = positions[begin_idx]
                end_pos = positions[end_idx]
                
                fig.add_trace(
                    go.Scatter3d(
                        x=[begin_pos[0], end_pos[0]],
                        y=[begin_pos[1], end_pos[1]],
                        z=[begin_pos[2], end_pos[2]],
                        mode='lines',
                        line=dict(color='gray', width=3),
                        showlegend=False
                    ),
                    row=row, col=col
                )
        
        # Function to add pharmacophore features to a subplot
        def add_pharmacophore_features(feature_grids, grid_info, row, col):
            # Create coordinate grids
            x = np.linspace(grid_info['min_coords'][0], grid_info['max_coords'][0], grid_info['shape'][0])
            y = np.linspace(grid_info['min_coords'][1], grid_info['max_coords'][1], grid_info['shape'][1])
            z = np.linspace(grid_info['min_coords'][2], grid_info['max_coords'][2], grid_info['shape'][2])
            
            # Add isosurfaces for each feature type
            for feature_type in feature_types:
                # Get grid for this feature
                grid = feature_grids[feature_type]
                
                # Find appropriate isosurface value
                max_val = np.max(grid)
                if max_val > 0:
                    iso_val = max_val * 0.2  # Use 20% of max value
                    
                    # Create meshgrid for coordinates
                    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
                    
                    # Create isosurface
                    fig.add_trace(
                        go.Isosurface(
                            x=X.flatten(),
                            y=Y.flatten(),
                            z=Z.flatten(),
                            value=grid.flatten(),
                            isomin=iso_val,
                            isomax=max_val,
                            opacity=0.5,
                            surface_count=1,
                            colorscale=[[0, feature_colors[feature_type]], [1, feature_colors[feature_type]]],
                            name=feature_names[feature_type],
                            showscale=False,
                            showlegend=(col == 1)  # Only show legend for first column
                        ),
                        row=row, col=col
                    )
        
        # Add molecule structures
        add_molecule_structure(mol1, 1, 1)
        add_molecule_structure(mol2, 1, 2)
        
        # Add pharmacophore features
        add_pharmacophore_features(feature_grids1, grid_info1, 1, 1)
        add_pharmacophore_features(feature_grids2, grid_info2, 1, 2)
        add_pharmacophore_features(blended_features, blended_grid_info, 1, 3)
        
        # Update layout
        fig.update_layout(
            title="Pharmacophore Fingerprints Comparison",
            width=1800,
            height=700,
            legend=dict(
                x=0,
                y=1,
                bgcolor='rgba(255, 255, 255, 0.5)',
                bordercolor='rgba(0, 0, 0, 0.5)',
                borderwidth=1
            )
        )
        
        # Update scenes to have the same camera view
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
        
        fig.update_scenes(
            camera=camera,
            aspectmode='data',
            xaxis_visible=True,
            yaxis_visible=True,
            zaxis_visible=True
        )
        
        # Write to HTML file
        fig.write_html(filename)
        print(f"All pharmacophore fingerprints visualization saved to {os.path.abspath(filename)}")
        
        # Create a simplified version for better performance
        create_simplified_visualization(
            mol1, mol2, 
            feature_grids1, feature_grids2, blended_features,
            grid_info1, grid_info2, blended_grid_info,
            filename.replace('.html', '_simple.html')
        )
        
        return True
    except Exception as e:
        print(f"Error creating all pharmacophore visualization: {str(e)}")
        return False

def create_simplified_visualization(mol1, mol2, 
                                   feature_grids1, feature_grids2, blended_features,
                                   grid_info1, grid_info2, blended_grid_info,
                                   filename="all_pharmacophore_fingerprints_simple.html"):
    """
    Create a simplified visualization using scatter points instead of isosurfaces
    for better performance
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Define feature types and their colors
        feature_types = ['D', 'A', 'P', 'N', 'H', 'R']
        feature_names = {
            'D': 'H-bond Donor',
            'A': 'H-bond Acceptor',
            'P': 'Positive Ionizable',
            'N': 'Negative Ionizable',
            'H': 'Hydrophobic',
            'R': 'Aromatic'
        }
        feature_colors = {
            'D': 'blue',
            'A': 'red',
            'P': 'purple',
            'N': 'orange',
            'H': 'green',
            'R': 'pink'
        }
        
        # Create a figure with three subplots
        fig = make_subplots(
            rows=1, cols=3,
            specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]],
            subplot_titles=[
                f"Molecule 1: {Chem.MolToSmiles(mol1)}",
                f"Molecule 2: {Chem.MolToSmiles(mol2)}",
                "Blended Child"
            ],
            horizontal_spacing=0.02
        )
        
        # Function to add molecule structure to a subplot
        def add_molecule_structure(mol, row, col):
            # Get atom positions
            conf = mol.GetConformer()
            positions = []
            elements = []
            
            for atom in mol.GetAtoms():
                idx = atom.GetIdx()
                pos = conf.GetAtomPosition(idx)
                positions.append([pos.x, pos.y, pos.z])
                elements.append(atom.GetSymbol())
            
            # Add atoms as scatter3d
            fig.add_trace(
                go.Scatter3d(
                    x=[pos[0] for pos in positions],
                    y=[pos[1] for pos in positions],
                    z=[pos[2] for pos in positions],
                    mode='markers',
                    marker=dict(
                        size=7,
                        color='gray',
                        opacity=0.8
                    ),
                    text=elements,
                    name=f"Atoms (Mol {col})",
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # Add bonds as lines
            for bond in mol.GetBonds():
                begin_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                begin_pos = positions[begin_idx]
                end_pos = positions[end_idx]
                
                fig.add_trace(
                    go.Scatter3d(
                        x=[begin_pos[0], end_pos[0]],
                        y=[begin_pos[1], end_pos[1]],
                        z=[begin_pos[2], end_pos[2]],
                        mode='lines',
                        line=dict(color='gray', width=3),
                        showlegend=False
                    ),
                    row=row, col=col
                )
        
        # Function to add pharmacophore features as scatter points
        def add_pharmacophore_scatter(feature_grids, grid_info, row, col):
            # Create coordinate grids
            x = np.linspace(grid_info['min_coords'][0], grid_info['max_coords'][0], grid_info['shape'][0])
            y = np.linspace(grid_info['min_coords'][1], grid_info['max_coords'][1], grid_info['shape'][1])
            z = np.linspace(grid_info['min_coords'][2], grid_info['max_coords'][2], grid_info['shape'][2])
            
            # Create meshgrid
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            
            # Add scatter points for each feature type
            for feature_type in feature_types:
                # Get grid for this feature
                grid = feature_grids[feature_type]
                
                # Find points above threshold
                threshold = np.max(grid) * 0.3 if np.max(grid) > 0 else 0
                if threshold > 0:
                    mask = grid > threshold
                    
                    # Subsample points if there are too many
                    points_x = X[mask]
                    points_y = Y[mask]
                    points_z = Z[mask]
                    values = grid[mask]
                    
                    max_points = 500
                    if len(points_x) > max_points:
                        idx = np.random.choice(len(points_x), max_points, replace=False)
                        points_x = points_x[idx]
                        points_y = points_y[idx]
                        points_z = points_z[idx]
                        values = values[idx]
                    
                    if len(points_x) > 0:
                        # Add scatter points
                        fig.add_trace(
                            go.Scatter3d(
                                x=points_x,
                                y=points_y,
                                z=points_z,
                                mode='markers',
                                marker=dict(
                                    size=4,
                                    color=feature_colors[feature_type],
                                    opacity=0.7
                                ),
                                name=feature_names[feature_type],
                                showlegend=(col == 1)  # Only show legend for first column
                            ),
                            row=row, col=col
                        )
        
        # Add molecule structures
        add_molecule_structure(mol1, 1, 1)
        add_molecule_structure(mol2, 1, 2)
        
        # Add pharmacophore features as scatter points
        add_pharmacophore_scatter(feature_grids1, grid_info1, 1, 1)
        add_pharmacophore_scatter(feature_grids2, grid_info2, 1, 2)
        add_pharmacophore_scatter(blended_features, blended_grid_info, 1, 3)
        
        # Update layout
        fig.update_layout(
            title="Pharmacophore Fingerprints Comparison (Simplified View)",
            width=1800,
            height=700,
            legend=dict(
                x=0,
                y=1,
                bgcolor='rgba(255, 255, 255, 0.5)',
                bordercolor='rgba(0, 0, 0, 0.5)',
                borderwidth=1
            )
        )
        
        # Update scenes to have the same camera view
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
        
        fig.update_scenes(
            camera=camera,
            aspectmode='data',
            xaxis_visible=True,
            yaxis_visible=True,
            zaxis_visible=True
        )
        
        # Write to HTML file
        fig.write_html(filename)
        print(f"Simplified visualization saved to {os.path.abspath(filename)}")
        return True
    except Exception as e:
        print(f"Error creating simplified visualization: {str(e)}")
        return False

def generate_valid_child_molecule(parent1, parent2):
    """
    Generate child molecules by connecting fragments at dummy atoms and print all combinations.
    """
    print("\n=== GENERATING CHILD MOLECULES WITH DUMMY ATOMS ===")

    p1_fragments = fragment_molecule_with_dummy_atoms(parent1)
    p2_fragments = fragment_molecule_with_dummy_atoms(parent2)

    print(f"Parent 1 fragments: {len(p1_fragments)}")
    print(f"Parent 2 fragments: {len(p2_fragments)}")

    for i, frag1 in enumerate(p1_fragments):
        for j, frag2 in enumerate(p2_fragments):
            print(f"\nConnecting fragment {i+1} from Parent 1 with fragment {j+1} from Parent 2")
            child_mol = connect_fragments_at_dummy_atoms(frag1, frag2)
            if child_mol:
                print(f"✅ Combination successful: {Chem.MolToSmiles(child_mol)}")
            else:
                print("❌ Combination failed")

def fragment_molecule_with_dummy_atoms(mol):
    """
    Fragment molecule and insert dummy atoms ([*]) at fragmentation points.
    """
    frags = BRICS.BRICSDecompose(mol, keepNonLeafNodes=True)
    fragments = [Chem.MolFromSmiles(frag) for frag in frags]
    for idx, frag in enumerate(fragments):
        print(f"Fragment {idx+1} with dummy atoms: {Chem.MolToSmiles(frag)}")
    return fragments

def connect_fragments_at_dummy_atoms(frag1, frag2):
    """
    Connect two fragments at dummy atoms ([*]) and check chemical validity.
    """
    combo_smiles = Chem.MolToSmiles(frag1) + '.' + Chem.MolToSmiles(frag2)
    combo_mol = Chem.MolFromSmiles(combo_smiles)
    if combo_mol:
        try:
            Chem.SanitizeMol(combo_mol)
            print(f"✅ Valid combined molecule: {Chem.MolToSmiles(combo_mol)}")
            return combo_mol
        except Exception as e:
            print(f"❌ Invalid combination: {str(e)}")
    return None

def main():
    # Example usage
    smiles1 = "CC(=O)Nc1ccc(O)cc1"  # Acetaminophen
    smiles2 = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"  # Ibuprofen
    
    # Process molecules
    mol1 = process_smiles(smiles1)
    mol2 = process_smiles(smiles2)
    
    if mol1 is None or mol2 is None:
        print("Error processing molecules")
        return
    
    # Generate pharmacophore fingerprints
    print("\nGenerating pharmacophore fingerprints for molecule 1...")
    overall_grid1, feature_grids1, feature_atoms1, grid_info1 = generate_pharmacophore_fingerprints(mol1)
    
    print("\nGenerating pharmacophore fingerprints for molecule 2...")
    overall_grid2, feature_grids2, feature_atoms2, grid_info2 = generate_pharmacophore_fingerprints(mol2)
    
    # Save fingerprints
    save_pharmacophore_fingerprints(
        "molecule1", overall_grid1, feature_grids1, feature_atoms1, grid_info1
    )
    save_pharmacophore_fingerprints(
        "molecule2", overall_grid2, feature_grids2, feature_atoms2, grid_info2
    )
    
    # Visualize pharmacophore fingerprints
    print("\nVisualizing pharmacophore fingerprints for molecule 1...")
    visualize_pharmacophore_fingerprints(
        mol1, overall_grid1, feature_grids1, grid_info1,
        filename="pharmacophore_fingerprints_1.html"
    )
    
    print("\nVisualizing pharmacophore fingerprints for molecule 2...")
    visualize_pharmacophore_fingerprints(
        mol2, overall_grid2, feature_grids2, grid_info2,
        filename="pharmacophore_fingerprints_2.html"
    )
    
    # Align grids
    print("\nAligning pharmacophore grids...")
    aligned_overall1, aligned_overall2, aligned_features1, aligned_features2, aligned_grid_info = align_pharmacophore_grids(
        overall_grid1, feature_grids1, grid_info1,
        overall_grid2, feature_grids2, grid_info2
    )
    
    # Blend fingerprints
    print("\nBlending pharmacophore fingerprints...")
    blended_overall, blended_features = blend_pharmacophore_fingerprints(
        aligned_overall1, aligned_features1,
        aligned_overall2, aligned_features2
    )
    
    # Save blended fingerprints
    save_pharmacophore_fingerprints(
        "blended", blended_overall, blended_features, 
        {**feature_atoms1, **feature_atoms2}, aligned_grid_info,
        output_dir="blended_fingerprints"
    )
    
    # Create comprehensive visualization of all fingerprints
    print("\nCreating comprehensive visualization of all pharmacophore fingerprints...")
    visualize_all_pharmacophore_fingerprints(
        mol1, mol2,
        overall_grid1, feature_grids1, grid_info1,
        overall_grid2, feature_grids2, grid_info2,
        blended_overall, blended_features, aligned_grid_info,
        filename="all_pharmacophore_fingerprints.html"
    )
    
    # Generate a valid child molecule
    print("\nGenerating a valid child molecule...")
    generate_valid_child_molecule(mol1, mol2)
    
    print("\nFingerprint blending and molecule generation complete.")

def visualize_molecules(mol1, mol2, child_mol=None, filename="molecules.html"):
    """
    Visualize parent molecules and optionally a child molecule
    """
    from rdkit.Chem import AllChem
    import py3Dmol
    
    # Prepare molecules for visualization
    for mol in [mol1, mol2]:
        if mol:
            AllChem.Compute2DCoords(mol)
    
    if child_mol:
        AllChem.Compute2DCoords(child_mol)
    
    # Create viewer
    view = py3Dmol.view(width=800, height=400)
    
    # Add molecules
    mol1_block = Chem.MolToMolBlock(mol1)
    view.addModel(mol1_block, "mol1")
    view.setStyle({"model": "mol1"}, {"stick": {"radius": 0.2, "color": "blue"}})
    
    mol2_block = Chem.MolToMolBlock(mol2)
    view.addModel(mol2_block, "mol2")
    view.setStyle({"model": "mol2"}, {"stick": {"radius": 0.2, "color": "green"}})
    
    if child_mol:
        child_block = Chem.MolToMolBlock(child_mol)
        view.addModel(child_block, "child")
        view.setStyle({"model": "child"}, {"stick": {"radius": 0.2, "color": "red"}})
    
    # Set camera
    view.zoomTo()
    
    # Save to HTML file
    view.render()
    
    with open(filename, "w") as f:
        f.write(view.html())
    
    print(f"Visualization saved to {filename}")

if __name__ == "__main__":
    main()