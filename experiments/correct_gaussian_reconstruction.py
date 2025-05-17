import argparse
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit import RDLogger
import threading
import time
import os
import py3Dmol

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

def calculate_gaussian_overlap(mol1, mol2, grid_spacing=0.5, sigma=1.0, return_fingerprints=False):
    """
    Calculate Gaussian shape overlap between two molecules
    
    This is a simplified version that:
    1. Aligns the molecules by their centers of mass
    2. Creates a grid around both molecules
    3. Computes Gaussian functions at each grid point for both molecules
    4. Calculates the overlap as the dot product of the Gaussian grids
    
    Args:
        mol1: First RDKit molecule
        mol2: Second RDKit molecule
        grid_spacing: Spacing between grid points
        sigma: Width of Gaussian functions
        return_fingerprints: Whether to return the fingerprints
        
    Returns:
        If return_fingerprints=False: Tanimoto similarity
        If return_fingerprints=True: (Tanimoto similarity, fingerprint1, fingerprint2, grid_info)
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
        
        for atom in mol1.GetAtoms():
            pos = conf1.GetAtomPosition(atom.GetIdx())
            pos1.append(np.array([pos.x, pos.y, pos.z]) - com1)
            radii1.append(Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()))
        
        for atom in mol2.GetAtoms():
            pos = conf2.GetAtomPosition(atom.GetIdx())
            pos2.append(np.array([pos.x, pos.y, pos.z]) - com2)
            radii2.append(Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()))
        
        # Determine grid dimensions
        all_pos = np.vstack(pos1 + pos2)
        min_coords = np.min(all_pos, axis=0) - 3.0
        max_coords = np.max(all_pos, axis=0) + 3.0
        x = np.arange(min_coords[0], max_coords[0], grid_spacing)
        y = np.arange(min_coords[1], max_coords[1], grid_spacing)
        z = np.arange(min_coords[2], max_coords[2], grid_spacing)
        
        print(f"Grid dimensions: {len(x)} x {len(y)} x {len(z)}")
        
        # Initialize grids
        grid1 = np.zeros((len(x), len(y), len(z)))
        grid2 = np.zeros((len(x), len(y), len(z)))
        
        # Compute Gaussian contributions for molecule 1
        for i, atom_pos in enumerate(pos1):
            radius = radii1[i]
            for ix, px in enumerate(x):
                for iy, py in enumerate(y):
                    for iz, pz in enumerate(z):
                        grid_pos = np.array([px, py, pz])
                        dist = np.linalg.norm(grid_pos - atom_pos)
                        grid1[ix, iy, iz] += np.exp(-(dist**2) / (2 * (sigma * radius)**2))
        
        # Compute Gaussian contributions for molecule 2
        for i, atom_pos in enumerate(pos2):
            radius = radii2[i]
            for ix, px in enumerate(x):
                for iy, py in enumerate(y):
                    for iz, pz in enumerate(z):
                        grid_pos = np.array([px, py, pz])
                        dist = np.linalg.norm(grid_pos - atom_pos)
                        grid2[ix, iy, iz] += np.exp(-(dist**2) / (2 * (sigma * radius)**2))
        
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

def main():
    # Use hardcoded SMILES strings
    smiles1 = SMILES1
    smiles2 = SMILES2
    
    print(f"Processing SMILES 1: {smiles1}")
    mol1 = process_smiles(smiles1)
    if mol1 is None:
        print("Failed to process first molecule")
        return
    
    print(f"Processing SMILES 2: {smiles2}")
    mol2 = process_smiles(smiles2)
    if mol2 is None:
        print("Failed to process second molecule")
        return
    
    print("\nCalculating USR descriptors...")
    usr_desc1 = calculate_usr_descriptors(mol1)
    usr_desc2 = calculate_usr_descriptors(mol2)
    
    if usr_desc1 is not None and usr_desc2 is not None:
        usr_sim = calculate_usr_similarity(usr_desc1["usr"], usr_desc2["usr"])
        usrcat_sim = calculate_usr_similarity(usr_desc1["usrcat"], usr_desc2["usrcat"])
        
        print(f"USR Similarity: {usr_sim:.4f}")
        print(f"USRCAT Similarity: {usrcat_sim:.4f}")
    
    print("\nCalculating shape moments...")
    moments1 = calculate_shape_moments(mol1)
    moments2 = calculate_shape_moments(mol2)
    
    if moments1 is not None and moments2 is not None:
        print("\nShape moments for molecule 1:")
        print(f"PMI: {moments1['pmi']}")
        print(f"NPR: {moments1['npr']}")
        
        print("\nShape moments for molecule 2:")
        print(f"PMI: {moments2['pmi']}")
        print(f"NPR: {moments2['npr']}")
        
        # Calculate Euclidean distance between NPR points
        npr_dist = np.sqrt((moments1['npr'][0] - moments2['npr'][0])**2 + 
                           (moments1['npr'][1] - moments2['npr'][1])**2)
        print(f"\nNPR Distance: {npr_dist:.4f}")
    
    print("\nCalculating Gaussian shape overlap...")
    overlap, fp1, fp2, grid_info, grid1_3d, grid2_3d = calculate_gaussian_overlap(mol1, mol2, return_fingerprints=True)
    if overlap is not None:
        print(f"Gaussian Shape Overlap (Tanimoto): {overlap:.4f}")
        
        # Print fingerprint information
        print(f"\nGSO Fingerprint 1 shape: {fp1.shape}")
        print(f"GSO Fingerprint 2 shape: {fp2.shape}")
        
        # Print first 20 values of each fingerprint
        print("\nFirst 20 values of GSO Fingerprint 1:")
        for i, val in enumerate(fp1[:20]):
            print(f"  {i}: {val:.6f}")
        
        print("\nFirst 20 values of GSO Fingerprint 2:")
        for i, val in enumerate(fp2[:20]):
            print(f"  {i}: {val:.6f}")
        
        # Calculate some statistics
        print("\nFingerprint statistics:")
        print(f"Fingerprint 1 - Min: {fp1.min():.6f}, Max: {fp1.max():.6f}, Mean: {fp1.mean():.6f}")
        print(f"Fingerprint 2 - Min: {fp2.min():.6f}, Max: {fp2.max():.6f}, Mean: {fp2.mean():.6f}")
        
        # Save fingerprints to file
        save_fingerprints_to_file(fp1, fp2, grid_info)
        
        # Demonstrate shape reconstruction
        print("\nReconstructing shapes from fingerprints...")
        points1, values1 = reconstruct_shape_from_fingerprint(fp1, grid_info)
        points2, values2 = reconstruct_shape_from_fingerprint(fp2, grid_info)
        
        print(f"Reconstructed shape 1: {len(points1)} points above threshold")
        print(f"Reconstructed shape 2: {len(points2)} points above threshold")
        
        # Visualize the fingerprint shapes
        try:
            visualize_fingerprint_shape([grid1_3d, grid2_3d], grid_info)
        except Exception as e:
            print(f"Could not create fingerprint visualization: {str(e)}")
    
    print("\nCreating 3D visualization...")
    visualize_molecules(mol1, mol2)
    
    print("\nSummary:")
    if usr_desc1 is not None and usr_desc2 is not None:
        print(f"USR Similarity: {usr_sim:.4f}")
        print(f"USRCAT Similarity: {usrcat_sim:.4f}")
    if moments1 is not None and moments2 is not None:
        print(f"NPR Distance: {npr_dist:.4f}")
    if overlap is not None:
        print(f"Gaussian Shape Overlap: {overlap:.4f}")
        
    print("\nCan the shape be reconstructed from the fingerprints?")
    print("Yes, the shape can be reconstructed from the GSO fingerprints because:")
    print("1. The fingerprints are flattened 3D grids that represent the Gaussian shape functions")
    print("2. We store the grid dimensions and coordinates along with the fingerprints")
    print("3. To reconstruct the shape, we:")
    print("   a. Reshape the flattened fingerprint back to a 3D grid")
    print("   b. Map each grid point to its 3D coordinates")
    print("   c. Apply a threshold to identify points that are part of the shape")
    print("   d. Visualize these points as an isosurface")
    print("4. The reconstructed shape represents the molecular volume as a Gaussian density")
    print("5. This is similar to electron density maps used in crystallography")
    print("\nThe fingerprint_shape.html file shows the reconstructed shapes from the fingerprints.")

if __name__ == "__main__":
    main()