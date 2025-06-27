import pickle as pkl
from shape_utils import get_atom_stamp
from shape_utils import get_shape
from rdkit import Chem
from shape_utils import ROTATIONS
from random import sample
from shape_utils import centralize
from shape_utils import get_mol_centroid
from shape_utils import trans
from rdkit.Chem import rdMolTransforms
import copy
from shape_utils import get_binary_features
from tfbio_data import make_grid
import numpy as np
import os
import glob
import time
import gc 
from concurrent.futures import ProcessPoolExecutor
from math import ceil

BASE_DIR = "/home/luost_local/sdivita/synformer"
CAVITY_DIR = "/home/luost_local/sdivita/synformer/posebuster/posebusters_benchmark_set/cavity"
RECEPTOR_DIR = "/home/luost_local/sdivita/synformer/posebuster/posebusters_benchmark_set"
OUTPUT_DIR = os.path.join(BASE_DIR, "data/desert/posebuster_shapes")
print("ok")
os.makedirs(OUTPUT_DIR, exist_ok=True)


data_path = os.path.join(BASE_DIR, 'data/desert/1.pkl')
with open(data_path, 'rb') as fr:
    data = pkl.load(fr)
atom_stamp = get_atom_stamp(grid_resolution=0.5, max_dist=4.0)

def process_molecule(cavity_file, protein_file):
    try:
        start_time = time.time()
        print(f"Processing {os.path.basename(cavity_file)}...")
        
       
        cavity = Chem.MolFromPDBFile(cavity_file, proximityBonding=False)
        if cavity is None:
            print(f"Failed to load cavity file: {cavity_file}")
            return None
        
        protein = Chem.MolFromPDBFile(protein_file, proximityBonding=False)
        if protein is None:
            print(f"Failed to load protein file: {protein_file}")
            return None

        cavity_centroid = get_mol_centroid(cavity)
        cavity = centralize(cavity)
        translation = trans(-cavity_centroid[0], -cavity_centroid[1], -cavity_centroid[2]) 
        protein_conformer = protein.GetConformer()
        rdMolTransforms.TransformConformer(protein_conformer, translation)

        sample_shapes = []
        sample_n_o_f = []
        
        for rotation_idx in range(24):
            rotation_start = time.time()
            print(f"  Rotation {rotation_idx+1}/24 (took {time.time() - start_time:.2f}s so far)")

            copied_cavity = copy.deepcopy(cavity)
            copied_protein = copy.deepcopy(protein)

            cavity_conformer = copied_cavity.GetConformer()
            protein_conformer = copied_protein.GetConformer()

            rotation_mat = ROTATIONS[rotation_idx]
            rotation = np.zeros((4, 4))
            rotation[:3, :3] = rotation_mat
            rdMolTransforms.TransformConformer(cavity_conformer, rotation)
            rdMolTransforms.TransformConformer(protein_conformer, rotation)

            # print("    Getting cavity shape...")
            curr_cavity_shape = get_shape(copied_cavity, atom_stamp, 0.5, 15)
            large_cavity_shape = np.zeros((61*3, 61*3, 61*3))
            large_cavity_shape[61*1:61*2,61*1:61*2,61*1:61*2] = curr_cavity_shape

            # print("    Getting protein features...")
            protein_coords, protein_features = get_binary_features(copied_protein, -1, False)
            protein_grid, feature_dict = make_grid(protein_coords, protein_features, 0.5, 15)
            protein_grid = protein_grid.squeeze()
            
            # print("    Creating N/O/F grid...")
            n_o_f_grid = np.zeros(protein_grid.shape)
            for xyz in feature_dict[(7.0,)] + feature_dict[(8.0,)] + feature_dict[(9.0,)]:
                x, y, z = xyz[0], xyz[1], xyz[2]
                
                x_left = x - 4 if x - 4 >=0 else 0
                x_right = x + 4 if x + 4 < protein_grid.shape[0] else protein_grid.shape[0] - 1
                
                y_left = y - 4 if y - 4 >=0 else 0
                y_right = y + 4 if y + 4 < protein_grid.shape[0] else protein_grid.shape[0] - 1

                z_left = z - 4 if z - 4 >=0 else 0
                z_right = z + 4 if z + 4 < protein_grid.shape[0] else protein_grid.shape[0] - 1

                tmp = n_o_f_grid[x_left: x_right + 1, y_left: y_right + 1, z_left: z_right + 1]
                tmp += 1
            large_n_o_f_grid = np.zeros((61*3, 61*3, 61*3))
            large_n_o_f_grid[61*1:61*2,61*1:61*2,61*1:61*2] = n_o_f_grid
            
            # print("    Sampling molecules...")
            seed_data = sample(data, 10)
            union_shape = np.zeros((61, 61, 61))
            for seed_idx, seed in enumerate(seed_data):
                # print(f"      Processing seed {seed_idx+1}/10")
                mol_shape = get_shape(centralize(seed[0]), atom_stamp, 0.5, 15)
                union_shape = union_shape + mol_shape
            union_shape[union_shape>1]=1

            # print("    Finding intersections...")
            flag = False
            inter_shape = None
            for j in range(0, 122):
                cavity_view_slice = large_cavity_shape[j: j + 61, j: j + 61, j: j + 61]
                current_inter_slice = cavity_view_slice * union_shape
                
                if current_inter_slice.sum() > 2400:
                    flag = True
                    inter_shape = np.zeros_like(large_cavity_shape)
                    inter_shape[j: j + 61, j: j + 61, j: j + 61] = current_inter_slice
                    break
            
            if flag and inter_shape is not None:
                # print("    Intersection found, extracting shape...")
                inter_idx = np.where(inter_shape > 0)
                x, y, z = inter_idx[0].mean(), inter_idx[1].mean(), inter_idx[2].mean()
                x, y, z = int(x.round()), int(y.round()), int(z.round())
                x_left, x_right = x - 13, x + 14 + 1
                y_left, y_right = y - 13, y + 14 + 1
                z_left, z_right = z - 13, z + 14 + 1
                inter_shape = inter_shape[x_left: x_right, y_left: y_right, z_left: z_right]
                inter_n_o_f = large_n_o_f_grid[x_left: x_right, y_left: y_right, z_left: z_right]
                sample_shapes.append(inter_shape)
                sample_n_o_f.append(inter_n_o_f)
            
            # Force garbage collection after each rotation to free memory
            gc.collect()

        # print("  Getting final cavity shape...")
        sample_shapes.append(get_shape(cavity, atom_stamp, 0.5, 6.75))

        # print("  Getting final protein features...")
        protein_coords, protein_features = get_binary_features(protein, -1, False)
        protein_grid, feature_dict = make_grid(protein_coords, protein_features, 0.5, 6.75)
        protein_grid = protein_grid.squeeze()
        n_o_f_grid = np.zeros(protein_grid.shape)
        for xyz in feature_dict[(7.0,)] + feature_dict[(8.0,)] + feature_dict[(9.0,)]:
            x, y, z = xyz[0], xyz[1], xyz[2]
            
            x_left = x - 4 if x - 4 >=0 else 0
            x_right = x + 4 if x + 4 < protein_grid.shape[0] else protein_grid.shape[0] - 1
            
            y_left = y - 4 if y - 4 >=0 else 0
            y_right = y + 4 if y + 4 < protein_grid.shape[0] else protein_grid.shape[0] - 1

            z_left = z - 4 if z - 4 >=0 else 0
            z_right = z + 4 if z + 4 < protein_grid.shape[0] else protein_grid.shape[0] - 1

            tmp = n_o_f_grid[x_left: x_right + 1, y_left: y_right + 1, z_left: z_right + 1]
            tmp += 1
        sample_n_o_f.append(n_o_f_grid)
        
        # print(f"  Processing completed in {time.time() - start_time:.2f}s")
        return sample_shapes, sample_n_o_f
        
    except Exception as e:
        print(f"Error processing molecule: {str(e)}")
        return None

def process_batch(batch_files):
    results = []
    for cavity_file in batch_files:
        cavity_basename = os.path.basename(cavity_file)
        output_basename = cavity_basename.replace("_protein_cavity.pdb", "")
        
        print(f"\nProcessing: {cavity_basename}")
        print(f"Output basename: {output_basename}")
        
        pdb_id = output_basename  
        protein_file = os.path.join(RECEPTOR_DIR, pdb_id, f"{pdb_id}_protein.pdb")
        print(f"Looking for protein file at: {protein_file}")
        
        if not os.path.exists(protein_file):
            print(f"Warning: No matching receptor file found for {cavity_file}")
            print(f"Directory exists: {os.path.exists(os.path.dirname(protein_file))}")
            continue
            
        result = process_molecule(cavity_file, protein_file)
        if result is None:
            continue
            
        sample_shapes, sample_n_o_f = result
        
        shapes_output = os.path.join(OUTPUT_DIR, f"{output_basename}_shapes.pkl")
        with open(shapes_output, 'wb') as fw:
            pkl.dump(sample_shapes, fw)
            
        gc.collect()

def main():
    cavity_files = glob.glob(os.path.join(CAVITY_DIR, "*_protein_cavity.pdb"))  
    print(f"Found {len(cavity_files)} cavity files")
    
    processed_files = set(os.path.splitext(f)[0].replace('_shapes', '') 
                        for f in os.listdir(OUTPUT_DIR) if f.endswith('_shapes.pkl'))
    
    unprocessed_files = [f for f in cavity_files 
                        if os.path.basename(f).replace("_protein_cavity.pdb", "") not in processed_files]  
    print(f"Found {len(unprocessed_files)} unprocessed files")
    
    if not unprocessed_files:
        print("No files to process")
        return
        
    batch_size = ceil(len(unprocessed_files) / 20)
    batches = [unprocessed_files[i:i + batch_size] for i in range(0, len(unprocessed_files), batch_size)]
    
    print(f"Split into {len(batches)} batches of approximately {batch_size} files each")
    
    with ProcessPoolExecutor(max_workers=20) as executor:
        executor.map(process_batch, batches)

if __name__ == "__main__":
    main()